"""Schneller Entwicklungs-Check: gleicher Ablauf wie benchmark.py, aber auf einer Teilmenge der Frequenzen.

Misst Tracen/Kompilieren und Laufzeit von Forward und Gradient, die Speicherspitze, und speichert
die Ergebnisse, damit man Code-Staende miteinander vergleichen kann.

    python tools/dev_check.py --stride 10 --save runs/dev/ref.npz
    python tools/dev_check.py --stride 10 --compare runs/dev/ref.npz

Die Kompilierzeit haengt kaum von der Frequenzzahl ab, die Laufzeit etwa linear.
Fuer die gewertete Zahl immer den vollen Benchmark nehmen.
"""
import argparse
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import benchmark as bm  # setzt dieselben Umgebungsvariablen wie der Benchmark
import jax
import jax.numpy as jnp
import numpy as np
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces


def peak_rss_gib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**30 if sys.platform == "darwin" else rss / 2**20


def timed(label, fn, times):
    start = perf_counter()
    result = jax.block_until_ready(fn())
    times[label] = perf_counter() - start
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=10, help="jede n-te der 101 Frequenzen (1 = alle)")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    data = REPO / "examples" / "data"
    fixture = json.loads((data / "uifo_800_3000.json").read_text())
    reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())

    times = {}
    start = perf_counter()
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, bm.PROPERTIES)
    initial = []
    for pair in pairs:
        name, prop = pair[0] if isinstance(pair[0], list) else pair
        initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop])
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    frequencies = jnp.asarray(reference["frequencies"], dtype=jnp.float64)[:: args.stride]
    expected = jnp.asarray(reference["sensitivities"], dtype=jnp.float64)[:: args.stride]
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays["optimized_parameters"] = parameters
    ports = jnp.asarray(ports)
    one = jnp.asarray(1.0, dtype=jnp.float64)
    times["build"] = perf_counter() - start

    lowered = timed("forward_lower", lambda: jax.jit(bm.forward_with_pullback).lower(parameters, arrays), times)
    forward = timed("forward_compile", lowered.compile, times)
    outputs, pullback = timed("forward_run", lambda: forward(parameters, arrays), times)
    sens_fn = jax.jit(bm.sensitivity_with_pullback).lower(outputs, ports, expected).compile()
    loss, readout_pullback, (sensitivity, powers) = sens_fn(outputs, ports, expected)
    lowered = timed("gradient_lower", lambda: jax.jit(bm.backward).lower(pullback, readout_pullback, one), times)
    gradient_fn = timed("gradient_compile", lowered.compile, times)
    gradient = timed("gradient_run", lambda: gradient_fn(pullback, readout_pullback, one), times)
    # zweiter Lauf = warm
    timed("forward_warm", lambda: forward(parameters, arrays), times)
    timed("gradient_warm", lambda: gradient_fn(pullback, readout_pullback, one), times)

    log_error = float(jnp.max(jnp.abs(jnp.log(sensitivity) - jnp.log(expected))))
    print(f"Frequenzen: {len(frequencies)}  |  max |log(s) - log(ref)| = {log_error:.2e}  (Grenze 1e-5)")
    for key in ("build", "forward_lower", "forward_compile", "forward_run", "forward_warm",
                "gradient_lower", "gradient_compile", "gradient_run", "gradient_warm"):
        print(f"  {key:18s} {times[key]:8.3f} s")
    compile_total = sum(times[k] for k in ("forward_lower", "forward_compile", "gradient_lower", "gradient_compile"))
    print(f"  {'kompilieren gesamt':18s} {compile_total:8.3f} s")
    print(f"  {'Speicherspitze':18s} {peak_rss_gib():8.2f} GiB")

    result = {
        "carrier": np.asarray(outputs[0]), "signal": np.asarray(outputs[1]), "noise": np.asarray(outputs[2]),
        "sensitivity": np.asarray(sensitivity), "powers": np.asarray(powers),
        "loss": np.asarray(loss), "gradient": np.asarray(gradient),
    }
    finite = all(np.all(np.isfinite(v)) for v in result.values())
    print(f"  alles endlich: {finite}, Gradient nicht null: {bool(np.any(result['gradient'] != 0))}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save, **result, times=json.dumps(times))
        print(f"gespeichert: {args.save}")
    if args.compare:
        ref = np.load(args.compare)
        print(f"Vergleich mit {args.compare} (max. Abweichung relativ zum Betrags-Maximum):")
        # Grenzen: deutlich unter der Benchmark-Toleranz (1e-5 im log der Sensitivitaet), aber
        # grosszuegig genug fuer Rundungsunterschiede, die schlecht konditionierte LU-Loesungen verstaerken.
        limits = {"sensitivity": 1e-7, "noise": 1e-7, "signal": 1e-6, "carrier": 1e-9, "powers": 1e-9, "gradient": 1e-5}
        ok = True
        for key, value in result.items():
            if key == "loss":  # der Loss liegt nahe null, daher absolut vergleichen
                diff = float(np.abs(value - ref[key]))
                ok &= diff < 1e-9
                print(f"  {key:12s} {diff:.2e} (absolut)")
                continue
            scale = max(np.max(np.abs(ref[key])), 1e-300)
            diff = float(np.max(np.abs(value - ref[key])) / scale)
            ok &= diff < limits[key]
            print(f"  {key:12s} {diff:.2e}")
        print("OK" if ok else "ABWEICHUNG pruefen")


if __name__ == "__main__":
    main()
