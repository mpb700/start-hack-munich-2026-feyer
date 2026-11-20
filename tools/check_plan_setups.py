"""Vergleicht den Plan-Pfad von `simulate` mit dem urspruenglichen Pfad (plan=None) auf mehreren Setups.

Abgedeckt: Frequenz-Sweep (aLIGO, Voyager mit Homodyn-Detektoren, uifo), Carrier-Sweep
(Laserleistung, mehrere Carrier-Zeilen) und Parameter-Linking. Verglichen werden Carrier,
Signal, Rauschen und der Gradient eines skalaren Werts nach einigen Designvariablen.

    python tools/check_plan_setups.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import differometor as df
from differometor.setups import aligo, uifo, voyager

jax.config.update("jax_enable_x64", True)


def nodes_of(setup, component):
    return [name for name, data in setup.nodes(data=True) if data["component"] == component]


def compare(label, setup, changing_pairs, changing_values, optimization_pairs):
    arrays, *_ = df.run_build_step(setup, changing_pairs, changing_values, optimization_pairs)
    # Startwerte der Designvariablen aus dem Setup, wie im Benchmark
    order = np.argsort(np.asarray(arrays["optimized_value_indices"]), kind="stable")
    indices = np.asarray(arrays["optimized_parameter_indices"])[order]
    _, first = np.unique(np.asarray(arrays["optimized_value_indices"])[order], return_index=True)
    start = jnp.real(arrays["parameters"][0, indices[first]])

    def scalar(p, plan):
        carrier, signal, noise = df.simulate(**dict(arrays, optimized_parameters=p, plan=plan))
        return jnp.sum(jnp.log(noise)) + jnp.sum(jnp.abs(signal) ** 2) + jnp.sum(jnp.abs(carrier) ** 2)

    plan = arrays["plan"]
    new = jax.jit(lambda p: df.simulate(**dict(arrays, optimized_parameters=p)))(start)
    old = jax.jit(lambda p: df.simulate(**dict(arrays, optimized_parameters=p, plan=None)))(start)
    worst = 0.0
    text = []
    for name, a, b in zip(("carrier", "signal", "noise"), new, old):
        diff = float(jnp.max(jnp.abs(a - b)) / max(float(jnp.max(jnp.abs(b))), 1e-300))
        worst = max(worst, diff)
        text.append(f"{name} {diff:.1e}")
    if len(optimization_pairs):
        g_new = jax.jit(jax.grad(lambda p: scalar(p, plan)))(start)
        g_old = jax.jit(jax.grad(lambda p: scalar(p, None)))(start)
        diff = float(jnp.max(jnp.abs(g_new - g_old)) / max(float(jnp.max(jnp.abs(g_old))), 1e-300))
        worst = max(worst, diff)
        text.append(f"gradient {diff:.1e}")
    status = "OK" if worst < 1e-6 else "ABWEICHUNG"
    print(f"{status:10s} {label:42s} " + ", ".join(text))
    return worst < 1e-6


frequencies = jnp.linspace(10, 3000, 9).reshape(1, -1)
results = []

for name, (setup, *_) in (("aLIGO", aligo()), ("Voyager (Homodyn)", voyager()), ("uifo 2x2 (zufaellig)", uifo(2, random=True, random_seed=3))):
    mirrors = nodes_of(setup, "mirror")[:3]
    pairs = [(m, "tuning") for m in mirrors] + [(m, "reflectivity") for m in mirrors]
    results.append(compare(f"{name}, Frequenz-Sweep", setup, [("f", "frequency")], frequencies, pairs))
    lasers = nodes_of(setup, "laser")
    if lasers:
        powers = jnp.linspace(1.0, 3.0, 4).reshape(1, -1)
        results.append(compare(f"{name}, Carrier-Sweep Laserleistung", setup, [(lasers[0], "power")], powers, pairs))

print("ALLE OK" if all(results) else "FEHLER")
sys.exit(0 if all(results) else 1)
