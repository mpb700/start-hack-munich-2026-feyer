"""Do the compile stages stay at zero when the setup changes from run to run?

Run from the repo directory, with the repo first on the path so that the argparse carrier is found:
    PYTHONPATH=. python runs/noel/switch_setups.py            all setups one after the other in ONE process
    PYTHONPATH=. python runs/noel/switch_setups.py --only N   only setup N (a cold run in a fresh process)
    PYTHONPATH=. python runs/noel/switch_setups.py --count    print the number of setups
The setups come from runs/noel/switch_cases.py: the public fixture, aLIGO in three modes, Voyager,
random grids of three sizes with random design values (about half of them read two detector ports),
the fixture with 37 frequencies, and the fixture once more.
Every setup goes through the stages of benchmark.py with fresh jit objects: lower, compile and run
of the forward program, of the readout and of the gradient. Per program it prints whether the
NumPy interpreter took it (i) or XLA compiled it (X), and the lower and compile times in ms."""
from time import perf_counter
T0 = perf_counter()
import argparse
from datetime import datetime, timezone
import hashlib, importlib.metadata, json, os
from pathlib import Path
import platform, statistics, subprocess
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
IMPORT = (perf_counter() - T0) * 1e3
import sys, importlib.util
from differometor import _interpret, fast
jax.config.update("jax_enable_x64", True); jax.config.update("jax_enable_compilation_cache", False); jax.devices("cpu")
spec = importlib.util.spec_from_file_location("kern_cases", os.environ.get("CASES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "switch_cases.py")))
kern_cases = importlib.util.module_from_spec(spec); spec.loader.exec_module(kern_cases)
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]

def forward_with_pullback(parameters, arrays):
    return jax.vjp(lambda p: df.simulate(**dict(arrays, optimized_parameters=p)), parameters)
def readout(outputs, detector_ports, reference):
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[detector_ports]
    detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
    sensitivity = (noise / jnp.abs(detected)).squeeze()
    loss = jnp.mean(jnp.log10(sensitivity / reference))
    return loss, (sensitivity, df.power_detector(carrier))
def sensitivity_with_pullback(outputs, detector_ports, reference):
    return jax.vjp(lambda values: readout(values, detector_ports, reference), outputs, has_aux=True)
def backward(simulation_pullback, readout_pullback, cotangent):
    output_cotangents = readout_pullback(cotangent)[0]
    return simulation_pullback(output_cotangents)[0]

def setups():
    cases = list(kern_cases.cases(3))
    repo = os.getcwd()
    fixture = json.load(open(f"{repo}/examples/data/uifo_800_3000.json"))
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    cases.append(("fixture, 37 frequencies", setup, pairs, jnp.linspace(800.0, 3000.0, 37)))
    cases.append(("fixture again", setup, pairs, cases[0][3]))
    return cases

def run(name, setup, pairs, frequencies):
    t = {}
    def stage(key, function):
        start = perf_counter(); result = function(); jax.block_until_ready(result); t[key] = (perf_counter() - start) * 1e3
        return result
    arrays, ports, *_ = stage("build", lambda: df.run_build_step(setup, [("f", "frequency")], frequencies, pairs))
    values = np.zeros(int(np.asarray(arrays["optimized_value_indices"]).max()) + 1)
    values[np.asarray(arrays["optimized_value_indices"])] = np.asarray(arrays["parameters"])[0, np.asarray(arrays["optimized_parameter_indices"])].real
    parameters = jnp.asarray(values); ports = jnp.asarray(ports); one = jnp.asarray(1.0, dtype=jnp.float64)
    expected = jnp.ones(frequencies.shape[0], dtype=jnp.float64)
    arrays["optimized_parameters"] = parameters
    path = "fast" if fast.supports(arrays) else "JAX"
    kinds = []
    def compiled(key, function, *args):
        lowered = stage(key + "_lower", lambda: jax.jit(function).lower(*args))
        program = stage(key + "_compile", lowered.compile)
        kinds.append("i" if type(program).__module__ == _interpret.__name__ else "X")
        return program
    forward = compiled("forward", forward_with_pullback, parameters, arrays)
    outputs, simulation_pullback = stage("forward", lambda: forward(parameters, arrays))
    sensitivity_fn = compiled("sensitivity", sensitivity_with_pullback, outputs, ports, expected)
    loss, readout_pullback, (sensitivity, powers) = stage("sensitivity", lambda: sensitivity_fn(outputs, ports, expected))
    gradient_fn = compiled("gradient", backward, simulation_pullback, readout_pullback, one)
    gradient = stage("gradient", lambda: gradient_fn(simulation_pullback, readout_pullback, one))
    finite = bool(np.isfinite(np.asarray(loss)) and np.isfinite(np.asarray(gradient)).all())
    lower = [t[k + "_lower"] for k in ("forward", "sensitivity", "gradient")]
    comp = [t[k + "_compile"] for k in ("forward", "sensitivity", "gradient")]
    print(f"{name:26s} {values.size:4d} var {int(ports.shape[0])} port {int(frequencies.shape[0]):5d} f {path:4s} | programs {'/'.join(kinds)} | lower {lower[0]:6.2f} {lower[1]:6.2f} {lower[2]:6.2f} | compile {comp[0]:7.2f} {comp[1]:7.2f} {comp[2]:7.2f} | forward {t['forward']:7.1f} gradient {t['gradient']:6.1f} | lower+compile {sum(lower) + sum(comp):7.2f} | {'finite' if finite else 'NOT FINITE'}", flush=True)
    return sum(comp), "X" in kinds

cases = setups()
if "--count" in sys.argv:
    print(len(cases)); sys.exit(0)
if "--only" in sys.argv:
    cases = [cases[int(sys.argv[sys.argv.index("--only") + 1])]]
worst, any_xla = 0.0, False
for case in cases:
    total, xla = run(*case)
    worst, any_xla = max(worst, total), any_xla or xla
print(f"imports {IMPORT:.0f} ms | largest compile total of a setup {worst:.2f} ms | a program went to XLA: {any_xla}")
