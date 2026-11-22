"""Parameter links with other functions than sqrt on the fast path (patches/philip-fastest-links.patch).

    PYTHONPATH=. python runs/noel/test_link_functions.py

aLIGO with amplitude modulation has one link: the signal amplitude follows the laser power, in
the constructor through `jnp.sqrt`. Here that function is replaced, one at a time, by the other
functions of `fast.LINK_RULES` and by plain formulas (lambda, def, closure), and the fast path is
compared with the dense JAX reference: log sensitivity, the three outputs and the gradient of the
benchmark's readout (the laser power is a design variable, so the gradient runs through the link).
Functions the fast path does not cover (branches, `jnp.where`, jit objects) must be reported as
unsupported, so that they keep using the JAX implementation.
"""
import math
import sys

import numpy as np, jax, jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import differometor as df
import differometor.fast as fast
import differometor.setups as setups
from differometor import build
from differometor._simulate_dicht import simulate_jax

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
F = jnp.asarray(np.geomspace(20.0, 5000.0, 7))
setup, allpairs = setups.aligo(mode="amplitude_modulation")
pairs = [list(p) for p in allpairs if p[1] in PROPERTIES]
arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], F, pairs)
ports = jnp.asarray(ports)
base = np.real(np.asarray(arrays["parameters"])[0])
slots, places = np.asarray(arrays["optimized_parameter_indices"]), np.asarray(arrays["optimized_value_indices"])
values = np.zeros(places.max() + 1); values[places] = base[slots]
x = jnp.asarray(values, dtype=jnp.float64)
arrays["optimized_parameters"] = x
link_slots = [int(i) for i in np.asarray(arrays["linking_function_indices"]).reshape(-1)]
assert len(link_slots) == 1 and build.LINKING_FUNCTIONS[link_slots[0]] is jnp.sqrt, "expected the one sqrt link of amplitude modulation"
source = int(np.asarray(arrays["linked_indices"]).reshape(-1)[0])
source_is_design = bool(np.isin(source, slots))
power = float(base[source])
print(f"link source: parameter {source} = {power:g}, design variable: {source_is_design}")
scale = 3.0


def smooth(p):
    return jnp.tanh(p / 200.0) * 4 + jnp.log10(p) - 1 / p


covered = {
    "jnp.sqrt (as built)": jnp.sqrt, "jnp.square": jnp.square, "jnp.log": jnp.log, "jnp.log10": jnp.log10, "jnp.sin": jnp.sin,
    "jnp.cos": jnp.cos, "jnp.tan": jnp.tan, "jnp.tanh": jnp.tanh, "jnp.arctan": jnp.arctan, "jnp.negative": jnp.negative,
    "jnp.reciprocal": jnp.reciprocal, "jnp.abs": jnp.abs, "jnp.radians": jnp.radians, "jnp.degrees": jnp.degrees,
    "lambda: exp(p / 100)": lambda p: jnp.exp(p / 100), "lambda: sqrt(p) / 2": lambda p: jnp.sqrt(p) / 2,
    "lambda: 0.1 * p + 3": lambda p: 0.1 * p + 3, "lambda: p ** 0.5": lambda p: p**0.5,
    "lambda: exp(-p/50) + cos(p)**2": lambda p: jnp.exp(-p / 50) + jnp.cos(p) ** 2,
    "closure: scale * sqrt(p)": lambda p: scale * jnp.sqrt(p), "def: tanh, log10, 1/p": smooth, "lambda: constant": lambda p: 2.5,
}
uncovered = {
    "jnp.where": lambda p: jnp.where(p > 0, jnp.sqrt(p), 0.0), "branch": lambda p: p if p == 0 else 2 * p,
    "jit object": jax.jit(lambda p: 2 * p), "math.sqrt": math.sqrt, "jnp.exp2 (not in the table)": jnp.exp2,
}


def loss(sim):
    def f(p):
        carrier, signal, noise = sim(**dict(arrays, optimized_parameters=p))
        signals = df.signal_detector(carrier, signal)[ports]
        detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
        s = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(s)) + 1e-3 * jnp.sum(df.power_detector(carrier)), (s, carrier, signal, noise)
    return f


rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300))
failed, worst = [], [0.0, 0.0, 0.0]
original = build.LINKING_FUNCTIONS[link_slots[0]]
try:
    for label, function in covered.items():
        build.LINKING_FUNCTIONS[link_slots[0]] = function
        supported = bool(fast.supports(arrays))
        gf, auxf = jax.grad(loss(df.simulate), has_aux=True)(x)
        gd, auxd = jax.grad(loss(simulate_jax), has_aux=True)(x)
        e_s = float(np.abs(np.log(np.asarray(auxf[0])) - np.log(np.asarray(auxd[0]))).max())
        e_out = max(rel(auxf[1], auxd[1]), rel(auxf[2], auxd[2]), rel(auxf[3], auxd[3]))
        e_g = rel(gf, gd)
        finite = bool(np.isfinite(np.asarray(gf)).all() and np.isfinite(np.asarray(auxf[0])).all())
        worst[:] = max(worst[0], e_s), max(worst[1], e_out), max(worst[2], e_g)
        if not supported or not finite or e_s > 1e-6 or e_g > 1e-5:
            failed.append(label)
        print(f"{label:32s} fast path {str(supported):5s} | ln s {e_s:.1e} | outputs {e_out:.1e} | gradient {e_g:.1e} | gradient of the power {float(np.asarray(gf)[places[slots == source]][0]) if source_is_design else float('nan'):+.3e}", flush=True)
    for label, function in uncovered.items():
        build.LINKING_FUNCTIONS[link_slots[0]] = function
        supported = bool(fast.supports(arrays))
        print(f"{label:32s} fast path {str(supported):5s} (expected False: stays with the JAX implementation)")
        if supported:
            failed.append(label)
finally:
    build.LINKING_FUNCTIONS[link_slots[0]] = original
print(f"WORST ln s {worst[0]:.2e} | outputs {worst[1]:.2e} | gradient {worst[2]:.2e}")
print("FAILED: %s" % failed if failed else "OK")
sys.exit(1 if failed else 0)
