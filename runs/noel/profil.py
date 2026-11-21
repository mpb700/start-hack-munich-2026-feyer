"""Profile the host callbacks of the fast path on the benchmark workload.

Usage: python runs/noel/profil.py <repo> [forward|backward|build]
"""
import functools
import cProfile
import json
import os
import pstats
import sys
from pathlib import Path
from time import perf_counter as pc

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
what = sys.argv[2] if len(sys.argv) > 2 else "forward"

import jax
import jax.numpy as jnp
import numpy as np
import differometor as df
from differometor import fast
from differometor.setups import constrain_inter_grid_cell_spaces

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data = repo / "examples" / "data"
fixture = json.loads((data / "uifo_800_3000.json").read_text())
reference = json.loads((data / "uifo_800_3000_test_data.json").read_text())


def build_inputs():
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    initial = []
    for pair in pairs:
        members = pair if isinstance(pair[0], list) else [pair]
        values = [fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop] for name, prop in members]
        initial.append(values[0])
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    frequencies = jnp.asarray(reference["frequencies"], dtype=jnp.float64)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    arrays["optimized_parameters"] = parameters
    return arrays, ports, parameters


if what == "build":
    t = pc(); build_inputs(); print("first build", pc() - t)
    profile = cProfile.Profile(); profile.enable(); build_inputs(); profile.disable()
    pstats.Stats(profile).sort_stats("cumulative").print_stats(35)
    sys.exit()

arrays, ports, parameters = build_inputs()
split = hasattr(fast, "_split_host")  # since round 8: static index arrays stay with the callbacks
sparse = hasattr(fast, "_host_leaves")  # since round 6: sparse system matrices, every host array packed
host_backward = fast._host_backward
if split:
    static, layout, traced = fast._split_host(arrays)
    host = [np.asarray(leaf) for leaf in traced]
    host_forward = functools.partial(fast._host_forward, arrays["carrier_matrix"].shape[0], static, layout)
    host_backward = functools.partial(fast._host_backward, static, layout)
elif sparse:
    host = [np.asarray(leaf) for leaf in fast._host_leaves(arrays)]
    host_forward = functools.partial(fast._host_forward, arrays["carrier_matrix"].shape[0])
else:
    host = [np.asarray(arrays[k]) for k in fast.HOST_KEYS]
    host_forward = fast._host_forward
p = np.asarray(parameters)

for _ in range(2):
    t = pc(); out = host_forward(p, *host); t_fwd = pc() - t
print("forward seconds", t_fwd)
carrier, signal, noise, *helpers = out
carrier_bar = np.zeros_like(carrier)
signal_bar = np.zeros_like(signal)
rng = np.random.default_rng(0)
# A cotangent as the benchmark readout gives it: at every frequency a multiple of conj(w), with the
# detector weights w (since round 10 any other cotangent costs a second factorization).
selection = np.asarray(arrays["noise_selection_vectors"])[0, 0]
d, n_carrier = np.nonzero(selection)[0], carrier.shape[0]
weights = selection.astype(complex)
weights[:n_carrier] *= carrier[:, 0]
weights[n_carrier : 2 * n_carrier] = np.conj(weights[n_carrier : 2 * n_carrier] * carrier[:, 0])
alpha = rng.normal(size=signal.shape[1]) + 1j * rng.normal(size=signal.shape[1])
signal_bar[d] = np.conj(weights[d])[:, None] * alpha
noise_bar = rng.normal(size=noise.shape)
backward_keys = () if sparse else getattr(fast, "BACKWARD_KEYS", fast.HOST_KEYS)  # older states pass host arrays
backward_host = [h for k, h in zip(fast.HOST_KEYS, host) if k in backward_keys]
args = (p, *backward_host, carrier, signal, noise, *helpers, carrier_bar, signal_bar, noise_bar)
for _ in range(2):
    t = pc(); g = host_backward(*args); t_bwd = pc() - t
print("backward seconds", t_bwd)

profile = cProfile.Profile()
profile.enable()
for _ in range(5):
    if what == "forward":
        host_forward(p, *host)
    else:
        host_backward(*args)
profile.disable()
stats = pstats.Stats(profile)
stats.sort_stats("cumulative").print_stats(38)
