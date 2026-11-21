"""How does a simulator tree scale with the size of the UIFO grid?

    python runs/noel/scaling.py <repo> <frequencies> <size> [<size> ...]

`differometor.setups.uifo(size)` builds the model of the public fixture as a size x size grid
(the fixture is 3x3: 48 mirrors, 193 design variables). For every size this script builds a
random setup with a plain detector, gives the design variables random values and times
`jit(value_and_grad(loss))` with the loss of benchmark.py: the first call (with compilation) and
the median of three warm calls. `<repo>` may be the unchanged original (`git archive 5eece1d`),
which needs far more memory: use few frequencies there.

Sizes of 10 and more are not usable: the generator builds node names from the digits of the
grid position, and they collide from there on.
"""
import os, resource, statistics, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
repo, F, sizes = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3:]]
sys.path.insert(0, repo)
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces, uifo

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
frequencies = jnp.asarray(np.geomspace(800.0, 3000.0, F))


def random_values(pairs, rng):
    draw = {
        "reflectivity": lambda: rng.uniform(0.05, 0.95),
        "tuning": lambda: rng.uniform(-360, 360),
        "db": lambda: rng.uniform(0, 10),
        "angle": lambda: rng.uniform(0, 360),
        "power": lambda: 10 ** rng.uniform(-3, 2),
        "mass": lambda: 10 ** rng.uniform(0, 2.3),
        "length": lambda: rng.uniform(0, 4000),
    }
    return np.array([draw[(pair if isinstance(pair[0], str) else pair[0])[1]]() for pair in pairs])


print(f"tree {repo}, {F} frequencies")
print(f"{'grid':>5s} {'mirrors':>8s} {'variables':>10s} {'carrier n':>10s} {'signal m':>9s} {'build ms':>9s} {'first call ms':>14s} {'warm ms':>9s} {'peak RSS GB':>12s}  finite")
for size in sizes:
    setup, _ = uifo(size, boundaries={"01": "detector"}, random=True, random_seed=size)
    mirrors = sum(1 for node in setup.to_data()["nodes"].values() if node.get("component") == "mirror")
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    start = time.perf_counter()
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    build = time.perf_counter() - start
    ports = jnp.asarray(ports)
    p = jnp.asarray(random_values(pairs, np.random.default_rng(size)))

    def loss(q):
        carrier, signal, noise = df.simulate(**dict(arrays, optimized_parameters=q))
        signals = df.signal_detector(carrier, signal)[ports]
        detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
        return jnp.mean(jnp.log10((noise / jnp.abs(detected)).squeeze()))

    function = jax.jit(jax.value_and_grad(loss))
    start = time.perf_counter(); value, gradient = function(p); gradient.block_until_ready(); first = time.perf_counter() - start
    warm = []
    for _ in range(3):
        start = time.perf_counter(); value, gradient = function(p); gradient.block_until_ready(); warm.append(time.perf_counter() - start)
    finite = bool(np.isfinite(float(value)) and np.isfinite(np.asarray(gradient)).all())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024) / 2**30  # of the process so far
    n, m = arrays["carrier_matrix"].shape[0], arrays["signal_matrix"].shape[0]
    print(f"{size:>3d}x{size:<1d} {mirrors:>8d} {p.shape[0]:>10d} {n:>10d} {m:>9d} {build * 1e3:>9.1f} {first * 1e3:>14.1f} {statistics.median(warm) * 1e3:>9.1f} {peak:>12.2f}  {finite}", flush=True)
    del function, arrays
