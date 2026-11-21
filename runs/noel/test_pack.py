"""Host arrays of the fast path: static index arrays and packed residuals.

- `_pack` and `_unpack` restore every array exactly: the traced arrays of the benchmark setup, all
  host arrays as if none were static, and synthetic arrays of other dtypes and shapes (boolean,
  32-bit, empty, zero-dimensional, seven dimensions). The statically computed sizes
  (`_packed_sizes`, used for the callback result shapes) match what `_pack` returns.
- Index arrays from the build step are `StaticArray`s: JAX sees no leaves for them, equal contents
  give equal pytree definitions, different contents different ones.
- The fast path gives the same result with static index arrays, with all index arrays as ordinary
  JAX arrays (they travel through JAX then), and follows a changed index array (new compilation,
  agreement with the original JAX code).

Usage: python runs/noel/test_pack.py <repo>
"""
import copy, json, os, sys
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
sys.path.insert(0, sys.argv[1])
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor import fast
from differometor.fast import SparseMatrix, StaticArray
from differometor.setups import constrain_inter_grid_cell_spaces
from differometor.simulate import simulate_jax

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data = os.path.join(sys.argv[1], "examples", "data")
fixture = json.load(open(os.path.join(data, "uifo_800_3000.json"))); reference = json.load(open(os.path.join(data, "uifo_800_3000_test_data.json")))
setup = df.Setup.from_data(fixture)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
frequencies = jnp.asarray(np.asarray(reference["frequencies"])[[0, 50, 100]])
arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)


def check(name, A, layout):
    packed = fast._pack(A, layout)
    sizes = fast._packed_sizes(layout)
    assert [x.shape[0] for x in packed] == [size for size in sizes if size], (name, [x.shape for x in packed], sizes)
    assert all(x.ndim == 1 and x.dtype in (np.int64, np.float64) for x in packed), name
    restored = fast._unpack(layout, packed)
    assert list(restored) == [key for key, _, _ in layout], name
    for key, _, _ in layout:
        a, b = np.asarray(A[key]), restored[key]
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), (name, key, a.dtype, b.dtype, a.shape, b.shape)
    print(f"{name}: {len(restored)} arrays restored exactly from {sum(x.nbytes for x in packed)} bytes")


layout_of = lambda A: tuple((key, np.asarray(value).dtype.char, np.asarray(value).shape) for key, value in A.items())
static, layout, traced = fast._split_host(arrays)
assert len(static) + len(layout) == len(fast.HOST_KEYS) and all(isinstance(value, StaticArray) for value in static.values())
benchmark = {key: np.asarray(leaf) for (key, _, _), leaf in zip(layout, traced)}
assert layout == layout_of(benchmark)
check("traced arrays of the benchmark setup", benchmark, layout)
everything = dict(zip(fast.HOST_KEYS, [np.asarray(leaf) for leaf in fast._host_leaves(arrays)]))
check("all host arrays", everything, layout_of(everything))
rng = np.random.default_rng(0)
other = dict(everything)
other["optimized_parameter_indices"] = np.array([True, False, True])
other["optimized_value_indices"] = rng.integers(-5, 5, (1, 2, 3, 1)).astype(np.int32)
other["signal_changing_values"] = rng.normal(size=(5, 1)).astype(np.float32)
other["parameters"] = (rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))).astype(np.complex64)
other["carrier_changing_values"] = np.empty((0, 3))
other["noise_function_indices"] = np.array(7, dtype=np.uint8)
other["noise_selection_vectors"] = rng.normal(size=(1, 2, 1, 2, 2, 3, 1))
other["carrier_function_indices"] = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max])
check("other dtypes and shapes", other, layout_of(other))
check("nothing to pack", {}, ())

# Static arrays in pytrees
leaves = jax.tree_util.tree_leaves(arrays)
indices = [key for key, value in arrays.items() if isinstance(value, StaticArray)]
print(f"{len(arrays)} entries from the build step: {len(indices)} static index arrays, {len(leaves)} leaves for JAX")
assert len(indices) >= 20 and len(leaves) <= 8 and not arrays["signal_matrix_indices"].flags.writeable
twin = copy.deepcopy(arrays)
assert jax.tree_util.tree_structure(arrays) == jax.tree_util.tree_structure(twin)
# A changed index array: two tunings of the design take each other's value.
initial, kinds = [], []
for pair in pairs:
    name, prop = (pair if isinstance(pair[0], list) else [pair])[0]
    initial.append(fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop]); kinds.append(prop)
tunings = [i for i, kind in enumerate(kinds) if kind == "tuning"]
first = tunings[0]; second = next(i for i in tunings[1:] if initial[i] != initial[first])
value_indices = np.asarray(arrays["optimized_value_indices"])
swapped = value_indices.copy(); swapped[value_indices == first] = second; swapped[value_indices == second] = first
changed = dict(arrays, optimized_value_indices=StaticArray(swapped))
assert jax.tree_util.tree_structure(arrays) != jax.tree_util.tree_structure(changed)

# The fast path with static, with traced and with changed index arrays
parameters = jnp.asarray(initial, dtype=jnp.float64)
ports = jnp.asarray(ports)


def loss(simulate):
    def function(p, arguments):
        carrier, signal, noise = simulate(**dict(arguments, optimized_parameters=p))
        detected = df.signal_detector(carrier, signal)[ports].squeeze()
        return jnp.mean(jnp.log10(noise / jnp.abs(detected))) + 1e-3 * jnp.sum(df.power_detector(carrier))

    return jax.jit(jax.value_and_grad(function))


def as_traced(value):
    if isinstance(value, StaticArray):
        return jnp.asarray(np.asarray(value))
    if isinstance(value, SparseMatrix):
        return SparseMatrix(as_traced(value.indices), value.values, value.shape)
    return value


rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())
fast_loss, jax_loss = loss(df.simulate), loss(simulate_jax)
value, gradient = fast_loss(parameters, arrays)
traced_arrays = {key: as_traced(item) for key, item in arrays.items()}
assert fast.supports(dict(traced_arrays, optimized_parameters=parameters)) and len(fast._split_host(traced_arrays)[0]) == 0
value_traced, gradient_traced = fast_loss(parameters, traced_arrays)
assert float(value) == float(value_traced) and np.array_equal(np.asarray(gradient), np.asarray(gradient_traced))
print("index arrays as ordinary JAX arrays: identical loss and gradient")
value_changed, gradient_changed = fast_loss(parameters, changed)
value_jax, gradient_jax = jax_loss(parameters, changed)
assert float(value_changed) != float(value), "the changed index array was ignored"
deviation = max(abs(float(value_changed) - float(value_jax)) / abs(float(value_jax)), rel(gradient_changed, gradient_jax))
print(f"changed index array: loss {float(value):.9e} -> {float(value_changed):.9e}, against the original JAX code {deviation:.1e}")
assert deviation < 1e-6
print("OK")
