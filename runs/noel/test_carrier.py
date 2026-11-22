"""Carrier system through the native kernel (patches/philip-fastest-carrier.patch).

    PYTHONPATH=. python runs/noel/test_carrier.py          (from a checkout that has the patch)

Checks, on the public fixture, aLIGO (two modes) and Voyager:

1. Importing differometor does not load SciPy's SuperLU module any more.
2. With the native kernel (x86-64 Linux) the carrier system goes through `_SingleLU`, and after a
   complete run SuperLU is still not loaded. Without the kernel SuperLU is used, as before.
3. The carrier solution and a transposed solve are as close to a dense solve as SuperLU's.
4. Both fallbacks give the same sensitivity and gradient: the kernel declining the carrier
   system only (SuperLU for the carrier, kernel for the signal systems), and no kernel at all.
"""
import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
import jax
import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor import fast
from differometor.setups import aligo, voyager, constrain_inter_grid_cell_spaces

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
ROOT = os.getcwd()
SUPERLU = "scipy.sparse.linalg._dsolve._superlu"
native = fast._pattern_lu() is not None
print("native kernel available:", native)
assert fast._superlu is fast._UNLOADED and SUPERLU not in sys.modules, "the import must not load SuperLU"

rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(float(np.abs(np.asarray(b)).max()), 1e-300))
fixture = json.load(open(os.path.join(ROOT, "examples", "data", "uifo_800_3000.json")))
grid = jnp.asarray(json.load(open(os.path.join(ROOT, "examples", "data", "uifo_800_3000_test_data.json")))["frequencies"], dtype=jnp.float64)
wide = jnp.asarray(json.load(open(os.path.join(ROOT, "examples", "data", "aligo_test_data.json")))["frequencies"], dtype=jnp.float64)
setup = df.Setup.from_data(fixture)
cases = {
    "fixture": (setup, constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES), grid),
    "aligo": (*aligo("space_modulation"), wide),
    "aligo linked": (*aligo("amplitude_modulation"), wide),
    "voyager": (*voyager(), wide),
}

kinds = []
original_matrix = fast.SystemPlan.carrier_matrix


def spying(self, entries):
    result = original_matrix(self, entries)
    kinds.append(type(result[0]).__name__)
    return result


fast.SystemPlan.carrier_matrix = spying
original_create = fast._PatternLU.create.__func__


def declining(cls, size, rows, cols, values, select=None, overwrite=False, samples=None):
    if values.shape[1] == 1:  # the carrier system: one matrix
        return None, 1
    return original_create(cls, size, rows, cols, values, select, overwrite, samples)


worst = 0.0
rng = np.random.default_rng(5)
for position, (name, (setup, pairs, frequencies)) in enumerate(cases.items()):
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    ports = jnp.asarray(ports)
    values = np.zeros(int(np.asarray(arrays["optimized_value_indices"]).max()) + 1)
    values[np.asarray(arrays["optimized_value_indices"])] = np.asarray(arrays["parameters"])[0, np.asarray(arrays["optimized_parameter_indices"])].real
    arrays = {k: v for k, v in arrays.items() if k != "optimized_parameters"}

    def loss(p):
        carrier, signal, noise = df.simulate(**arrays, optimized_parameters=p)
        signals = df.signal_detector(carrier, signal)[ports]
        detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(sensitivity)), (carrier, sensitivity)

    run = lambda: jax.value_and_grad(loss, has_aux=True)(jnp.asarray(values))
    kinds.clear()
    (_, (carrier, sensitivity)), gradient = run()
    kind = kinds[0]
    assert kind == ("_SingleLU" if native else "SuperLU"), kind
    if position == 0 and native:
        assert SUPERLU not in sys.modules, "a run with the kernel must not load SuperLU"
    assert all(np.isfinite(np.asarray(x)).all() for x in (carrier, sensitivity, gradient))

    # accuracy against a dense solve, forward and transposed
    A = {key: arrays[key] for key in fast.HOST_KEYS}
    entry_plan, plan = fast.EntryPlan(A), fast.SystemPlan(A)
    carrier_entries = entry_plan.values(values)[0]
    factors, rhs = original_matrix(plan, carrier_entries)
    stored = plan.carrier_base_values.copy()
    stored[plan.carrier_matrix_slots] = carrier_entries[plan.carrier_matrix_entries]
    dense = np.zeros((plan.n, plan.n), complex)
    dense[plan.carrier_rows, plan.carrier_cols] = stored
    b = rng.normal(size=plan.n) + 1j * rng.normal(size=plan.n)
    forward_error = rel(factors.solve(rhs), np.linalg.solve(dense, rhs))
    transposed_error = rel(factors.solve(b, trans="T"), np.linalg.solve(dense.T, b))
    residual = float(np.abs(dense @ factors.solve(rhs) - rhs).max() / (np.abs(dense).sum(axis=1).max() * np.abs(factors.solve(rhs)).max()))
    bound = 1e3 * np.finfo(float).eps * np.linalg.cond(dense)
    assert forward_error < bound and transposed_error < bound and residual < 1e-14, (forward_error, transposed_error, residual, bound)

    row = {}
    if native:
        fast._PatternLU.create = classmethod(declining)
        kinds.clear()
        (_, (carrier_d, sensitivity_d)), gradient_d = run()
        fast._PatternLU.create = classmethod(original_create)
        assert kinds[0] == "SuperLU", kinds[0]
        row.update({"carrier": rel(carrier, carrier_d), "sensitivity": rel(sensitivity, sensitivity_d), "gradient": rel(gradient, gradient_d)})
    fast.NATIVE_LU = False
    (_, (carrier_s, sensitivity_s)), gradient_s = run()
    fast.NATIVE_LU = True
    row.update({"sensitivity, no kernel": rel(sensitivity, sensitivity_s), "gradient, no kernel": rel(gradient, gradient_s)})
    worst = max(worst, *row.values())
    print(f"{name:13s} {kind:9s} dense {forward_error:.1e}, transposed {transposed_error:.1e}, residual {residual:.1e} | " + ", ".join(f"{k} {v:.1e}" for k, v in row.items()))

assert worst < 1e-6, worst
print(f"OK, worst relative difference between the paths {worst:.1e}")
