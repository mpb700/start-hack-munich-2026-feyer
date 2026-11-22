"""Compares the native formula kernel (entries.cpp) with the dual-number evaluation in NumPy.

Run from the repository root: python differometor/_native/test_entries.py

1. Every component function, random inputs: values and derivatives of all four outputs for every
   choice this script draws of differentiated inputs and swept inputs, including which outputs
   have a derivative and which arrays have one column instead of one per row.
2. The public grid setup and aLIGO: every group of calls of the forward pass with derivatives,
   then outputs and gradient of the whole simulation with the kernel on and off.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import differometor as df
from differometor import components, fast
from differometor.setups import aligo, constrain_inter_grid_cell_spaces

TOLERANCE = 1e-12
SIMULATION_TOLERANCE = 1e-7
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
worst = 0.0


def close(a, b, what):
    global worst
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"{what}: shape {a.shape} against {b.shape}")
    if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.array_equal(np.isinf(a), np.isinf(b)):
        raise AssertionError(f"{what}: nan or inf in different places")
    finite = np.isfinite(a)
    if finite.any():
        scale = max(np.abs(b[finite]).max(), 1e-300)
        error = np.abs(a[finite] - b[finite]).max() / scale
        worst = max(worst, error)
        if error > TOLERANCE:
            raise AssertionError(f"{what}: relative difference {error:.2e}")


def numpy_group(function_index, arguments, lead, columns):
    """What `_EntryBlock.run` computes for one group: value and derivative of each of the four outputs."""
    arguments = list(arguments)
    width = len(columns)
    for position, j in enumerate(columns):
        seed = np.zeros((width, 1, 1))
        seed[position] = 1.0
        arguments[j] = fast.Dual(arguments[j], seed)
    with np.errstate(all="ignore"):
        outputs = fast.FUNCTIONS_NUMPY[function_index](arguments)
    result = []
    for item in outputs:
        value, slope = (item.val, item.tan) if isinstance(item, fast.Dual) else (item, 0.0)
        value = np.broadcast_to(value, lead)
        slope = None if np.ndim(slope) == 0 else np.broadcast_to(slope, (width, lead[0], slope.shape[-1]))
        result.append((value, slope))
    return result


def compare_group(function_index, arguments, lead, swept, columns, what):
    parts = [(o, None, None, None, None) for o in range(4)]
    native = fast._native_entries(function_index, arguments, lead, swept, np.asarray(columns, np.intp), parts)
    if native is None:
        raise AssertionError(f"{what}: the kernel declined")
    values, tangents, flags = native
    for o, (value, slope) in enumerate(numpy_group(function_index, arguments, lead, columns)):
        close(values[o], value, f"{what} output {o} value")
        if slope is None:
            if flags[o] & 1:
                raise AssertionError(f"{what} output {o}: the kernel reports a derivative, NumPy has none")
            continue
        if not flags[o] & 1:
            raise AssertionError(f"{what} output {o}: NumPy has a derivative, the kernel reports none")
        mine = tangents[o] if flags[o] & 4 else tangents[o][:, :, :1]
        close(mine, slope, f"{what} output {o} derivative")


def random_functions(rng, rounds=40):
    members, rows = 5, 6
    count = 0
    for function_index, function in enumerate(components.FUNCTIONS):
        for _ in range(rounds):
            swept = rng.random(7) < 0.3
            columns = np.nonzero(rng.random(7) < 0.35)[0]
            lead = (members, rows if swept.any() else 1)
            # Physical ranges: fractions in (0, 1), indices near 1, angles in degrees, frequencies in Hz.
            base = np.stack([rng.uniform(0.05, 0.95, members), rng.uniform(0.05, 0.95, members), rng.uniform(0.05, 0.95, members),
                             rng.uniform(0.05, 0.95, members), rng.uniform(1.0, 1.6, members), rng.uniform(1.0, 1.6, members),
                             rng.uniform(0.0, 60.0, members)])
            if function.__name__ in ("space", "space_lower", "space_modulation", "space_modulation_lower", "susceptibility", "laser_frequency_modulation", "laser_frequency_modulation_lower"):
                base[0] = rng.uniform(10.0, 5000.0, members)
                base[1] = rng.uniform(1.0, 4000.0, members)
            arguments = [base[j][:, None] * rng.uniform(0.9, 1.1, (members, rows)) if swept[j] else base[j][:, None].copy() for j in range(7)]
            compare_group(function_index, arguments, lead, swept, columns, f"{function.__name__} {columns.tolist()} swept {np.nonzero(swept)[0].tolist()}")
            count += 1
    # Total internal reflection and the ends of the reflectivity range: nan and inf in the same places.
    for name in ("surface", "force_calculation_right", "corrected_optomechanical_phase_right"):
        function_index = [f.__name__ for f in components.FUNCTIONS].index(name)
        for reflectivity in (0.0, 1.0, 0.5):
            base = np.array([[0.0, reflectivity, 30.0, 100.0, 1.45, 1.0, 225.0], [0.1, reflectivity, 0.0, 100.0, 1.0, 1.45, 60.0]]).T
            if name != "surface":
                base = np.array([[225.0, 1.45, 1.0, 0.1, 1.45, 1.0, 225.0], [100.0, 30.0, reflectivity, 0.0, 1.45, 1.0, 225.0]]).T
            arguments = [base[j][:, None].copy() for j in range(7)]
            for columns in ([], [1, 2], [0, 1, 2, 6]):
                compare_group(function_index, arguments, (2, 1), np.zeros(7, bool), columns, f"{name} edge {reflectivity} {columns}")
                count += 1
    return count


def build(kind):
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "examples", "data")
    if kind == "grid":
        fixture = json.load(open(os.path.join(data, "uifo_800_3000.json")))
        reference = json.load(open(os.path.join(data, "uifo_800_3000_test_data.json")))
        setup = df.Setup.from_data(fixture)
        pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
        store = fixture
    else:
        reference = json.load(open(os.path.join(data, "aligo_test_data.json")))
        setup, _ = aligo(mode="space_modulation")
        store = setup.to_data()
        pairs = [list(pair) for pair in setup.parameters if pair[1] in PROPERTIES]
    initial = []
    for pair in pairs:
        name, prop = (pair[0] if isinstance(pair[0], list) else pair)
        initial.append(store["nodes" if name in store["nodes"] else "edges"][name]["properties"][prop])
    parameters = np.asarray(initial, float)
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], np.asarray(reference["frequencies"], float), pairs)
    arrays["optimized_parameters"] = parameters
    return tuple(np.asarray(arrays[key]) if not isinstance(arrays[key], tuple) and hasattr(arrays[key], "__jax_array__") else arrays[key] for key in fast.HOST_KEYS), parameters


def setups():
    count = 0
    for kind in ("grid", "aligo"):
        host, parameters = build(kind)
        plan = fast.EntryPlan(dict(zip(fast.HOST_KEYS, host)))
        carrier_rows, signal_rows = plan._rows(parameters)
        for name, block, rows in (("carrier", plan.carrier, carrier_rows), ("signal", plan.signal, signal_rows), ("noise", plan.noise, signal_rows)):
            for (function_index, members, _, columns), swept in zip(block.groups, block.swept):
                arguments, lead = block._inputs(rows, members, swept)
                compare_group(function_index, arguments, lead, swept, columns, f"{kind} {name} {components.FUNCTIONS[function_index].__name__}")
                count += 1
        results = {}
        for switch in (True, False):
            fast.NATIVE_ENTRIES = switch
            (carrier, signal, noise, Y, MU), work = fast._run_forward(parameters, host, taped=True)
            rng = np.random.default_rng(1)
            bars = (rng.normal(size=(carrier.shape[0], 1)) + 0j, np.zeros(signal.shape, complex), rng.normal(size=noise.shape))
            gradient = fast._run_backward(parameters, (carrier, signal, noise, Y, MU), work, *bars)
            results[switch] = (carrier, signal, noise, gradient)
        fast.NATIVE_ENTRIES = True
        # The linear systems amplify differences in the last digit of the entries (the SuperLU and
        # the native factorization differ by about 1e-9 for the same reason), so the whole
        # simulation is held to SIMULATION_TOLERANCE instead of TOLERANCE.
        for label, a, b in zip(("carrier", "signal", "noise", "gradient"), results[True], results[False]):
            if np.abs(a - b).max() > SIMULATION_TOLERANCE * np.abs(b).max():
                raise AssertionError(f"{kind} simulation {label}: relative difference {np.abs(a - b).max() / np.abs(b).max():.2e}")
            print(f"  {kind:5s} {label:8s} kernel against NumPy: relative difference {np.abs(a - b).max() / np.abs(b).max():.2e}")
    return count


if __name__ == "__main__":
    if fast._entries_kernel() is None:
        raise SystemExit("libentries.so could not be loaded; build it with differometor/_native/build_entries.sh")
    functions = random_functions(np.random.default_rng(0))
    groups = setups()
    print(f"PASS: {functions} random function evaluations, {groups} groups of the two setups, largest relative difference {worst:.2e} (tolerance {TOLERANCE:.0e})")
