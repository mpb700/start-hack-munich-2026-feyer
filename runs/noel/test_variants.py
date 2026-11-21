"""Setup variants: the fast path against the original JAX code on random UIFO setups.

    python runs/noel/test_variants.py <repo> [--quick] [--size N] [--seeds N]

The final evaluation uses "a similar optical setup ... with slightly different parameters and
components". `differometor.setups.uifo(random=True)` draws such setups: beamsplitters or
directional beamsplitters in the centers, lasers or squeezers at the boundaries and one readout,
which is a plain detector or a balanced homodyne detector (two ports, a local oscillator and a
`qhd` noise detector) with equal probability.

For every variant the design variables get random values in the ranges of the public fixture.
`simulate` has to take the fast path, and carrier, signal, noise, sensitivity and the gradient of
the benchmark loss have to agree with `simulate_jax`. The sensitivity is compared as the
benchmark does it: max |ln s - ln s_ref|, which must stay below 1e-5 there.

Random values can give a setup without a usable readout. Two kinds are recognized and not
counted as failures: the reference itself is not finite (no light reaches the detector), or the
detected signal is a difference of nearly equal numbers (`cancellation`: the sum of the
magnitudes of its parts over its own magnitude). In the second case sensitivity and gradient are
rounding noise in both implementations, and only the fields at the detector are compared.

Variants: space modulation with several seeds (the benchmark kind), forced plain and homodyne
readouts, frequency and amplitude modulation (the latter has linked parameters), other grid
sizes, and a homodyne phase as a design variable. `--quick` runs a small selection.
"""
import os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
repo = sys.argv[1]
sys.path.insert(0, repo)
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor import fast
from differometor.setups import constrain_inter_grid_cell_spaces, uifo
from differometor.simulate import simulate_jax

option = lambda name, default: int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
QUICK, SIZE, SEEDS = "--quick" in sys.argv, option("--size", 3), option("--seeds", 6)
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
FREQUENCIES = jnp.asarray(np.geomspace(800.0, 3000.0, 5))
# Limits: the benchmark tolerance for the sensitivity. Random values give badly conditioned signal
# systems now and then (condition number 1e12 after equilibration for seed 4), where any two
# solvers differ by 1e-7; a mistake in the fast path shows as a deviation of order one.
LIMIT_SENSITIVITY, LIMIT_GRADIENT, LIMIT_FIELDS, LIMIT_CANCELLATION = 1e-5, 1e-4, 1e-6, 1e6
rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300))


def random_values(pairs, rng):
    """Random design values in the ranges of the public fixture (reflectivities also close to 0 and 1)."""
    draw = {
        "reflectivity": lambda: [rng.uniform(0, 1), 10 ** rng.uniform(-9, -1), 1 - 10 ** rng.uniform(-6, -1)][rng.integers(3)],
        "tuning": lambda: rng.uniform(-360, 360),
        "db": lambda: rng.uniform(0, 10),
        "angle": lambda: rng.uniform(0, 360),
        "power": lambda: 10 ** rng.uniform(-6, 2.3),
        "mass": lambda: 10 ** rng.uniform(-2, 2.3),
        "length": lambda: rng.uniform(0, 4000),
        "phase": lambda: rng.uniform(0, 360),
    }
    return np.array([draw[(pair if isinstance(pair[0], str) else pair[0])[1]]() for pair in pairs])


def evaluation(simulate, arrays, ports):
    """Loss as in benchmark.py (reference 1), its gradient and the physical outputs."""

    def loss(p):
        carrier, signal, noise = simulate(**dict(arrays, optimized_parameters=p))
        signals = df.signal_detector(carrier, signal)[ports]
        detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
        sensitivity = (noise / jnp.abs(detected)).squeeze()
        return jnp.mean(jnp.log10(sensitivity)), (carrier, signal, noise, sensitivity, df.power_detector(carrier))

    return jax.jit(jax.value_and_grad(loss, has_aux=True))


failures, worst = [], dict(sensitivity=0.0, gradient=0.0)


def case(label, size, seed, mode="space_modulation", boundaries=None, extra_pairs=()):
    setup, _, _, resolved = uifo(size, boundaries=boundaries, random=True, mode=mode, random_seed=seed, verbose=True)
    readout = "homodyne" if "balanced_homodyne" in resolved.values() else "detector"
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES) + [list(pair) for pair in extra_pairs]
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], FREQUENCIES, pairs)
    ports, n = np.asarray(ports), arrays["carrier_matrix"].shape[0]
    p = jnp.asarray(random_values(pairs, np.random.default_rng(1000 + seed)))
    supported = bool(fast.supports(dict(arrays, optimized_parameters=p)))
    text = f"{label:34s} {readout:8s} n={n:4d} variables={p.shape[0]:4d} fast={supported!s:5s}"
    # The noise detector reads the rows of the detector ports, so the gradient needs no second factorization.
    rows = np.concatenate([ports, ports + n])
    same_rows = set(np.nonzero(np.asarray(arrays["noise_selection_vectors"])[0, 0])[0].tolist()) == set(rows.tolist())
    start = time.perf_counter(); (_, reference), gradient_reference = evaluation(simulate_jax, arrays, jnp.asarray(ports))(p); seconds_jax = time.perf_counter() - start
    start = time.perf_counter(); (_, outputs), gradient = evaluation(df.simulate, arrays, jnp.asarray(ports))(p); seconds = time.perf_counter() - start
    carrier, signal = np.asarray(reference[0]), np.asarray(reference[1])
    parts = (np.abs(carrier[ports]) * (np.abs(signal[ports]) + np.abs(signal[ports + n]))).sum(axis=0)
    detected = np.asarray(df.signal_detector(reference[0], reference[1]))[ports]
    detected = np.abs(detected[0] if len(ports) == 1 else detected[0] - detected[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        cancellation = float(np.max(parts / detected))
        sensitivity = float(np.abs(np.log(np.asarray(outputs[3])) - np.log(np.asarray(reference[3]))).max())
    finite = all(bool(np.isfinite(np.asarray(x)).all()) for x in (*reference, gradient_reference))
    fields = max(rel(outputs[0], reference[0]), rel(np.asarray(outputs[1])[rows], signal[rows]), rel(outputs[2], reference[2]), rel(outputs[4], reference[4]))
    gradient_deviation = rel(gradient, gradient_reference)
    text += f"  fields {fields:.0e} |dln s| {sensitivity:.1e} gradient {gradient_deviation:.1e} cancellation {cancellation:.0e}  {seconds:5.2f} s (JAX {seconds_jax:4.1f} s)"
    ok = supported and same_rows
    if not finite:
        text += "  [no usable readout: reference not finite, skipped]"
    elif cancellation > LIMIT_CANCELLATION:
        ok = ok and fields <= LIMIT_FIELDS
        text += "  [readout cancels: only the fields are compared]"
    else:
        ok = ok and sensitivity <= LIMIT_SENSITIVITY and gradient_deviation <= LIMIT_GRADIENT and bool(np.any(np.asarray(gradient) != 0))
        worst["sensitivity"], worst["gradient"] = max(worst["sensitivity"], sensitivity), max(worst["gradient"], gradient_deviation)
    if not ok:
        failures.append(label)
        text += "  <-- FAILED"
    print(text, flush=True)


first_boundary = "01"
homodyne, detector = {first_boundary: "balanced_homodyne"}, {first_boundary: "detector"}
if QUICK:
    case(f"{SIZE}x{SIZE} space seed 0", SIZE, 0)
    case(f"{SIZE}x{SIZE} space homodyne seed 1", SIZE, 1, boundaries=homodyne)
else:
    for seed in range(SEEDS):
        case(f"{SIZE}x{SIZE} space seed {seed}", SIZE, seed)
    for seed in (10, 11):
        case(f"{SIZE}x{SIZE} space detector seed {seed}", SIZE, seed, boundaries=detector)
        case(f"{SIZE}x{SIZE} space homodyne seed {seed}", SIZE, seed, boundaries=homodyne)
    case(f"{SIZE}x{SIZE} frequency detector", SIZE, 20, "frequency_modulation", detector)
    case(f"{SIZE}x{SIZE} frequency homodyne", SIZE, 21, "frequency_modulation", homodyne)
    case(f"{SIZE}x{SIZE} amplitude detector (links)", SIZE, 22, "amplitude_modulation", detector)
    case(f"{SIZE}x{SIZE} amplitude homodyne (links)", SIZE, 23, "amplitude_modulation", homodyne)
    case(f"{SIZE}x{SIZE} homodyne, phase is a variable", SIZE, 24, boundaries=homodyne, extra_pairs=[(f"boundary{first_boundary}noise", "phase")])
    for size in (1, 2, 4):
        if size != SIZE:
            case(f"{size}x{size} space detector", size, 30 + size, boundaries=detector)
            case(f"{size}x{size} space homodyne", size, 50 + size, boundaries=homodyne)
print(f"WORST |dln s| {worst['sensitivity']:.2e} (limit {LIMIT_SENSITIVITY:.0e}), gradient {worst['gradient']:.2e} (limit {LIMIT_GRADIENT:.0e})")
if failures:
    sys.exit(f"FAILED: {failures}")
print("OK")
