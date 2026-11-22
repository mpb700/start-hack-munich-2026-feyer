"""A beamsplitter beyond the critical angle (total internal reflection): fast path against the dense JAX path.

    PYTHONPATH=. python runs/philip/test_tir.py      (prints one line per case, exits non-zero on failure)

With glass (n = 1.45) behind the 45 degree beamsplitter `center12` of the public UIFO, Snell's law has no
real solution: `arcsin(n_left / n_right * sin(alpha))` gets an argument of magnitude 1.025. The dense JAX
path computes with complex parameters and takes the value on the branch cut. The component functions of
the fast path get real numbers, and NumPy's real arcsin returned nan there: the carrier matrix became
singular ("Factor is exactly singular"). `fast._complex_outside` hands such arguments to NumPy as complex
numbers. Found by other teams (SudeepJoshi21 911074c, YuechenWang331 623b0a6); the cases are those of
their "edge-cases bundle" plus the mirrored angles.

Part 1 compares sqrt and arcsin of the NumPy shim with jax.numpy on the complex numbers the dense path
sees (zero imaginary part of either sign), values and derivatives. Part 2 runs the setups.

Not covered, and not a difference to the original code: when the beamsplitter has no design variable at
all, its matrix comes from the build step alone, which computes with real numbers in the original code
too; the original then returns nan everywhere.
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
from differometor.setups import Setup, constrain_inter_grid_cell_spaces
from differometor.simulate import simulate_jax

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "examples", "data")
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
SPACES = ("center12_ml12", "center12_mt12")
LIMIT = 1e-6  # ln(sensitivity) per frequency, the tolerance of benchmark_aligo.py; gradient relative to its largest entry


def branch_cuts():
    """sqrt and arcsin of the shim on real numbers against jax.numpy on the same numbers with +0j and -0j."""
    points = np.array([-4.0, -1.025, -1.0000001, -0.3, 0.0, 0.3, 0.9999999, 1.025, 1.5, 2.0])
    shim, worst = fast._NumpyShim, 0.0
    for name in ("sqrt", "arcsin"):
        inside = points[(points > 0) if name == "sqrt" else (np.abs(points) != 1)]  # finite slope
        ours = getattr(shim, name)(fast.Dual(inside, np.ones((1,) + inside.shape)))
        plain = getattr(shim, name)(inside)
        for sign in (0.0, -0.0):
            z = jnp.asarray(inside + 1j * np.full(inside.shape, sign))
            value, slope = jax.jvp(getattr(jnp, name), (z,), (jnp.ones_like(z),))
            worst = max(worst, float(np.abs(np.asarray(ours.val) - np.asarray(value)).max()),
                        float(np.abs(np.asarray(plain) - np.asarray(value)).max()),
                        float((np.abs(np.asarray(ours.tan[0]) - np.asarray(slope)) / np.abs(np.asarray(slope))).max()))
    ok = worst <= 1e-12
    print(f"branch cuts of sqrt and arcsin against jax.numpy: largest deviation {worst:.1e} -> {'ok' if ok else 'FAILED'}", flush=True)
    return ok


def detected(outputs, ports):
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[ports]
    return signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]


def check(name, index, alpha):
    fixture = json.load(open(os.path.join(DATA, "uifo_800_3000.json")))
    for space in SPACES:
        fixture["edges"][space]["properties"]["refractive_index"] = index
    if alpha is not None:
        fixture["nodes"]["center12"]["properties"]["alpha"] = alpha
    setup = Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    frequencies = jnp.asarray(json.load(open(os.path.join(DATA, "uifo_800_3000_test_data.json")))["frequencies"])
    with np.errstate(invalid="ignore"):  # the build step evaluates the constant matrix with real numbers, as the original does
        arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    ports = jnp.asarray(ports)
    values = np.zeros(int(arrays["optimized_value_indices"].max()) + 1)
    values[np.asarray(arrays["optimized_value_indices"])] = np.asarray(arrays["parameters"][0, arrays["optimized_parameter_indices"]]).real
    p0 = jnp.asarray(values)
    arrays = {k: v for k, v in arrays.items() if k != "optimized_parameters"}
    on_fast = bool(fast.supports(dict(arrays, optimized_parameters=p0)))

    def run(sim):
        def f(p):
            out = sim(**arrays, optimized_parameters=p)
            sens = (out[2] / jnp.abs(detected(out, ports))).squeeze()
            return jnp.mean(jnp.log10(sens)), (out[0], sens)
        return jax.jit(jax.value_and_grad(f, has_aux=True))(p0)

    try:
        (_, (carrier, sens)), grad = run(df.simulate)
    except Exception as error:  # noqa: BLE001 - the failure is the finding
        print(f"{name}: fast path {on_fast}, FAILED with {type(error).__name__}: {str(error).strip().splitlines()[-1][:120]}", flush=True)
        return False
    (_, (carrier_j, sens_j)), grad_j = run(simulate_jax)
    carrier, sens, grad, carrier_j, sens_j, grad_j = (np.asarray(a) for a in (carrier, sens, grad, carrier_j, sens_j, grad_j))
    finite = all(np.all(np.isfinite(a)) for a in (carrier, sens, grad, sens_j, grad_j))
    e_sens = float(np.abs(np.log(sens) - np.log(sens_j)).max())
    e_carrier = float(np.abs(carrier - carrier_j).max() / np.abs(carrier_j).max())
    e_grad = float(np.abs(grad - grad_j).max() / np.abs(grad_j).max())
    ok = on_fast and finite and e_sens <= LIMIT and e_grad <= LIMIT and e_carrier <= LIMIT
    print(f"{name}: fast path {on_fast}, finite {finite}, ln(sens) {e_sens:.1e}, carrier {e_carrier:.1e}, gradient {e_grad:.1e} -> {'ok' if ok else 'FAILED'}", flush=True)
    return ok


if __name__ == "__main__":
    cases = {
        "glass n=1.2 (refraction, no total reflection)": (1.2, None),
        "n=1.45, alpha 45 (arcsin of +1.025)": (1.45, None),
        "n=1.45, alpha 135": (1.45, 135.0),
        "n=1.45, alpha 225 (arcsin of -1.025)": (1.45, 225.0),
        "n=1.45, alpha -45": (1.45, -45.0),
        "n=1.45, alpha -135": (1.45, -135.0),
        "n=sqrt(2), alpha 45 (at the critical angle)": (float(np.sqrt(2.0)), None),
    }
    results = [branch_cuts()] + [check(name, *case) for name, case in cases.items()]
    sys.exit(0 if all(results) else 1)
