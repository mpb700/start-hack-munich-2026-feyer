"""Dispatcher check on other public setups: fast path vs original JAX (outputs and full gradient).

aLIGO (space / frequency modulation): one noise detector, no qhd, no links  -> fast path
aLIGO (amplitude modulation): linked parameter                             -> JAX fallback
Voyager: balanced homodyne detector (qhd)                                  -> fast path
"""
import os
import json

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
import jax
import jax.numpy as jnp
import numpy as np

import differometor as df
from differometor import fast
from differometor.setups import aligo, voyager
from differometor.simulate import simulate_jax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300))
aligo_data = json.load(open(os.path.join(ROOT, "examples", "data", "aligo_test_data.json")))
frequencies = jnp.asarray(aligo_data["frequencies"])


def sensitivity(outputs, ports):
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[ports]
    detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
    return (noise / jnp.abs(detected)).squeeze()


def check(name, setup, pairs, expect_fast):
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], frequencies, pairs)
    ports = jnp.asarray(ports)
    values = np.zeros(int(arrays["optimized_value_indices"].max()) + 1)
    values[np.asarray(arrays["optimized_value_indices"])] = np.asarray(arrays["parameters"][0, arrays["optimized_parameter_indices"]]).real
    p0 = jnp.asarray(values)
    arrays = {k: v for k, v in arrays.items() if k != "optimized_parameters"}
    assert fast.supports(dict(arrays, optimized_parameters=p0)) == expect_fast, name

    def loss(sim):
        def f(p):
            out = sim(**arrays, optimized_parameters=p)
            sens = sensitivity(out, ports)
            return jnp.mean(jnp.log10(sens)), (out, sens)
        return jax.jit(jax.value_and_grad(f, has_aux=True))

    (_, (out, sens)), grad = loss(df.simulate)(p0)
    (_, (out_j, sens_j)), grad_j = loss(simulate_jax)(p0)
    print(f"{name}: path={'fast' if expect_fast else 'jax'}, {p0.shape[0]} optimized values")
    for key, a, b in zip(("carrier", "signal", "noise", "sensitivity", "gradient"), (*out, sens, grad), (*out_j, sens_j, grad_j)):
        print(f"  {key:12s} vs JAX {rel(a, b):.2e}")
    return sens


s = check("aLIGO space_modulation", *aligo("space_modulation"), True)
err = np.abs(np.log(np.asarray(s)) - np.log(aligo_data["sensitivities"])).max()
print(f"  aLIGO vs public test data: max |log diff| {err:.2e} (example uses atol 1e-6)")
check("aLIGO frequency_modulation", *aligo("frequency_modulation"), True)
check("aLIGO amplitude_modulation (linked)", *aligo("amplitude_modulation"), True)
check("Voyager (qhd)", *voyager(), True)
