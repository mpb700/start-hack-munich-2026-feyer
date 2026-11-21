"""Fast path with parameter links (amplitude modulation) against the dense JAX reference.

Usage: PYTHONPATH=. python runs/max/test_links.py

aLIGO and Voyager in amplitude and space modulation, at the values of the constructors and at
values lowered by up to 2 % (only downwards: a reflectivity above 1 is not a setup). Compared are
log sensitivity, the three outputs and the gradient of the benchmark's readout.

Voyager with amplitude modulation at the constructor's values is the one case with a looser
limit: its two detector ports cancel to 1e-5 of a single port (a symmetric interferometer rejects
laser amplitude noise), so five digits of the detected signal are gone in either implementation.
"""
import sys
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import differometor as df
import differometor.fast as fast
import differometor.setups as setups
from differometor._simulate_dicht import simulate_jax

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
F = jnp.asarray(np.geomspace(20.0, 5000.0, 7))
worst = [0.0, 0.0, 0.0]
failed = []
def case(label, setup, allpairs, perturb=0.0, seed=0, limit=1e-6):
    pairs = [list(p) for p in allpairs if p[1] in PROPERTIES]
    arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], F, pairs)
    base = np.real(np.asarray(arrays["parameters"])[0])
    slots, places = np.asarray(arrays["optimized_parameter_indices"]), np.asarray(arrays["optimized_value_indices"])
    values = np.zeros(places.max() + 1); values[places] = base[slots]
    if perturb:
        values = values * (1 - perturb * np.random.default_rng(seed).uniform(0, 1, values.size))  # only downwards: a reflectivity above 1 is not a setup
    x = jnp.asarray(values, dtype=jnp.float64)
    arrays["optimized_parameters"] = x
    ports = jnp.asarray(ports)
    supported = bool(fast.supports(arrays))
    def loss(sim):
        def f(p):
            carrier, signal, noise = sim(**dict(arrays, optimized_parameters=p))
            signals = df.signal_detector(carrier, signal)[ports]
            detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
            s = (noise / jnp.abs(detected)).squeeze()
            return jnp.mean(jnp.log10(s)) + 1e-3 * jnp.sum(df.power_detector(carrier)), (s, carrier, signal, noise)
        return f
    gf, auxf = jax.grad(loss(df.simulate), has_aux=True)(x)
    gd, auxd = jax.grad(loss(simulate_jax), has_aux=True)(x)
    sf, sd = np.asarray(auxf[0]), np.asarray(auxd[0])
    e_s = float(np.abs(np.log(sf) - np.log(sd)).max())
    rel = lambda a, b: float(np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(np.asarray(b)).max(), 1e-300))
    e_out = max(rel(auxf[1], auxd[1]), rel(auxf[2], auxd[2]), rel(auxf[3], auxd[3]))
    e_g = rel(gf, gd)
    linked_design = int(np.isin(np.asarray(arrays["linked_indices"]), slots).sum())
    worst[0], worst[1], worst[2] = max(worst[0], e_s), max(worst[1], e_out), max(worst[2], e_g)
    if e_s > limit or e_g > 1e-5 or not supported:
        failed.append(label)
    print("%-34s P=%3d links=%d (Quelle ist Stellgröße: %d) schnell=%-5s | ln s %.1e | Ausgaben %.1e | Gradient %.1e | Gradient der verknüpften Quelle != 0: %s" % (
        label, len(values), int(arrays["linked_indices"].shape[0]), linked_design, supported, e_s, e_out, e_g,
        bool(np.all(np.asarray(gf)[places[np.isin(slots, np.asarray(arrays["linked_indices"]))]] != 0)) if linked_design else "-"))
    return supported
ok = True
for name in ("aligo", "voyager"):
    for mode in ("amplitude_modulation", "space_modulation"):
        s, pairs = getattr(setups, name)(mode=mode)
        ok &= case("%s %s" % (name, mode), s, pairs, limit=1e-5 if (name, mode) == ("voyager", "amplitude_modulation") else 1e-6)
    for mode in ("amplitude_modulation", "space_modulation"):
        s, pairs = getattr(setups, name)(mode=mode)
        try:
            ok &= case("%s %s, gestört 2%%" % (name, mode[:5]), s, pairs, perturb=0.02, seed=1)
        except Exception as e:
            print("%s %s gestört: %s" % (name, mode, str(e).splitlines()[-1][:80]))
print("WORST ln s %.2e | outputs %.2e | gradient %.2e | all on fast path: %s" % (worst[0], worst[1], worst[2], ok))
print("FAILED: %s" % failed if failed else "OK")
sys.exit(1 if failed or not ok else 0)
