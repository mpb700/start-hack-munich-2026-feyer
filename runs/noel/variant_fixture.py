"""Make a variant of the public fixture: a setup file and reference sensitivities for benchmark.py.

    python runs/noel/variant_fixture.py <upstream tree> <output folder> <kind> [seed]

The final evaluation uses "a similar optical setup ... with slightly different parameters and
components". This script writes such a setup as `uifo_800_3000.json` and its reference as
`uifo_800_3000_test_data.json` (the names benchmark.py reads) into the output folder. The
reference comes from the simulator in `<upstream tree>`, which should be the unchanged original
(`git archive 5eece1d | tar -x -C <folder>`), so it does not depend on the code under test.
`runs/noel/variant_run.py` runs the benchmark on the result.

Kinds:
  perturbed   the public fixture with every design value changed a little
  homodyne    the public fixture with a balanced homodyne readout in place of its detector
  swapped     the public fixture with one center and one boundary source of the other type
  random      `uifo(3, random=True, random_seed=seed)`, as the public fixture was drawn, with
              random design values; half of the seeds give a balanced homodyne readout

The public fixture stores the `centers` and `boundaries` of the generator call it came from,
which is how the kinds `homodyne` and `swapped` rebuild it with one thing changed. Design values
of parts that exist in the public fixture are taken from there.
"""
import json, os, sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
upstream, out, kind = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
sys.path.insert(0, str(upstream))
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces, uifo

PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
data_dir = upstream / "examples" / "data"
public = json.loads((data_dir / "uifo_800_3000.json").read_text())
frequencies = json.loads((data_dir / "uifo_800_3000_test_data.json").read_text())["frequencies"]
rng = np.random.default_rng(seed)
section = lambda data, name: data["nodes" if name in data["nodes"] else "edges"]


def generated(centers, boundaries, random=False):
    setup, _, centers, boundaries = uifo(3, centers=centers, boundaries=boundaries, random=random, random_seed=seed, verbose=True)
    data = setup.to_data()
    data["centers"] = {key: list(value) for key, value in centers.items()}
    data["boundaries"] = dict(boundaries)
    return json.loads(json.dumps(data))  # plain JSON types, as a stored fixture has them


def take_public_values(data):
    """Design values of the parts that the public fixture has as well."""
    for part in ("nodes", "edges"):
        for name, item in data[part].items():
            original = public[part].get(name)
            if original is not None and original.get("component") == item.get("component"):
                for prop in item["properties"]:
                    if prop in original["properties"]:
                        item["properties"][prop] = original["properties"][prop]


def set_design_values(data, change):
    """Give every design variable `change(property, old value)`; grouped lengths keep one value."""
    setup = df.Setup.from_data(data)
    for pair in constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES):
        members = pair if isinstance(pair[0], list) else [pair]
        name, prop = members[0]
        value = float(change(prop, section(data, name)[name]["properties"][prop]))
        for name, prop in members:
            section(data, name)[name]["properties"][prop] = value


def slightly(prop, value):
    if prop == "reflectivity":  # stays inside (0, 1): a small step of the logit
        value = min(max(value, 1e-12), 1 - 1e-12)
        return 1 / (1 + (1 / value - 1) * np.exp(rng.normal(0, 0.05)))
    if prop in ("tuning", "angle"):
        return value + rng.normal(0, 2.0)
    return value * np.exp(rng.normal(0, 0.05))


def anything(prop, value):
    return {
        "reflectivity": lambda: [rng.uniform(0, 1), 10 ** rng.uniform(-9, -1), 1 - 10 ** rng.uniform(-6, -1)][rng.integers(3)],
        "tuning": lambda: rng.uniform(-360, 360),
        "db": lambda: rng.uniform(0, 10),
        "angle": lambda: rng.uniform(0, 360),
        "power": lambda: 10 ** rng.uniform(-6, 2.3),
        "mass": lambda: 10 ** rng.uniform(-2, 2.3),
        "length": lambda: rng.uniform(0, 4000),
    }[prop]()


if kind == "perturbed":
    data = json.loads(json.dumps(public))
    set_design_values(data, slightly)
elif kind == "homodyne":
    readout = next(key for key, value in public["boundaries"].items() if value == "detector")
    data = generated(public["centers"], dict(public["boundaries"], **{readout: "balanced_homodyne"}))
    take_public_values(data)
elif kind == "swapped":
    centers, boundaries = dict(public["centers"]), dict(public["boundaries"])
    center = sorted(centers)[seed % len(centers)]
    centers[center] = ["directional_beamsplitter" if centers[center][0] == "beamsplitter" else "beamsplitter", centers[center][1]]
    sources = sorted(key for key, value in boundaries.items() if value in ("laser", "squeezer"))
    source = sources[seed % len(sources)]
    boundaries[source] = "squeezer" if boundaries[source] == "laser" else "laser"
    data = generated(centers, boundaries)
    take_public_values(data)
    print(f"center {center} is now a {centers[center][0]}, boundary {source} a {boundaries[source]}")
elif kind == "random":
    data = generated(None, None, random=True)
    set_design_values(data, anything)
else:
    sys.exit(__doc__)

# Reference sensitivities with the simulator of the upstream tree, as benchmark.py reads them out.
setup = df.Setup.from_data(data)
pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
values = [section(data, (pair if isinstance(pair[0], str) else pair[0])[0])[(pair if isinstance(pair[0], str) else pair[0])[0]]["properties"][(pair if isinstance(pair[0], str) else pair[0])[1]] for pair in pairs]
arrays, ports, *_ = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(frequencies, dtype=jnp.float64), pairs)
arrays["optimized_parameters"] = jnp.asarray(values, dtype=jnp.float64)
ports = jnp.asarray(ports)


@jax.jit
def sensitivity(arrays):
    carrier, signal, noise = df.simulate(**arrays)
    signals = df.signal_detector(carrier, signal)[ports]
    detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
    return (noise / jnp.abs(detected)).squeeze()


reference = np.asarray(sensitivity(arrays))
if reference.shape != (len(frequencies),) or not (np.isfinite(reference).all() and (reference > 0).all()):
    sys.exit(f"no usable reference for this variant: shape {reference.shape}, finite {bool(np.isfinite(reference).all())}")
out.mkdir(parents=True, exist_ok=True)
(out / "uifo_800_3000.json").write_text(json.dumps(data))
(out / "uifo_800_3000_test_data.json").write_text(json.dumps({"frequencies": frequencies, "sensitivities": reference.tolist()}))
components = {}
for item in data["nodes"].values():
    components[item["component"]] = components.get(item["component"], 0) + 1
print(f"{kind} (seed {seed}): {len(pairs)} design variables, readout with {ports.shape[0]} port(s), simulator {df.__file__}")
print(f"components: {dict(sorted(components.items()))}")
print(f"sensitivity {reference.min():.3e} to {reference.max():.3e}, written to {out}")
