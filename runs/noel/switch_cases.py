"""The setups the kernel probes run on: fixture, aLIGO, Voyager and random grids with random design values."""
import json, os
import jax.numpy as jnp, numpy as np
import differometor as df
from differometor.setups import aligo, voyager, uifo, constrain_inter_grid_cell_spaces
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
repo = os.getcwd()
section = lambda data, name: data["nodes" if name in data["nodes"] else "edges"]
def anything(rng, prop):
    return {
        "reflectivity": lambda: [rng.uniform(0, 1), 10 ** rng.uniform(-9, -1), 1 - 10 ** rng.uniform(-6, -1)][rng.integers(3)],
        "tuning": lambda: rng.uniform(-360, 360),
        "db": lambda: rng.uniform(0, 10),
        "angle": lambda: rng.uniform(0, 360),
        "power": lambda: 10 ** rng.uniform(-6, 2.3),
        "mass": lambda: 10 ** rng.uniform(-2, 2.3),
        "length": lambda: rng.uniform(0, 4000),
    }[prop]()

def cases(seeds=8):
    fixture = json.load(open(f"{repo}/examples/data/uifo_800_3000.json"))
    grid = jnp.asarray(json.load(open(f"{repo}/examples/data/uifo_800_3000_test_data.json"))["frequencies"], dtype=jnp.float64)
    wide = jnp.asarray(json.load(open(f"{repo}/examples/data/aligo_test_data.json"))["frequencies"], dtype=jnp.float64)
    setup = df.Setup.from_data(fixture)
    yield "fixture", setup, constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES), grid
    for mode in ("space_modulation", "frequency_modulation", "amplitude_modulation"):
        yield f"aligo {mode}", *aligo(mode), wide
    yield "voyager", *voyager(), wide
    for size, count in ((3, seeds), (2, 3), (4, 2)):
        for seed in range(count):
            rng = np.random.default_rng(1000 * size + seed)
            setup, *_ = uifo(size, random=True, random_seed=seed, verbose=True)
            data = json.loads(json.dumps(setup.to_data()))
            setup = df.Setup.from_data(data)
            for pair in constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES):
                members = pair if isinstance(pair[0], list) else [pair]
                value = float(anything(rng, members[0][1]))
                for name, prop in members:
                    section(data, name)[name]["properties"][prop] = value
            setup = df.Setup.from_data(data)
            yield f"uifo{size} seed {seed}", setup, constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES), grid

