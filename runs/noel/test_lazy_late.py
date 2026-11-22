"""`jax.experimental.multihost_utils` stays unloaded through the first array creation
(patches/philip-fastest-lazy-late.patch).

    python runs/noel/test_lazy_late.py          (from the root of a checkout that has the patch)

JAX imports that module inside the function behind the first `device_put` and uses it for
sharded arrays only. The checks run in fresh processes that start like `python benchmark.py`
(same header, started in the repo directory so that the `argparse` carrier is found):

1. After the header the module is in `sys.modules` as a lazy module, and it still is after the
   first `jnp.asarray`, a jitted function and `jax.device_put`.
2. Whoever uses the module gets the real one: its functions work.
3. With the lazy modules switched off (DIFFEROMETOR_STARTUP without `lazy`), and with the
   carrier switched off (DIFFEROMETOR_STARTUP_SHADOW=0), nothing is created ahead of time and
   JAX behaves as it always did.
"""
import os
import subprocess
import sys

ROOT = os.getcwd()
HEADER = """
import argparse
from datetime import datetime, timezone
import hashlib, importlib.metadata, json, os
from pathlib import Path
import platform, statistics, subprocess
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
import jax, jax.numpy as jnp, numpy as np
import differometor as df
import sys, importlib.util
state = getattr(sys, "_differometor_startup", None) or {}
name = "jax.experimental.multihost_utils"
lazy = importlib.util._LazyModule
kind = lambda: "absent" if name not in sys.modules else ("lazy" if type(sys.modules[name]) is lazy else "loaded")
before = kind()
x = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
y = jax.jit(lambda v: v * 2)(x)
z = jax.device_put(np.arange(3.0))
after = kind()
from jax.experimental import multihost_utils
imported = kind()
values = np.asarray(multihost_utils.broadcast_one_to_all(x))
print("RESULT", state.get("carrier"), "lazy" in state.get("early", []), before, after, imported, kind(), float(values.sum()), float(y.sum() + z.sum()))
"""
failures = []


def fresh(label, **environment):
    result = subprocess.run([sys.executable, "-c", HEADER], cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, **environment))
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT")]
    if result.returncode != 0 or not lines:
        failures.append(f"{label}: process failed: {result.stderr[-500:]}")
        return None
    carrier, lazy_on, before, after, imported, used, total, others = lines[-1].split()[1:]
    print(f"{label:34s} carrier {carrier:7s} lazy modules {lazy_on:5s} | before the first array: {before:6s} after: {after:6s} after the import statement: {imported:6s} after use: {used}")
    if (used, total, others) != ("loaded", "6.0", "15.0"):
        failures.append(f"{label}: the module does not work when used ({used}, {total}, {others})")
    return carrier, lazy_on == "True", before, after, imported


answer = fresh("default")
if answer:
    carrier, lazy_on, before, after, imported = answer
    if carrier != "None" and lazy_on:
        if (before, after, imported) != ("lazy", "lazy", "lazy"):
            failures.append(f"default: expected a lazy module throughout, got {before}, {after}, {imported}")
    else:
        print("   (the start-up measures are not active in this checkout: nothing to check for the default)")

answer = fresh("lazy modules switched off", DIFFEROMETOR_STARTUP="gc,defer,dataclass,jits,report")
if answer and answer[2] != "absent":
    failures.append(f"switched off: the module was created anyway ({answer[2]})")

answer = fresh("carrier switched off", DIFFEROMETOR_STARTUP_SHADOW="0")
if answer and answer[0] == "None" and answer[2] != "absent":  # (a .pth hook would be a carrier of its own)
    failures.append(f"carrier switched off: the module was created anyway ({answer[2]})")

print("OK" if not failures else "FAILED\n  " + "\n  ".join(failures))
sys.exit(1 if failures else 0)
