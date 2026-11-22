"""The heap measure of the start-up module (patches/*-heap.patch): HEAP_IN_LARGE_STEPS, HEAP_HUGE_PAGES.

    python runs/noel/test_heap.py          (from the root of a checkout that has the patch)

Fresh processes that start like `python benchmark.py` (same header, repo directory):

1. By default the measure reports what it did. On glibc that is "large steps, ..."; anywhere else
   (macOS) it reports that it is off, and nothing else changes.
2. On glibc a freed block of 8 MB is reused: allocating it again touches no new memory.
3. `DIFFEROMETOR_STARTUP` without `heap` switches it off, without `hugepages` leaves out the
   advice for huge pages, and with the carrier off it runs when differometor is imported.
4. NumPy computes the same in every case.
"""
import os
import subprocess
import sys

ROOT = os.getcwd()
CODE = """
import argparse
from datetime import datetime, timezone
import hashlib, importlib.metadata, json, os
from pathlib import Path
import platform, statistics, subprocess
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["JAX_ENABLE_X64"] = "true"
import jax, jax.numpy as jnp, numpy as np
import differometor as df
import sys, resource
state = getattr(sys, "_differometor_startup", None) or {}
faults = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_minflt
a = np.ones(1 << 20); del a          # 8 MB, freed again
f = faults(); b = np.ones(1 << 20); again = faults() - f
rng = np.random.default_rng(0); m = rng.normal(size=(300, 300))
check = float(np.linalg.solve(m, m @ np.arange(300.0)).sum()) + float(jnp.asarray(b[:5]).sum())
print("RESULT|%s|%s|%d|%.6f" % (state.get("carrier"), state.get("heap"), again, check))
"""
sys.path.insert(0, ROOT)
from differometor import _startup

default = sorted(_startup._early_options())
assert "heap" in default and "hugepages" in default, default
failures, answers = [], {}


def fresh(label, **environment):
    result = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, **environment))
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT|")]
    if result.returncode != 0 or not lines:
        failures.append(f"{label}: process failed: {result.stderr[-400:]}")
        return None
    carrier, heap, again, check = lines[-1].split("|")[1:]
    print(f"{label:22s} carrier {carrier:7s} heap: {heap:38s} page faults for a second 8 MB block: {again:>5s}")
    answers[label] = check
    return carrier, heap, int(again)


answer = fresh("default")
if answer:
    carrier, heap, again = answer
    if heap == "None":
        failures.append("default: the measure did not report")
    elif heap.startswith("large steps"):
        if again > 16:
            failures.append(f"default: a freed block of 8 MB is not reused ({again} page faults)")
    elif not heap.startswith("off: "):
        failures.append(f"default: unexpected report {heap!r}")
glibc = bool(answer) and answer[1].startswith("large steps")

answer = fresh("without heap", DIFFEROMETOR_STARTUP=",".join(o for o in default if o not in ("heap", "hugepages")))
if answer and answer[1] != "None":
    failures.append(f"without heap: the measure ran anyway ({answer[1]})")

answer = fresh("without huge pages", DIFFEROMETOR_STARTUP=",".join(o for o in default if o != "hugepages"))
if answer and glibc and answer[1] != "large steps":
    failures.append(f"without huge pages: {answer[1]!r}")

answer = fresh("carrier switched off", DIFFEROMETOR_STARTUP_SHADOW="0")
if answer and answer[1] == "None":
    failures.append("carrier switched off: the measure did not run with the import of differometor")

if len(set(answers.values())) > 1:
    failures.append(f"results differ between the cases: {answers}")
print("OK" if not failures else "FAILED\n  " + "\n  ".join(failures))
sys.exit(1 if failures else 0)
