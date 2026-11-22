"""LAZY_STDLIB_MODULES and LAZY_LATE_MODULES (differometor/_startup.py).

Usage: python runs/max/test_lazy_stdlib.py        (from the repository root, so that the carrier is found)

Each check runs in a process of its own that starts like benchmark.py: the carrier first, then the
rest of the header, then jax. "Has not run" means: lazy, or not imported at all (which modules the
standard library imports differs between patch versions of Python 3.12, e.g. ipaddress in urllib.parse).
A module that the interpreter had loaded before the carrier ran (a .pth file of an editable install can
do that) is an ordinary module and is left out of the comparison.
"""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAMES = ["bisect", "difflib", "glob", "gzip", "ipaddress", "random", "socketserver"]
HEADER = ("import sys\nearlier = set(sys.modules)\nimport argparse, json\nfrom pathlib import Path\nimport jax, jax.numpy as jnp\nimport numpy as np\nimport differometor\n"
          "names = %r\nlazy = lambda: [n for n in names if n not in sys.modules or type(sys.modules[n]).__name__ == '_LazyModule']\n"
          "expected = [n for n in names if n not in earlier]\n" % NAMES)
ALL_BUT_LAZY = "gc,defer,dataclass,jits,report,tpuprobe,npdocs,hlo"
failures = []
if sys.version_info < (3, 12, 3):
    print("skipped: the standard-library modules are lazy from Python 3.12.3 on")
    sys.exit(0)


def run(body, header=HEADER, **environment):
    env = dict(os.environ, **environment)
    env.pop("PYTHONPATH", None)
    out = subprocess.run([sys.executable, "-c", header + body], cwd=ROOT, env=env, capture_output=True, text=True)
    if out.returncode:
        failures.append("process failed: " + (out.stderr.strip().splitlines() or ["?"])[-1])
        return {}
    return json.loads(out.stdout.strip().splitlines()[-1])


def check(name, condition, detail=""):
    print(("ok   " if condition else "FAIL ") + name + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# What a program gets when it uses the modules: the same values with and without the measure.
USE = """
import bisect, difflib, glob, gzip, ipaddress, random, socketserver, urllib.parse
random.seed(7)
with socketserver.TCPServer(("127.0.0.1", 0), socketserver.BaseRequestHandler) as server:
    port_ok = server.server_address[1] > 0
values = {
    "random": [random.random(), random.randint(0, 99), random.Random(3).random()],
    "bisect": [bisect.bisect([1, 3, 5], 4), bisect.bisect_left([1, 3, 5], 3)],
    "difflib": list(difflib.unified_diff(["a", "b"], ["a", "c"], lineterm="")),
    "glob": sorted(glob.glob("benchmark*.py")),
    "gzip": gzip.decompress(gzip.compress(b"differometor", mtime=0)).decode(),
    "ipaddress": [str(ipaddress.ip_address("::1")), ipaddress.ip_network("10.0.0.0/30").num_addresses],
    "urlsplit": urllib.parse.urlsplit("http://[::1]:8080/x").hostname,
    "socketserver": port_ok,
    "einsum": float(jnp.einsum("ab,bc,cd,de,ef->af", *[jnp.eye(3) * (k + 1) for k in range(5)], optimize="greedy")[1, 1]),
}
"""

r = run("state = getattr(sys, '_differometor_startup', {})\nprint(json.dumps({'early': state.get('early'), 'lazy': lazy(), 'expected': expected, "
        "'late': type(sys.modules.get('jax.experimental.multihost_utils')).__name__}))")
check("after the header none of the seven modules has run", "lazy" in (r.get("early") or []) and len(r.get("lazy", [])) >= 6 and r.get("lazy") == r.get("expected"), str(r))

r = run("x = jnp.asarray([1.0, 2.0])\nprint(json.dumps({'late': type(sys.modules.get('jax.experimental.multihost_utils')).__name__, 'value': float(x.sum())}))")
check("multihost_utils is still unloaded after the first device_put", r.get("late") == "_LazyModule" and r.get("value") == 3.0, str(r))

on = run(USE + "print(json.dumps({'values': values, 'lazy': lazy(), 'same': all(sys.modules[n] is globals()[n] for n in names), "
         "'types': sorted({type(sys.modules[n]).__name__ for n in names})}))")
off = run(USE + "print(json.dumps({'values': values, 'lazy': lazy()}))", DIFFEROMETOR_STARTUP=ALL_BUT_LAZY)
check("used for real, every module is an ordinary module", on.get("lazy") == [] and on.get("types") == ["module"] and on.get("same"), str(on))
check("without `lazy` they are ordinary modules from the start", off.get("lazy") == [], str(off))
check("their results are the same with and without the measure", bool(on.get("values")) and on.get("values") == off.get("values"), f"{on.get('values')} != {off.get('values')}")

r = run("import threading\nseen = []\ndef touch():\n    import random\n    seen.append(random.Random(1).random())\n"
        "threads = [threading.Thread(target=touch) for _ in range(8)]\n[t.start() for t in threads]\n[t.join() for t in threads]\n"
        "print(json.dumps({'count': len(seen), 'equal': len(set(seen)) == 1, 'lazy': 'random' in lazy()}))")
check("eight threads touching a lazy module at once all get the real one", r.get("count") == 8 and r.get("equal") and not r.get("lazy"), str(r))

r = run("print(json.dumps({'lazy': lazy()}))", header="import random, gzip\n" + HEADER)
check("a module that was imported before the carrier stays as it is", "random" not in r.get("lazy", ["random"]) and "gzip" not in r.get("lazy", ["gzip"]), str(r))

# The property the measure rests on: a benchmark run does not touch the modules while it is timed.
BENCH = """
import sys
earlier = set(sys.modules)
import builtins, os, runpy
names = [n for n in %r if n not in earlier]
lazy = lambda: [n for n in names if n not in sys.modules or type(sys.modules[n]).__name__ == '_LazyModule']
real = builtins.print
seen = {}
def listen(*a, **k):
    if a and isinstance(a[0], str) and a[0].startswith('Repeat 01') and not seen:
        seen['lazy'] = lazy()   # printed after the timed first run and the first repetition
    real(*a, **k)
builtins.print = listen
sys.argv = ['%s'] + (['--cpus', '1'] if hasattr(os, 'sched_setaffinity') else []) + ['--repeats', '3', '--output', os.path.join(%r, 'r.json')]
sys.path.insert(0, os.getcwd())
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit:
    pass
import json
real(json.dumps({'after_first_run': seen.get('lazy'), 'expected': names}))
"""
import tempfile

with tempfile.TemporaryDirectory() as directory:
    for script in ("benchmark_aligo_original.py", "benchmark_original.py"):
        r = run("", header=BENCH % (NAMES, script, directory))
        check(f"{script}: none of them has run after the timed first run", len(r.get("expected", [])) >= 6 and r.get("after_first_run") == r.get("expected"), str(r))

print("all checks passed" if not failures else f"{len(failures)} failed")
sys.exit(1 if failures else 0)
