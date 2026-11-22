"""Checks for the start-up measures in differometor/_startup.py that put work off or reuse compiled code.

    python runs/philip/test_startup.py      (from anywhere; prints one line per check, exits non-zero on failure)

Runs the header of benchmark.py in this process (with the carrier argparse.py), then uses the three
deferred modules for real and compares with what an ordinary import gives. The modules of the header that
are put off until the report (DEFER_REPORT_MODULES) are checked in a process of their own.
"""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.chdir(root)
sys.path.insert(0, str(root))
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
for name in ("DIFFEROMETOR_STARTUP", "DIFFEROMETOR_STARTUP_SHADOW"):
    os.environ.pop(name, None)

failures = []


def check(name, condition, detail=""):
    print(("ok   " if condition else "FAIL ") + name + (f"  [{detail}]" if detail and not condition else ""))
    if not condition:
        failures.append(name)


import argparse  # noqa: E402  the carrier, found next to benchmark.py as in a benchmark run

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import differometor  # noqa: E402, F401

state = sys._differometor_startup
kind = lambda name: type(sys.modules[name]).__name__ if name in sys.modules else "absent"
deferred = ("http.server", "numpy.ma", "opt_einsum")

check("the real argparse module is in place", argparse.ArgumentParser(add_help=False).parse_known_args(["--x"])[1] == ["--x"] and "carrier" not in (argparse.__doc__ or "")
      and Path(argparse.__file__).resolve().parent != root)
check("the three modules are deferred after the imports", all(kind(name) == "_DeferredModule" for name in deferred), str({n: kind(n) for n in deferred}))
check("nothing they import has been loaded", all(kind(name) == "absent" for name in ("http.client", "ssl", "numpy.ma.core", "opt_einsum.contract")))
import http.server  # noqa: E402
import numpy.ma  # noqa: E402
import opt_einsum  # noqa: E402, F401

check("importing them again does not load them", all(kind(name) == "_DeferredModule" for name in deferred))
check("numpy.unique does not load numpy.ma", np.array_equal(np.unique(np.array([3, 1, 3])), [1, 3]) and kind("numpy.ma") == "_DeferredModule")

# ---- dataclass code ----------------------------------------------------------------------------
import dataclasses  # noqa: E402

check("dataclass code is installed", str(state.get("dataclass_code", "")).startswith("on:"), str(state.get("dataclass_code")))


@dataclasses.dataclass(frozen=True, order=True)
class Fresh:
    a: int
    b: str = "x"
    c: list = dataclasses.field(default_factory=list, compare=False)


check("a dataclass that is not in the code file works", repr(Fresh(1)) == "Fresh(a=1, b='x', c=[])" and Fresh(1) == Fresh(1, c=[2]) and Fresh(1) < Fresh(2)
      and hash(Fresh(3)) == hash(Fresh(3)))
probe = """
import dataclasses, json, os, sys
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, os.getcwd())
if sys.argv[1] == "with":
    import argparse
import jax
from jax._src import core, pjit
out = {}
for cls in (pjit.PjitInfo, core.JaxprDebugInfo if hasattr(core, "JaxprDebugInfo") else core.DebugInfo, jax.ShapeDtypeStruct):
    for name in ("__init__", "__repr__", "__eq__", "__hash__", "__replace__", "__setattr__"):
        method = cls.__dict__.get(name)
        code = getattr(method, "__code__", None)
        if code is not None:
            out[f"{cls.__name__}.{name}"] = [code.co_code.hex(), repr(code.co_names), repr(code.co_varnames), code.co_filename, code.co_firstlineno]
out["installed"] = str(getattr(sys, "_differometor_startup", {}).get("dataclass_code"))
print(json.dumps(out))
"""
import json  # noqa: E402

runs = {}
for mode in ("with", "without"):
    env = dict(os.environ, DIFFEROMETOR_STARTUP_SHADOW="1" if mode == "with" else "0")
    result = subprocess.run([sys.executable, "-c", probe, mode], cwd=root, env=env, capture_output=True, text=True)
    runs[mode] = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 else {"error": result.stderr[-500:]}
methods = [key for key in runs["with"] if key != "installed"]
check("methods of JAX dataclasses have the same code with and without the code file",
      runs["with"].get("installed", "").startswith("on:") and runs["without"].get("installed") == "None" and len(methods) >= 3
      and all(runs["with"][key] == runs["without"].get(key) for key in methods), str({k: v for k, v in runs["with"].items() if k == "installed"}) + str(runs["without"].get("error", "")))

# ---- lazy jit objects ---------------------------------------------------------------------------
from jax._src.numpy import reductions  # noqa: E402

created, built = state.get("lazy_jits", [0, 0])
check("JAX's own jit objects are lazy and almost none is built after the imports", state.get("lazy_jits_installed") == "on" and created > 150 and built < 25,
      f"{state.get('lazy_jits_installed')} created {created} built {built}")
lazy = reductions._mean
check("a lazy jit object is built by its first call, once", type(lazy).__name__ == "_LazyJit" and lazy._lazy_real is None
      and float(jnp.mean(jnp.arange(4.0))) == 1.5 and type(lazy._lazy_real).__name__ == "PjitFunction" and lazy._lazy_pending is None)
check("a function from outside JAX gets JAX's jit object at once", type(jax.jit(lambda x: x)).__name__ == "PjitFunction")

# The same battery of JAX calls with and without all start-up measures.
import re  # noqa: E402

battery = {}
for mode in ("1", "0"):
    result = subprocess.run([sys.executable, str(root / "runs" / "philip" / "jax_battery.py")], cwd=root, env=dict(os.environ, DIFFEROMETOR_STARTUP_SHADOW=mode),
                            capture_output=True, text=True)
    battery[mode] = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 and result.stdout.strip() else {"error": result.stderr[-800:]}


def normalized(value):
    """Without memory addresses; and without the one traceback frame a lazy jit object adds to an error."""
    if isinstance(value, dict):
        return {key: ([frame for frame in item if frame != "_startup.py"] if key == "frames" else normalized(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return re.sub(r"0x[0-9a-f]+", "0x", value) if isinstance(value, str) else value


different = [key for key in sorted(set(battery["1"]) | set(battery["0"])) if normalized(battery["1"].get(key)) != normalized(battery["0"].get(key))]
check("83 JAX calls give the same numbers, names, signatures and errors with and without the measures", len(battery["1"]) >= 80 and not different,
      f"{len(battery['1'])} entries, different: {different[:6]} {battery['1'].get('error', '')}")
extra = [key for key in battery["1"] if key.startswith("error.") and isinstance(battery["1"][key], dict)
         and battery["1"][key]["frames"].count("_startup.py") > 1]
check("an error passes through at most one frame of the start-up measures", not extra, str(extra))

# ---- opt_einsum ---------------------------------------------------------------------------------
from jax._src.numpy import einsum as jax_einsum  # noqa: E402

a, b = np.arange(6.0).reshape(2, 3), np.arange(12.0).reshape(3, 4)
check("jnp.einsum works and loads opt_einsum", np.allclose(jnp.einsum("ij,jk->ik", a, b), a @ b) and kind("opt_einsum") == "module")
check("JAX's path optimizer is a subclass of the real class", issubclass(jax_einsum.Unoptimized, opt_einsum.paths.PathOptimizer)
      and opt_einsum.paths is sys.modules["opt_einsum.paths"] and callable(opt_einsum.contract))

# ---- numpy.ma -----------------------------------------------------------------------------------
masked = np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0])
check("a masked array loads numpy.ma", kind("numpy.ma") == "module" and float(masked.sum()) == 4.0 and np.ma.is_masked(masked))
from jax._src import core, dtypes  # noqa: E402
from jax._src.interpreters import mlir, pxla  # noqa: E402

tables = (core.pytype_aval_mappings, dtypes.canonicalize_value_handlers, mlir._constant_handlers, pxla.shard_arg_handlers)
check("JAX's tables know the real MaskedArray and no stand-in", all(np.ma.MaskedArray in table for table in tables)
      and not any(type(key).__name__ == "_StandInMeta" for table in tables for key in table))
messages = []
for call in (lambda: jnp.asarray(masked), lambda: jnp.sin(masked), lambda: jax.jit(lambda x: x + 1)(masked)):
    try:
        call()
        messages.append("no error")
    except Exception as error:
        messages.append(f"{type(error).__name__}: {error}")
check("masked arrays are rejected with JAX's own error", all(message.startswith("ValueError: numpy masked arrays are not supported") for message in messages), str(messages))

# ---- http.server --------------------------------------------------------------------------------
from jax._src import profiler  # noqa: E402

check("the profiler's handler class still waits for http.server", kind("http.server") == "_DeferredModule")


class Request:
    def makefile(self, *args, **kwargs):
        import io

        return io.BytesIO()

    def sendall(self, data):
        pass


handler = profiler._PerfettoServer(Request(), ("127.0.0.1", 0), type("Server", (), {})())
check("creating the handler loads http.server and gives a real request handler", kind("http.server") == "module"
      and isinstance(handler, http.server.SimpleHTTPRequestHandler)
      and [c.__name__ for c in profiler._PerfettoServer.__mro__][:3] == ["_PerfettoServer", "SimpleHTTPRequestHandler", "BaseHTTPRequestHandler"])
check("all three were loaded exactly once", all(state.get("deferred_loaded", []).count(name) == 1 for name in deferred), str(state.get("deferred_loaded")))

# ---- modules of the benchmark header that wait for the report -----------------------------------
import tempfile  # noqa: E402

header = """
import json, os, sys
sys.path.insert(0, os.getcwd())
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
import differometor
kind = lambda name: type(sys.modules[name]).__name__ if name in sys.modules else "absent"
names = ("importlib.metadata", "statistics", "hashlib", "subprocess")
out = {"after_imports": {name: kind(name) for name in names}}
jax.devices("cpu")
state = sys._differometor_startup
out["after_backend"] = {name: kind(name) for name in names + ("email.message", "importlib.metadata._meta")}
out["answered"] = list(state.get("entry_points_answered") or [])
out["sum"] = float(jnp.sum(jnp.arange(3.0)))
nothing = importlib.metadata.entry_points(group="differometor-no-such-group")
out["nothing"] = [list(nothing), len(nothing), kind("importlib.metadata")]
out["nothing_names"] = [sorted(nothing.names), kind("importlib.metadata")]
out["version"] = importlib.metadata.version("jax") == jax.__version__
out["median"] = statistics.median([3, 1, 2])
out["sha"] = hashlib.sha256(b"abc").hexdigest()[:8]
out["echo"] = subprocess.run([sys.executable, "-c", "print(7)"], capture_output=True, text=True).stdout.strip()
out["after_use"] = {name: kind(name) for name in names}
out["loaded"] = state.get("deferred_loaded")
print(json.dumps(out))
"""
runs = {}
for mode, value in (("on", None), ("off", "gc,lazy,defer,dataclass,jits")):
    env = {key: item for key, item in os.environ.items() if key != "DIFFEROMETOR_STARTUP"}
    if value:
        env["DIFFEROMETOR_STARTUP"] = value
    result = subprocess.run([sys.executable, "-c", header], cwd=root, env=env, capture_output=True, text=True)
    runs[mode] = json.loads(result.stdout.strip().splitlines()[-1]) if result.returncode == 0 and result.stdout.strip() else {"error": result.stderr[-600:]}
on, off = runs["on"], runs["off"]
report = ("importlib.metadata", "statistics", "hashlib", "subprocess")
check("the four report modules are put off by the header of benchmark.py", all(on.get("after_imports", {}).get(name) == "_DeferredModule" for name in report), str(on)[:400])
check("starting the CPU backend loads none of them and nothing importlib.metadata imports",
      all(on.get("after_backend", {}).get(name) == "_DeferredModule" for name in report)
      and on["after_backend"]["email.message"] == on["after_backend"]["importlib.metadata._meta"] == "absent", str(on.get("after_backend")))
check("JAX's question about plugins was answered without the module", on.get("answered") == ["jax_plugins"], str(on.get("answered")))
check("a group nobody advertises gives an empty tuple; its methods load the module", on.get("nothing") == [[], 0, "_DeferredModule"] and on.get("nothing_names") == [[], "module"],
      str((on.get("nothing"), on.get("nothing_names"))))
check("the four work when used and are then ordinary modules, loaded once", on.get("version") is True and on.get("median") == 2 and on.get("sha") == "ba7816bf" and on.get("echo") == "7"
      and all(on.get("after_use", {}).get(name) == "module" for name in report) and all((on.get("loaded") or []).count(name) == 1 for name in report), str(on)[:500])
check("with the switch off the header loads them as always, with the same results", all(off.get("after_imports", {}).get(name) == "module" for name in report)
      and off.get("answered") == [] and on.get("sum") == 3.0 and all(off.get(key) == on.get(key) for key in ("version", "median", "sha", "echo", "sum")) and off.get("nothing_names", [None])[0] == [], str(off)[:500])

# The cheap answer against the real one, with and without a distribution that advertises the group.
hook = sys.modules["_differometor_startup"]
import importlib.metadata  # noqa: E402

group = "differometor-test-group"
check("no distribution advertises the test group: certain", hook._no_entry_point_group(group) is True and len(importlib.metadata.entry_points(group=group)) == 0)
with tempfile.TemporaryDirectory() as folder:
    info = Path(folder) / "fake_plugin-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fake-plugin\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(f"[{group}]\nfake = fake_plugin_module\n")
    sys.path.append(folder)
    try:
        found = importlib.metadata.entry_points(group=group)
        check("a distribution that advertises it: not certain, and the real search finds it", hook._no_entry_point_group(group) is False and [entry.value for entry in found] == ["fake_plugin_module"])
        check("another group stays certain next to it", hook._no_entry_point_group("differometor-other-group") is True)
    finally:
        sys.path.remove(folder)
archive = Path(tempfile.mkdtemp()) / "archive.zip"
archive.write_bytes(b"")
sys.path.append(str(archive))
try:
    check("a file on sys.path: not certain", hook._no_entry_point_group(group) is False)
finally:
    sys.path.remove(str(archive))


# ---- prefetch table ------------------------------------------------------------------------------
import marshal  # noqa: E402

table = marshal.loads((root / "differometor" / "_prefetch.bin").read_bytes())
check("the prefetch table lists files of installed packages with size and byte ranges", len(table) > 100
      and all(type(relative) is str and not relative.startswith(("/", "..")) and type(size) is int and ranges
              and all(0 <= offset and 0 < length and offset + length <= size for offset, length in ranges) for relative, size, ranges in table)
      and table[0][0] == "jaxlib/libjax_common.so", str(table[0][:2]))
note = state.get("prefetch") or {}
expected = ("requested", "warm cache, nothing requested") if hasattr(os, "posix_fadvise") else ("off: no posix_fadvise on this platform",)
check("the prefetch ran or said why not, without an error", "prefetch" in state.get("early", []) and note.get("state") in expected, str(note))
switched_off = subprocess.run([sys.executable, "-c", "import os, sys; sys.path.insert(0, os.getcwd()); import argparse; print(sys._differometor_startup.get('prefetch'))"],
                              cwd=root, env=dict(os.environ, DIFFEROMETOR_STARTUP="gc,lazy,defer,dataclass,jits,report"), capture_output=True, text=True)
check("without `prefetch` in DIFFEROMETOR_STARTUP nothing is requested", switched_off.stdout.strip().splitlines()[-1:] == ["None"], switched_off.stdout[-200:] + switched_off.stderr[-200:])

print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
sys.exit(1 if failures else 0)
