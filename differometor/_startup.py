"""Process-level start-up measures: what happens around `import jax` and with the garbage collector.

This file has two entry points and is deliberately standalone (standard library only, nothing
from differometor, jax or numpy at module level), because the first entry point runs before any
of those can be imported:

``early()``
    Has to run *before* ``import jax`` to have any effect. `differometor` itself is imported
    after `jax` by the benchmark, so `early()` needs one of two carriers:

    * the start-up hook ``differometor_startup.pth`` (installed into site-packages by
      ``pip install -e .`` when the opt-in in ``setup.py`` is on). It runs at interpreter
      start, i.e. before the first timer of ``benchmark.py``;
    * the module ``argparse.py`` next to ``benchmark.py`` (see there). It runs inside the
      timed window, at the ``import argparse`` line of the benchmark, its first import.

    Without a carrier `early()` never runs and nothing below marked "early" happens.

``late()``
    Called at the end of ``differometor/__init__.py``. Needs no carrier.

Every measure has a switch below. Two kinds of measure are kept strictly apart:

kind (A)  removes work from the process (collector passes that find nothing, modules that the
          run never touches). Nothing is computed ahead of time.
kind (B)  does NOT remove work. It moves the imports in front of the first timer of
          ``benchmark.py`` ("interpreter launch", which the benchmark excludes). Off by default.
          Only possible with the ``.pth`` carrier.
kind (C)  does not remove work either. It leaves a module that the benchmark script imports in
          its header, but uses only when it writes its report, unloaded until that use. The
          module's code then runs after the timed phases instead of inside the timed imports.
          See DEFER_REPORT_MODULES; the part of it that concerns JAX is of kind (A).
kind (D)  removes waiting for the disk, not work: file pages the process is going to read are
          requested from the kernel early, so that they are read side by side. See PREFETCH.

For measurements the switches can be overridden per process without editing this file:
``DIFFEROMETOR_STARTUP`` is a comma separated list out of ``gc``, ``lazy``, ``defer``,
``dataclass``, ``jits``, ``report``, ``prefetch``, ``preimport`` (or ``none``) and replaces the eight early switches; ``DIFFEROMETOR_GC`` replaces ``GC_RUN``.
"""
import sys

# --------------------------------------------------------------------------------------------
# Switches
# --------------------------------------------------------------------------------------------

# early, kind (A): the cyclic garbage collector does not run while `import jax` executes.
# Importing JAX allocates about 110 000 long-lived objects (functions, classes, dicts) and no
# garbage worth mentioning; the collector nevertheless runs about 150 times over them. When the
# import is done, the objects are frozen (moved to the permanent generation, so no later pass
# has to look at them either) and the collector is switched back on.
GC_PAUSE_JAX_IMPORT = True

# early, kind (A): the JAX modules in LAZY_MODULES are created empty and executed on first use
# (importlib.util.LazyLoader). `import jax` pulls them in, this workload never touches them.
# Anything that does touch them later gets the real module, just later.
LAZY_JAX_MODULES = True

# early, kind (A): three modules outside JAX that `import jax` executes only to define things a
# run like this never touches (DEFERRED_MODULES below): `http.server` (base class of a helper of
# the JAX profiler; pulls in http.client, ssl, html), `numpy.ma` (JAX registers "masked arrays are
# not supported" handlers for its array class) and `opt_einsum` (JAX subclasses one class of
# `opt_einsum.paths`). Their code runs on first real use instead. Until then JAX sees a stand-in
# class where it only needs a class object; when the module is really loaded, every use of the
# stand-in is moved over to the real class (see `_DeferredModule`). Measured with
# `python -X importtime` on the Xeon Platinum 8468: 7.6 + 4.9 + 5.0 ms inside `import jax`.
# `numpy.unique` asks `numpy.ma.is_masked` on every call; that one function is answered without
# loading the module (`_is_masked`).
DEFER_UNUSED_MODULES = True

# early, kind (A): compiled code for the methods of dataclasses. `dataclasses` writes __init__,
# __repr__, __eq__ and so on of every dataclass as source text and compiles that text when the
# class is created. `import jax` creates 68 dataclasses with 340 such methods; compiling them
# takes 17 ms on an M4, more than a tenth of the import. The file DATACLASS_CODE_FILE next to
# this module holds the compiled code for those texts (as `__pycache__` does for module
# sources), keyed by the complete text: a method whose text is in the file is not compiled
# again, any other text is compiled as always. The text is built exactly as
# `dataclasses._create_fn` of Python 3.12 builds it; with another Python, or if that function is
# not the expected one, nothing is installed. Rebuild the file with `runs/philip/bytecode.sh`.
DATACLASS_CODE = True
DATACLASS_CODE_FILE = "_dataclass_code.bin"

# early, kinds (A) and (C): four modules from the header of `benchmark.py` are loaded when
# somebody really needs them (REPORT_MODULES below). `statistics`, `hashlib` and `subprocess`
# are first used by the script itself, between or after its timed phases (medians for the
# report, checksums of the fixtures, `git` for the report). JAX wants `hashlib` too, for the
# keys of its compilation cache at the first compilation; the script's checksums come before
# that. This part is purely of kind (C). For `importlib.metadata` two places ask in a
# benchmark run:
#   * the header of `benchmark.py` imports it and uses it once, for the package versions in the
#     report it writes after all timed phases. That import pulls in `email`, `zipfile`, `csv`,
#     `tempfile`, `shutil` and thirty more modules nothing else in the run needs. With this
#     switch the module object exists after the header and its code runs at that first use
#     (kind (C): the work is still done, after the timers).
#   * JAX looks for backend plugins when it creates the CPU backend: it asks
#     `importlib.metadata.entry_points(group="jax_plugins")`, which parses the metadata of every
#     installed distribution. That one question is answered here without the module (kind (A)):
#     `_no_entry_point_group` reads the `entry_points.txt` files of the same distributions and
#     looks for the group name. Only when no file mentions it is the answer "none" given here;
#     if one does, or anything about the installation is unusual (a zip file on `sys.path`, a
#     finder with distributions of its own), the module is loaded and asked as always. A machine
#     with a JAX plugin installed therefore still gets its plugin.
# Needs a carrier that runs before the header's `import hashlib` (argparse.py or the .pth hook).
# A module that something does need earlier is simply loaded then.
DEFER_REPORT_MODULES = True

# early, kind (D): ask the kernel early for the file pages a start with a cold file cache will
# read. Neither work nor results change; what goes away is waiting. A process that starts
# with a cold cache (first run on a machine, fresh job) reads about 160 MB from disk before
# the timed phases end, 120 MB of them pages of jaxlib/libjax_common.so, and it reads them
# one page fault after the other. The table PREFETCH_FILE lists for the files of the installed
# packages which byte ranges a benchmark run touched (recorded on the Xeon VM with
# runs/philip/prefetch_map.py, 277 files, 145 MB). When the carrier runs, these ranges are
# requested with posix_fadvise(WILLNEED): the kernel starts all the reads at once and the
# imports find the pages in the cache or on their way.
#   * A file is used only if it exists under the same relative path with the recorded size;
#     another package version is simply skipped. A wrong or missing hint costs or saves
#     waiting time, never a result.
#   * With a warm cache nothing is requested: one read of a single byte that must not wait
#     (preadv with RWF_NOWAIT) at three places of the largest file tells whether its pages are
#     in memory; only if one is not does the prefetch run.
# Measured on the Xeon Platinum 8468 with the cache emptied before each process (job
# prefetch-0114): first run UIFO 1528 -> 512 ms, aLIGO 1501 -> 475 ms.
PREFETCH = True
PREFETCH_FILE = "_prefetch.bin"
PREFETCH_STEP = 128 * 1024  # bytes per request; the kernel serves at most one read-ahead window per call

# early, kind (A): the jit objects of JAX's own functions are built on first use. `import jax`
# decorates 323 of its functions with `jax.jit` (most of jax.numpy and lax); for each one JAX
# reads the signature, resolves static arguments and builds the C++ dispatch object, 30 to 40
# microseconds apiece and 10 ms altogether on an M4, although a run calls a dozen of them.
# With this switch `jax.jit` hands back a `_LazyJit` for functions defined in a `jax.` module:
# it carries the name, docstring and attributes JAX would have copied and builds the real jit
# object with JAX's own `make_jit` on the first call or the first use of anything else
# (`.lower`, `.trace`, `._jit_info`, ...). Functions from anywhere else get JAX's object at once.
LAZY_JITS = True

# early, kind (B): MOVES WORK OUTSIDE THE TIMED WINDOW. When the process is `benchmark.py`, the
# hook repeats the environment set-up of its header (JAX_PLATFORMS, JAX_ENABLE_X64, CPU affinity,
# thread limits) and imports everything the header imports, before the benchmark starts its
# timer. The timed imports then find all modules in `sys.modules`. Same work, not measured.
PREIMPORT = False

# late, kind (A): what the collector does after `differometor` has been imported.
#   "default"  leave it alone
#   "relaxed"  freeze everything that exists now and raise the thresholds (GC_RELAXED_THRESHOLDS):
#              the collector keeps working, but it looks at young objects far less often
#   "off"      freeze and gc.disable(): no collector passes for the rest of the process
GC_RUN = "relaxed"
GC_RELAXED_THRESHOLDS = (50_000, 20, 100)

# Modules that `import jax` (0.9.0) loads and that a plain CPU run never touches. Found by
# loading all of jax and jaxlib lazily and listing what was still unloaded after the timed
# phases of the benchmark. A name that does not exist in another JAX version is simply never asked for.
LAZY_MODULES = frozenset({
    # public namespaces that only re-export; the implementation behind them is loaded with them
    "jax.api_util", "jax.custom_batching", "jax.custom_transpose", "jax.debug", "jax.distributed",
    "jax.dlpack", "jax.dtypes", "jax.errors", "jax.export", "jax.ffi", "jax.image",
    "jax.interpreters.ad", "jax.interpreters.batching", "jax.interpreters.mlir",
    "jax.interpreters.partial_eval", "jax.interpreters.pxla", "jax.interpreters.xla",
    "jax.lax", "jax.lib", "jax.memory", "jax.monitoring", "jax.nn", "jax.numpy.fft",
    "jax.numpy.linalg", "jax.ops", "jax.profiler", "jax.random", "jax.scipy", "jax.sharding",
    "jax.stages", "jax.tree", "jax.typing",
    "jax.experimental.compilation_cache.compilation_cache", "jax.experimental.multihost_utils",
    # internals nobody asks for on one CPU
    "jax._src.clusters", "jax._src.error_check", "jax._src.mesh_utils", "jax._src.numpy.linalg",
    "jax._src.pickle_util",
    # accelerator and LAPACK bindings of jaxlib
    "jaxlib.cpu_sparse", "jaxlib.gpu_linalg", "jaxlib.gpu_prng", "jaxlib.gpu_rnn",
    "jaxlib.gpu_solver", "jaxlib.gpu_sparse", "jaxlib.gpu_triton", "jaxlib.lapack",
    "jaxlib.mlir.dialects.builtin", "jaxlib.mosaic.python.mosaic_gpu", "jaxlib.mosaic.python.tpu",
})

# What the header of benchmark.py imports (lines 6 to 40), in its order. Used by PREIMPORT only.
_BENCHMARK_HEADER = (
    "argparse", "datetime", "hashlib", "importlib.metadata", "json", "os", "pathlib", "platform",
    "statistics", "subprocess", "jax", "jax.numpy", "numpy", "differometor", "differometor.setups",
)

# Filled in by early() and late(): which measures are active in this process, and through
# which carrier. Shared between the two possible copies of this module through `sys`.
_STATE_ATTRIBUTE = "_differometor_startup"


def state():
    """What the start-up measures did in this process (a dict, empty when nothing ran)."""
    return getattr(sys, _STATE_ATTRIBUTE, None) or {}


def _early_options():
    import os

    value = os.environ.get("DIFFEROMETOR_STARTUP")
    if value is None:
        return {
            name
            for name, on in (
                ("gc", GC_PAUSE_JAX_IMPORT),
                ("lazy", LAZY_JAX_MODULES),
                ("defer", DEFER_UNUSED_MODULES),
                ("dataclass", DATACLASS_CODE),
                ("jits", LAZY_JITS),
                ("report", DEFER_REPORT_MODULES),
                ("prefetch", PREFETCH),
                ("preimport", PREIMPORT),
            )
            if on
        }
    return {part.strip().lower() for part in value.split(",")} & {
        "gc", "lazy", "defer", "dataclass", "jits", "report", "prefetch", "preimport"}


# --------------------------------------------------------------------------------------------
# Lazy jit objects
# --------------------------------------------------------------------------------------------

_MethodType = type((lambda self: None).__get__(0))
_FunctionType = type(lambda: None)
_jit_lock = None  # one lock for building jit objects, created with the first `_LazyJit`


class _LazyJit:
    """What `jax.jit` returns for a function of JAX itself until its jit object is needed.

    JAX's jit object is built by `_build_jit` on the first call and on the first read of an
    attribute this object does not have. Until then this object answers what does not need it:
    the attributes JAX copies from the function (name, module, docstring, annotations,
    `__wrapped__`, `_fun`), the signature, and binding as a method, which works like that of
    JAX's object. Calls always go through `__call__` here and then to JAX's object.
    """

    __slots__ = ("_lazy_pending", "_lazy_real", "__dict__", "__weakref__")

    def __init__(self, make, fun, options):
        self._lazy_pending = (make, fun, options)
        self._lazy_real = None

    def __call__(self, /, *args, **kwargs):
        real = self._lazy_real
        if real is None:
            real = _build_jit(self)
        return real(*args, **kwargs)

    def __get__(self, instance, owner=None):
        return self if instance is None else _MethodType(self, instance)

    def __getattr__(self, name):
        # Reached only for names that are neither set on this object nor defined by the class.
        if name[:2] == "__" and name[-2:] == "__":
            raise AttributeError(name)
        return getattr(_build_jit(self), name)

    @property
    def __signature__(self):
        import inspect

        return inspect.signature(self.__dict__["__wrapped__"])

    def __reduce_ex__(self, protocol):
        return _build_jit(self).__reduce_ex__(protocol)

    def __repr__(self):
        return f"<PjitFunction of {self.__dict__.get('__wrapped__')!r}>"


def _build_jit(lazy):
    """JAX's jit object for a `_LazyJit`, built once."""
    with _jit_lock:
        real = lazy._lazy_real
        if real is None:
            make, fun, options = lazy._lazy_pending
            real = make(fun, **options)
            lazy._lazy_real = real
            lazy._lazy_pending = None
            state()["lazy_jits"][1] += 1
    return real


def _install_lazy_jits(pjit):
    """Replace `jax._src.pjit.make_jit`, through which `jax.jit` and `pjit` build every jit object."""
    global _jit_lock
    import _thread

    original = pjit.make_jit
    code = original.__code__
    if code.co_argcount != 1 or code.co_varnames[0] != "fun" or not code.co_kwonlyargcount:
        raise TypeError("make_jit has another signature than expected")
    wraps = pjit.wraps
    _jit_lock = _thread.RLock()
    counts = state().setdefault("lazy_jits", [0, 0])  # created, built

    def make_jit(fun, **options):
        if type(fun) is _FunctionType and (fun.__module__ or "").startswith("jax."):
            lazy = _LazyJit(original, fun, options)
            wraps(fun)(lazy)  # what JAX copies from the function to its jit object
            lazy._fun = fun
            counts[0] += 1
            return lazy
        return original(fun, **options)

    make_jit.__doc__ = original.__doc__
    make_jit.__wrapped__ = original
    pjit.make_jit = make_jit
    # Frames of this file are plumbing, as JAX's own frames are: keep them out of filtered
    # tracebacks and out of the source locations of traced operations.
    pjit.traceback_util.register_exclusion(__file__)
    pjit.source_info_util.register_exclusion(__file__)


class _AfterExec:
    """Loader wrapper: run the module's code as usual, then call `after(module)`."""

    def __init__(self, loader, after, note):
        self._loader, self._after, self._note = loader, after, note

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        try:
            module.__spec__.loader = self._loader
            module.__loader__ = self._loader
        except Exception:
            pass
        self._loader.exec_module(module)
        try:
            self._after(module)
            state()[self._note] = "on"
        except Exception as error:  # another JAX: leave it alone
            state()[self._note] = f"off: {type(error).__name__}: {error}"

    def __getattr__(self, name):
        return getattr(self._loader, name)


# --------------------------------------------------------------------------------------------
# Compiled code for dataclass methods
# --------------------------------------------------------------------------------------------

# What identifies `dataclasses._create_fn` of Python 3.12: its variables, the names it uses and
# its constants (see `_create_fn_shape`). The replacement below repeats that function line by line.
_CREATE_FN_SHAPE = (
    ("name", "args", "body", "globals", "locals", "return_type", "return_annotation", "txt", "local_vars", "ns"),
    ("MISSING", "join", "keys", "exec"),
    (None, "", "__dataclass_return_type__", "->__dataclass_return_type__", ",", "\n", " def ", "(", ")", ":\n", ", ",
     "def __create_fn__(", "):\n", "\n return ", "__create_fn__", ()),
)
_RECORDING = "_differometor_dataclass_recording"


def _create_fn_shape(function):
    code = function.__code__
    return (code.co_varnames, code.co_names, tuple(c for c in code.co_consts if c is None or isinstance(c, (str, tuple))))


def _dataclass_code_path():
    import os

    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DATACLASS_CODE_FILE)


def _read_dataclass_code(path=None):
    """The table text -> code object from the code file, or {} if the file is not for this Python."""
    import marshal

    import _frozen_importlib_external

    try:
        with open(path or _dataclass_code_path(), "rb") as file:
            data = file.read()
    except OSError:
        return {}
    magic = _frozen_importlib_external.MAGIC_NUMBER
    if data[: len(magic)] != magic:
        return {}
    try:
        table = marshal.loads(data[len(magic):])
    except Exception:
        return {}
    return table if isinstance(table, dict) else {}


def _replace_create_fn(code_for):
    """Put a `_create_fn` into `dataclasses` that takes the code for a text from `code_for(text)`.

    Returns None, or the reason why `dataclasses` was left alone.
    """
    import dataclasses

    if sys.version_info[:2] != (3, 12):
        return "written for the dataclasses module of Python 3.12"
    if sys.flags.optimize:
        return "not with python -O"
    if _create_fn_shape(dataclasses._create_fn) != _CREATE_FN_SHAPE:
        return "dataclasses._create_fn is not the function this was written for"
    MISSING = dataclasses.MISSING

    def _create_fn(name, args, body, *, globals=None, locals=None, return_type=MISSING):
        # Line by line what dataclasses._create_fn of Python 3.12 does, up to the last two lines:
        # there the text is not always compiled, the code for a known text comes from the table.
        if locals is None:
            locals = {}
        return_annotation = ''
        if return_type is not MISSING:
            locals['__dataclass_return_type__'] = return_type
            return_annotation = '->__dataclass_return_type__'
        args = ','.join(args)
        body = '\n'.join(f'  {b}' for b in body)
        txt = f' def {name}({args}){return_annotation}:\n{body}'
        local_vars = ', '.join(locals.keys())
        txt = f"def __create_fn__({local_vars}):\n{txt}\n return {name}"
        ns = {}
        code = code_for(txt)
        exec(txt if code is None else code, globals, ns)
        return ns['__create_fn__'](**locals)

    dataclasses._create_fn = _create_fn
    return None


def _install_dataclass_code():
    """Use the code file for dataclass methods. Returns what happened, for `state()`."""
    if getattr(sys, _RECORDING, None) is not None:
        return "off: the code file is being recorded"
    table = _read_dataclass_code()
    if not table:
        return "off: no code file for this Python"
    reason = _replace_create_fn(table.get)
    return f"off: {reason}" if reason else f"on: {len(table)} texts"


def record_dataclass_code(path=None):
    """Tool: record the dataclass methods this process compiles and add them to the code file at exit.

    Call it before anything that defines dataclasses is imported (see `runs/philip/bytecode.sh`).
    Texts that are in the file already stay there, so the file can be filled from several
    environments (Python 3.12 patch versions, platforms).
    """
    import atexit
    import marshal

    import _frozen_importlib_external

    path = path or _dataclass_code_path()
    recorded = {}

    def code_for(text):
        code = recorded.get(text)
        if code is None:
            # What exec(text) compiles: file name "<string>", no future features (dataclasses has none).
            code = recorded[text] = compile(text, "<string>", "exec", dont_inherit=True)
        return code

    setattr(sys, _RECORDING, recorded)
    reason = _replace_create_fn(code_for)
    if reason:
        raise RuntimeError(f"cannot record dataclass code: {reason}")

    def write():
        table = _read_dataclass_code(path)
        before = len(table)
        table.update(recorded)
        with open(path, "wb") as file:
            file.write(_frozen_importlib_external.MAGIC_NUMBER + marshal.dumps(table))
        print(f"{path}: {len(table)} texts ({len(table) - before} new, {len(recorded)} used by this process)")

    atexit.register(write)


# --------------------------------------------------------------------------------------------
# Deferred modules
# --------------------------------------------------------------------------------------------

# Attribute names that describe a module rather than use it. The import machinery reads them on
# every further `import` of a module that is already in `sys.modules`.
_DESCRIPTIVE = frozenset({
    "__dict__", "__class__", "__spec__", "__name__", "__loader__", "__package__", "__path__",
    "__file__", "__cached__", "__doc__", "__builtins__",
})
_PLAN = "__deferred_plan__"
_OWNER = "__deferred_owner__"


class _StandInMeta(type):
    """Metaclass of stand-in classes: creating an instance loads the real module first."""

    def __call__(cls, *args, **kwargs):
        owner = getattr(cls, _OWNER, None)
        if owner is not None:
            _load_deferred(owner)  # moves `cls` (a subclass of the stand-in) under the real class
        return type.__call__(cls, *args, **kwargs)


def _stand_in(name):
    """Early attribute: a class object that takes the place of `module.<name>` until the load."""

    def early(module):
        return _StandInMeta(name, (), {"__module__": module.__name__, _OWNER: module, "__doc__": f"Stand-in for {module.__name__}.{name}."})

    early.stand_in = True
    return early


def _submodule(name):
    """Early attribute: the real submodule `<package>.<name>`, imported without the package's own code."""

    def early(module):
        import importlib

        return importlib.import_module(module.__name__ + "." + name)

    early.stand_in = False
    return early


def _is_masked(module):
    """Early attribute `numpy.ma.is_masked`, which `numpy.unique` asks on every call.

    The real function answers False for every object without a `_mask` attribute. Only for an
    object that has one is `numpy.ma` loaded and asked.
    """
    missing = object()

    def is_masked(x):
        if getattr(x, "_mask", missing) is missing:
            return False
        _load_deferred(module)
        return module.is_masked(x)

    return is_masked


_is_masked.stand_in = False


def _rekey_masked_array(stand_in, real):
    """JAX keeps tables from Python types to handlers; move the entries of the stand-in to the real class."""
    for module_name, table_name in (
        ("jax._src.core", "pytype_aval_mappings"),
        ("jax._src.dtypes", "canonicalize_value_handlers"),
        ("jax._src.interpreters.mlir", "_constant_handlers"),
        ("jax._src.interpreters.pxla", "shard_arg_handlers"),
    ):
        table = getattr(sys.modules.get(module_name), table_name, None)
        if isinstance(table, dict) and stand_in in table:
            table[real] = table.pop(stand_in)


def _no_entry_point_group(group):
    """True when it is certain that no installed distribution advertises entry points in `group`.

    `importlib.metadata.entry_points` looks at every `*.dist-info` and `*.egg-info` directory in
    the directories of `sys.path` (and at `EGG-INFO` inside an `.egg` directory) and parses
    their `entry_points.txt`. This reads the same files and only looks for the group name in
    them. False means "not certain", never "there is one": the caller then asks the real module.
    """
    import os
    from importlib.machinery import PathFinder

    for finder in sys.meta_path:
        if finder is not PathFinder and getattr(finder, "find_distributions", None) is not None:
            return False  # distributions that are not found through sys.path
    needle = group.encode("utf-8")
    for entry in sys.path:
        if type(entry) is not str:
            return False
        root = entry or os.curdir
        try:
            children = os.listdir(root)
        except NotADirectoryError:
            return False  # a file on sys.path: a zip archive or an egg, which the real search opens
        except OSError:
            continue  # nothing there, as for the real search
        in_egg = os.path.basename(root).lower().endswith(".egg")
        for child in children:
            low = child.lower()
            if not (low.endswith((".dist-info", ".egg-info")) or (in_egg and low == "egg-info")):
                continue
            try:
                with open(os.path.join(root, child, "entry_points.txt"), "rb") as file:
                    text = file.read()
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError):
                continue  # this distribution has no entry points, as for the real search
            except OSError:
                return False
            if needle in text:
                return False
    return True


def _entry_points(module):
    """Early attribute `importlib.metadata.entry_points`: "none" for a group nobody advertises.

    Only the call `entry_points(group=<name>)` is answered here, and only when
    `_no_entry_point_group` is certain; the answer is an empty tuple, where the real function
    gives an empty `EntryPoints` (a tuple with a few methods). Anything asked of that tuple
    beyond what a tuple has loads the module and asks a real empty `EntryPoints`. Every other
    call loads the module and is passed on.
    """

    class NoEntryPoints(tuple):
        __slots__ = ()

        def __getattr__(self, name):
            _load_deferred(module)
            return getattr(module.EntryPoints(()), name)

    def entry_points(**params):
        if len(params) == 1 and type(params.get("group")) is str:
            try:
                certain = _no_entry_point_group(params["group"])
            except Exception:
                certain = False
            if certain:
                state().setdefault("entry_points_answered", []).append(params["group"])
                return NoEntryPoints()
        _load_deferred(module)
        return module.entry_points(**params)

    return entry_points


_entry_points.stand_in = False

# Modules the benchmark script imports in its header and uses only for its report (see
# DEFER_REPORT_MODULES). Same form as DEFERRED_MODULES.
REPORT_MODULES = {
    "importlib.metadata": ({"entry_points": _entry_points}, None),
    "statistics": ({}, None),
    "hashlib": ({}, None),
    "subprocess": ({}, None),
}

# name -> (early attributes, what else to do with a (stand-in, real class) pair after the load)
DEFERRED_MODULES = {
    "http.server": ({"SimpleHTTPRequestHandler": _stand_in("SimpleHTTPRequestHandler")}, None),
    "numpy.ma": ({"MaskedArray": _stand_in("MaskedArray"), "is_masked": _is_masked}, _rekey_masked_array),
    "opt_einsum": ({"paths": _submodule("paths")}, None),
}


class _DeferredPlan:
    def __init__(self, loader, early, after):
        import _thread

        self.loader, self.early, self.after = loader, early, after
        self.served = {}  # early attributes handed out so far
        self.stand_ins = {}  # the stand-in classes among them
        self.lock = _thread.RLock()


class _DeferredModule(type(sys)):
    """A module whose code has not run yet. It runs on the first real use.

    Not a use: reading what describes the module (`_DESCRIPTIVE`), reading a name that is already
    in the module's namespace (a submodule the import system has put there) and reading one of
    the early attributes of its plan. Everything else runs the module's code in this very module
    object, which then is an ordinary module (`_load_deferred`).
    """

    def __getattribute__(self, name):
        namespace = object.__getattribute__(self, "__dict__")
        if name in _DESCRIPTIVE or name in namespace:
            return object.__getattribute__(self, name)
        plan = namespace.get(_PLAN)
        if plan is not None:
            early = plan.early.get(name)
            if early is not None:
                with plan.lock:
                    if namespace.get(_PLAN) is plan:
                        if name not in plan.served:
                            plan.served[name] = early(self)
                            if early.stand_in:
                                plan.stand_ins[name] = plan.served[name]
                        return plan.served[name]
            _load_deferred(self)
        return getattr(self, name)


def _load_deferred(module):
    """Run the code of a deferred module now and move every use of its stand-ins to the real classes."""
    namespace = object.__getattribute__(module, "__dict__")
    plan = namespace.get(_PLAN)
    if plan is None:
        return
    with plan.lock:
        if namespace.get(_PLAN) is not plan:
            return  # loaded meanwhile, or being loaded further up on this thread
        del namespace[_PLAN]
        object.__setattr__(module, "__class__", type(sys))
        try:
            plan.loader.exec_module(module)
        except BaseException:
            # As after a failed ordinary import: the half-built module must not stay around.
            if sys.modules.get(namespace.get("__name__")) is module:
                del sys.modules[namespace["__name__"]]
            raise
        for name, stand_in in plan.stand_ins.items():
            real = namespace.get(name)
            if not isinstance(real, type):
                continue
            for subclass in stand_in.__subclasses__():
                subclass.__bases__ = tuple(real if base is stand_in else base for base in subclass.__bases__)
            if plan.after is not None:
                plan.after(stand_in, real)
        state().setdefault("deferred_loaded", []).append(namespace.get("__name__"))


class _DeferredLoader:
    """Loader wrapper: instead of running the module's code, turn the module into a `_DeferredModule`."""

    def __init__(self, loader, early, after):
        self._loader, self._early, self._after = loader, early, after

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        try:
            module.__spec__.loader = self._loader
            module.__loader__ = self._loader
        except Exception:
            pass
        if type(module) is not type(sys):
            self._loader.exec_module(module)  # not a plain module object: load it the ordinary way
            return
        module.__dict__[_PLAN] = _DeferredPlan(self._loader, self._early, self._after)
        module.__class__ = _DeferredModule
        state().setdefault("deferred", []).append(module.__name__)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _PausedCollector:
    """Loader wrapper for the top-level `jax` package: no collector passes while it executes."""

    def __init__(self, loader, watcher):
        self._loader = loader
        self._watcher = watcher

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        import gc

        # Put the real loader back first: nothing outside ever sees this wrapper.
        try:
            module.__spec__.loader = self._loader
            module.__loader__ = self._loader
        except Exception:
            pass
        pause = self._watcher.pause and gc.isenabled()
        if pause:
            gc.disable()
        try:
            self._loader.exec_module(module)
        finally:
            self._watcher.remove()
            if pause:
                # Everything allocated meanwhile is module-level state that lives as long as
                # the process. Freezing it keeps later passes from walking it once more.
                gc.freeze()
                gc.enable()
                state()["gc_paused_for_jax"] = True

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _Watcher:
    """Entry on `sys.meta_path` that waits for the import of `jax` and removes itself afterwards.

    It never loads anything itself. For `jax` it hands back the normal spec with a loader that
    pauses the collector, for names in `lazy` the normal spec with a lazy loader, and for every
    other name `None`, which sends the import machinery on to the normal finders.
    """

    def __init__(self, pause, lazy, deferred=None, lazy_jits=False):
        self.pause = pause
        self.lazy = lazy
        self.deferred = deferred or {}
        self.after = {"jax._src.pjit": (_install_lazy_jits, "lazy_jits_installed")} if lazy_jits else {}
        self._busy = False

    def remove(self):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass

    def find_spec(self, name, path=None, target=None):
        if name != "jax" and name not in self.lazy and name not in self.deferred and name not in self.after:
            return None
        if self._busy:
            return None
        self._busy = True
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                find = getattr(finder, "find_spec", None)
                if find is None:
                    continue
                spec = find(name, path, target)
                if spec is not None:
                    break
            if spec is None or spec.loader is None:
                return None
            if name == "jax":
                spec.loader = _PausedCollector(spec.loader, self)
            elif name in self.deferred:
                from importlib.machinery import SourceFileLoader

                if type(spec.loader) is not SourceFileLoader:
                    return None
                spec.loader = _DeferredLoader(spec.loader, *self.deferred[name])
            elif name in self.after:
                spec.loader = _AfterExec(spec.loader, *self.after[name])
            else:
                from importlib.machinery import SourceFileLoader

                if type(spec.loader) is not SourceFileLoader:
                    return None
                from importlib.util import LazyLoader

                spec.loader = LazyLoader(spec.loader)
                state().setdefault("lazy", []).append(name)
            return spec
        except Exception:
            # Never stand in the way of an import: the normal finders take over.
            return None
        finally:
            self._busy = False


def _prefetch(note):
    """Request the pages listed in PREFETCH_FILE if the file cache is cold. Fills `note`."""
    import marshal
    import os
    import time

    started = time.perf_counter()
    if not hasattr(os, "posix_fadvise"):
        note["state"] = "off: no posix_fadvise on this platform"
        return
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), PREFETCH_FILE), "rb") as file:
        table = marshal.loads(file.read())
    roots = [root for root in sys.path if type(root) is str and root]
    advise, willneed = os.posix_fadvise, os.POSIX_FADV_WILLNEED
    files = calls = total = 0
    for relative, size, ranges in table:
        fd = -1
        for root in roots:
            path = root + "/" + relative
            try:
                if os.stat(path).st_size == size:
                    fd = os.open(path, os.O_RDONLY)
                    break
            except OSError:
                continue
        if fd < 0:
            continue
        try:
            if not files and hasattr(os, "preadv") and hasattr(os, "RWF_NOWAIT"):
                # The first file found (the table starts with the largest): are its pages in memory?
                try:
                    for offset, _ in (ranges[0], ranges[len(ranges) // 2], ranges[-1]):
                        os.preadv(fd, [bytearray(1)], offset, os.RWF_NOWAIT)
                    note.update(state="warm cache, nothing requested", seconds=round(time.perf_counter() - started, 5))
                    return
                except BlockingIOError:
                    pass  # not in the cache: go on
                except OSError:
                    pass  # this file system cannot answer: request, it costs little
            files += 1
            for offset, length in ranges:
                end = offset + length
                total += length
                while offset < end:
                    advise(fd, offset, min(PREFETCH_STEP, end - offset), willneed)
                    offset += PREFETCH_STEP
                    calls += 1
        finally:
            os.close(fd)
    note.update(state="requested", files=files, calls=calls, megabytes=round(total / 1e6, 1), seconds=round(time.perf_counter() - started, 5))


def _is_benchmark_process():
    import os

    argv = getattr(sys, "argv", None) or [""]
    return os.path.basename(argv[0]) == "benchmark.py"


def _preimport():
    """Kind (B): import what the header of benchmark.py imports, before its timer starts.

    The header prepares the process before it imports the numerical libraries (platform and
    precision for JAX, CPU affinity and thread limits before the libraries start workers). The
    same preparation has to happen here first, otherwise the libraries would be initialized
    differently than in an ordinary run. Only done for a process whose script is benchmark.py.
    """
    import importlib
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ.setdefault("MPLBACKEND", "Agg")
    cpus = None
    argv = sys.argv[1:]
    for index, argument in enumerate(argv):
        if argument == "--cpus" and index + 1 < len(argv):
            cpus = int(argv[index + 1])
        elif argument.startswith("--cpus="):
            cpus = int(argument.split("=", 1)[1])
    if cpus is not None:
        allowed = sorted(os.sched_getaffinity(0))
        if not 1 <= cpus <= len(allowed):
            return False  # the benchmark reports the error itself
        os.sched_setaffinity(0, allowed[:cpus])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(cpus)
    # The script directory is put on sys.path after the hook has run. The benchmark insists that
    # differometor comes from the checkout next to it, so look there first, as the script will.
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    added = script_dir not in sys.path
    if added:
        sys.path.insert(0, script_dir)
    try:
        for name in _BENCHMARK_HEADER:
            importlib.import_module(name)
    finally:
        if added:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass
    return True


def early(carrier="pth"):
    """Apply the early measures. Safe to call more than once; only the first call acts."""
    if getattr(sys, _STATE_ATTRIBUTE, None) is not None:
        return state()
    info = {"carrier": carrier, "early": []}
    setattr(sys, _STATE_ATTRIBUTE, info)
    options = _early_options()
    if "prefetch" in options:
        note = info["prefetch"] = {}
        try:
            _prefetch(note)
        except Exception as error:  # never stand in the way of the process
            note["state"] = f"off: {type(error).__name__}: {error}"
        info["early"].append("prefetch")
    if "dataclass" in options:
        try:
            info["dataclass_code"] = _install_dataclass_code()
        except Exception as error:  # never stand in the way of the process
            info["dataclass_code"] = f"off: {type(error).__name__}: {error}"
    if "jax" in sys.modules:
        # Too late for anything that concerns the import of JAX.
        return info
    pause = "gc" in options
    lazy = LAZY_MODULES if "lazy" in options else frozenset()
    # A module that is loaded already stays as it is.
    deferred = {name: plan for name, plan in DEFERRED_MODULES.items() if name not in sys.modules} if "defer" in options else {}
    if "report" in options:
        deferred.update((name, plan) for name, plan in REPORT_MODULES.items() if name not in sys.modules)
    lazy_jits = "jits" in options
    if pause or lazy or deferred or lazy_jits:
        sys.meta_path.insert(0, _Watcher(pause, lazy, deferred, lazy_jits))
        info["early"] += [name for name in ("gc", "lazy", "defer", "jits", "report") if name in options]
    if "preimport" in options and carrier == "pth" and _is_benchmark_process():
        if _preimport():
            info["early"].append("preimport")
    return info


def late():
    """Apply the collector policy for the run (GC_RUN). Called when differometor has been imported."""
    import gc
    import os

    info = getattr(sys, _STATE_ATTRIBUTE, None)
    if info is None:
        info = {"carrier": None, "early": []}
        setattr(sys, _STATE_ATTRIBUTE, info)
    if "late" in info:
        return info
    mode = os.environ.get("DIFFEROMETOR_GC", GC_RUN).strip().lower()
    info["late"] = mode
    if mode not in ("relaxed", "off") or not gc.isenabled():
        return info
    gc.freeze()
    if mode == "off":
        gc.disable()
    else:
        gc.set_threshold(*GC_RELAXED_THRESHOLDS)
    return info
