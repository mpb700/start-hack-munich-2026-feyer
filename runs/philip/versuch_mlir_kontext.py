"""Experiment, not part of the library: benchmark.py with one MLIR context per thread.

JAX creates a new MLIR context for every program it lowers (`jax._src.interpreters.mlir.
make_ir_context`, 2-3 ms each, 11 ms for the first one). benchmark.py lowers six programs in the
timed part: three conversions in its build step, forward, sensitivity and gradient. This script
replaces that function by one that hands out the same context again and then runs benchmark.py
unchanged. It only measures what the idea would bring; changing JAX from inside the library is
a question for Feyer (like XLA flags, B6 in docs/notes/FRAGEN_AN_FEYER.md).

Measured on x86 (6 alternating runs each): build 56-65 -> 52 ms, trace/lower 35.5 -> 29 ms,
compile unchanged. `import jax` runs before benchmark.py starts its clock here, so compare the
phases in the JSON report, not the printed total.

Usage (from the repository root): python runs/philip/versuch_mlir_kontext.py 1 --cpus 1 --repeats 3 --output x.json
The first argument switches the shared context on (1) or off (0).
"""

import runpy
import sys
import threading
from pathlib import Path

root = Path(__file__).resolve().parents[2]
share = sys.argv[1] == "1"
sys.argv = [str(root / "benchmark.py")] + sys.argv[2:]
sys.path.insert(0, str(root))
if share:
    import jax  # noqa: F401  (benchmark.py imports it anyway; the replacement has to be in place before the first lowering)
    from jax._src.interpreters import mlir

    original, local = mlir.make_ir_context, threading.local()

    def shared_context():
        context = getattr(local, "context", None)
        if context is None:
            context = local.context = original()
        return context

    mlir.make_ir_context = shared_context
runpy.run_path(str(root / "benchmark.py"), run_name="__main__")
