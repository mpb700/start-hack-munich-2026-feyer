# SX — `XLA_FLAGS=--xla_backend_optimization_level=1` auf Basis `ac7aedc` (Feyer-gated, Vorbereitung)

**Status:** Vorbereitung/Experiment. Kein Merge, `ac7aedc`/`submission` unangetastet.
Feyer-Freigabe nötig, weil `XLA_FLAGS` im Benchmark-Report (`environment.xla_flags`) sichtbar ist.

## Änderung (`differometor/__init__.py`, früh — vor Backend-Init)

```python
import os as _os

_sx_flags = _os.environ.get("XLA_FLAGS", "")
if "xla_backend_optimization_level" not in _sx_flags:
    _os.environ["XLA_FLAGS"] = (_sx_flags + " --xla_backend_optimization_level=1").strip()
```

Wirkt, weil `benchmark.py` `import differometor` **vor** `initialize()`/`jax.devices()` ausführt.
Kein `sitecustomize.py`, keine Datei außerhalb `differometor/`. Vom Benchmark-Report als
`"xla_flags": "--xla_backend_optimization_level=1"` ausgewiesen (transparent).

## Messung

_(offizielle A/B-Zahlen auf Basis `ac7aedc` folgen nach dem Messfenster)_

Erwartung aus früherem Lauf auf `ab82e42`: Δ e2e ≈ −0,03…−0,04 s, Ursache
`sensitivity_compile`/`gradient_compile` je ≈ −0,017 s; Exec/warm unverändert;
Arrays bitgleich/letzte Bits; Validation PASS.

## Offenlegungstext (EN)

> **Implementation note (SX):** the candidate sets
> `XLA_FLAGS=--xla_backend_optimization_level=1` (default 3) in `differometor/__init__.py` before the
> XLA backend is initialized. This only reduces compiler backend optimisation effort; no fast-math or
> reduced-precision flags are set, all frequencies/parameters are evaluated in double precision and
> the compiled semantics are unchanged. The flag is visible in the benchmark report under
> `environment.xla_flags`.
