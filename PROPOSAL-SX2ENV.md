# SX2-env — XLA-Flags `--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true` (Feyer-gated, Vorbereitung)

**Status:** Vorbereitung/Experiment. Kein Merge, `ac7aedc`/`submission` unangetastet.
Bevorzugte Feyer-Variante gegenüber SX2 (`jax.config`-Flag), weil die Wirkung **im Benchmark-Report
sichtbar** ist (`environment.xla_flags`) — vollständig transparent, kein interner JAX-Schalter.

## Änderung (`differometor/__init__.py`, früh — vor Backend-Init)

```python
import os as _os

_sx2_flags = "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true"
_current = _os.environ.get("XLA_FLAGS", "")
if "--xla_backend_optimization_level" not in _current:
    _os.environ["XLA_FLAGS"] = (_current + " " + _sx2_flags).strip()
```

- **Anhängen**, nicht Überschreiben: ein von außen gesetztes `XLA_FLAGS` bleibt erhalten.
- Wirkt, weil `benchmark.py` `import differometor` **vor** `initialize()`/`jax.devices()` ausführt
  (XLA-Flags werden erst bei der Backend-Initialisierung gelesen).
- Erscheint im Report als
  `"xla_flags": "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true"`.

## Erwartung

Semantisch identisch zu SX2: SX2 setzt in jax 0.9.0 genau diese beiden Felder
(`_src/compiler.py:218–220`: `xla_backend_optimization_level = 0`,
`xla_llvm_disable_expensive_passes = True`, `xla_test_all_input_layouts = False`.
`xla_test_all_input_layouts=False` ist in jax der Default; ohne Effekt). Daher:
erwartetes Δ e2e ≈ −0,08 s warm / −0,07 s kalt, **numerisch == SX2** (nur kurzer Bestätigungslauf
bzw. Standard-Gate nötig).

## Messung

_(offizielle A/B-Zahlen folgen nach dem Messfenster)_

## Offenlegungstext (EN)

> **Implementation note (SX2-env):** the candidate appends
> `XLA_FLAGS=--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true` to the
> process environment in `differometor/__init__.py` before the XLA backend is initialised
> (inherited XLA_FLAGS from the caller are preserved, not overwritten). These flags only reduce
> compiler optimisation effort: no fast-math or reduced-precision options are enabled, the
> simulation is still compiled and executed by XLA in double precision with all 101 frequencies and
> 193 design variables, and the compiled-graph semantics are unchanged. Numerical outputs are
> bit-identical for carrier, signal, noise, powers and parameters; sensitivity/gradient differ only
> in the last bits and all sensitivity/FD checks pass. The flags are visible in the benchmark report
> under `environment.xla_flags`.

## Risiken

- Sichtbar im Report → offenlegungspflichtig, aber transparent (kein versteckter Eingriff).
- Optimierungsstufe 0 kann auf anderen Maschinen das Exec etwas verlangsamen; auf diesem Workload
  ist der Graph klein (Warm-Runtime unverändert in SX2-Messungen).
- Kein interner JAX-Schalter, keine zusätzlichen Dependencies.
