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

## Offizielle A/B-Zahlen auf Basis `ac7aedc` (Fenster 03:40–03:46, Kern 14 exklusiv)

A = `ac7aedc` (`/root/wt-10-ms-imports`), B = SX (`520ec7a`), je 10 warme + 4 kalte Paare
(pycache gelöscht + drop_caches), `--cpus 1 --repeats 10`, frische Prozesse.
Idle-Check vor/nach: **kein fremder benchmark/quick_probe**.

| A/B | A median | σ_A | B median | σ_B | Δ (B−A) | pooled σ | Urteil |
|---|---:|---:|---:|---:|---:|---:|---|
| warm, 10 Paare | 0,7834 s | 0,0026 | 0,7746 s | 0,0146 | **−0,0088 s (−1,1 %)** | 0,0105 | **NOISE** |
| kalt, 4 Paare | 1,0146 s | 0,0022 | **0,9978 s** | 0,0021 | **−0,0168 s (−1,7 %)** | 0,0022 | **GAIN** |

Compile-Effekt (Median): `sensitivity_compile` 0,1425 → **0,1325 s** (−0,010);
`gradient_compile` 0,0705 → 0,0723 s (+0,002, Rauschen); `imports` ≈ unverändert.
`environment.xla_flags` im JSON: **`--xla_backend_optimization_level=1`** ✓.

**Arrays:** alle Felder (carrier, signal, noise, sensitivity, powers, gradient, parameters,
frequencies) **bit-identisch** zu `ac7aedc`.
**Gate:** Δerr = 0 (Limit 1e-8), Doppellauf bit-identisch, gradient rel_L2 = 2,29e-15 (Limit 1e-5),
finite/sens/FD PASS; Gate-Kaltlauf 1,0030 s. → **GATE PASS**.

**Bewertung:** auf der neuen Basis (`ac7aedc`) ist SX nur noch ein **Rückfall**:
warm NOISE, kalt kleiner GAIN (−1,7 %); deutlich schwächer als SX2 (−10,4 %/−7,5 %).
Artefakte: `measurements/ms/sxvariants-ab/` (`sx-*.json/npz`, `sx-gate-*`).
