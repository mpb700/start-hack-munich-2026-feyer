# SX2 — `jax_disable_most_optimizations` (deferred / NOT submission)

**Status:** validierter, **nicht aktiver** Kandidat. **Kein Merge, keine Änderung an
`ac7aedc`/`submission`.** Tag: `s-group-d-det-superlu-sx2` (Commit `99e0909` = Code `4e63631` + finalisierte Doku, Branch `track/10-sx2`).
**Feyer-Freigabe steht aus** — ohne sie wird SX2 nicht eingereicht (Lead-Entscheid B, 03:29 UTC).

## Idee
In `differometor/__init__.py` **vor** dem Backend-Init:
```python
import jax as _jax
_jax.config.update("jax_disable_most_optimizations", True)
```
Wirkung in jax 0.9.0 (verifiziert): `xla_backend_optimization_level=0`,
`xla_llvm_disable_expensive_passes=True`, `xla_test_all_input_layouts=False` — die
Optimierungspipeline wird global zurückgefahren; der Graph wird weiterhin von XLA kompiliert und in
double precision ausgeführt, alle 101 Frequenzen / 193 Variablen werden berechnet.
**Grauzone:** interne/experimentelle JAX-Config (`jax._src.compiler`), global wirkend, erscheint
**nicht** in `environment.xla_flags`/Benchmark-Report → Offenlegung nötig; Regel v4 (Compiler-Optionen
via `XLA_FLAGS` + Feyer-Freigabe vorher) ist **nicht erfüllt**.

## Offizielle Messung (00, Kern 14, leere Maschine, `--cpus 1`, 10 rep, 03:30–03:36 UTC)
| Lauf | end_to_end | compile | warm_median | err |
|---|---:|---:|---:|---|
| **SX2 `S-sx2-c01`** (4e63631, warm) | **0,691 s** | **0,194 s** | 0,153 s | 5,374248516e-06 |
| **SX2 `S-sx2-c02`** | **0,681 s** | **0,185 s** | 0,147 s | 5,374248516e-06 |
| **SX2 `S-sx2-cold`** (COLD=1) | **0,920 s** | 0,209 s | 0,149 s | 5,374248516e-06 |
| Referenz Submission `ac7aedc` (offiziell) | 0,762 / 0,769 s | 0,268 / 0,272 s | 0,145 s | 5,374248516e-06 |
| Referenz `ac7aedc` kalt | 0,995 s | 0,293 s | 0,151 s | 5,374248516e-06 |

**Delta:** warm ≈ **−0,080 s (−10,4 %)**, kalt ≈ **−0,075 s (−7,5 %)**; compile 0,27 → 0,19 s
(hauptsächlich `sensitivity_compile` −0,060 s, `gradient_compile` −0,025 s), imports/runtime unverändert.

## Unabhängige A/B (`ms_gate`, 10 warme + 4 kalte Paare, Kern 14, MS-AB PASS)
| Größe | A = `ac7aedc` | B = SX2 `4e63631` | Δ | Urteil |
|---|---:|---:|---:|---|
| warm e2e median | 0,7623 s (σ 0,0128) | **0,6833 s (σ 0,0052)** | **−0,0790 s (−10,36 %)** | **GAIN** |
| kalt e2e median | 0,9877 s (σ 0,0011) | **0,8998 s (σ 0,0066)** | **−0,0879 s (−8,90 %)** | **GAIN** |

Report: `measurements/ms/sx2-ac7aedc/ab-report.md`.

## Korrektheit / Compliance
- Arrays vs `ac7aedc`: carrier/signal/noise/powers/parameters/frequencies **bitgleich**;
  sensitivity max_abs 9,2e-41, gradient max_abs 1,3e-10 (rel ≈1,9e-15, letzte Bits durch
  Optimierungsreihenfolge). Lauf1 vs Lauf2 bitgleich (deterministisch).
- Harness-Validierung: finite / upstream-sensitivity / FD **PASS**; `max|ln S − ln S_ref|` =
  **5,374248516e-06 unverändert** (Limit 1e-5). `benchmark.py`/Fixtures/requirements unverändert;
  Diff nur `differometor/__init__.py` (+8) und diese Doku.
- **Einziger offener Punkt: Feyer-Freigabe + Offenlegung des globalen Flags.**

## Offenlegungstext (EN, für Feyer-Frage / SUBMISSION.md „deferred" Abschnitt)
> **Deferred experimental variant (SX2, not part of the submitted result):** an optional variant sets
> the JAX internal configuration flag `jax_disable_most_optimizations=True` before backend
> initialisation. In JAX 0.9.0 this configures the compiler with `xla_backend_optimization_level=0`
> and disables LLVM's expensive passes. It removes optimisation work only: the simulation is still
> compiled and executed by XLA in double precision, all 101 frequencies and all 193 design variables
> are evaluated, and the compiled graph semantics are unchanged. Numerical outputs are bit-identical
> for carrier, signal, noise, powers, parameters and frequencies; sensitivity and gradient differ only
> in the last bits (gradient max |Δ| = 1.3e-10, relative L2 ≈ 2e-15) and all sensitivity/finite-
> difference checks pass. Measured (one logical CPU, isolated core, 10 repeats): **0.691/0.681 s warm,
> 0.920 s cold** vs 0.762/0.769 s / 0.995 s for the submitted variant (−10.4 % / −7.5 %). This variant
> is **not** part of the submission; it is documented for transparency and awaits team approval.

## „Ready to swap"-Paket
`/root/faster-physics/submission-sx2-ready/` (+ `submission-sx2-ready.tar.gz`, SHA256 in
`/root/states/`): Code-Tar/Bundle des Tags, offizielle Messartefakte, dieses Proposal, Disclosure-Text,
DEFERRED-VARIANTS-Katalog, SHA256SUMS. Aktivierung erfordert nur: Tag-Wechsel auf
`s-group-d-det-superlu-sx2` + Repack + Feyer-Freigabe.
