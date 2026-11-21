# SX2 — `jax_disable_most_optimizations` (Grauzone, NUR Vorbereitung)

**Status:** Vorbereitung/Experiment. **Kein Merge, keine Änderung an `ac7aedc`/`submission`.**
Feyer-Freigabe nötig, weil der Schalter **global** wirkt und **nicht** in
`environment.xla_flags`/Benchmark-Report auftaucht (Offenlegungsfrage).

## Idee

In `differometor/__init__.py` **vor** dem Backend-Init:

```python
import jax as _jax

_jax.config.update("jax_disable_most_optimizations", True)
```

Wirkung in jax 0.9.0 (verifiziert über `jax._src.compiler._DISABLE_MOST_OPTIMIZATIONS`):
`xla_backend_optimization_level = 0`, `xla_llvm_disable_expensive_passes = True`,
`xla_test_all_input_layouts = False` — d. h. die JAX/XLA-Optimierungspipeline wird global
zurückgefahren. Das Flag ist in JAX eine **interne/experimentelle Config-Option**
(`jax._src.compiler.bool_flag("jax_disable_most_optimizations", ...)`, auch per
`JAX_DISABLE_MOST_OPTIMIZATIONS=1` setzbar); es ist semantikerhaltend (nur Optimierungen entfallen),
aber es ist **kein** offizielles Feature und erscheint nicht im Benchmark-Report.

## Smoke-Messung (spontan, Kern 12, `--cpus 1 --repeats 3`, keine offizielle Messung)

| | `ac7aedc` (A) | **SX2 (B)** | Δ |
|---|---:|---:|---:|
| e2e | 0,7575 / 0,7536 s | **0,6757 / 0,6759 s** | **−0,080 s (−10,6 %)** |
| forward_compile | 0,0077 / 0,0074 | 0,0076 / 0,0075 | ≈0 |
| **sensitivity_compile** | 0,1397 / 0,1468 | **0,0819 / 0,0831** | **−0,060 s** |
| **gradient_compile** | 0,0744 / 0,0722 | **0,0506 / 0,0474** | **−0,025 s** |
| imports | 0,2713 / 0,2612 | 0,2679 / 0,2686 | ≈0 |
| warm runtime | 0,1383 / 0,1486 | 0,1428 / 0,1321 | Rauschen |

**Arrays** (SX2 vs `ac7aedc`): carrier/signal/noise/powers/parameters/frequencies **bitgleich**;
sensitivity max_abs 9,2e-41; gradient max_abs 1,3e-10 (rel ≈1,9e-15, letzte Bits).
**Determinismus:** SX2-Lauf1 vs Lauf2 bitgleich. **Validation:** finite / upstream-sens / FD **PASS**,
`max|ln S − ln S_ref|` = 5,374e-06 unverändert.

## Offenlegungstext (für Feyer-Frage, EN)

> **Implementation note (SX2, experimental):** the candidate additionally sets the JAX internal
> configuration flag `jax_disable_most_optimizations=True` before the backend is initialized. In
> JAX 0.9.0 this configures the compiler with `xla_backend_optimization_level=0` and disables
> LLVM's expensive passes. It removes optimisation work only: the simulation is still compiled and
> executed by XLA in double precision, all 101 frequencies and 193 variables are evaluated, and the
> compiled-graph semantics are unchanged. Numerical outputs are bit-identical for carrier, signal,
> noise, powers and parameters; sensitivity/gradient differ only in the last bits
> (gradient max |Δ| = 1.3e-10, rel. L2 ≈ 2e-15) and the FD/sensitivity checks pass. The flag does not
> appear in the benchmark's `environment.xla_flags` report because it is a JAX config option, not an
> XLA_FLAGS entry; it is set in `differometor/__init__.py` and is therefore part of the submitted code.

## Risiken

- **Grauzone:** internes JAX-Flag (`jax._src...`), kein dokumentiertes API; kann in anderen
  jax-Versionen fehlen/anders heißen (in 0.9.0 vorhanden, gepinnt in requirements.txt).
- Global für **alle** Compilations im Prozess (auch Jury-Fremdcode), nicht auf differometor begrenzt.
- Nicht im Report sichtbar → nur über Code-Diff/Offenlegung nachvollziehbar.
- Warm-Runtime unverändert (Codegen weniger optimiert, aber Graph ist klein) → kein Exec-Risiko.

## Nächster Schritt

Offizielles A/B (10 warm + 4 kalt, Kern 14, Fenster von 00) + Gate (Δerr, bitgleich, gradient rel_L2);
bei PASS eigener Tag `s-group-d-det-superlu-sx2` (nicht in der Hauptlinie), dann Feyer-Frage.
