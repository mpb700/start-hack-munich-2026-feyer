# PROPOSAL — SX: `XLA_FLAGS=--xla_backend_optimization_level=1` auf der Gruppen-Basis

**Status:** env-only gemessen, **KEIN Commit** der Flag-Setzung. Feyer-Freigabe offen.
Basis: Gruppen-Tree `ab82e42` (`/root/wt-10-ms-imports`), FINAL FREEZE unberührt.

## Kandidat (nur falls freigegeben)

In `differometor/__init__.py` ganz oben, vor dem JAX-Backend-Init (benchmark.py importiert
differometor vor `initialize()`):

```python
import os
_flags = os.environ.get("XLA_FLAGS", "")
if "xla_backend_optimization_level" not in _flags:
    os.environ["XLA_FLAGS"] = (_flags + " --xla_backend_optimization_level=1").strip()
```

Kein `sitecustomize.py`, keine Dateien außerhalb `differometor/`.

## Messung (Kern 14, `--cpus 1 --repeats 10`, frische Prozesse, `env -u XLA_FLAGS` als Basis)

| Lauf | Gruppe `ab82e42` | + SX level=1 | Δ |
|---|---:|---:|---:|
| warm #1 e2e | 1.0398 s | **1.0112 s** | −0.029 s |
| warm #2 e2e | 1.0606 s | **1.0112 s** | −0.049 s |
| warm ø | 1.0502 s | 1.0112 s | **−0.039 s (−3.7 %) |
| kalt #1 (pycache gelöscht + drop_caches) | 1.3361 s | 1.2983 s | −0.038 s |
| imports (warm) | 0.381/0.384 | 0.378/0.380 | ≈0 |
| sensitivity_compile | 0.200/0.208 | 0.189/0.185 | −0.017 s |
| **gradient_compile** | **0.169/0.170** | **0.1515/0.1521** | **−0.017 s (−10 %)** |
| warm runtime | 0.1474/0.1514 | 0.1461/0.1542 | ≈0 |

Referenz `S-gruppe-official` = 1.0633 s → SX **−0.052 s**. Streuung der SX-Läufe selbst: 0.0001 s.
`environment.xla_flags` im JSON nachgewiesen (`--xla_backend_optimization_level=1`).

## Numerik

Arrays **bit-identisch** zu `measurements/group-d/group-b.npz` (alle 6 Felder, max_abs 0.0);
`max|ln S − ln S_ref|` = 5.374249e-06 unverändert (Δerr 0). Kein Fast-Math-Flag; XLA-Optimierungs-
stufe ist semantikerhaltend (weniger Backend-Pässe/Codegen, keine Reassoziation).

## Risiken / Offenlegung

1. `xla_flags` im Benchmark-Report sichtbar → ein Disclosure-Satz in SUBMISSION/REPRODUCE:
   *"XLA backend optimization level 1 instead of default 3; no fast-math flags; outputs bit-identical."*
2. Gewinn ~−0.04 s (−3.7 %) auf ~1.05 s — über der SX-Streuung, aber durch `ms_gate.py`-A/B
   (10 warm + 4 kalt) zu bestätigen; erst danach Merge-Entscheidung.
3. `level=2`/`level=0` bzw. Pass-Deaktivierung waren auf der alten Basis schlechter/neutral —
   hier nicht erneut getestet (nur level=1 ist der Kandidat).

## Nächster Schritt

1. `ms_gate.py --wt /root/wt-10-ms-imports --base /root/wt-10-group-d --ab-pairs 10 --cold-pairs 4`
   mit `XLA_FLAGS` env-only (Achtung: ms_gate setzt `env -u XLA_FLAGS` → für SX entweder
   Code-Variante im Worktree-Branch nötig oder separater A/B-Lauf mit env; Gate-PASS für
   Code-Variante Lazy erfassen).
2. Nur bei bestätigtem GAIN + PASS: Feyer-Freigabe → minimaler `os.environ`-Block, Bundle, EXPERIMENTS `kept`.
