# Faster-Physics — Branch `Joel` (submission status)

> Schnellüberblick fürs Team + KI-Agenten. Fakten zuerst, Erklärung unten.

## TL;DR (die harten Fakten)
- **Submission:** Commit `ac7aedc` · Tag `s-group-d-det-superlu` · **Alias `joels-fastest`** · Branch `Joel`.
- **Laufzeit = 0,762 s warm / 0,995 s kalt** (AMD EPYC-Turin, 1 CPU, Kern 14, `--cpus 1`, 10 reps).
- **Baseline → Submission: 33,5 s → 0,76 s = −97,7 % (~44×)** auf identischer Hardware.
- Jury (Intel Xeon 8468) ≈ **3,2 s statt 44,2 s**; ~2,6 s davon ist der Pflicht-`import jax`, den **jedes** Team zahlt.
- **PASS** · Fehler `5,3742485e-06` (Limit 1e-5) · Gradient vs. Baseline `9,1e-8` (cos 1,0) · Doppellauf bit-identisch · Frisch-Install reproduziert.

## Was hier passiert ist (in Worten)
Das Original kompiliert pro Frequenz dichte 751×751-JAX-Systeme: allein das Kompilieren kostet ~20 s, das Ausführen ~17 s. Unsere Lösung rechnet **gar nicht mehr im JAX-Graphen**, sondern:

1. **Physik auf dem Host**: Die unveränderten Bauteilformeln werden über einen „Dual-Zahl"-Trick in **NumPy** ausgewertet — Werte *und* Ableitungen, ohne dass XLA je etwas tracen muss.
2. **Löser = SuperLU pro Frequenz**, und zwar **direkt aus scipys `_superlu*.so` geladen** (kein `import scipy` → 0 scipy-Module, spart den Import).
3. **Handgeschriebener Adjoint** als `jax.custom_vjp` um `pure_callback` — der Gradient entsteht aus dem geteilten LU-Faktor, mit Detektor-Spalten, die der Forward schon vorlöst (kein zweites LU).
4. **Detektoren als Callback** mit eigenem Rückweg.
5. **Fallback auf dichtes JAX** für Setups, die der Fast-Path nicht abdeckt (Parameter-Linking, Homodyn, mehrere Detektoren, Carrier-Sweeps).

Netto: der JAX-Graph ist auf **1 Callback** geschrumpft; übrig bleiben nur der Pflicht-`import jax` (~0,28 s), fester XLA-Overhead (~0,23 s) und die eigentliche Rechnung (~0,15 s).

## Wo die 0,76 s jetzt stecken
| Phase | Zeit | Anmerkung |
|---|---:|---|
| imports | 0,28 s | `jax` (Pflicht); scipy raus, plotly lazy |
| compile | ~0,23 s | fester XLA-Aufwand + Benchmark-Readout |
| exec (fwd+grad) | ~0,15 s | NumPy/SciPy |
| build | 0,06 s | NumPy |

## Fertig für Feyer (noch NICHT aktiv)
| Branch | Tag | warm / kalt | Anmerkung |
|---|---|---|---|
| `sx2-env` | `s-group-d-det-superlu-sx2-env` | **0,699 s / 0,926 s** (−10,5 % / −8,45 %) | **bevorzugt**: hängt `--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true` an `XLA_FLAGS` an → **im Report sichtbar** |
| `sx2` | `s-group-d-det-superlu-sx2` | ≡ SX2-env | interner `jax_disable_most_optimizations=True`, im Report unsichtbar |
| `sx` | `s-group-d-det-superlu-sx` | warm NOISE / kalt −1,7 % | `level=1`, nur Rückfall |

Alle gated (bit-identisch, Δerr = 0, A/B-GAIN). Aktivierung ≈ 10 min nach Feyer-Zusage. Entscheidungsblatt: `measurements/FEYER-DECISION.md`.

## Pakete (sha256)
- `submission-final.tar.gz` → `5c36bc879954959f3bb3d4fa9d9a31c14dc612da8bab2b4107ff22397ff521bf`
- `submission-deferred-ready.tar.gz` → `7e2600b7f1bd6330f7c5e35d1bcf485ce1a867a9826021daf2b7050e3342b5e1`

## Wichtige Dateien
- `SUBMISSION.md` — Ansatz, Phasen-/FD-Tabelle, Offenlegung, Rangliste.
- `REPRODUCE.md` — Env, Install (kein Build; scipy nur im Fallback), Messbefehle, Hashes.
- `README.optimizations.md` — ausführlicher Optimierungs-Bericht (vom Initiator).
- `measurements/` — `server-baseline.*`, `benchmark-candidate-final.*`, `benchmark-candidate-cold-final.*`, `DEFERRED-VARIANTS.md`, `FEYER-DECISION.md`, `GROUP-D-GATE.md`.
- `runs/check_setups.py`, `runs/check_fast.py`, `runs/compare_npz.py` — Robustheits-Checks (aLIGO/Voyager/Fallback).

## Reproduzieren
```bash
pip install -r requirements.txt && pip install -e .
taskset -c 14 python benchmark.py --cpus 1 --repeats 10 --output out.json --save-arrays out.npz
# kalt:  sync; echo 3 > /proc/sys/vm/drop_caches   (vor dem Lauf)
```

## SX2-env aktivieren (nur nach Feyer-Zusage)
```bash
git checkout s-group-d-det-superlu-sx2-env   # oder Branch sx2-env mergen
# dann: Offenlegung in SUBMISSION/REPRODUCE -> Audit-Delta -> offizielle Messung -> Repack
```

## Ausprobiert und verworfen (gemessen, kein Netto-Gewinn / unter Rauschen)
Graph-Verkleinerung (Forward = 1 Callback) · X+MU-Mehrspalten-Solve · `jnp` statt Callback · SuperLU-Optionen · triviale Zeilen · `gc.disable()` · SX-P (`posix_fadvise`-Prefetch) · nativer C-Code (~10–20 ms, freigabepflichtig) · `lax.switch`-Fusion (schon drin).
