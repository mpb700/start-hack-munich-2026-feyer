# MS-Diagnose: Lazy Imports + SX (XLA level=1) auf der Gruppen-Basis `ab82e42`

Stand: 2026-11-21 02:25 UTC · Agent 06 · Worktree `/root/wt-10-ms-imports` (Branch `track/10-ms-imports`)
Fenster: 02:21:29–02:22:02 UTC, **Kern 14 exklusiv** (von 00 genehmigt), `env -u XLA_FLAGS`,
`taskset -c 14`, `benchmark.py --cpus 1 --repeats 10`, frische Prozesse.
Idle-Check vor/nach allen Läufen: **kein fremder `benchmark.py`/`quick_probe.py`** (pgrep, `/tmp/ms_window2.log`).

## Varianten

| Variante | Code-/Env-Änderung |
|---|---|
| **Gruppe (A)** | `ab82e42` unverändert (Vorlage) |
| **Lazy (B)** | nur `differometor/__init__.py`: `visualize_setup`/Plotly via PEP-562-`__getattr__` erst bei Bedarf (Commit `9ec11ed` auf `track/10-ms-imports`) |
| **GroupSX** | `ab82e42` + **env-only** `XLA_FLAGS=--xla_backend_optimization_level=1` (kein Commit, Proposal) |
| Lazy+SX | **nicht gemessen** (Zeitfenster) — nächster Schritt |

## Ergebnisse (s, e2e = First-Run end_to_end)

| Variante | warm #1 | warm #2 | warm ø | kalt #1 | imports warm | sens_compile | grad_compile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gruppe `ab82e42` | 1.0398 | 1.0606 | 1.0502 | 1.3361 | 0.381 / 0.384 | 0.200 / 0.208 | 0.169 / 0.170 |
| **Lazy** | **1.0216** | **1.0228** | **1.0222** | **1.2936** | 0.3625 / 0.3633 | 0.196 / 0.199 | 0.172 / 0.166 |
| **GroupSX (level=1)** | **1.0112** | **1.0112** | **1.0112** | 1.2983 | 0.378 / 0.380 | 0.189 / 0.185 | **0.1515 / 0.1521** |

Kaltläufe: vor jedem Lauf `rm -rf differometor/__pycache__` + `sync` + `drop_caches=3` (Kalt-Realismus inkl. neuem Bytecode).

### Δ vs. Referenz `S-gruppe-official` (Kern 14, e2e 1.0633 s, warm 0.1460 s)

- Lazy: **−0.041 s warm**, **−0.043 s kalt** (vs `S-gruppe-cold` 1.2705 s)
- GroupSX: **−0.052 s warm**, **−0.038 s kalt**
- Beide klar getrennt von der Gruppen-Streuung (Gruppe warm spread 0.021 s; Lazy 0.001 s; SX 0.0001 s)

## Numerik / Arrays

`validate`-Vergleich aller Varianten (auch SX!) gegen `measurements/group-d/group-b.npz`:
**alle 6 Felder (carrier, signal, noise, sensitivity, gradient, powers) max_abs = 0.0, rel_L2 = 0.0 → bit-identisch.**
`max|ln S − ln S_ref|` = 5.374249e-06 (unverändert, Δerr 0 vs candidate-group-d).
Lazy-Imports ändert nur Ladereihenfolge; XLA level=1 ist semantikerhaltend (kein Fast-Math).

## Interpretation

1. **Lazy-Imports**: Gewinn ~0.02–0.04 s, fast vollständig in `imports` (warm −0.019 s; kalt −0.043 s,
   weil Plotly-Submodule im Kaltfall von Platte gelesen werden). Der Rest der Import-Phase
   (0.36 s) ist jax (~0.24 s, vom unveränderlichen `benchmark.py` importiert) + `scipy.sparse`
   (~0.06 s, für den Fast-Path gebraucht) — nicht lazy-fähig ohne e2e-Verlagerung.
2. **SX level=1**: −0.03…−0.05 s warm, Ursache `gradient_compile` −0.017 s und
   `sensitivity_compile` −0.017 s (−8…10 %). Exec/warm nahezu unverändert (0.146→0.146–0.154).
   Zahlen oben sind Einzelmessungen (2 warm, 1 kalt) — **Bestätigung über `ms_gate.py`-A/B
   (10 warme + 4 kalte Paare) steht aus**, Fenster bei 00 angefragt.
3. Kombination Lazy+SX sollte ~0.05–0.07 s unter Gruppe liegen (zu messen).

## Status / Regeln

- Lazy-Patch: **committet** auf `track/10-ms-imports` (`9ec11ed`, ausschließlich `differometor/__init__.py`).
- SX: **NUR env-only** gemessen, **kein Commit** einer `XLA_FLAGS`-Setzung (Feyer-Freigabe offen,
  `xla_flags` erscheint im Report → Disclosure nötig).
- Basis-Repo `/root/faster-physics/repo` und FINAL-FREEZE `ab82e42` wurden **nicht** angefasst
  (alle Läufe im eigenen Worktree; Basis-Repo nur als Mess-/Referenzquelle gelesen).
- Keine Merge-Grundlage ohne `ms_gate.py` PASS + Gewinn über Rauschen (Lead-Regel).

## Modul-Analyse (Lead-Auftrag): was lohnt lazy, was nicht

`taskset -c 12 python -X importtime -c "import differometor"`, frische Prozesse, `env -u XLA_FLAGS`.
Gesamt: **353.2 ms (Basis eager) → 348.1 ms (lazy)**, d. h. **−5.2 ms** im warmen Microbench.

### (a) Im Benchmark **nie** benutzt → lazy = echter Gewinn (nur das ist committet)

| Modul | self | cum | Bewertung |
|---|---:|---:|---|
| `differometor.plot` | 3.61 ms | 4.73 ms | nur `visualize_setup` (Plotly-HTML); vom Benchmark nie aufgerufen |
| `plotly` + `plotly.graph_objects` + `_plotly_utils` | ~1.0 ms | ~1.1 ms | nur von `differometor.plot` gebraucht |
| **Summe lazy-fähig** | | **~5.2 ms** (warm) | im Fenster-Kontext gemessen: imports −19 ms warm / −43 ms kalt (Page-Cache-Effekt) |

### (b) Im Benchmark **benutzt** → Lazy würde die Zeit nur in eine andere gemessene Phase
verschieben (kein e2e-Gewinn) → **nicht** angefasst

| Modul | self | cum | warum nicht lazy |
|---|---:|---:|---|
| `differometor.simulate` → `jax` (via `jax.numpy`) | 2.67 ms | 242 ms | benchmark.py importiert `jax`/`jax.numpy` selbst **vor** differometor; nicht unser Anteil, Verschieben nutzlos |
| `differometor.fast` → `scipy.sparse` | 0.58 ms | 60–92 ms | Fast-Path wird im Benchmark benutzt (build/simulate) |
| `differometor.setups`, `.build`, `.components` | <1 ms | je <2 ms | im Benchmark benutzt |
| `numpy` | 43.7 ms (in jax.numpy enthalten) | — | vom Benchmark selbst importiert |

**Fazit:** Nur (a) `visualize_setup`/Plotly ist lazy-fähig; das ist umgesetzt (Commit `9ec11ed`).
Weitere Lazy-Umbauten in `differometor/` bringen keinen e2e-Gewinn (nur Phasen-Verschiebung) →
werden nicht gemacht. Der große Import-Block (jax ~242 ms) ist benchmark-eigener Code (unveränderlich).
