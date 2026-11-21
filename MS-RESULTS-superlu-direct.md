# MS: SuperLU direkt laden ohne scipy zu importieren (Lead-Prio 1)

Stand: 2026-11-21 02:42 UTC · Agent 06 · Worktree `/root/wt-10-ms-imports`, Branch `track/10-ms-imports`
Basis: **`df2a5a3`** (Tag `s-group-d-det`, Submission) · Commits: `b170ef1` (SuperLU), `6a3fafe` (Lazy, NOISE — nicht Merge-Kandidat)

## Umsetzung

Neu: `differometor/host_superlu.py` (~150 Zeilen):
1. `importlib.util.find_spec("scipy")` (führt scipy **nicht** aus) → Pfad zu
   `scipy/sparse/linalg/_dsolve/_superlu*.so` → `ExtensionFileLoader("_superlu", path)` →
   `exec_module` (**1,3 ms**, 0 scipy-Module).
2. `gstrf(N, nnz, data, indices_intc, indptr_intc, csc_construct_func=_CSCShim, ilu=False,
   options=dict(DiagPivotThresh=None, ColPerm=permc_spec, PanelSize=None, Relax=None))` —
   **exakt die Argumente von `scipy.sparse.linalg.splu` (scipy 1.18.1)**; bei
   `permc_spec="NATURAL"` zusätzlich `SymmetricMode=True` (wie scipy).
3. Eigene `CSC`-Klasse (`data/indices/indptr/shape`, kein `scipy.sparse`); Indizes werden auf
   `np.intc` gecastet (wie scipy `safely_cast_index_arrays`).
4. Rückfall: bei jedem Fehler (kein scipy, kein .so, ABI, `gstrf`-Exception) → lazy
   `from scipy.sparse.linalg import splu`; das ist die **einzige** scipy-Importstelle im Paket.

`differometor/fast.py`: scipy-Imports entfernt; Zuweisungen `sp.csc_matrix` → `_HostCSC`;
`scipy`-CSR-Scatter der Rückwärtspassage → `np.add.at` (bitgleiche Summationsreihenfolge
geprüft); Carrier-/Signal-Faktorisierung über `host_superlu.splu`.

## Import-Kosten (`python -X importtime -c "import differometor"`, frisch, Kern 12)

| | Basis `df2a5a3` | mit `host_superlu` | Δ |
|---|---:|---:|---:|
| `differometor` gesamt | 362,5 ms | **271,4 ms** | **−91 ms** |
| importtime-Zeilen mit "scipy" | 156 | **0** (nur `jax.scipy`) | — |
| `sys.modules` mit `scipy*` nach `import differometor` | 156 | **0** | — |
| `_superlu`-Ladezeit | — | 1,3 ms | — |

## Verhalten (spontane Läufe Kern 12, `--cpus 1 --repeats 3`, keine offizielle Messung)

| | `df2a5a3` (A) | SuperLU (B) | Δ |
|---|---:|---:|---:|
| e2e | 0,8436 s | **0,7662 s** | **−0,077 s** |
| imports | 0,3519 s | **0,2720 s** | **−0,080 s** |
| build | 0,0659 | 0,0641 | −0,002 |
| gradient_compile / sensitivity_compile | 0,0691 / 0,1403 | 0,0748 / 0,1446 | Rauschen |
| Arrays vs A | — | carrier/signal/noise/sensitivity/powers/parameters/frequencies **bitgleich**; gradient 25/193 Komponenten max_rel **9,4e-15** | nur letzte Bits |

Doppellauf deterministisch (bitgleich); lnS unverändert 5,37424851643209e-06; `direct_available=True`.
Rückfallpfad getestet (`_gstrf=None` → scipy splu): läuft, gleiche Arrays (bis auf dieselben Gradient-Bits).


## Offizielles A/B (Fenster 02:46:00–02:51:00 UTC, Kern 14 exklusiv, von 00 genehmigt)

`benchmark.py --cpus 1 --repeats 10`, frische Prozesse, `env -u XLA_FLAGS`, Idle-Check vor/nach
(**kein fremder benchmark/quick_probe**, pgrep-Log in `measurements/ms/superlu-audit/window.log`).
A = `/root/wt-10-ms-compile` (`df2a5a3`), B = `/root/wt-10-ms-imports` (`b170ef1`), abwechselnd.

| A/B | A median | σ_A | B median | σ_B | Δ (B−A) | pooled σ | Urteil |
|---|---:|---:|---:|---:|---:|---:|---|
| **warm (10 Paare)** | 0,8848 s | 0,0123 | **0,7832 s** | 0,0055 | **−0,1016 s (−11,5 %)** | 0,00955 | **GAIN** |
| **kalt (4 Paare, pycache gelöscht + drop_caches)** | 1,1528 s | 0,0058 | **1,0238 s** | 0,0020 | **−0,1290 s (−11,2 %)** | 0,00430 | **GAIN** |

Einzelpaare warm (Δ je Paar): −0,0915 / −0,0909 / −0,1117 / −0,0999 / −0,1086 / −0,1013 /
−0,1050 / −0,0995 / −0,0908 / −0,0876 s — **jedes der 10 Paare positiv**.
`imports`: warm 0,374–0,389 → 0,286–0,295 (−0,09 s); kalt 0,592–0,600 → 0,467–0,470 (−0,13 s).

## Gate (ms_gate Legacy-Checks auf denselben Läufen; Tool-Report-Code crasht in der A/B-Version — Zahlen aus Konsole + `a/b/cold.npz`)

| Prüfung | Ergebnis | Limit |
|---|---|---|
| Δerr vs `candidate-group-d` | **0,0** | ≤ 1e-8 |
| Doppellauf bit-identisch | **True** | ja |
| gradient rel_L2 vs `group-d/group-b.npz` | **2,29e-15** | ≤ 1e-5 |
| affinity | [14] | Kern 14 |
| Kaltlauf e2e | 0,9816 s | — |
| Benchmark-Validierung | finite/upstream-sens/FD **PASS** | PASS |

⇒ **GAIN + Gate PASS.**

## Audit-Nachweise (00-Vorgaben)

1. **Rückfall absichtlich ausgelöst** (`host_superlu._gstrf = None`): 154 scipy-Module werden
   geladen, Benchmark **PASS** (finite/sens/FD ok), Arrays identisch zum Kandidaten
   (gradient max_abs 7,3e-12 wie im Kandidaten selbst) — Log `measurements/ms/superlu-audit/fallback-run.log`.
2. **0 scipy-Module**: `python -X importtime -c "import differometor"` → 362,5 ms/156 scipy-Zeilen
   (Basis) vs **271,4 ms/0 scipy-Zeilen** (nur `jax.scipy`); `sys.modules` scipy-Anzahl = **0**.
   Rohlogs in `measurements/ms/superlu-audit/importtime-*.txt`.
3. **Kaltes A/B**: 4 Paare im Fenster (s. o.).
4. **Mechanik-Notiz** (für Regel-Audit): `importlib.util.find_spec("scipy")` (führt scipy nicht aus)
   → Glob auf `scipy/sparse/linalg/_dsolve/_superlu*.so` → `importlib.machinery.ExtensionFileLoader("_superlu", pfad)`
   → `spec_from_loader`/`module_from_spec`/`exec_module` (**1,3 ms**) → `module.gstrf(...)`.
   `gstrf`-Argumente **exakt wie `scipy.sparse.linalg.splu` 1.18.1**:
   `(N, nnz, data, indices.astype(intc), indptr.astype(intc), csc_construct_func=shim, ilu=False,
   options=dict(DiagPivotThresh=None, ColPerm=permc_spec, PanelSize=None, Relax=None))`,
   `SymmetricMode=True` bei `"NATURAL"`. **Keine neuen Dependencies** (scipy/numpy bleiben wie in
   `requirements.txt`); keine Caches/Precomputed-Arrays. Fallback: `from scipy.sparse.linalg import splu`
   nur im Fehlerfall — genau eine lazy scipy-Importstelle im Paket.
   Randfall „direkter `_superlu`-Kern + späteres `import scipy.sparse.linalg`" ist extern (Lead)
   verifiziert.

## Artefakte

`/root/states/06-0249-superlu.bundle` (sha256 `aedbed2ad6cf32b0a92462f47f0a4f936a4d9a6a62dae87c5d92740e589b7c03`),
Commit `b170ef1` auf `track/10-ms-imports` (Basis `df2a5a3`). Lazy-Commit `6a3fafe` bleibt
NOISE/nicht Merge-Kandidat.
