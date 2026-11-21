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

## Offizielle Messung im Fenster 03:48:00–03:53:00 (Kern 14 exklusiv) — Basis `ac7aedc`

A = `ac7aedc` (`/root/wt-10-ms-imports`), B = SX2-env (`275a8ee`), 10 warme + 4 kalte Paare,
`--cpus 1 --repeats 10`, `env -u XLA_FLAGS`. Idle-Check vor/nach je **„none"**.

### (a) Report-Nachweis (Anforderung 00)

`environment.xla_flags` in `s2e-B-w01.json` / `s2e-B-w02.json`:
```
--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true
```
(non-null und exakt wie vorgesehen) ✓

### (b) Append statt Überschreiben (Anforderung 00)

Test mit externem Flag: `XLA_FLAGS="--xla_cpu_max_isa=AVX2"` → nach `import differometor`:
```
--xla_cpu_max_isa=AVX2 --xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true
```
→ vorhandene Flags bleiben erhalten ✓

### (c) Decksungsgleich mit SX2 (`036194a`)

- **Arrays SX2-env vs offizielle SX2-Outputs** (`measurements/S-sx2-cold.npz`):
  carrier/signal/noise/sensitivity/powers/gradient/parameters/frequencies **alle bit-identisch (max_abs = 0,0)** ✓
- Gegen `ac7aedc`: sensitivity 9,2e-41, gradient 1,31e-10 (wie SX2; nur letzte Bits).
- **Absolute Zeiten** B-minima: warm **0,6880 s**, kalt **0,9206 / 0,9221 s**
  ↔ 00-offizielle SX2-Werte warm 0,681/0,691 s, kalt 0,920 s → **deckungsgleich** ✓
- Compile: `sensitivity_compile` 0,1625 → **0,0952 s** (−0,067), `gradient_compile`
  0,0816 → **0,0595 s** (−0,022) — wie bei SX2.

### (d) Gate (B)

Δerr = 0 (≤1e-8), Doppellauf bit-identisch, gradient rel_L2 = 1,21e-15 (≤1e-5),
finite/sens/FD PASS; gate-e2e warm 0,7001/0,7466 s, kalt 0,8457 s → **GATE PASS**.

### A/B-Verdicts: NOISE — wegen **Fremdlast im Fenster kontaminiert**

| A/B | A median | σ_A | B median | σ_B | Δ | pooled σ | Urteil |
|---|---:|---:|---:|---:|---:|---:|---|
| warm, 10 Paare | 0,8499 s | 0,0897 | 0,8376 s | 0,0809 | −0,0123 s | 0,0854 | NOISE (kontaminiert) |
| kalt, 4 Paare | 1,0157 s | 0,0391 | 0,9853 s | 0,0949 | −0,0303 s | 0,0726 | NOISE (kontaminiert) |

Die A-Werte drifteten im Fenster von 0,7725 → 0,9937 s (Systemlast durch andere Agenten; Lastavg
stieg auf >12), 1 Kaltpaar hatte einen Ausreißer. Die Paare selbst zeigen den erwarteten Effekt
(9 von 10 warmen Paaren: B −0,03…−0,13 s). **Sauberer A/B-Rerun bei 00 angefragt** (Fenster);
die Nachweise (a)–(d) sind davon unabhängig bereits vollständig grün.

## FINALES sauberes Rerun (Fenster 03:58:00–04:03:00 UTC, Kern 14 exklusiv, strikte Ruhe)

A = `ac7aedc` (`/root/wt-10-ms-imports`), B = SX2-env (`396df84`, Code `275a8ee`), 10 warme +
4 kalte Paare, `--cpus 1 --repeats 10`, `env -u XLA_FLAGS`, frische Prozesse.
Idle-Check vor/nach: **kein fremder benchmark/quick_probe**. Lauf: 03:58:43–04:00:25.

| A/B | A median | σ_A | B median | σ_B | Δ (B−A) | pooled σ | Urteil |
|---|---:|---:|---:|---:|---:|---:|---|
| **warm, 10 Paare** | 0,7813 s | 0,0044 | **0,6993 s** | 0,0075 | **−0,0820 s (−10,50 %)** | 0,00611 | **GAIN** |
| **kalt, 4 Paare** | 1,0117 s | 0,0008 | **0,9263 s** | 0,0054 | **−0,0855 s (−8,45 %)** | 0,00384 | **GAIN** |

**Alle 10 warmen Paare negativ** (−0,0749…−0,0903 s); alle 4 kalten Paare negativ (−0,0802…−0,0929 s).
Compile-Effekt (Median): `sensitivity_compile` 0,1417 → **0,0843 s (−0,0574)**;
`gradient_compile` 0,0721 → **0,0487 s (−0,0234)**; `imports` 0,2889 → 0,2881 (neutral).
`environment.xla_flags`: exakt `--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true` ✓.

**Arrays:** gegen `ac7aedc` carrier/signal/noise/powers/parameters/frequencies bitgleich,
sensitivity 9,2e-41, gradient 1,31e-10 (letzte Bits); **gegen offizielle SX2-Outputs
(`S-sx2-cold.npz`) alle Felder bitgleich inkl. gradient = 0,0** ✓ (SX2-env ≡ SX2).

**Gate (B):** Δerr = 0 (≤1e-8), Doppellauf **bit-identisch**, gradient rel_L2 = 1,21e-15 (≤1e-5),
finite/sens/FD **PASS**; Gate-e2e warm 0,7085/0,7060 s, kalt 0,9205 s → **GATE PASS**.

**Fazit:** SX2-env reproduziert SX2 exakt (Arrays bitgleich, Zeiten 0,6993/0,9263 vs
0,681–0,691/0,920) und ist die **transparente, report-sichtbare Feyer-Variante**:
**−10,5 % warm / −8,5 % kalt**, bitgleich zu SX2, Gate PASS.
