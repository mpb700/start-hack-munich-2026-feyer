# Evaluation modes — welcher Schalter je Feyer-Antwort

Stand: 22.11., Xeon Platinum 8468 (Nebius-VM), kalt gemessen. Drei Varianten, in Sekunden umschaltbar, derselbe Code.

| Variante | Schalter | kalt | warm |
|---|---|---|---|
| **A – safe** | `DIFFEROMETOR_STARTUP=gc,lazy,defer,jits` (default, ohne preimport) | ~1,30 s | ~0,23 s |
| **B – preimport** | A + `preimport` (braucht `packaging/differometor_startup.pth` in site-packages) | ~0,13 s | ~0,13 s |
| **C – max** | B + nicht-bit-identisches XLA-Tuning (`opt0`+`summen`, Frage Q6) | ~0,11 s | ~0,11 s |

## Welche Feyer-Antwort → welche Variante

| Feyer sagt | nutze |
|---|---|
| „benchmark.py darf geändert werden" (I/O gleich) → Import-Timer verschiebbar | **B** (oder C) |
| „.pth / Preimport vor dem Timer erlaubt" | **B** |
| „XLA-Tuning opt0+summen erlaubt" (Q6) | **C** |
| keines davon / konservativ | **A** |

## Hinweise

- **B/C verschieben die Imports vor den ersten Timer** („kind B"). Das ist die Grauzone; ohne Freigabe **A**.
- **C ist zusätzlich nicht bit-identisch** (opt0+summen), deshalb getrennt von B.
- Der `.pth` wird bei `pip install -e .` **nicht** automatisch installiert; von Hand nach `site-packages` kopieren.
- Ehrliche Pitch-Zahl: **kalt gegen kalt Faktor 34** (A). Mit B/C sinkt sie weiter — dann aber als Grauzone klar benennen.
