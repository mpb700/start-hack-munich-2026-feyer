# Morgenbericht Nachtlauf Differometor (Schlussbericht)

Stand: nach Runde 4 von 4 (21.11.2026, Start 07:06 Uhr, Ende ca. 07:14 Uhr).

## Kurz

- **Noel hat unseren Commit `f1000e9` übernommen** (Merge `7279114` auf `origin/noel`) und darauf zwei weitere Runden gebaut. Neuer Stand: `origin/noel@6b67cfd`.
- **Bester Stand ist jetzt `noel@6b67cfd`**, lokal als `nacht/basis-1`. Er besteht alle Prüfungen und ist gegen unseren bisherigen `f1000e9` messbar schneller: Forward −10 ms, Gradient −5,5 ms, warm −15 ms.
- In Runde 4 gab es **keine neue Variante**. Die Profilierung des neuen Stands findet keinen Hebel ≥ 5 ms, der ohne Regelfrage umsetzbar wäre (Details unten).
- **Empfehlung für das Team:** `origin/noel@6b67cfd` abgeben. Unser Beitrag aus der Nacht steckt darin.

## Bester Stand

- **`noel@6b67cfd`** (lokaler Branch `nacht/basis-1`, Worktree `fayer-nacht/kandidaten/noel-6b67cfd`). Enthält `f1000e9`.
- Wechselmessung gegen den bisherigen besten Stand `f1000e9`. Setup: M2 Pro, macOS, Threads auf 1, ohne .pyc. Werte als Median / Minimum in ms.

  | Messreihe | f1000e9 | noel@6b67cfd | Δ Median / Min |
  | --- | ---: | ---: | ---: |
  | 8 Wechsel (`mess/k-noel-6b67cfd/ab.txt`) | 935,1 / 929,0 | **917,2 / 907,5** | −1,91 % / −2,3 % |
  | 12 Wechsel (`mess/k-noel-6b67cfd/ab12.txt`) | 933,1 / 926,3 | **916,8 / 911,7** | −1,75 % / −1,6 % |

  Phasen aus der 12er-Reihe, Median in ms. Imports 722,3, Build 67,5, Compile 79,0, Forward 34,7, Gradient 13,4, warm 40,0.
- **Übernahme-Entscheidung:** Der Median liegt knapp unter der 2-%-Schwelle aus Schritt 1.4. Ich habe deshalb die Grenzfallregel aus 2.5 analog angewandt: beide Messreihen ≥ 1 % Gewinn, das Minimum sinkt ebenfalls. Außerdem gilt laut Anleitung Noels Stand, wenn er unsere Idee schon enthält. Das ist hier der Fall, denn `f1000e9` ist ein Vorfahr von `6b67cfd`.
- Prüfungen auf `6b67cfd` (`mess/k-noel-6b67cfd/`):
  - check_fast: Log-Fehler 5,374e-6, Signal 6,9e-8 (3 Frequenzen), Rauschen 7,25e-8, Sensitivität 2,165e-9, Gradient 1,172e-8, Powers 1,39e-11, Carrier 9,7e-12
  - check_setups: alles 0,0 gegen JAX
  - test_detectors: 2,86e-14
  - test_adjoint: 5,83e-8
  - benchmark: **PASS** (Einzellauf 922 ms)
- Vergleich mit `origin/noel`:
  - Gegen den Stand zu Beginn der Nacht (`c261b63`, ca. 950–980 ms in unseren Reihen) ist der beste Stand jetzt etwa 3–4 % schneller. Eine direkte gemeinsame Messreihe dazu gibt es nicht.
  - Gegen das aktuelle `origin/noel` gibt es keinen Unterschied: Der beste Stand ist dieser Commit.

## Kollegen-Log

| Zeit | Branch | Stand | Befund |
| --- | --- | --- | --- |
| 01:04 / 01:31 | alle | – | Runde 1: nichts Neues |
| 03:08 | origin/Joel | 5eece1d | neu, identisch mit `main` |
| 05:08 | origin/noel | dba2828 | Noel Runde 2. Gemessen: PASS, −0,8 % gegen f1000e9, nicht übernommen |
| 05:08 | origin/Joel | ac7aedc | alles schon in `noel` enthalten |
| 05:08 | origin/sx, sx2, sx2-env | – | XLA-Flags bzw. `jax_disable_most_optimizations` aus der Library. Regelfrage, nicht gemessen |
| **07:07** | **origin/noel** | **6b67cfd** | Noel Runden 3+4: **Merge unseres `f1000e9`**, `_BlockLU` (16 Frequenzen pro SuperLU-Aufruf), keine dichte Signalmatrix im Rückwärtslauf, Komponenten schreiben direkt in die Einträge, Rückwärtslauf speichert nur echte Tangenten. **Gemessen, alle Prüfungen bestanden, übernommen** (siehe oben) |
| 07:07 | origin/Joel | 76e799f | nur README (Status-Übersicht), kein Code |
| 07:07 | philip, max/lu-once, main, sx*, Fork Haniubub | – | unverändert |

## Varianten der Nacht

Zeiten: TOTAL Median / Min in ms (Basis in Klammern), ohne .pyc.

| Nr | Runde | Idee | Ergebnis | Entscheidung | Grund |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | K3: MMD_ATA statt COLAMD | 987,3 / 965,8 (979,6 / 959,6) | verworfen | ~1 ms, im Rauschen |
| 2 | 1 | K4/K8: Callback-Haushalt, bitgleich | 952,1 / 941,0 (951,8 / 942,8) | verworfen | TOTAL +0,3 ms, Patch aufbewahrt |
| 3 | 1 | N2/K6: Build/Import straffen | – | zurückgestellt | kein Teil > 1 ms |
| 4 | 1 | Nr. 2 mit .pyc (nur Info) | 366,4 / 362,0 (368,5 / 362,8) | nur Info | −0,6 % |
| 5 | 1 | N3: triviale Unbekannte | Mikro −2,5 ms | zurückgestellt | nur ohne Pivotprüfung, Regelfrage |
| 6 | 2 | Bündel Nr. 1+2 + `_EntryBlock.back` | 947,7 / 942,8 (949,3 / 941,7) | verworfen | −0,17 %, Min schlechter |
| 7 | 2 | Nr. 6 + schlanker Import | 944,6 / 926,5 (956,3 / 936,9); 938,5 / 932,0 (951,4 / 931,1) | **akzeptiert, gepusht** | Grenzfall, 2× ≥ 1 %. **Von Noel übernommen** |
| 8 | 2 | Solver-Schleife profiliert | – | zurückgestellt | einziger Hebel ≥ 8 ms sind Threads, Regelfrage |
| 9 | 3 | noel@dba2828 + unser Import-Teil | 930,9 / 918,9 (936,7 / 930,4) | verworfen | −0,62 % < 1 %. Inzwischen von Noel per Merge erledigt |
| 10 | 4 | Profil von noel@6b67cfd, Suche nach Hebel ≥ 5 ms | – | keine Variante gestartet | siehe unten |

## Erkenntnisse aus Runde 4 (Profil, `fayer-nacht/analyse/r4/`)

- `import differometor` kostet nur noch **13,9 ms** (5 Prozesse, Median, `-X importtime`).
  - Davon sind ca. 11,5 ms reines Übersetzen der Quelltexte ohne .pyc: fast 4,3, build 4,3, components 1,8, setups 0,7, simulate 0,4 ms.
  - Ungenutzter Code in components.py und simulate.py macht höchstens ca. 0,5 ms aus.
- `import jax` kostet ca. 717 ms. Dazu lädt JAX selbst Module nach, z. B. MLIR-Dialekte (28 ms beim ersten Lowering). Das ist für jedes JAX-Programm gleich, darauf haben wir keinen Einfluss.
- Die Forward- und Gradient-Callbacks sind flach profiliert. Größter Einzelposten ist die SuperLU-Zerlegung (ca. 8–10 ms). Noels offene Ideen 1–3 sind gemessen und bringen nur 1,4–3,5 ms.
- **Ergebnis:** Kein Hebel ≥ 5 ms (0,5-%-Grenzfall) ohne Regelfrage. Die verbleibenden großen Hebel sind alle Regelfragen (siehe unten).

## Zurückgestellt / nächste Kandidaten (nur nach Klärung)

1. `jax.config.update("jax_disable_most_optimizations", True)`: laut Noel −13 ms auf dem Mac, −42 ms in Docker. Regelfrage B6 an Feyer.
2. Threads für die 7 `_BlockLU`-Blöcke: Gewinn nicht gemessen. Ob SuperLU die GIL freigibt, ist nicht geprüft. Regelfrage.
3. Feste glibc-Schwellen über `mallopt`: ca. 3,5 ms in Docker, auf dem Mac nicht gemessen. Regelfrage B6.
4. Kleine Ideen unter der Schwelle: Noels Ideen 1–3 und Auslagern ungenutzter Funktionen (≤ 0,5 ms).

## Fragen an Max

1. Messung mit Bytecode-Cache als Entscheidungsgrundlage? Mit .pyc ca. 367 statt ca. 950 ms, die 2-%-Schwelle läge dann bei ca. 7 ms.
2. Darf die Library intern Python-Threads nutzen (LU-Blöcke parallel)?
3. Die sx-Branches (Joel/agent06) und Noels Idee 4 setzen JAX- oder XLA-Schalter aus der Library. Das sollte vor der Abgabe mit Feyer geklärt sein (B6).
4. Übernahme von 6b67cfd bei −1,75 bis −1,91 % (knapp unter 2 %): einverstanden mit der analogen Grenzfallregel?

## Auf origin/max/nacht

- `f1000e9` „Lighter differometor import and leaner sparse callbacks“ (Basis `c261b63`). Messung liegt in `mess/7/`. Gepusht 03:25, ohne Force. Seitdem nichts Neues gepusht.
- **Noel hat ihn übernommen** (Merge `7279114`), mit einer Ausnahme: Statt unserer Spaltenordnung `MMD_ATA` gilt seine Ordnung `MMD_AT_PLUS_A` mit Skalierung.
- Runde 4 hat nichts gepusht. Der beste Stand enthält keine eigenen Commits außer dem schon gemergten `f1000e9`. Ein Push von Noels Commits auf `max/nacht` hätte nichts Eigenes hinzugefügt.

## Aufräumliste (alles vom Nachtlauf angelegt, nichts davon ist gelöscht)

Lokale Branches in `fayer-muc`:
- `nacht/7-import-schlank` (f1000e9, identisch mit `origin/max/nacht`, in `noel` enthalten)
- `nacht/basis-1` (6b67cfd = `origin/noel`)

Worktrees unter `/Users/maxjackle/Projekte/fayer-nacht` (zusammen ca. 614 MB mit venv):
- `wt/7-import-schlank` (nacht/7-import-schlank)
- `kandidaten/noel-6b67cfd` (nacht/basis-1, aktueller bester Stand)
- `kandidaten/noel-dba2828` (detached)
- `kandidaten/noel-c261b63` (detached)
- außerdem die Ordner `analyse/`, `laeufe/`, `pyc-messung/`, `ref/`, `verworfen/`, `.venv/`

Remote: `origin/max/nacht` (f1000e9). Kann bleiben oder gelöscht werden, da der Inhalt in `noel` steckt.

Befehle zum Aufräumen (erst ausführen, wenn nichts mehr gebraucht wird):

```bash
cd /Users/maxjackle/Projekte/fayer-muc
for w in wt/7-import-schlank kandidaten/noel-6b67cfd kandidaten/noel-dba2828 kandidaten/noel-c261b63; do git worktree remove --force /Users/maxjackle/Projekte/fayer-nacht/$w; done
git branch -D nacht/7-import-schlank nacht/basis-1
git worktree prune
# optional: rm -rf /Users/maxjackle/Projekte/fayer-nacht ; git push origin --delete max/nacht
```

Zustand, Ledger, Messdateien und Patches liegen unter `runs/nacht/`, also `state.json`, `ledger.jsonl`, `mess/` und `verworfen/`.
