# Optimierung auf Branch `noel`: Stand, Messmethode, offene Ideen

Stand: Sa 21.11.2026, nach der siebten Runde (Commit `e311d5c`). Diese Datei ist das Gedächtnis für die nächste Optimierungsrunde.

## Messwerte

Vier Stände im Wechsel gemessen (je frischer Prozess), Erstlauf gesamt in ms, Median / Minimum. Alle Läufe `PASS`. Die absoluten Werte hängen vom Zustand der Maschine ab. Vergleichbar sind nur Werte aus derselben Messreihe.

| Umgebung | Philip + Max (`11b696f`) | Nach Runde 1 (`c261b63`) | Nach Runde 2 (`49d386b`) | Nach Runde 3 (`1442ef3`) |
| --- | ---: | ---: | ---: | ---: |
| Apple M4, macOS, Threads auf 1 begrenzt, 12 Läufe | 606,5 / 602,7 | 460,3 / 457,9 | 454,4 / 450,1 | **445,8 / 441,3** |
| Docker auf demselben M4, Linux aarch64, `--cpus 1`, 10 Läufe | 716,5 / 707,2 | 542,8 / 537,4 | 532,3 / 528,2 | **519,7 / 517,3** |

Das Original-Kit braucht auf diesem Mac 12'827 ms (gemessen in Runde 1). Eine Messung von `noel` auf x86-Linux fehlt weiterhin. Joel hat eine solche Maschine (siehe «Stand der anderen Branches»).

Phasen (Median, ms):

| Phase | Mac: Philip + Max | Mac: Runde 1 | Mac: Runde 2 | Mac: Runde 3 | Docker: Philip + Max | Docker: Runde 1 | Docker: Runde 2 | Docker: Runde 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Imports | 407,1 | 314,7 | 313,5 | 312,4 | 386,2 | 308,9 | 308,4 | 309,0 |
| Build (inkl. Init, Laden) | 32,1 | 31,7 | 29,9 | 30,3 | 32,6 | 32,8 | 29,2 | 29,0 |
| Kompilation | 94,3 | 65,0 | 65,0 | 65,3 | 205,4 | 129,9 | 130,0 | 130,1 |
| Forward | 47,4 | 33,3 | 31,5 | 26,4 | 50,4 | 43,4 | 42,1 | 35,0 |
| Gradient (inkl. Sensitivity, Transfer) | 26,9 | 16,4 | 13,2 | 11,7 | 39,3 | 27,7 | 21,9 | 17,7 |
| Warm (Median) | 66,9 | 43,6 | 38,3 | 32,8 | 84,0 | 62,0 | 51,9 | 38,6 |

Runde 4 gegen Runde 3, eigene Messreihe (Mac 16 Läufe, Docker 12 Läufe, Median):

| Umgebung | Erstlauf Runde 3 (`6635856`) | Erstlauf Runde 4 (`ba8f1f2`) | Gradient-Phase | Warm |
| --- | ---: | ---: | --- | --- |
| Mac | 437,9 ms | **435,9 ms** | 10,7 → 9,8 ms | 32,4 → 31,3 ms |
| Docker `--cpus 1` | 519,3 ms | **515,0 ms** | 16,4 → 14,8 ms | 40,3 → 38,4 ms |

Runde 5 gegen Runde 4 (`ba8f1f2` → `45a2328`), eigene Messreihen, Median. Der Laptop lief dabei auf Akku und wurde benutzt. Die absoluten Werte liegen darum rund 5 % höher als oben:

| Umgebung | Erstlauf gesamt | Erstlauf ohne Imports | Gradient-Phase (inkl. Sensitivity, Transfer) | Warm |
| --- | --- | --- | --- | --- |
| Mac, 20 Läufe | 461,2 → 461,6 ms | 139,5 → **138,4 ms** | 11,1 → 11,2 ms | 32,8 → 32,5 ms |
| Docker `--cpus 1`, 12 Läufe | 534,7 → 535,3 ms | 219,4 → **216,1 ms** | 17,1 → 14,6 ms | 41,6 → 44,8 ms |
| Docker `--cpus 1`, 16 Läufe, Reihenfolge getauscht | 548,5 → 535,5 ms | 221,1 → **214,6 ms** | 17,7 → 14,5 ms | 39,7 → 46,6 ms |

Die Import-Phase gehört zu `benchmark.py` und JAX und streut um ±3 ms, also mehr, als die Änderung bringt (Imports im Median: Mac 321,4 → 322,6 ms, Docker 314,7 → 318,6 und 320,5 → 319,3 ms). Darum steht der Erstlauf hier auch ohne Imports. In der Gesamtzeit ist der Gewinn nicht nachweisbar.

Runde 6 gegen Runde 5 (`45a2328` → `7d12ba7`), eigene Messreihen, Median, weiterhin auf Akku:

| Umgebung | Erstlauf gesamt | Erstlauf ohne Imports | Forward-Phase | Gradient-Phase (inkl. Sensitivity, Transfer) | Warm |
| --- | --- | --- | --- | --- | --- |
| Mac, 16 Läufe, in Runde 7 nachgemessen | 446,3 → **444,9 ms** | 131,5 → 128,8 ms | 27,0 → 24,0 ms | 10,4 → 10,0 ms | 31,6 → 29,8 ms |
| Docker `--cpus 1`, 16 Läufe | 515,4 → **511,5 ms** | 206,7 → 200,9 ms | 35,1 → 29,7 ms | 13,6 → 13,3 ms | 43,3 → 35,2 ms |
| Docker `--cpus 1`, 12 Läufe, Reihenfolge getauscht, Image aus dem Commit | 518,3 → **506,7 ms** | 209,3 → 196,6 ms | 34,8 → 29,4 ms | 14,0 → 13,1 ms | 43,4 → 35,4 ms |

Die Benchmark-Ausgaben (Carrier, Signal, Noise, Sensitivity, Powers, Gradient) sind bitgleich zum Stand davor.

Die Mac-Zeile ersetzt die erste Messung von Runde 6 (441,3 → 433,4 ms gesamt). Dort war die Import-Phase um rund 3 ms zugunsten des neuen Stands verfälscht (siehe «Erkenntnisse aus Runde 7», Bytecode). Die Phasen Forward, Gradient und Warm stimmen in beiden Messungen überein, die Docker-Reihen sind nicht betroffen.

Runde 7 gegen Runde 6 (`d0e0fa8` → `e311d5c`), je 16 Läufe im Wechsel, Median:

| Umgebung | Erstlauf gesamt | Build-Phase | Erstlauf ohne Imports | Warm |
| --- | --- | --- | --- | --- |
| Mac | 438,3 → **437,8 ms** | 25,8 → 23,6 ms | 126,2 → 124,5 ms | 29,3 → 29,7 ms |
| Docker `--cpus 1` | 511,3 → **507,5 ms** | 26,2 → 23,0 ms | 201,9 → 198,5 ms | 35,8 → 35,5 ms |

Die Änderung betrifft nur den Build. Forward, Gradient und Warm sind innerhalb der Streuung gleich. Auf dem Mac verschwindet der Gewinn von gut 2 ms in der Streuung der Import-Phase (±3 ms).

Die Reports `mac-neu.json`, `mac-neu.png` und `docker-neu.json` zeigen den Stand `e311d5c`, erzeugt am 21.11. um 10:54 auf Akku bei ruhigem Rechner: Mac 431,6 ms im Erstlauf und 29,2 ms warm (der mittlere von fünf Läufen, 427,1 bis 434,3 ms), Docker 499,1 ms und 35,7 ms (der mittlere von fünf, 496,9 bis 510,1 ms). Alle `PASS`. Der Mac-Report lief ohne Bytecode-Cache für `differometor` (siehe Runde 7). Mit Cache wären es rund 10 ms weniger. Die `*-vorher`-Reports zeigen Philip + Max.

## Was in Runde 7 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 20 | Build-Schritt verschlankt, bei identischer Ausgabe. `set_instructions` brauchte 2,7 von 5 ms, fast alles in der Erweiterung auf zwei Seitenbänder (170 Aufrufe mit je rund 35 kleinen NumPy-Schritten). Die Erweiterung läuft jetzt in `_sideband_layout` auf Python-Listen. Dazu: Einheitsdiagonale der Trägermatrix ohne temporäre Einheitsmatrix, Suche der Nichtnull-Einträge auf der Float-Sicht, Auffüllen der Funktionseingänge mit einer Allokation, konstante Zeilen mit `np.full`, Wörterbuch statt `list.index` für die optimierten Parameter | `build.py`, neu `runs/noel/test_build.py` | Build-Phase: Docker 26,2 → 23,0 ms, Mac 25,8 → 23,6 ms. `run_build_step` allein: Mac warm 6,7 → 4,2 ms, erster Aufruf im Docker-Ablauf 8,5 → 5,9 ms |

Zu Nr. 20, wie die identische Ausgabe geprüft ist:

- Die Arrays aus `run_build_step` sind für 27 Setups gleich (946 Arrays, `PYTHONHASHSEED=0`): Benchmark-Setup in drei Varianten, aLIGO und Voyager in je drei Modi, zufällige UIFOs der Grössen 1 bis 3.
- `set_instructions` alt gegen neu in 20'000 zufälligen Aufrufen: gleiche gespeicherte Anweisungen und gleiche Änderungen an den übergebenen Arrays. Das Original verschiebt Zeilen- und Spaltenindizes des Aufrufers an Ort und Stelle, auch wenn beide dasselbe Array sind (so beim Verlustrauschen). Das ist nachgebildet.
- `runs/noel/test_build.py` prüft `_sideband_layout` gegen eine Referenz in der Schreibweise des Originals und kann Build-Ausgaben zweier Stände vergleichen.

## Was in Runde 6 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 19 | Der Build übergibt Träger-, Signal- und Rauschmatrix in Koordinatenform statt als dichte Arrays: `differometor.fast.SparseMatrix` (Zeilen und Spalten, Werte, feste Form, als JAX-Pytree registriert). Die dichte Signal- und Rauschmatrix (je 9 MB) entstehen gar nicht mehr. Der Fast-Path rechnet direkt mit den Einträgen. Der JAX-Weg (`simulate_jax`) macht daraus am Eingang wieder dichte Arrays. Wer `simulate` dichte Matrizen übergibt, landet im JAX-Weg | `fast.py`, `build.py`, `_simulate_dicht.py`, `simulate.py`, neu `runs/noel/test_sparse.py` | Forward-Phase: Docker 35,1 → 29,7 ms, Mac 26,7 → 23,6 ms. Gradient-Phase: Docker 13,6 → 13,3 ms, Mac 10,3 → 9,8 ms. Warm: Docker 43,3 → 35,2 ms, Mac 31,3 → 29,5 ms |

Zu Nr. 19, was damit wegfällt:

- JAX kopiert keine 19 MB dichte Matrizen mehr in den Forward-Callback, und der Callback durchsucht sie nicht mehr nach Einträgen.
- Die Trägermatrix (1,3 MB) ist kein Residuum mehr. Alle 28 Arrays, die der Rückwärtslauf braucht, stecken in den zwei gepackten Residuen. Der Rückwärts-Callback hat 12 statt 16 Argumente.
- Die feste Obergrenze für konstante Einträge (`BASE_ENTRIES_PER_UNKNOWN`) und ihre Fehlermeldung gibt es nicht mehr. Der allgemeine Rückwärtslauf bekommt die Einträge jetzt vollständig.
- Das ist die frühere offene Idee 1 in anderer Form. Der Einwand von damals (zwei Darstellungen derselben Matrix könnten auseinanderlaufen) entfällt, weil es nur noch eine Darstellung gibt. Die frühere Idee 2 (Trägermatrix) ist mit erledigt.

Was sich für andere ändert: `arrays["carrier_matrix"]`, `arrays["signal_matrix"]` und `arrays["noise_matrix"]` aus `run_build_step` sind `SparseMatrix`-Objekte. `np.asarray(matrix)` und `matrix.dense()` liefern die dichte Matrix wie bisher, `matrix.shape` bleibt gleich. `SparseMatrix.from_dense(...)` wandelt eine dichte Matrix um. Zurücknehmen: `git revert 7d12ba7`.

## Was in Runde 5 geändert wurde

Noel hat am 21.11. um 08:30 entschieden, die in Runde 4 gemessene Bündelung einzubauen, obwohl die warmen Wiederholungen in Docker damit langsamer werden.

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 18 | Der Forward-Callback packt die 22 kleinen Arrays, die der Rückwärtslauf braucht (Index-Arrays, Parameter, Frequenzen), in zwei Residuen: ein `int64`-Array mit Kopfteil (Datentyp und Form jedes Arrays) und ein `float64`-Array. Der Rückwärts-Callback stellt sie als Sichten exakt wieder her (`_pack`, `_unpack`, `_packed_sizes`). Der Rückwärts-Callback bekommt 16 statt 36 Argumente, die Programme reichen 1 statt 23 Host-Arrays durch. Die Trägermatrix (1,3 MB) geht weiter einzeln | `fast.py`, neu `runs/noel/test_pack.py` | Erstlauf ohne Imports: Docker 219,4 → 216,1 ms und 221,1 → 214,6 ms, Mac 139,5 → 138,4 ms. Warm in Docker 41,6 → 44,8 und 39,7 → 46,6 ms, auf dem Mac gleich |

Zu Nr. 18: Die Ausgaben aller Prüfskripte sind Zeile für Zeile gleich wie vorher (bis auf Zeitangaben). Die Phasen einzeln, als Unterschied der Mediane, für die zwei Docker-Reihen und die Mac-Reihe:

| Phase | Docker | Mac |
| --- | --- | --- |
| Forward: Lowering und Kompilation | −0,5 und −0,6 ms | −0,3 ms |
| Forward | −0,1 und +0,2 ms | +0,2 ms |
| Sensitivity (Lauf) | −0,6 und −0,6 ms | 0,0 ms |
| Gradient: Lowering und Kompilation | 0,0 und −0,5 ms | −0,8 ms |
| Gradient (Lauf) | −2,1 und −2,7 ms | 0,0 ms |
| Sensitivity: Kompilation (gehört zu `benchmark.py`) | −2,0 und −0,2 ms | −1,0 ms |

## Was in Runde 4 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 17 | Der Rückwärtslauf speichert Ableitungen nur noch für Funktionsausgänge, die wirklich von einer Designvariable abhängen, und zwar in der Form, in der die Dual-Zahlen sie liefern. Vorher füllte `_EntryBlock.run` pro Gruppe ein dichtes Array (Zeilen × Aufrufe × 4 Ausgänge × Eingänge), grösstenteils Nullen. Ableitungen, die nicht von der Frequenz abhängen, behalten eine einzige Zeile | `fast.py`, `_EntryBlock.run` und `back` | Rückwärts-Callback einzeln auf dem Mac 8,0 → 7,2 ms. Gradient-Phase: Mac 10,7 → 9,8 ms, Docker 16,4 → 14,8 ms |

In dieser Runde gemessen und nicht übernommen:

| Versuch | Ergebnis |
| --- | --- |
| Die 22 kleinen Arrays für den Rückwärtslauf im Forward-Callback zu zwei Residuen bündeln (36 → 16 Callback-Argumente, 23 → 1 durchgereichtes Host-Array) | Erstlauf: Docker 514,6 → 511,4 ms (Gradient-Phase 16,1 → 13,8 ms), Mac 438,0 → 436,6 ms, je 12 Läufe. Aber warm in Docker 38,9 → 44,8 ms. Grund: Ohne die vielen kleinen Arrays stutzt glibc den Heap nach jedem Aufruf, und jeder Aufruf fasst wieder rund 4'400 (Forward) und 3'300 (Rückwärts) frische Seiten an. Auf dem Mac bleibt warm gleich. Eine Variante, die nur die 18 Index-Arrays bündelt, verhält sich warm genauso. In Runde 4 zurückgenommen, in Runde 5 auf Noels Entscheid eingebaut (Nr. 18) |
| Alle Ergebnis-Arrays des Forward-Laufs in einem gemeinsamen Speicherblock anlegen | 886 → 472 Seitenfehler, aber nur 0,1 bis 0,3 ms. Nicht der Mühe wert |
| SuperLU mit anderen Werten für `PanelSize` (1, 2, 4, 8) und `Relax` (1, 2, 3, 4, 6), mit und ohne `SymmetricMode` | Keine Einstellung ist messbar schneller als `PanelSize=1`, `Relax=1` (12,96 ms im Median für Zerlegung und beide Lösungen auf dem Mac). Die beste andere liegt bei 12,83 ms, also innerhalb der Streuung. Ab `Relax=2` wird es langsamer, bis 17,8 ms |
| Idee 1 als Wegwerf-Versuch: Forward-Callback ohne die zwei dichten Matrizen | Forward-Phase im Erstlauf in Docker 32,7 → 29,2 ms (−3,5 ms), rund 3'000 Seitenfehler weniger, warm −0,8 ms. Damit ist die Schätzung für Idee 1 durch eine Messung ersetzt |

## Was in Runde 3 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 11 | Max' Commit `f1000e9` zusammengeführt (Merge `7279114`): Beispiel-Setups, `Setup.add`/`Setup.space` und der dichte JAX-Weg liegen in Modulen, die erst bei Bedarf geladen werden. `_EntryBlock.back` füllt nur die Slots der jeweiligen Gruppe, Produkte im Rückwärtslauf in-place | `setups.py`, `simulate.py`, drei neue Module, `fast.py` | `import differometor` ohne `.pyc` in Docker 15,5 → 13,0 ms (einzeln gemessen, je 15 Prozesse). Rückwärts-Callback auf dem Mac 8,9 → 8,5 ms |
| 12 | `_BlockLU`: 16 Frequenzen bilden ein Block-Diagonalsystem, also ein SuperLU-Aufruf pro Block für die Zerlegung und einer pro Lösung. Ergebnis bitgleich | `fast.py` | Zerlegung und beide Lösungen: Mac 13,4 → 12,1 ms, Docker (erster Aufruf) 18,5 → 16,8 ms |
| 13 | Die dichte Signalmatrix (9 MB) ist nicht mehr in den Residuen. Der Forward-Callback gibt dem Rückwärtslauf über `SystemPlan.base_residuals` nur die Werte mit, die er braucht | `fast.py` | Nr. 11 bis 13 und 16 zusammen: Docker Forward 42,0 → 38,1 ms, Gradient 21,3 → 19,2 ms |
| 14 | Forward- und Rückwärtslauf verarbeiten je 16 Frequenzen am Stück statt alle 101 auf einmal | `fast.py` | Docker: Forward 37,5 → 36,0 ms, Gradient 19,3 → 18,8 ms, warm 49,7 → 46,1 ms. Mac (Callbacks einzeln): Forward gleich, Rückwärts 8,5 → 9,0 ms |
| 15 | Die Komponentenfunktionen schreiben ihre Ausgänge direkt in die Matrixeinträge. Eingänge, die nicht von der Frequenz abhängen, gehen als eine Zeile hinein statt als 101 | `fast.py`, `_EntryBlock` | Callbacks einzeln, Mac: Forward 21,6 → 20,6 ms, Rückwärts 9,1 → 7,8 ms. Bitgleich |
| 16 | Suche der Nichtnull-Einträge auf der flachen Float-Sicht. Vorarbeit, die nur der Rückwärtslauf braucht, läuft nicht mehr im Forward | `fast.py` | zusammen ca. 0,5 ms pro Forward-Aufruf |

Zu Nr. 13: Die Residuen haben eine feste Grösse (16 konstante Einträge pro Unbekannte, das Benchmark-Setup hat 2). Hat eine Signalmatrix mehr, rechnet der gewöhnliche Rückwärtslauf trotzdem richtig. Nur der allgemeine Rückwärtslauf (Gewichte ausserhalb der Detektorzeilen) meldet dann einen klaren Fehler, statt falsch zu rechnen. `test_adjoint.py` prüft beides.

Zu Nr. 14: Der Gradient ändert sich in der 14. Stelle, weil die Summen über die Frequenzen anders geklammert sind. Blockgrösse 16 ist gemessen: Mit 8 ist der Gradient langsamer (Docker 19,8 statt 17,5 ms), mit 32 beide Phasen (Docker Forward 35,5 statt 34,8 ms, Gradient 18,4 statt 17,5 ms).

## Was in Runde 2 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 8 | Rückwärtslauf ohne dichte Matrizen: `SystemPlan(A, signal_base=False, noise_base=False)` baut den Plan nur aus den Index-Arrays. Die Rauschmatrix geht nicht mehr an den Rückwärts-Callback (`BACKWARD_KEYS`) | `fast.py` | Gradient: Docker 26,9 → 21,4 ms, Mac 16,2 → 13,3 ms |
| 9 | Signal-Systeme skaliert (`_balance`, `D⁻¹·S·D` mit Zweierpotenzen), symmetrisch nach `MMD_AT_PLUS_A` angeordnet, Schwellen-Pivotsuche mit `DiagPivotThresh = 0,1`. Faktoren: ca. 6'100 statt 8'330 Einträge | `fast.py`, `SystemPlan._analyze` | Forward: Docker 42,3 → 39,3 ms, Mac 32,8 → 30,6 ms |
| 10 | Dichte Matrizen im Build 64-Byte-ausgerichtet anlegen (`_aligned_zeros`), dann übernimmt `jnp.asarray` sie ohne Kopie. Einheitsdiagonale direkt setzen statt über `np.eye` | `build.py` | Build −3,4 ms, Forward +2,3 ms in Docker (netto −1,1 ms). Mac netto −0,6 ms |

Zu Nr. 9: Die Skalierung macht den Löser auch genauer. Bei zufällig veränderten Designvariablen (`test_perturbed.py`, 6 Versuche) weicht die Sensitivität um höchstens 2,1e-8 (im Logarithmus) vom Original-JAX ab. Der Stand davor lag in Versuch 4 um 8,2e-6 daneben.

## Was in Runde 1 geändert wurde

| Nr. | Änderung | Datei | Gewinn (Mac, Messung vom 20.11.) |
| --- | --- | --- | --- |
| 1 | SuperLU-Modul direkt laden statt `import scipy.sparse.linalg` | `fast.py`, `_load_superlu` | 633 → 551 ms |
| 2 | `power_detector` und `signal_detector` als Host-Callback mit eigener Rückwärtsregel, abschaltbar über `components.HOST_DETECTORS` | `components.py` | Kompilation 100 → 68 ms |
| 3 | Ableitungen nur für Funktionseingänge, die Designvariablen sind | `fast.py`, `_EntryBlock` | Rückwärtslauf 24 → 15 ms |
| 4 | SuperLU mit `Relax=1`, `PanelSize=1` (keine aufgefüllten Supernodes) | `fast.py`, `_splu` | Zerlegung 20,9 → 12,5 ms |
| 5 | Zwei Löser-Aufrufe pro Frequenz statt drei, Rauschen für alle Frequenzen auf einmal | `fast.py`, `forward` | Teil von 43 → 30 ms |
| 6 | Schnellerer Aufbau des Matrix-Musters (Schlüssel + `searchsorted` statt dichter Masken) | `fast.py`, `SystemPlan` | 3,6 → 1,8 ms je Aufruf |
| 7 | Plotly erst bei Bedarf laden (Max' Fassung) | `__init__.py` | Import `differometor` 107 → 15 ms (zusammen mit Nr. 1) |

## Wo die restliche Zeit steckt (Docker, Stand Runde 3, Build Runde 7, Forward und Gradient Runde 6)

| Phase | ms | Unser Anteil |
| --- | ---: | --- |
| Imports | 309 | 13 ms für `import differometor`, fast alles Übersetzen der Quelltexte, weil das Image keine `__pycache__`-Ordner hat: `fast.py` 4,8, `build.py` 4,4, `components.py` 2,0, `setups.py` 0,9, `simulate.py` 0,6. Mit Cache: 1,4 ms |
| Initialisierung, Laden | 3 | nichts |
| Build | 23 | 5,9 ms (`run_build_step`, vor Runde 7 waren es 8,5 ms). Der Rest gehört zu `benchmark.py`: Der erste `jnp.asarray`-Aufruf startet das JAX-Backend (ca. 9 ms), und jede neue Form in `jnp.asarray(liste, dtype=...)` kompiliert ein kleines XLA-Programm (je rund 3,5 ms, dreimal) |
| Forward: Lowering, Kompilation | 7,5 | alles, aber fast nur JAX-Grundaufwand (31 Argumente) |
| Forward | 29,5 | alles. Callback im Erstlauf 26,8 bis 27,4 ms (Aufteilung aus Runde 3: Zerlegung 11, Lösen 4,6, Schleife in `forward` 3,6, Einträge 2,1, Plan 1,4, Matrixwerte 1,2). Ausserhalb des Callbacks ca. 2 ms: JAX kopiert 6 MB Ergebnisse heraus. Vor Runde 6 waren es 5 bis 6 ms, weil 19 MB Argumente hineinkopiert wurden |
| Sensitivity: Lowering | 11 | ca. 2 ms (zwei Callbacks). Der Rest ist das Tracing des Readouts |
| Sensitivity: Kompilation | 66 | nichts. Das ist der Readout aus `benchmark.py` in XLA |
| Gradient: Lowering | 5 | wenig |
| Gradient: Kompilation | 36 | nichts |
| Gradient | 12,3 | fast alles. Callback im Erstlauf 10,3 bis 10,6 ms: Funktionen mit Ableitung, Schleife in `backward`, Rückrechnung auf die Parameter. Ausserhalb ca. 2 ms Übergabe (12 Argumente, darunter rund 7 MB Residuen und Kotangenten) |

Rund 430 von 505 ms sind fix, solange `benchmark.py` unverändert bleibt und keine JAX-Schalter gesetzt werden dürfen.

## Erkenntnisse aus Runde 7 (alle gemessen)

| Thema | Befund |
| --- | --- |
| Bytecode verfälschte Mac-Vergleiche | In dieser Claude-Sitzung ist `PYTHONDONTWRITEBYTECODE=1` gesetzt. Python legt darum keine `.pyc`-Dateien an und übersetzt `differometor` bei jedem Start neu (12,1 ms). Syntaxprüfungen mit `python -m py_compile` schreiben aber trotzdem `.pyc`-Dateien. So hatte der Arbeitsbaum für einzelne Dateien einen Cache, der Vergleichsstand nie. Folge: Die Import-Phase des neuen Stands war zu schnell, in Runde 7 um 9 ms (vier Dateien), in Runde 6 um rund 3 ms (`fast.py`). Betroffen sind nur Gesamtzeiten und Import-Phase von Mac-Reihen. Phasen wie Build, Forward und Gradient und alle Docker-Reihen sind nicht betroffen. Ob Mac-Gesamtzeiten der Runden 1 bis 4 betroffen waren, lässt sich nicht mehr prüfen |
| Abhilfe | Vor Mac-Messungen `differometor/__pycache__` in allen verglichenen Ständen löschen. `runs/noel/ab.py` bricht jetzt ab, wenn die Caches verschieden sind. Syntax ohne `.pyc` prüfen: `python -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" datei.py` |
| Import mit und ohne Cache | `import differometor` auf dem Mac: 12,1 ms ohne Cache, 1,9 ms mit vollständigem Cache. In einer gewöhnlichen Shell legt Python den Cache beim ersten Lauf an. Ab dem zweiten Prozess ist der Erstlauf dort also rund 10 ms schneller als in unseren Messungen. Das Docker-Image setzt `PYTHONDONTWRITEBYTECODE=1` und misst immer ohne Cache (Frage B4) |
| Wo der Build-Schritt Zeit braucht | Vorher, Mac warm: `build` 5,0 ms, `pairs_to_arrays` 1,5 ms, `prepare_arrays` 0,1 ms. `set_instructions`: 519 Aufrufe, 2,74 ms, davon 2,27 ms in den 170 Aufrufen mit Seitenband-Erweiterung (13,4 µs pro Aufruf) und 0,36 ms in den übrigen 349 (1,0 µs). Nachher: `build` 3,2 ms, `pairs_to_arrays` 0,95 ms, `set_instructions` 1,25 ms (5,7 µs pro Aufruf mit Erweiterung) |
| Seitenfehler im Build | Docker, erster Aufruf: 2'066 → 1'750 |
| Was im Build noch bleibt | Der Rest von `build` sind die Schleife über Bauteile und Kanten (1,4 ms im Profil), rund 2'400 `np.array`-Aufrufe für die Argumente (0,7 ms) und 41 Übergaben an JAX (0,6 ms). Weitere 0,5 bis 1 ms wären nur mit einem Umbau der Aufrufstellen zu holen |
| XLA-Programme | Das Forward-Programm enthält nur unseren Callback. Sensitivity und Gradient enthalten je zwei Callbacks von uns und sonst nur den Readout aus `benchmark.py` (unter anderem `gather`, komplexes `abs`, `log`, `scatter`). An den rund 106 ms Kompilierzeit dieser zwei Programme in Docker können wir ohne JAX-Schalter nichts ändern |
| `jnp.asarray` in `benchmark.py` | Jeder Aufruf mit `dtype` und einer neuen Form kompiliert ein kleines XLA-Programm: auf dem Mac 3,3 ms für die zweite Liste, 2,9 ms für den Skalar, danach 0,03 bis 0,09 ms. Das erklärt rund 10 ms der Build-Phase in Docker, die nicht uns gehören |

## Erkenntnisse aus Runde 6 (alle gemessen)

| Thema | Befund |
| --- | --- |
| Was der Build in die grossen Matrizen schreibt | In die Rauschmatrix nichts. In die Signalmatrix nur die Trägermatrix (ohne rechte Seite) in beiden Seitenband-Blöcken und eine Einheitsdiagonale für die Signal-Freiheitsgrade. Benchmark-Setup: Trägermatrix 687 Einträge, Signalmatrix 1'539 von 564'752, Rauschmatrix 0. Die konstanten Einträge der Signalmatrix lassen sich also direkt aus der kleinen Trägermatrix ableiten |
| Build-Ausgabe alt gegen neu | Mit gleichem `PYTHONHASHSEED` sind alle 155 Arrays aus fünf Setups (Benchmark, aLIGO in drei Varianten, Voyager) als dichte Arrays identisch |
| Build ist nicht deterministisch (schon vorher) | `noise_system_matrix_indices` hat bei aLIGO und Voyager von Prozess zu Prozess eine andere Reihenfolge, weil der Build über eine Menge iteriert (`all_ports - used_ports`). Beim Benchmark-Setup tritt das nicht auf. Für Bit-Vergleiche von Builds darum `PYTHONHASHSEED=0` setzen |
| Seitenfehler im Erstlauf, Docker, je drei Prozesse, vorher → nachher | Build 2'392 → 2'054 bis 2'064. Forward-Phase 10'995 bis 12'494 → 6'941 bis 7'890, davon ausserhalb des Callbacks rund 5'490 → 1'310 bis 1'820. Gradient-Phase 2'790 bis 3'828 → 2'484 bis 3'188. Sensitivity-Kompilation unverändert (2'150 bis 2'570) |
| Seitenfehler pro warmem Aufruf, Docker | Forward-Callback 1'480 bis 5'680 → 1'030 bis 2'500. Rückwärts-Callback 0 bis 4'970 → 120 bis 2'630. Warme Phasen im Messskript: Forward 28,8 bis 29,1 → 24,1 bis 24,5 ms, Gradient 12,8 bis 13,0 → 9,5 bis 10,0 ms. Der Verlust im Warmlauf aus Runde 5 ist damit mehr als aufgeholt: Im Wechselvergleich liegt warm jetzt bei 35 ms, der Stand vor der Bündelung lag in den Reihen von Runde 5 bei 40 bis 42 ms |
| Callbacks einzeln auf dem Mac (`profil.py`) | Forward 22,7 → 20,5 ms, Rückwärts 7,6 → 7,1 ms |
| Import | Unverändert: `import differometor` ohne `__pycache__` in Docker 13,26 gegen 13,30 ms (Median aus je 9 Prozessen). Übersetzen von `fast.py` 4,02 → 3,96 ms, `build.py` 3,60 → 3,57 ms |
| Lowering | Der Forward-Callback hat jetzt 29 statt 26 Argumente (jede Matrix zwei kleine Arrays). Forward-Lowering in Docker 3,5 → 3,8 ms. Im Gewinn oben ist das enthalten |
| Ruhezustand | `caffeinate -is` verhindert den Ruhezustand nicht, wenn der Deckel zu ist. Ab etwa 09:24 wechselte der Laptop auf Akku zwischen «Maintenance Sleep» und kurzen Wachphasen (`pmset -g log`). Läufe aus solchen Minuten erkennt man an Erstlaufzeiten über 500 ms auf dem Mac und über 900 ms in Docker. Alle Messreihen dieser Runde liefen davor (bis 09:23) |

## Erkenntnisse aus Runde 5 (Messungen, Erklärungen sind als solche bezeichnet)

| Thema | Befund |
| --- | --- |
| Woher der Gewinn der Bündelung kommt | Seitenfehler im Erstlauf in Docker, je drei Prozesse, vorher → nachher: Sensitivity-Kompilation 4'390 bis 5'340 → 2'180 bis 2'370, Sensitivity 820 bis 1'220 → 70 bis 370, Gradient 5'690 bis 6'700 → 3'100 bis 3'390 (davon im Rückwärts-Callback 3'180 bis 4'160 → 2'570 bis 2'880). Die Forward-Phase bleibt gleich (11'360 bis 11'750 gegen 11'200 bis 11'990). Das sind im Mittel rund 6'000 Seitenfehler weniger, bei 0,6 µs also rund 3,5 ms. Dazu kommt rund 1 ms Grundaufwand der 20 eingesparten Argumente beim Lowering und Kompilieren, der auch auf dem Mac sichtbar ist |
| Warum warm langsamer | Seitenfehler pro warmem Aufruf in Docker, vorher → nachher: Forward-Callback 380 bis 1'940 → 2'600 bis 5'900. Rückwärts-Callback 0 → rund 3'000 in zwei von drei Prozessen, im dritten fast keine |
| Einordnung | Der grössere Teil des Gewinns im Erstlauf und der ganze Verlust im Warmlauf sind Folgen davon, wann glibc Heap-Speicher ans System zurückgibt (siehe Runde 4). Erklärung, nicht bewiesen: Mit weniger kleinen Arrays liegt der Speicher anders, im Erstlauf bleibt mehr davon für die folgenden Phasen erhalten, zwischen den warmen Aufrufen weniger. Das hängt von der Reihenfolge aller Anforderungen ab, auch von jenen des Compilers, und kann auf x86 anders ausfallen. Sicher übertragbar ist nur rund 1 ms. Darum beide Stände auf x86 messen (offene Idee 5) |
| Kopien der dichten Matrizen | Von den rund 11'500 Seitenfehlern der Forward-Phase fallen 3'500 bis 5'500 ausserhalb des Callbacks an. Die zwei 9-MB-Kopien, die JAX für den Callback anlegt, entsprechen 4'400 Seiten (18 MB in 4-kB-Seiten), dazu kommen die Kopien der Ergebnisse. Das passt zum Hebel von Idee 1 |
| Quelltext-Länge | Ohne `__pycache__` kostet das Übersetzen von `fast.py` in Docker 3,72 ms, mit den 73 neuen Zeilen 3,98 ms (40 Wiederholungen im selben Prozess). Jede Zeile kostet dort also rund 3,5 µs im Import |
| Akku | Auf Akku und bei benutztem Rechner liegt der Erstlauf 4 bis 6 % höher (Mac 461 statt 436 ms, Docker 535 statt 515 ms für denselben Stand `ba8f1f2`). Reports nur am Netz erzeugen |
| `vmap` | `jax.vmap(jax.value_and_grad(...))` über drei Parametersätze liefert dieselben Werte wie Einzelaufrufe (Gradient auf 1e-9), vor und nach der Änderung identisch. Ein Test mit Reflektivitäten über 1 scheitert in beiden Ständen mit «Factor is exactly singular». Das ist kein gültiger Parametersatz |

## Erkenntnisse aus Runde 4 (Messungen, Erklärungen sind als solche bezeichnet)

| Thema | Befund |
| --- | --- |
| Warum warme Aufrufe in Docker mal schnell und mal langsam sind | Gemessen: Mit einer festen Stutz-Schwelle von 19 MB fallen pro warmem Forward-Aufruf 2'900 bis 4'700 Seitenfehler an, mit einer sehr grossen Schwelle 700 bis 1'600. Erklärung: glibc passt zwei Schwellen laufend an. Die mmap-Schwelle wird so gross wie der grösste freigegebene Block (hier die 9-MB-Kopien der dichten Matrizen), die Stutz-Schwelle doppelt so gross (18 MB). Ab dem zweiten Aufruf liegen die zwei 9-MB-Kopien im Heap des Arbeits-Threads. Werden sie und unsere Zwischenarrays freigegeben, ist oben mehr als 18 MB frei, und glibc gibt den Speicher ans System zurück. Der nächste Aufruf fasst ihn wieder frisch an |
| Warum das bisher meist nicht passierte | Vermutung, passend zu den Messungen: Kleine, länger lebende Arrays (die durchgereichten Index-Arrays, der zurückgegebene Gradient) liegen zufällig über den freigegebenen Bereichen und verhindern das Stutzen. Mit weniger kleinen Arrays (Bündelungs-Versuch) trat es in 10 von 12 Läufen auf, vorher in 2 von 12. Das ist ein Zufall des Speicherlayouts, kein Verdienst des Codes |
| Feste Schwellen als Gegenprobe | Mit `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=9500000:glibc.malloc.trim_threshold=2000000000` (nur zur Messung gesetzt, zusammen mit dem Bündelungs-Versuch): Forward-Phase im Erstlauf 33,4 → 31,7 ms, Gradient-Phase 13,4 → 11,6 ms (Mittel aus zwei bis drei Läufen im selben Messskript). Der Rückwärts-Callback hat dann schon im ersten Aufruf keine Seitenfehler mehr. Rund 3,5 ms im Erstlauf, aber eine Einstellung der Umgebung, also Regelfrage B6 |
| Kosten pro Callback-Argument | Rund 11 µs pro Argument und Aufruf, dazu je rund 15 µs beim Lowering und 20 µs bei der Kompilation (Versuch mit 0, 10 und 40 Argumenten). Ein durchgereichtes Residuum kostet rund 10 µs Lowering und 20 µs Kompilation |
| Build | `run_build_step` braucht warm 6,5 bis 7,5 ms: `build` 4,8 ms (davon `set_instructions` geschätzt 2 ms in 519 Aufrufen), `pairs_to_arrays` 1,5 ms, `prepare_arrays` 0,1 ms. Die Übergabe der 38 Arrays an JAX kostet zusammen nur 0,6 ms |

## Erkenntnisse aus Runde 3 (alle gemessen)

| Thema | Befund |
| --- | --- |
| JAX-Schalter | `jax.config.update("jax_disable_most_optimizations", True)` spart in Docker rund 42 ms (Sensitivity-Kompilation 64 → 36,5 ms, Gradient-Kompilation 35,3 → 20,6 ms), auf dem Mac 13 ms (24 → 17 und 17,8 → 12,1 ms). Loss und Gradient stimmen auf 11 Stellen. Nicht eingebaut: Regelfrage B6 in `FRAGEN_AN_FEYER.md`. Joel hat dieselbe Variante vorbereitet (`sx2`, `sx2-env`) |
| Threads | Der Forward-Callback läuft in einem XLA-Arbeits-Thread, der Rückwärts-Callback und die Detektoren im Haupt-Thread. Im Arbeits-Thread fasste jeder Forward-Aufruf wieder frische Speicherseiten an, auch im warmen Zustand (rund 4'200 Seitenfehler pro Aufruf, jetzt 250 bis 1'800) |
| Seitenfehler | In Docker kostet ein Seitenfehler rund 0,6 µs. Erstlauf vorher: 16'000 bis 16'800 in der Forward-Phase und 8'100 bis 9'300 in der Gradient-Phase. Jetzt: 11'000 bis 11'650 und 6'400 bis 6'500. Was bleibt: 19 MB Argument-Kopien von JAX, die Faktoren der Zerlegung (2'900), die Ergebnis-Arrays |
| Allocator-Umgebungsvariablen | `MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_` und `MALLOC_TOP_PAD_` ändern im echten Ablauf nichts an den Seitenfehlern (im Minimalbeispiel im Haupt-Thread schon). Auch darum bringt nur weniger Speicher etwas |
| Docker-VM | Seitengrösse 4 kB, Transparent Huge Pages auf `always`, glibc 2.41 |
| SuperLU | Rund 110 ns pro Spalte, dazu rund 10 µs Grundaufwand pro Aufruf (aus den Messungen abgeleitet). Die Zerlegung ist mit 11 ms weiterhin der grösste Einzelposten |
| Speicher zwischen den Blöcken freigeben (`del`) | Kein Gewinn (26,7 bis 27,2 gegen 26,9 bis 27,9 ms) |
| Nachgeladene Module | Auf dem Mac (mit `.pyc`) wandern ca. 0,5 ms vom Import in den Build, weil JAX einige eigene Module jetzt erst dort lädt. Netto dort neutral |
| Streuung | Imports ±3 ms, Kompilation ±3 ms pro Messreihe. Gewinne unter 3 ms sind in der Gesamtzeit nicht sichtbar, nur in den Phasen Forward und Gradient |

## Erkenntnisse aus Runde 2 (alle gemessen)

| Thema | Befund |
| --- | --- |
| Callback-Argumente | JAX kopiert jedes Argument eines `pure_callback` in ein neues NumPy-Array. Zwei 9-MB-Matrizen kosten pro Aufruf ca. 1,4 ms in Docker und beim ersten Aufruf auch auf dem Mac. Darum zählt jede grosse Eingabe |
| `jnp.asarray` auf der CPU | Kopiert ein gewöhnliches NumPy-Array (16-Byte-Ausrichtung). Ein 64-Byte-ausgerichtetes übernimmt es ohne Kopie. Ein Aufruf kostet ca. 15 µs |
| Durchgereichte Residuen | Ein Jit-Programm, das eine Eingabe unverändert zurückgibt, kopiert sie |
| Diagonale als Pivot | Mit symmetrischer Anordnung ist reine Diagonal-Pivotierung hier stabil (Abweichung 7,8e-8). Die Notiz aus Runde 1 («Fehler 1e75») stammte aus einem Versuch, bei dem nur die Spalten umsortiert waren |
| Unsymmetrische Anordnung | Eine Markowitz-Reihenfolge in Python bringt nicht weniger Füllung (6'230 statt 6'060 Einträge) und kostet 3,8 ms |

## Messmethode

```bash
source .venv/bin/activate
# Wechselvergleich lokal: Vergleichsstand als Kopie (runs/noel/vergleich) oder Git-Worktree.
# Vorher differometor/__pycache__ in allen Ständen löschen (siehe Runde 7), sonst bricht ab.py ab.
python runs/noel/ab.py 12 vorher=runs/noel/vergleich neu=.
# Wechselvergleich in Docker (Linux, --cpus 1), beliebig viele Images
docker build -q -t differometor-bench:neu . && python runs/noel/ab_docker.py 10 vorher=differometor-bench:runde7 neu=differometor-bench:neu
# Profil der Host-Callbacks, lokal oder im Image
source runs/env.sh && python runs/noel/profil.py . forward
docker run --rm --cpuset-cpus 0 --entrypoint python differometor-bench:neu /app/runs/noel/profil.py /app backward
```

Einzelne Läufe streuen um bis zu 40 ms. Darum immer abwechselnd messen und Median und Minimum ansehen. Für kleine Änderungen die Phasen einzeln vergleichen: Die Berichte der Wechselvergleiche liegen in `runs/noel/out/` (`ab-<Name>-<Nr>.json`) und `runs/noel/out/docker/`. Der erste Lauf eines frisch gebauten Images ist ein Ausreisser (562 statt 520 ms) und zählt nicht.

Vorhandene Docker-Images: `differometor-bench:base` (Philip + Max), `:vorher` (`c261b63`), `:kandidat3` (`49d386b`), `:runde3` (`1442ef3`), `:runde4` (`ba8f1f2`), `:runde5` (`45a2328`), `:runde6` (`7d12ba7`), `:runde7` (`e311d5c`), `:joel` (`76e799f`).

## Korrektheit prüfen (nach jeder Änderung)

```bash
source .venv/bin/activate && source runs/env.sh
python runs/check_fast.py runs/noel/orig.npz --fallback
python runs/check_setups.py
python runs/noel/test_detectors.py .
python runs/noel/test_adjoint.py .      # allgemeiner Rückwärtslauf, Blockgrösse, Residuen-Grenze
python runs/noel/test_perturbed.py .    # veränderte Designvariablen gegen das Original-JAX
python runs/noel/test_pack.py .         # gepackte Residuen: exakte Wiederherstellung
python runs/noel/test_sparse.py .       # Sparse-Matrizen: Form, dichte Eingaben, von Hand veränderte Matrizen
python runs/noel/test_build.py .        # Seitenband-Erweiterung im Build gegen eine Referenz
```

Nach Änderungen am Build zusätzlich die Ausgaben zweier Stände vergleichen (in jedem Stand einmal `--dump`, dann `--compare`):

```bash
PYTHONHASHSEED=0 python runs/noel/test_build.py . --dump neu.npz
python runs/noel/test_build.py . --compare alt.npz neu.npz   # 946 arrays compared, 0 different
```

`runs/noel/orig.npz` ist nicht im Repo (`*.npz` steht in `.gitignore`). Neu erzeugen im unveränderten Kit (Branch `main`), braucht rund 14 GiB RAM:

```bash
python benchmark.py --repeats 3 --output orig.json --save-arrays orig.npz
```

Sollwerte (Stand `e311d5c`, auf Noels M4, unverändert seit Runde 2):

| Grösse | Abweichung gegen das Original-JAX (max, relativ zum Maximum) |
| --- | ---: |
| carrier | 9,7e-12 |
| signal | 1,161e-07 |
| noise | 3,787e-08 |
| sensitivity | 5,735e-09 |
| powers | 1,394e-11 |
| gradient | 7,985e-08 |
| Fehler gegen Feyers Referenz | 5,3743e-06 (Grenze 1e-5) |

| Prüfung | Sollwert |
| --- | --- |
| `check_setups.py` | aLIGO (schnell) höchstens 1,1e-8, aLIGO verknüpft und Voyager 0 |
| `test_detectors.py` | WORST 2,9e-14 |
| `test_adjoint.py` | WORST 4,0e-08, Blockgrössen 1 und 2 gegen 16: Abweichung 0. Die Prüfung der Residuen-Grenze läuft nur noch bei Ständen vor Runde 6 |
| `test_perturbed.py` | WORST 2,1e-08 |
| `test_pack.py` | zweimal «28 arrays restored exactly», dann «OK» |
| `test_build.py` | «20000 random cases identical to the reference», dann «OK» |
| `test_sparse.py` | dichte Eingaben: Abweichung 0. Veränderte Matrizen, Fast-Path gegen JAX: Carrier 9,7e-12, Signal 1,3e-08, Noise 1,8e-08, Gradient 5,3e-08. Am Ende «OK» |

## Verworfen

| Idee | Grund |
| --- | --- |
| Eigene, über die Frequenzen gebündelte LU in NumPy | Etwa 15 ms reiner Speicherverkehr, also nicht schneller als SuperLU |
| Dichte Teilblöcke mit `np.linalg.solve` | Um Grössenordnungen zu teuer |
| Reine Diagonal-Pivotierung ohne Absicherung | Geschätzt 1 ms schneller, aber kein Schutz, wenn ein Pivot klein wird |
| Einfache Unbekannte vorab in NumPy herausrechnen (751 → 324 Unbekannte) | In Runde 3 nachgerechnet (Schätzung): Zerlegung und Lösen würden rund 5,6 ms sparen, die Vorab-Rechnung und das Zurückrechnen für beide Löserichtungen kosten rund 5 ms. Dafür rund 150 Zeilen neue Sparse-Algebra |
| Zweiter Rückwärts-Callback hinter `lax.cond` | Durch Nr. 13 erledigt: Die Signalmatrix wird gar nicht mehr übergeben |
| Zwischenergebnisse des Forward-Laufs im Prozess aufheben (Token) | Wäre ein Cache. Abmachung: keine Caches |
| XLA-Flags, Kompilier-Cache, `jax.config`-Schalter, `mallopt`, Allocator mit einer grossen Anforderung «vorwärmen» | Regelfrage, siehe `docs/notes/FRAGEN_AN_FEYER.md` (B6) |
| Rückgabe von Stellvertreter-Objekten statt Arrays, um den Readout zu umgehen | Wäre Schummeln |
| Prüfung der dichten Matrizen in XLA statt im Callback | Jede echte XLA-Operation kostet 5 bis 15 ms Kompilierzeit, mehr als die Kopie. Seit Runde 6 gegenstandslos: Es gibt keine dichten Matrizen mehr |

## Offene Ideen

| Nr. | Idee | Schätzung | Haken |
| --- | --- | --- | --- |
| 1 | Index-Arrays als feste Struktur statt als Daten: Die rund 20 ganzzahligen Arrays im `arrays`-Dict ändern sich für ein Setup nie. Als Pytree-Knoten ohne Blätter wären sie keine Programm-Argumente mehr, und die Callbacks bekämen sie ohne Kopie | geschätzt 1,5 bis 2 ms in Docker (Lowering und Kompilation pro Argument rund 35 µs, Kopie pro Argument und Aufruf rund 11 µs, kein Packen mehr), nicht gemessen | Grösserer Eingriff in die Schnittstelle als Nr. 19: Rund 20 Werte im Dict wären keine Arrays mehr. Der JAX-Weg müsste sie als Konstanten einbetten. Vorher im Wegwerf-Versuch messen |
| 2 | Residuum `Y` weglassen: Mit der Detektor-Basis ist `Y = conj(Z · conj(w))`, der Rückwärtslauf kann es aus `Z` berechnen | geschätzt 0,3 ms in Docker (2,4 MB weniger Kopien, dafür eine Rechnung über 1,2 MB), auf dem Mac nichts | Unter der Messgrenze. Nur zusammen mit anderem sinnvoll |
| 3 | `build.py` weiter straffen: Aufrufstellen von `set_instructions` (rund 2'400 `np.array`-Aufrufe), Schleife über Bauteile | geschätzt 0,5 bis 1 ms. Der grössere Teil (2,5 bis 3 ms) ist seit Runde 7 erledigt | Jede Aufrufstelle einzeln anfassen. Prüfbar mit `test_build.py --dump` und `--compare` |
| 4 | Falls Feyer Einstellungen im Code erlaubt (B6): eine Zeile `jax.config.update("jax_disable_most_optimizations", True)` in `differometor/__init__.py`, dazu feste glibc-Schwellen über `mallopt` | gemessen: 42 ms in Docker und 13 ms auf dem Mac für den JAX-Schalter. Die rund 3,5 ms für die glibc-Schwellen stammen aus Runde 4 und sind nach Runde 6 neu zu messen | Nur nach Zusage von Feyer |
| 5 | `noel` auf Joels x86-Maschine messen (AMD EPYC, `taskset -c 14 python benchmark.py --cpus 1`), im Wechsel mit `6b67cfd` (Stand vor Runde 5). Die Bündelung aus Runde 5 lässt sich nach Runde 6 nicht mehr einzeln zurücknehmen. Sie ist jetzt Teil des Entwurfs, und ihr Nachteil im Warmlauf ist weg | – | Joel fragen |
| 6 | Falls ohne `__pycache__` gewertet wird (Frage B4): Code, den der Benchmark nie ausführt, in Module auslagern, die erst bei Bedarf geladen werden (Teile von `build.py` und `components.py`) | unbekannt, höchstens einige der 13 ms | Mit `__pycache__` bringt es nichts. Erst nach Feyers Antwort |

## Stand der anderen Branches

Vor jeder Runde: `git fetch --all --prune`, dann alle fremden Branches auf neue Commits prüfen.

| Branch | Zuletzt gesehen | Stand in `noel` |
| --- | --- | --- |
| `origin/philip` | `ab82e42` (20.11. 19:58), unverändert am 21.11. 10:35 | vollständig enthalten |
| `origin/max/lu-once` | `79ae580` (20.11. 20:08), unverändert | vollständig enthalten |
| `origin/max/nacht` | `f1000e9` (21.11. 03:25), neu in Runde 3 | zusammengeführt in `7279114`. Übernommen: nachgeladene Module, `_EntryBlock.back`, In-place-Produkte, `signal_values` ohne Zwischenkopie. Ersetzt durch den Stand von `noel`: Spaltenordnung `MMD_ATA` (7'916 Faktoreinträge gegen 6'100) |
| `origin/Joel` | `76e799f` (21.11. 05:19), unverändert am 21.11. 10:35, baut auf Philips `ab82e42` auf | nichts zu übernehmen. Joels drei Code-Änderungen (Host-Detektoren, Plotly bei Bedarf, SuperLU ohne SciPy-Import) hat `noel` seit Runde 1 in eigener Fassung |
| `origin/sx`, `origin/sx2`, `origin/sx2-env` | `2c9e8bf`, `036194a`, `bf6bf60` (21.11. 03:33 bis 04:00), unverändert am 21.11. 10:35 | nicht übernommen. Das sind Joels vorbereitete Varianten mit XLA-Flags bzw. `jax_disable_most_optimizations`, laut seinem README erst nach Zusage von Feyer zu aktivieren. Gleiche Regelfrage wie unser B6 |
| `origin/main` | `5eece1d` | unverändertes Kit |

Joels Stand gegen `noel` auf Noels Rechner (je 10 Läufe im Wechsel, Median):

| Umgebung | `origin/Joel` (`76e799f`) | `noel` (`1442ef3`) |
| --- | ---: | ---: |
| Mac | 484,9 ms (Forward 45,8, Gradient 29,1, warm 67,2) | 442,9 ms (Forward 26,5, Gradient 11,7, warm 32,7) |
| Docker `--cpus 1` | 565,3 ms (Forward 49,2, Gradient 42,9, warm 84,4) | 516,8 ms (Forward 35,2, Gradient 17,8, warm 39,4) |

Joels README nennt für seinen Stand auf einem AMD EPYC (x86-Linux, 1 CPU) 762 ms im Erstlauf, davon Imports 280, Kompilation 230, Forward und Gradient 150, Build 60 ms. Forward und Gradient haben dort also einen viel grösseren Anteil als auf dem Mac. Der Vorsprung von `noel` dürfte dort entsprechend grösser ausfallen. Gemessen ist das nicht.

## Abmachungen

- Zu Beginn jeder Runde die Branches von Max, Philip und Joel holen und sichten. Was messbar hilft, in `noel` übernehmen (Merge oder Cherry-Pick) und wie jede Änderung messen und prüfen. Fremde Branches nie verändern.
- Noel hat am 21.11. freigegeben: Jede Runde committet ihre behaltenen Änderungen und pusht sie auf `origin/noel`.
- Nur der Branch `noel`. Kein Merge von `noel` in andere Branches, kein Pull Request, kein Force-Push, keine anderen Branches anfassen.
- Noel hat am 21.11. um 08:30 entschieden: Die Bündelung der kleinen Arrays (Nr. 18) wird eingebaut, obwohl die warmen Wiederholungen in Docker damit langsamer sind. Für die Wertung zählt der Erstlauf.
- Seit Runde 6 (21.11., `7d12ba7`) liefert `run_build_step` die drei Systemmatrizen als `SparseMatrix`. Wer Branches zusammenführt, die den Build oder den JAX-Weg ändern, muss das beachten. Die Runde hat das im Rahmen der Freigabe für behaltene Änderungen eingebaut. Noel hat am 21.11. um 10:10 bestätigt: bleibt drin.
- Verworfene Versuche werden nicht committet.
- Eine Änderung bleibt nur, wenn sie im Wechselvergleich schneller ist und alle Prüfungen unverändert gut ausfallen.
- `benchmark.py`, Fixtures und `requirements.txt` bleiben unangetastet. Keine Caches, nichts vorberechnen.
