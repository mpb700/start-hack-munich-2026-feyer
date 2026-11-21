# Optimierung auf Branch `noel`: Stand, Messmethode, offene Ideen

Stand: Sa 21.11.2026, 23:45, nach Runde 19. **Zwei neue Patches für `philip-fastest` (Basis `9a31fa1`, noch nicht übernommen): `patches/philip-fastest-rows.patch` (Lösungen im Layout des C-Kerns) und danach `patches/philip-fastest-import.patch` (Import von `differometor`).** Die Patches aus Runde 15 und 16 sind seit 21:49 in `philip-fastest` und `main` (`9a31fa1`). **Die Schleife arbeitet seit 21:02 für Philips Stand (`philip-fastest`), nicht mehr an `noel`.** Diese Datei ist das Gedächtnis für die nächste Optimierungsrunde.

**Zuerst lesen:** «Arbeit für Philips Stand» gleich unten, dann «Runde 14», «Stand 20:12» und «Runde 12», dann «Runde 9: Setup-Varianten» und «Stand der anderen Branches». (Die Zeitangaben «15:40» und «15:50» in den Commits `0ada44e` und `8a4d2fe` waren falsch geschätzt. Runde 9 war um 15:13 fertig. Hier ist das korrigiert.) Philips Stand ist auf dem echten Xeon deutlich schneller als `noel`, fällt aber bei einem Balanced-Homodyne-Setup auf JAX zurück.

## Messwerte

Vier Stände im Wechsel gemessen (je frischer Prozess), Erstlauf gesamt in ms, Median / Minimum. Alle Läufe `PASS`. Die absoluten Werte hängen vom Zustand der Maschine ab. Vergleichbar sind nur Werte aus derselben Messreihe.

| Umgebung | Philip + Max (`11b696f`) | Nach Runde 1 (`c261b63`) | Nach Runde 2 (`49d386b`) | Nach Runde 3 (`1442ef3`) |
| --- | ---: | ---: | ---: | ---: |
| Apple M4, macOS, Threads auf 1 begrenzt, 12 Läufe | 606,5 / 602,7 | 460,3 / 457,9 | 454,4 / 450,1 | **445,8 / 441,3** |
| Docker auf demselben M4, Linux aarch64, `--cpus 1`, 10 Läufe | 716,5 / 707,2 | 542,8 / 537,4 | 532,3 / 528,2 | **519,7 / 517,3** |

Das Original-Kit braucht auf diesem Mac 12'827 ms (gemessen in Runde 1). Eine Zeitmessung von `noel` auf echter x86-Hardware fehlt weiterhin. Joel hat eine solche Maschine (siehe «Stand der anderen Branches»). Die Korrektheit auf x86-64 ist seit dem 21.11. emuliert geprüft (siehe «x86-64, emuliert»).

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

Runde 8 gegen Runde 7 (`57e20fe` → `b6b1119`), je 16 Läufe im Wechsel, Median. Der Rechner war am Nachmittag rund 3 % langsamer als am Vormittag (Imports 317 bis 324 statt 304 bis 312 ms), darum sind die absoluten Werte höher:

| Umgebung | Erstlauf gesamt | Erstlauf ohne Imports | Forward: Lowering + Kompilation | Forward | Gradient (Lauf) | Warm |
| --- | --- | --- | --- | --- | --- | --- |
| Mac | 454,6 → **453,0 ms** | 131,5 → 130,5 ms | 7,7 → 6,8 ms | 24,4 → 24,0 ms | 9,8 → 9,7 ms | 30,8 → 30,4 ms |
| Docker `--cpus 1` | 520,6 → **516,2 ms** | 204,7 → 201,4 ms | 8,2 → 6,5 ms | 30,3 → 30,2 ms | 13,1 → 12,5 ms | 37,1 → 37,4 ms |

Runde 10 gegen Runde 9 (`8a4d2fe` → Runde 10), je 12 Läufe im Wechsel, Median / Minimum, auf Akku:

| Umgebung | Erstlauf gesamt | Erstlauf ohne Imports | Forward | Gradient (inkl. Sensitivity) | Warm |
| --- | --- | --- | --- | --- | --- |
| Mac | 447,4 / 442,2 → **444,2 / 439,4 ms** | 126,8 → 124,2 ms | 23,7 → 21,5 ms | 10,3 → 10,2 ms | 30,0 → 27,6 ms |
| Docker `--cpus 1` | 518,6 / 513,2 → **512,7 / 508,6 ms** | 202,6 → 199,1 ms | 29,7 → 26,6 ms | 13,6 → 13,2 ms | 36,7 → 33,5 ms |
| Docker, Fixture mit Balanced Homodyne (6 Läufe) | 536,3 → **525,0 ms** | 219,0 → 207,9 ms | 34,0 → 27,1 ms | 14,5 → 12,9 ms | 41,1 → 33,7 ms |
| Docker, Zufalls-UIFO Seed 0 mit Homodyne (6 Läufe) | 530,3 → **520,8 ms** | 213,9 → 205,7 ms | 33,9 → 27,6 ms | 13,5 → 12,4 ms | 40,1 → 33,4 ms |

Runde 11 gegen Runde 10 (`10ffc61` → `2e1cef1`), je 14 Läufe im Wechsel, Median / Minimum, auf Akku:

| Umgebung | Erstlauf gesamt | Gradient (inkl. Sensitivity) | Warm |
| --- | --- | --- | --- |
| Mac | 443,7 / 435,3 → **443,4 / 434,7 ms** | 10,2 / 9,9 → 9,7 / 9,3 ms | 27,7 → 27,2 ms |
| Docker `--cpus 1` | 513,5 / 508,8 → **512,9 / 507,4 ms** | 13,3 / 12,3 → 12,8 / 12,1 ms | 33,5 → 33,2 ms |

Der Gewinn von rund 0,5 ms liegt in der Gesamtzeit unter der Streuung. Sichtbar ist er in der Gradient-Phase und direkt am Rückwärts-Callback (Mac, je 40 Aufrufe, dreimal im Wechsel: 7,28 bis 7,36 → 7,00 bis 7,03 ms).

Die Reports `mac-neu.json`, `mac-neu.png` und `docker-neu.json` zeigen weiterhin den Stand `e311d5c` (für `b6b1119` nicht neu erzeugt: fünf Läufe lagen am Nachmittag bei 449,0 bis 450,4 ms auf dem Mac und 516,8 bis 529,7 ms in Docker, alle `PASS`, aber wegen des langsameren Rechners nicht mit den Vormittagswerten vergleichbar). Erzeugt wurden sie am 21.11. um 10:54 auf Akku bei ruhigem Rechner: Mac 431,6 ms im Erstlauf und 29,2 ms warm (der mittlere von fünf Läufen, 427,1 bis 434,3 ms), Docker 499,1 ms und 35,7 ms (der mittlere von fünf, 496,9 bis 510,1 ms). Alle `PASS`. Der Mac-Report lief ohne Bytecode-Cache für `differometor` (siehe Runde 7). Mit Cache wären es rund 10 ms weniger. Die `*-vorher`-Reports zeigen Philip + Max.

## Arbeit für Philips Stand (seit 21.11., 21:02)

Noel um 21:02 auf die Frage, ob die Schleife Patches für Philips Stand bauen soll statt `noel` zu optimieren: «Ja, für Philips Stand». Der Auftrag der Schleife ist entsprechend neu (`01bbf197`, alle 30 Minuten um :13 und :43).

**So läuft eine Runde:** `git fetch`, dann Philips neue Commits und seine `runs/philip/NOTIZEN.md` lesen (was ist offen, was baut er gerade). Arbeitskopie des neuesten `origin/philip-fastest` als `git worktree` im Scratchpad (zweimal: unverändert und geändert), Wechselvergleich mit `runs/noel/ab.py` und in Docker mit `runs/noel/ab_docker.py` (Images aus beiden Kopien), Prüfungen: `benchmark.py` `PASS` und Arrays bitgleich, alle `runs/philip/test_*.py` und `runs/flags/test_tuning.py` mit `PYTHONPATH=.`, `runs/check_setups.py`, `runs/noel/test_variants.py <kopie>` (dort sind die zwei Varianten mit Amplitudenmodulation erwartungsgemäss über JAX), `runs/noel/variant_run.py` auf Homodyne-Varianten. Den mitgelieferten Bytecode der geänderten Module neu erzeugen. Dann `git format-patch -1 --binary` nach `runs/noel/patches/`. `main`, `philip-fastest` und alle anderen Branches bleiben unberührt.

**Patches** (Basis `8e7c4f1`, das ist seit 21:18 sowohl `philip-fastest` als auch `main`). Beide lassen sich einzeln oder zusammen anwenden, sie berühren verschiedene Dateien. Geprüft: beide nacheinander auf `8e7c4f1`, danach `PASS` auf Fixture und Homodyne-Variante.

| Patch | Inhalt | Gemessen (M4, Median) | Stand |
| --- | --- | --- | --- |
| [`patches/philip-fastest-build.patch`](patches/philip-fastest-build.patch) | Build-Schritt (`build.py`): gemeinsam genutzte konstante Index-Arrays, Funktionseingaben ohne Auffüllen, kein `np.tile` bei einem Signal pro Raum, kurzer Weg in der Seitenband-Erweiterung | `run_build_step` 3,97 → 3,38 ms (erster Aufruf 4,66 → 4,12 ms). Build-Phase von `benchmark.py`, 14 Läufe im Wechsel: Mac 10,9 → 10,5 ms, Docker `--cpus 1` 9,9 → 9,3 ms. **Xeon: 18,4 → 17,2 ms** | übernommen als `13bd58d` |
| [`patches/philip-fastest-forward.patch`](patches/philip-fastest-forward.patch) | Vorwärtslauf (`fast.py`): Dual-Zahlen ohne Null-Terme, Rausch-Summen direkt in die rechte Seite | Einträge mit Ableitungen 2,33 → 1,96 ms, Vorwärts-Callback 17,3 → 16,8 ms. Forward-Phase von `benchmark.py`: Mac 22,0 → 21,1 ms (14 Läufe), Docker `--cpus 1` 27,5 → 26,3 ms (16 Läufe), dort Gradient-Phase 5,4 → 5,7 ms. **Xeon: Forward 34,5 → 33,3 ms, Gradient 11,1 → 11,5 ms** | übernommen als `9a31fa1` |

| [`patches/philip-fastest-rows.patch`](patches/philip-fastest-rows.patch) | Vorwärts- und Rückwärtslauf (`fast.py`): rechte Seiten und Lösungen im Layout des C-Kerns (m, F), keine Umkopier-Schritte zwischen (F, m, k) und (m, F) mehr. Basis `9a31fa1` | x86-64 emuliert mit C-Kern, 6 Läufe im Wechsel: warm Forward 18,8 → 17,1 ms, warm Gradient 4,8 → 4,5 ms, warm gesamt 24,7 → 22,6 ms. Seitenfehler im ersten Aufruf: Vorwärts 5'703 → 5'036, Rückwärts 1'997 → 1'516. Ohne C-Kern (Mac, Docker arm64) Forward gleich, Gradient-Phase 0,6 bis 1,2 ms schneller. **Xeon: noch nicht gemessen**, Schätzung 3 ms im Erstlauf | **offen**, siehe «Runde 19» |

| [`patches/philip-fastest-import.patch`](patches/philip-fastest-import.patch) | Import von `differometor` (`_tuning.py`, `fast.py`): die zwei Prüfungen von JAX-Funktionen in `_tuning.apply()` lesen den kompilierten Code statt den Quelltext (`inspect.getsource`), `_sparsetools` wird erst bei Bedarf geladen. **Nach dem Rows-Patch anwenden** (beide ändern `fast.py` und seinen Bytecode) | Docker arm64 `--cpus 1` mit Bytecode, zweimal 25 frische Prozesse je Stand: `import differometor` **3,86 → 2,68 ms**. Die beiden `getsource`-Aufrufe allein: 2,57 ms → 0,02 ms (M4). Im x86-Lauf wird `_sparsetools` nie geladen. **Xeon: noch nicht gemessen**, Schätzung 2 bis 2,5 ms | **offen**, siehe «Runde 19» |

Die ersten beiden sind seit 21:49 in `philip-fastest` und `main`. Die Xeon-Zahlen stehen unter «Runde 18». Die zwei neuen setzen auf `9a31fa1` auf, in dieser Reihenfolge:

```bash
git fetch origin && git checkout philip-fastest
git show origin/noel:runs/noel/patches/philip-fastest-rows.patch | git am
git show origin/noel:runs/noel/patches/philip-fastest-import.patch | git am
```

Die beiden älteren (schon angewendet, nur zur Vollständigkeit):

```bash
git show origin/noel:runs/noel/patches/philip-fastest-build.patch | git am
git show origin/noel:runs/noel/patches/philip-fastest-forward.patch | git am
```

Der frühere `patches/main-homodyne.patch` ist gelöscht: `main` steht seit 21:18 auf `8e7c4f1` und hat damit Philips eigenen Homodyne-Umbau (`708803d`). Der Patch liegt noch im Verlauf (`b3030c4`).

Zum Vorwärts-Patch (Runde 16, 21:18 bis 21:29):

- **Dual-Zahlen.** Die meisten Rechenschritte in den Bauteil-Formeln verknüpfen eine Dual-Zahl mit einer gewöhnlichen Zahl oder einem gewöhnlichen Array (ein Parameter, der keine Designvariable ist, eine Konstante). Bisher wurde daraus eine Dual-Zahl mit Ableitung 0 gemacht, und Produkt- und Summenregel haben ein Null-Array ausgerechnet und addiert. Jetzt braucht das Produkt einen Term und die Summe lässt die Ableitung stehen. Die Werte sind dieselben, nur das Vorzeichen einer Null kann sich unterscheiden. Dieselbe Klasse steckt in `noel` (dort im Rückwärts-Callback).
- **Rausch-Summen.** `N y` und `N^H y` haben wenige Einträge. Sie gehen direkt in die zweite rechte Seite, und die Rauschleistung `y^H N y` wird über die Rausch-Einträge summiert statt über alle Unbekannten. Zwei Felder (F, m), ihre Summe und eine konjugierte Kopie von `Y` entfallen. Die rechte Seite ist bitgleich, die Leistung wird in anderer Reihenfolge summiert.
- **Zahlen:** Carrier, Signal und Powers des Benchmarks bitgleich, Noise 6,3e-16, Sensitivity 4,1e-16, Gradient 1,6e-14 (relativ zum Maximum), Fehler gegen die Referenz unverändert 5,3742808816537035e-06. Alle elf Skripte unter `runs/philip` und `runs/flags` laufen durch, `test_variants.py` wie vorher (WORST 4,95e-07 und 2,41e-08), `variant_run.py` `PASS`. Im emulierten x86-64-Image mit der C-Zerlegung: Fixture und fünf Varianten `PASS`.
- **Gradient-Phase in Docker** in zwei Messreihen 0,2 bis 0,3 ms langsamer (5,4 → 5,7 ms), auf dem Mac gleich (3,5 ms). Der Rückwärts-Callback selbst ist gleich schnell (1,68 ms). Vermutung, nicht bewiesen: Der Vorwärtslauf gibt weniger Speicher frei, den der Rückwärtslauf im ersten Aufruf sonst wiederverwendet hätte. Netto bleiben in Docker rund 0,9 ms.
- Die nicht mehr benutzte Hilfsfunktion `_scatter_sum` ist entfernt. Der Patch enthält den neu erzeugten Bytecode von `fast.py`.

Zum Build-Patch (Runde 15, 21:02 bis 21:18):

- Alle Ausgaben von `run_build_step` sind für 27 Setups identisch (946 Arrays, `test_build.py --dump` und `--compare` mit `PYTHONHASHSEED=0`), die Benchmark-Arrays bitgleich, `_sideband_layout` gegen die Referenz auf 20'000 Zufallsfällen identisch. Alle elf Skripte unter `runs/philip` und `runs/flags` laufen durch, `variant_run.py` meldet `PASS` für Fixture, Homodyne-Variante und zwei Zufalls-UIFOs.
- Der Patch enthält den neu erzeugten Bytecode von `build.py` (checked-hash, Python 3.12.15). Er berührt weder den nativen Pfad noch `fast.py`.
- Geteilt werden nur Arrays, die `set_instructions` nie verändert (Ausgabe-Auswahl und Funktionseingaben). Sie sind schreibgeschützt, ein versehentliches Ändern würde also sofort auffallen. Positionen in den Systemmatrizen bleiben eigene Arrays, weil die Seitenband-Erweiterung sie an Ort und Stelle verschiebt.
- Wo die Build-Zeit danach steckt (M4, 3,4 ms): rund 2'400 `np.array`-Aufrufe für Positionslisten, die Seitenband-Erweiterung (224 Aufrufe), der Rumpf von `build()` und das Zusammenführen in `pairs_to_arrays`. Mehr wäre nur mit einem Umbau zu holen, der pro Bauteil-Art alle Bauteile auf einmal behandelt.

**Runde 19 (21.11., 22:20 bis 23:45): Noels Auftrag «nochmal alles geben, innovative Ansätze, testen und verwerfen».** Basis `9a31fa1` (`philip-fastest` und `main`, unverändert). Zwei Patches behalten, zwei Mess-Erkenntnisse, eine Regelfrage.

*1. Behalten: `patches/philip-fastest-rows.patch`.* Der C-Kern (`pattern_lu.c`) rechnet auf Feldern mit einer Zeile pro Unbekannter, Form (m, F). Python hat ihm rechte Seiten der Form (F, m, k) gegeben. `plu_solve_fmk` kopiert jede davon in ein Arbeitsfeld um und die Lösung wieder zurück (je 1,2 MB, quer zum Speicher), danach hat `forward` die Lösung für `signal` noch einmal transponiert und `backward` `conj(Y)` und `MU` wieder zurück. Jetzt:

- Die rechten Seiten sind ausserhalb weniger Zeilen null und werden als Zeilen und Werte übergeben (`solve_scattered`). `_PatternLU` baut sie in seinem Layout (ein Arbeitsfeld für alle rechten Seiten, der Kern überspringt Nullzeilen) und ruft `plu_solve` direkt auf. `_CondensedLU` baut sein Feld (F, m, k) wie bisher.
- Die Hilfsfelder sind `Yc` = conj(Y) und `MU`, beide (m, F), also so, wie der Rückwärtslauf sie braucht. Dessen zwei transponierte Kopien entfallen auf beiden Wegen. Die Rausch-Summen laufen im selben Layout, der Kotangent der Rausch-Einträge wird nur noch für die Einträge aus Rausch-Funktionen gebildet.
- Der C-Code und die Binärdatei sind unverändert (`plu_solve` gab es schon).

| Messung | vorher | nachher |
| --- | ---: | ---: |
| x86-64 emuliert (Rosetta, C-Kern mit AVX2), 6 Läufe im Wechsel: warm gesamt, Median / Minimum | 24,7 / 23,4 ms | **22,6 / 21,2 ms** |
| dort warm Forward | 18,75 ms | 17,06 ms |
| dort warm Gradient | 4,82 ms | 4,50 ms |
| dort Erstlauf Forward / Gradient (von der Übersetzung durch Rosetta dominiert, darum wenig aussagekräftig) | 53,1 / 15,3 ms | 52,6 / 15,1 ms |
| Seitenfehler im ersten Vorwärts-Aufruf (x86, C-Kern) | 5'703 | 5'036 |
| Seitenfehler im ersten Rückwärts-Aufruf | 1'997 | 1'516 |
| Lösungen allein, emuliert, warm: erste (für Y) | 0,49 ms | 0,38 ms |
| zweite (für signal und MU, inkl. Kopien) | 1,50 ms | 1,03 ms |
| Docker arm64 `--cpus 1` (ohne C-Kern), 12 Läufe: Forward, Median / Minimum | 30,2 / 26,9 ms | 29,1 / 26,7 ms |
| dort Gradient-Phase | 6,4 / 5,9 ms | 5,2 / 4,3 ms |
| Mac (ohne C-Kern), 12 Läufe: Forward | 22,9 / 21,3 ms | 22,8 / 21,9 ms |
| dort Gradient-Phase | 4,2 / 3,8 ms | 3,6 / 3,3 ms |

Schätzung für den Xeon: rund 2,4 ms aus den entfallenen Kopien (warm Forward dort 27 ms, hier −9 %) plus rund 1,1 ms aus 1'150 Seitenfehlern weniger, zusammen etwa **3 ms im Erstlauf**. Das muss Philip auf der VM messen, die Erstlauf-Zeiten unter Rosetta taugen dafür nicht.

Geprüft: `benchmark.py` `PASS` auf Mac, Docker arm64 und x86-64 emuliert. Arrays gegen `9a31fa1`: auf x86 mit dem C-Kern **alles bitgleich, auch der Gradient**. Ohne C-Kern Carrier, Signal, Noise, Sensitivity und Powers bitgleich, Gradient 3,8e-16 (die Summe `MU @ g` läuft in BLAS in anderer Reihenfolge). Alle zwölf Skripte unter `runs/philip` und `runs/flags` plus `runs/check_setups.py` laufen durch, mit denselben Kennzahlen wie vorher (`test_reverse` WORST 2,64e-13, `test_homodyne` 2,6e-09 und 4,8e-09, `test_condensed` 9,6e-08). `test_variants.py --quick`: WORST 1,99e-11 und 3,02e-12. `variant_run.py`: `PASS` im x86-Image für Homodyne-Fixture, Zufalls-UIFO 0 und 1, veränderte Werte und getauschte Bauteile, im arm64-Image für Homodyne-Fixture und Zufalls-UIFO 4. Der Patch enthält den neu erzeugten Bytecode von `fast.py` (`runs/philip/bytecode.sh`, Python 3.12.15).

*2. Messfehler in Docker gefunden.* Das `Dockerfile` setzt `PYTHONDONTWRITEBYTECODE=1`, bevor `pip install` läuft. pip übersetzt dann keine `.pyc`, und jeder Lauf im Container kompiliert JAX, NumPy und alles andere neu. Dazu schliesst `.dockerignore` `**/__pycache__` aus, der mitgelieferte Bytecode von `differometor` fehlt im Image also auch (18 ms pro Lauf). Gemessen (6 Läufe im Wechsel, `9a31fa1`): Imports **258,3 ms ohne, 116,0 ms mit Bytecode**, Erstlauf 336,9 gegen 189,4 ms. Alle bisherigen Docker-Zahlen haben also einen um rund 150 ms zu hohen Import-Anteil. Die Vergleiche zwischen zwei Ständen stimmen trotzdem, weil beide gleich betroffen waren. Auf Philips VM (Git-Checkout, pip mit Bytecode) gibt es das Problem nicht. Für Messungen ab jetzt: Dockerfile ohne die Variable beim Installieren und ohne den Ausschluss von `__pycache__` (so gebaut: `differometor-bench:pf5-*`). Vorschlag an Philip: im `Dockerfile` `PYTHONDONTWRITEBYTECODE` weglassen und in `.dockerignore` die Zeile `**/__pycache__` streichen.

*3. Wo die Zeit steckt, mit Bytecode* (Docker arm64, `--cpus 1`, `9a31fa1`, warmer Container, ms): Imports 94 (davon Kopf von `benchmark.py` bis `platform` 16, `statistics`-Träger 2,5, `jax` und `numpy` 74, `differometor` 3,8), Initialisieren 1,9 (`jax.devices`), Laden 1,1, Build 6,1 (davon `run_build_step` 4,4, erstes `jnp.asarray` 1,3), Lower 4,7 + 6,5 + 2,1, Kompilieren 3,9 + 7,2 + 5,0, Forward 24,5 (Callback 24, darin 8'200 bis 9'000 Seitenfehler), Sensitivity 0,6, Gradient 4,5 (Callbacks 3,4, in der Phase 2'400 bis 2'700 Seitenfehler). Die drei Programme haben 1, 31 und 19 Operationen, an ihnen ist von unserer Seite nichts mehr zu kürzen.

*4. Seitenfehler als Mass.* Zeilengenaues Profil mit `ru_minflt` (Skript im Scratchpad, `faultlines.py`). Kalibrierung aus Philips Xeon-Zahlen: Erstlauf minus warm sind 6,2 ms im Forward und 4 ms im Gradient, bei rund 5'500 und 2'700 Seitenfehlern, also **rund 1 µs pro Seitenfehler auf dem Xeon**. Der Vorwärts-Patch aus Runde 16 passt dazu (1'170 Seiten weniger, 1,2 ms). `mallopt` hilft nicht (Philips Fehlversuch 5, meine Runden 5 bis 8), es zählt nur weniger frisch angefasster Speicher. Die grössten Posten nach dem Patch, x86 mit C-Kern:

| Stelle | Seitenfehler | Was |
| --- | ---: | --- |
| `signal_values`, Zeilen mit `pre` | 1'379 | `pre = V[placing]` und `-pre[live] * carrier[...]`: drei Zwischenfelder von je 1,6 MB. Der Versuch aus Runde 17 hat zwei davon entfernt und war in Docker nicht messbar, wäre nach dieser Rechnung aber rund 0,8 ms auf dem Xeon |
| `solve_rows` | 876 | die drei Ergebnisfelder `Yc`, `signal`, `MU` (je 1,2 MB), nötig |
| Dual-Zahlen `__mul__` und `run` | 940 | Einträge mit Ableitungen |
| `carrier_matrix` und `SystemPlan.__init__` | rund 620 | die Carrier-Matrix wird dicht aufgebaut und kopiert (zweimal 1,3 MB), obwohl nur das dünn besetzte Muster gebraucht wird |
| `backward`, `np.take(right, cols)` und `block` | 760 | zwei Felder (Einträge, F) von 2,1 MB, in Zeilenblöcken wären es rund 100 |
| `_signal_adjoint` | 310 | `right = MU * (-2g)` und `signal * alpha` |

*5. Regelfrage, nicht gebaut: `import importlib.metadata` im Kopf von `benchmark.py`.* Das Modul wird nur für den Report gebraucht, zieht aber 36 Module nach (zipfile, email, csv, tempfile, bz2, lzma). Gemessen mit dem Kopf von `benchmark.py` in Docker `--cpus 1`, 25 Läufe im Wechsel: Imports 97,7 ms mit, **92,9 ms ohne** (−4,8 ms, hochgerechnet rund 9 ms auf dem Xeon, 3 % des Erstlaufs). `subprocess` bringt nur 0,3 ms, weil `jax/version.py` es ohnehin lädt. Der Haken: Philips Träger `statistics.py` läuft erst nach dieser Import-Zeile, man bräuchte also die `.pth`-Datei (bei Philip bewusst aus) oder ein zweites Schatten-Modul. Und es wäre ein Eingriff in die Imports von `benchmark.py`, der Arbeit hinter den Zeitnehmer schiebt. Ohne Noels Ja nicht gebaut.

*6. Behalten: `patches/philip-fastest-import.patch`* (auf dem Rows-Patch). `import differometor` kostet mit Bytecode 3,86 ms in Docker. Ein Drittel davon waren zwei Aufrufe von `inspect.getsource` in `_tuning.apply()`: Bevor `_install_lowering_in_place` und `_install_bind_jit` JAX patchen, prüfen sie am Quelltext, ob `jaxpr_subcomp` die Tabelle `_uncacheable_primitives` benutzt und ob `pjit_staging_rule` die erwarteten Parameter kennt. `getsource` liest dafür die ganze Datei und zerlegt sie in Tokens. Jetzt werden dieselben Fragen dem kompilierten Code gestellt (`co_names` für die benutzten Namen, `co_consts` für die Zeichenketten, jeweils samt inneren Funktionen). Für jax 0.9.0 kommen dieselben Antworten heraus, `_tuning.STATUS` meldet weiter alle Massnahmen als «on». Das ist kein neuer Eingriff in JAX, nur dieselbe Sicherheitsprüfung auf billigerem Weg. Ein Unterschied: Ein Parametername, der nur als Schlüsselwort eines Aufrufs vorkommt (`f(jaxpr=...)`), zählt jetzt auch, im Quelltext zählten nur Namen in Anführungszeichen. Dazu `_sparsetools`: Das Modul braucht nur `_CondensedLU`. Es wird beim ersten Zugriff geladen (`_csr_kernels`), mit dem C-Kern also nie. `fast._sparsetools = None` schaltet es weiterhin ab (`test_compat.py`).

Gemessen in Docker arm64 `--cpus 1` mit Bytecode, zweimal 25 frische Prozesse je Stand im Wechsel: `import differometor` 3,87 und 3,85 ms → **2,69 und 2,67 ms** (Minimum 3,72 → 2,60). Auf arm64 wandern davon 0,4 ms in den Vorwärtslauf (dort wird `_sparsetools` gebraucht), auf x86 mit C-Kern nicht. Geprüft: `test_tuning`, `test_startup`, `test_compat`, `test_condensed`, `test_forward_cache`, `test_reverse`, `check_setups` laufen durch. Arrays auf dem Mac bitgleich zum Stand mit dem Rows-Patch, im x86-Image bitgleich zu `9a31fa1`, `PASS`. Nach einem ganzen Benchmark-Lauf im x86-Image ist `scipy.sparse._sparsetools` nicht geladen. Der Patch enthält den neu erzeugten Bytecode von `_tuning.py` und `fast.py`.

*7. Sonst gemessen.* Im C-Kern kostet die Analyse (Pivotfolge) 3,0 ms gegen 2,3 ms Numerik (emuliert): Die Markowitz-Suche besucht 279'000 Einträge für 379 Pivots, eine auf wenige Zeilen und Spalten begrenzte Suche wäre der übliche Ausweg (Philips Code, hier nicht messbar).

*Stand der Branches um 23:13:* `philip-fastest` und `main` unverändert `9a31fa1`. `origin/philip` neu auf `6f26dec` (22:31), nur Ergebnisdateien: `v2-2209` (das andere Team mit neuem Stand `final/combined-v2` und eigenem C++-Löser: 763,9 ms gegen unsere 321,1 ms auf dem Xeon, Arrays auf 1e-9 gleich), `tuning-2220` (Schalter einzeln: nur bitgleiche Kompilieroptionen 377,6 ms, ohne XLA-Optionen 455,3, `_tuning` aus 489,5, auch Start-Massnahmen aus 593,9 gegen 315,8 ms mit allem) und `schalter-2226` (`INLINE_DISPATCH` und `SHORT_LOCATIONS` bringen nichts, dazu eine Aufschlüsselung der Import-Phase). Philip misst also gerade selbst an den Imports. `Joel`, `max/lu-once` und `sx*` unverändert.

**Nächste Hebel für Philips Linie**, in dieser Reihenfolge: (a) `signal_values` ohne die Zwischenfelder (rund 800 Seitenfehler), (b) Carrier-Matrix dünn besetzt aufbauen (rund 600), (c) `backward` in Zeilenblöcken (rund 700), (d) begrenzte Pivotsuche im C-Kern. Zusammen geschätzt 2 bis 3 ms auf dem Xeon.

**Runde 18 (21.11., 22:13 bis 22:15): Philip hat beide Patches übernommen und auf dem Xeon gemessen.** `origin/philip-fastest` und `origin/main` stehen auf `9a31fa1` (21:49): `13bd58d` ist der Build-Patch, `9a31fa1` der Vorwärts-Patch, unverändert übernommen. `origin/philip` (`3983044`, 21:59) enthält die Messungen dazu.

Philips Messung auf dem Xeon Platinum 8468 (Auftrag `patches-2145`, 14 Läufe im Wechsel, Median / Minimum in ms, alle `PASS`):

| Stand | Erstlauf | Build | Forward | Gradient | Warm Forward | Warm Gradient |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Basis `8e7c4f1` | 320,7 / 314,0 | 18,4 | 34,5 | 11,1 | 28,0 | 6,6 |
| nur Build-Patch | 318,4 / 314,7 | 17,2 | 34,8 | 11,0 | 28,1 | 6,7 |
| nur Vorwärts-Patch | 320,7 / 315,9 | 18,6 | 33,3 | 11,5 | 27,1 | 7,5 |
| beide | **317,8 / 312,0** | 17,2 | 33,3 | 11,5 | 27,1 | 7,6 |

Also: Build −1,2 ms, Forward −1,2 ms, Gradient +0,4 ms, zusammen −2,9 ms im Median. Meine Schätzung war 3 bis 5 ms.

**Warum die Gradient-Phase langsamer wird** (in Docker nachgemessen, Seitenfehler des ersten Rückwärts-Aufrufs): 1'452 vorher, 2'044 nachher. Der Vorwärtslauf legt seit dem Patch 4,8 MB weniger Zwischenfelder an und gibt sie darum auch nicht mehr frei. Der Rückwärtslauf hatte genau diesen freigegebenen Speicher wiederverwendet und muss jetzt 2,4 MB frisch anfassen. Der Rückwärts-Callback selbst rechnet gleich schnell (M4: 1,68 ms vorher und nachher). Eine billige Abhilfe habe ich nicht gefunden: Die drei Felder (m, F) in `_signal_adjoint` lassen sich nicht an Ort und Stelle wiederverwenden, ohne die Ergebnisse im Zwischenspeicher zu verändern, die ein zweiter Pullback noch braucht. In kleinen Zeilenblöcken zu rechnen spart höchstens 300 Seitenfehler (rund 0,2 ms auf dem Xeon). Nicht gebaut.

**Zur Einordnung, aus Philips Auftrag `konkurrenz-2050` (20:01 UTC, nicht von uns gemessen):** das öffentliche Repo eines anderen Teams (`Kartikdangi1/vibevoxels-starthack`, drei Branches) auf demselben Xeon mit demselben `benchmark.py`: 1'071 bis 1'246 ms im Erstlauf gegen 313,8 ms für `philip-fastest`. Deren Imports 295 ms (unsere 187), Kompilieren 390 (35), Forward 157 (34), Gradient 77 bis 94 (11). Das ist ein Stand von 20:13 und sagt nichts über Teams, die nicht öffentlich arbeiten.

**Runde 17 (21.11., 21:43 bis 21:48): kein neuer Patch.** `philip-fastest` und `main` stehen unverändert auf `8e7c4f1`, die beiden Patches sind dort noch nicht angewendet. Gearbeitet auf `8e7c4f1` plus beiden Patches.

| Versuch | Ergebnis |
| --- | --- |
| Signalwerte (`SystemPlan.signal_values`) mit weniger Zwischenfeldern: konstante Werte nur in die Zeilen ohne Eintrag, Vorzeichen der Platzierung auf dem kurzen Carrier-Vektor statt auf dem grossen Feld (zwei Felder von 1,6 MB weniger) | Die Funktion selbst 0,486 → 0,368 ms (M4, warm), Ergebnis bitgleich. Im Benchmark nicht messbar: Forward-Phase in Docker `--cpus 1`, 16 Läufe im Wechsel, 25,9 / 24,6 → 26,0 / 24,8 ms (Median / Minimum). Verworfen |

Gemessen, ohne etwas zu ändern (M4, auf beiden Patches):

- Strukturpläne pro Vorwärtslauf: `EntryPlan` 0,40 ms, `SystemPlan` 0,60 ms, Rückwärts-Layout 0,03 ms. In `_EntryBlock.__init__` stecken pro Vorwärtslauf rund 50 `np.unique`- und 70 `np.nonzero`-Aufrufe auf kleinen Arrays. Mehr als 0,2 ms sind dort nicht zu holen.
- Signalwerte: 0,49 ms warm (Platzierung 0,22, Konjugation 0,10, Einträge 0,08, Basiswerte 0,07). Philips 5,8 ms auf dem Xeon im ersten Aufruf sind also vor allem frisch angefasster Speicher.
- `pattern_lu.c` kennt schon einen Parameter `chunk` für die Numerik und die Lösungen. Laut Kommentar im Code brachte er in keiner Messung etwas, weil die Werte Eintrag für Eintrag über alle Frequenzen liegen. Frequenzblöcke nach dem Muster von `noel` bräuchten also eine andere Ablage der Faktoren (pro Block wiederverwendet) und eine Analyse, die über die Blöcke hinweg gilt. Das ist ein Umbau im C-Code und nur auf x86 messbar.

**Weitere Hebel für Philips Linie:** der genannte Umbau im C-Code (3 bis 8 ms laut Philips Schätzung), `signal` und `MU` ohne Umkopieren aus `solutions` (rund 0,5 ms, bräuchte zwei getrennte Lösungen, deren Kosten im C-Code hier nicht messbar sind), ein Umbau des Builds pro Bauteil-Art (geschätzt weitere 20 bis 30 % des Builds, gross), Imports (Philips Gebiet). Host-seitig sehe ich darüber hinaus nichts mehr über 0,3 ms auf dem Mac.

## Runde 14 (21.11., 20:55 bis 21:00): Schleife alle 30 Minuten, Blick in Philips Stand

Noel um 20:54: «starte wieder caffeinate und arbeite alle 30 minuten, die anderen Teams holen auf». `caffeinate -is` läuft, der Laptop hängt am Strom, die Schleife feuert jetzt um :13 und :43 (Auftrag `a794ba8a`, der alte Zwei-Stunden-Auftrag ist gelöscht).

**Branches:** `origin/philip-fastest` steht auf `8e7c4f1` (20:37, nur `runs/philip/test_genauigkeit.py` neu). `main` (`50341f9`) und alle anderen unverändert. Runde 13 (20:35) war ohne Änderung: 8 % Akku, nichts gemessen.

**`noel`:** keine Änderung. Regelkonform ist oberhalb von 0,5 ms nichts mehr offen (siehe Runde 11). Für die Rangliste zählt Philips Linie: `philip-fastest` 305,9 ms auf dem Xeon, `noel` (`91baf9c`) dort 688 bis 718 ms auf Philips Varianten.

**Vergleich der Host-Arbeit** (Mac, je 20 Aufrufe, Median, öffentliches Fixture, `philip-fastest` hier ohne seine C-Zerlegung, die nur unter x86 kompiliert):

| Stand | Build | Vorwärts-Callback | Rückwärts-Callback | Vorwärts + Rückwärts |
| --- | ---: | ---: | ---: | ---: |
| `noel` (`2b792de`) | 4,4 ms | 18,6 ms | 7,1 ms | 25,7 ms |
| `philip-fastest` (`8e7c4f1`) | 4,45 ms | 17,2 ms (wertet die Einträge schon hier mit Ableitungen aus) | 1,65 ms | 18,9 ms |

Die 6,8 ms Unterschied kommen aus Philips Zwischenspeicher: Sein Rückwärtslauf verwendet Pläne, Einträge samt Ableitungen und beide Zerlegungen des Vorwärtslaufs wieder. Die einzelnen Bausteine sind gleich schnell wie bei uns (Einträge mit Ableitungen je 2,8 ms, Pläne 1,6 gegen 1,4 ms, Signalwerte 0,8 gegen 1,0 ms). Aus `noel` lässt sich dort also nichts Schnelleres einsetzen.

**Wo auf dem Xeon noch Zeit liegt** (aus Philips Zahlen, Vorwärtslauf im ersten Aufruf 36,6 ms: Einträge mit Ableitungen 10, native Zerlegung 7,7, Signalwerte 5,8, Lösungen 5,3, Pläne 4,5, Rausch-Summen 2,1, Carrier 1,2):

| Hebel | Schätzung auf dem Xeon | Stand |
| --- | --- | --- |
| Frequenzblöcke im Vorwärtslauf wie in `noel` (je 16 Frequenzen, Arbeitsfelder werden wiederverwendet statt für 101 Frequenzen frisch angelegt). Philip zählt rund 11'000 Seitenfehler im ersten Vorwärtslauf | 3 bis 8 ms (Philips Schätzung, seine offene Idee 1) | Braucht in `pattern_lu.c` eine Trennung von Analyse (3,2 ms, einmal) und Numerik (pro Block). `plu_factor` macht heute beides in einem Aufruf. Der C-Code kompiliert nur unter x86, hier also nicht messbar |
| Rausch-Summen `q` und `qh` direkt in die rechte Seite streuen statt zwei Felder (F, m) anzulegen, `signal` und `MU` ohne Umkopieren aus `solutions` | rund 1 ms (etwa 1'500 Seitenfehler weniger) | Nicht gebaut. Auf dem Mac nicht messbar |
| Build weiter straffen (offene Idee 3). Der Build ist reines Python und kostet auf dem Xeon rund 17 ms, hier 4,4 ms | 2 bis 3 ms | Offen, in beiden Linien derselbe Code. Jede der 519 Aufrufstellen von `set_instructions` einzeln |
| Imports (183 von 306 ms) | unbekannt | Daran arbeitet Philip (`_startup.py`). Auf dem Mac gemessen: Von den Standardmodulen, die `benchmark.py` vor `jax` lädt, ist `importlib.metadata` mit Abstand das teuerste |

Die Kopie von `main` (`r14`) und die Arbeitskopien dieser Runde sind wieder entfernt. `main` und `philip-fastest` sind unverändert.

## Stand 20:12: Philip hat Balanced Homodyne selbst eingebaut, `main` noch nicht

- **`origin/philip-fastest`** steht auf `11e96b4`. Neu seit `efad894`: `708803d` (19:51, «Fast path covers balanced homodyne detectors», Philips eigene Fassung nach dem Muster von `8cb5eff`, ohne Verknüpfungen) und `11e96b4` (20:04, Dokumente aus dem offiziellen Repo übernommen). Philips Messung auf dem Xeon (Auftrag `homodyn-1944`): Fixture mit Homodyne-Auslese 35'505 → 323 ms, Zufalls-UIFOs mit Homodyne 314 bis 325 ms, Fixture selbst 319,2 → 320,3 ms mit bitgleichen Ausgaben.
- **Von hier aus nachgeprüft** (Mac, Stand `11e96b4`): `test_variants.py` zeigt 19 von 21 Varianten über den schnellen Weg mit denselben Abweichungen wie `noel` (WORST `|dln s|` 4,95e-07, Gradient 2,41e-08). Die zwei Varianten mit Amplitudenmodulation laufen dort richtig, aber über JAX (verknüpfte Parameter, kommen in einem JSON-Fixture nicht vor). `variant_run.py`: Fixture, Homodyne-Variante und zwei Zufalls-UIFOs mit Homodyne `PASS` bei 307 bis 346 ms.
- **`origin/main`** steht weiter auf `50341f9` und hat den Umbau nicht. `main` ist ein Vorfahr von `philip-fastest`, Philip kann also ohne Merge vorspulen.
- **Der Patch aus Runde 12 ist für `philip-fastest` überholt und darf dort nicht mehr angewendet werden** (er ändert dieselben Stellen wie `708803d`). Für `main` gilt: Käme er dort hinein, gäbe es beim nächsten Zusammenführen mit `philip-fastest` Konflikte in `fast.py`, `simulate.py`, `check_setups.py` und zwei `.pyc`-Dateien. Noel hat um 20:06 «Ja, auf main pushen» gesagt, als Philips Commit schon 15 Minuten auf dem Server lag. Darum ist nichts gepusht, und die Frage liegt wieder bei Noel.
- **Offizielles Repo** (feyerlabs `2bb437a`, 21.11. 14:26, von Philip übernommen): Das README sagt jetzt «faster for general optical setups» und «Final evaluation uses a different optical setup, a slight variation of the supplied example». Die Geschwindigkeitspreise gelten für den «complete first-run total on the final optical setup». Korrekt ist eine Abgabe, wenn sie Ergebnisse und Gradienten «on both the development example and the final setup» erhält.

## Runde 12 (21.11., 18:36 bis 19:08): `main` ist jetzt Philips Stand, Patch für Balanced Homodyne

**Lage:** `origin/main` steht seit 17:30 auf Philips Linie (`50341f9`: Tuning-Modul, eigene C-Zerlegung, Zwischenspeicher, Start-Massnahmen, mitgelieferter Bytecode). `origin/philip-fastest` (`efad894`, 18:26) geht noch weiter. Philips Messungen auf dem Xeon Platinum 8468: `main` 355,8 ms, `philip-fastest` 315,3 ms im Erstlauf (Auftrag `neu-1804`). `noel` lag dort bei 726 ms (Stand Runde 8). Die Abgabe wird also aus dieser Linie kommen, nicht aus `noel`.

**Das Problem dort:** `fast.supports` schliesst in beiden Branches weiter Balanced Homodyne und verknüpfte Parameter aus. Philip hat es in `runs/philip/NOTIZEN.md` selbst als offen vermerkt: Ein Homodyne-Setup braucht auf dem Xeon 35,5 s (Auftrag `stand-1642`). Der Zufallsgenerator, aus dem das Fixture stammt, wählt Homodyne in der Hälfte der Fälle (siehe Runde 9).

**Was diese Runde geliefert hat:** [`runs/noel/patches/main-homodyne.patch`](patches/main-homodyne.patch), ein Commit gegen `origin/main` (`50341f9`), der den Umbau aus Runde 9 auf Philips `fast.py` überträgt. `main` und `philip-fastest` selbst sind unverändert. Anwenden (Stand 19:08. Seit 19:51 nur noch für `main` sinnvoll, siehe «Stand 20:12»):

```bash
git fetch origin
git checkout main            # nicht mehr auf philip-fastest: dort ist Philips eigener Umbau
git show origin/noel:runs/noel/patches/main-homodyne.patch | git am
```

Inhalt des Patches:

- `differometor/fast.py`: `EntryPlan.selection` (Phasenfaktor des Homodyne-Detektors am Auswahlvektor), Verknüpfungen mit `jnp.sqrt` auf der Parameterzeile samt Kettenregel, Ableitung nach der Phase, `supports` ohne die beiden Ausschlüsse, fünf weitere Host-Arrays (bei Philip `Constant`, also ohne Kosten in JAX). Den Auswahlvektor bekommt der Plan direkt nach dem Bau (`plan.selection = ...`), in beiden Wegen (mit und ohne Zwischenspeicher).
- `differometor/simulate.py`: nur der Docstring.
- Der mitgelieferte Bytecode der beiden Module ist neu erzeugt (`compileall --invalidation-mode checked-hash`, Python 3.12.15, Hash stimmt mit der Quelle überein).
- `runs/check_setups.py` erwartet für aLIGO verknüpft und Voyager den schnellen Weg. Neu dazu: `runs/noel/test_variants.py`, `variant_fixture.py`, `variant_run.py`.

Geprüft auf Noels Mac (dort ohne die C-Zerlegung, die nur unter x86-Linux lädt) und in einem emulierten x86-64-Container:

| Prüfung | Ergebnis mit Patch |
| --- | --- |
| `benchmark.py` auf dem öffentlichen Fixture | `PASS`, alle Arrays bitgleich zum ungepatchten `main` (Carrier, Signal, Noise, Sensitivity, Powers, Gradient) |
| `runs/noel/test_variants.py` | 21 Varianten, alle über den schnellen Weg, WORST `|dln s|` 4,95e-07 und Gradient 2,41e-08 (dieselben Werte wie auf `noel`) |
| `runs/check_setups.py` | alle vier Setups schnell, aLIGO verknüpft 1,8e-08, Voyager 7,3e-13 (Sensitivity) |
| Philips Tests `test_reverse`, `test_forward_cache`, `test_static`, `test_compat`, `test_condensed`, `test_ports` | laufen durch (`test_reverse` WORST 2,6e-13) |
| Homodyne-Variante des Fixtures (`variant_run.py`) | `PASS`, 340 ms statt 9'837 ms, Fehler gegen die Referenz aus dem Original-Code 8,3e-08 |
| Zufalls-UIFO Seed 0 (Homodyne) und Seed 2 (Detektor) | `PASS`, 345 und 341 ms |
| `philip-fastest` mit Patch | Fixture `PASS` 299 ms, Homodyne-Variante `PASS` 305 ms |
| `main` mit Patch als x86-64-Image unter Rosetta, C-Zerlegung geladen (`NATIVE_LU True`), `variant_run.py --image` | Fixture und 14 Varianten `PASS` (6 davon mit Homodyne). Fehler gegen die Referenz aus dem Original-Code höchstens 8,3e-08, Fixture 5,374e-06 wie überall. Die Zeiten aus der Emulation sagen nichts aus |

Nicht gemessen ist der Patch auf dem echten Xeon. Das kann nur Philip mit seinem VM-Skript. Die Zeiten dieser Runde sind nur Grössenordnungen: Der Deckel war bis etwa 19:00 zu, der Akku bei 20 bis 23 %. Aus demselben Grund gab es in dieser Runde keine Optimierung an `noel` und keinen Wechselvergleich.

Was an `main` auffällt und in die Regelfragen gehört (Feststellung, keine Wertung): Eine Datei `statistics.py` im Wurzelverzeichnis fängt das `import statistics` von `benchmark.py` ab, um vor `import jax` Start-Massnahmen auszuführen. Dazu kommen mitgelieferte `.pyc`-Dateien, eine `.pth`-Datei unter `packaging/`, Eingriffe in JAX und XLA für den ganzen Prozess (`_tuning.py`) und eine vorkompilierte Bibliothek (`_native/libpattern_lu.so`). Das sind die Fragen B4, B5 und B6 aus `FRAGEN_AN_FEYER.md`.

## Was in Runde 11 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 25 | Ein Produkt statt zwei pro Matrixeintrag im Rückwärtslauf (offene Idee 9). Gilt die Abkürzung aus Runde 10 (`lam = alpha * conj(Y)`), dann ist der Kotangent eines Eintrags `-conj(Y[row]) * (alpha * X + 2 g * MU)[column]`. Die Adjungierte `lam` wird nicht mehr als Feld angelegt, und pro Frequenzblock gibt es zwei statt vier Zugriffe mit Indexlisten | `fast.py` (`backward`, `_adjoint_factors`) | Gradient-Phase: Mac 10,2 → 9,7 ms, Docker 13,3 → 12,8 ms. Rückwärts-Callback allein: 7,3 → 7,0 ms |

Zu Nr. 25: Carrier, Signal, Noise, Sensitivity und Powers des Benchmarks sind bitgleich, der Gradient ändert sich um 1,7e-14 (relativ zum Maximum). Alle Prüfskripte und `test_variants.py` mit denselben Werten. Für jeden anderen Kotangenten (zweite Zerlegung) rechnet die Schleife wie bisher mit zwei Produkten, `test_adjoint.py` prüft diesen Zweig. Zurücknehmen: `git revert 2e1cef1`.

Damit sind die regelkonformen Ideen oberhalb von rund 0,5 ms aufgebraucht. Was bleibt, steht unter «Offene Ideen»: Nr. 1 bis 3 liegen bei 0,3 bis 1 ms, Nr. 8 lässt sich nur auf x86 beurteilen, der Rest braucht eine Zusage von Feyer oder Noel.

## Skalierung mit der Gittergrösse (21.11., 16:10, auf Noels Frage)

`uifo(size)` baut dasselbe Modell wie das Fixture als N×N-Gitter. Gemessen mit `runs/noel/scaling.py` auf dem Mac (Akku), Zufalls-Setup mit einfachem Detektor und Zufallswerten, `jit(value_and_grad(loss))` mit dem Loss aus `benchmark.py`, 101 Frequenzen, Stand `55ecbc3`:

| Gitter | Spiegel | Designvariablen | Unbekannte im Signalsystem | Warm (Forward + Gradient) | Erster Aufruf mit Kompilation | Speicher (Spitze) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2×2 | 24 | 103 | 373 | 13,8 ms | 46 ms | 0,20 GB |
| 3×3 (wie das Fixture) | 48 | 190 | 749 | 27,9 ms | 62 ms | 0,22 GB |
| 4×4 | 80 | 304 | 1'247 | 45,6 ms | 81 ms | 0,25 GB |
| 5×5 | 120 | 459 | 1'883 | 72,6 ms | 109 ms | 0,30 GB |
| 6×6 | 168 | 619 | 2'629 | 106,6 ms | 146 ms | 0,37 GB |
| 8×8 | 288 | 1'055 | 4'525 | 212,9 ms | 254 ms | 0,48 GB |

Die Zeit wächst etwa linear mit der Zahl der Unbekannten (Exponent rund 1,1). Der unveränderte Original-Code (`5eece1d`) im selben Skript, wegen des Speichers mit 21 statt 101 Frequenzen (er rechnet auf dem Mac mit mehreren Kernen, unser Weg mit einem):

| Gitter | Original warm | `noel` warm | Faktor | Original: Speicher | `noel`: Speicher |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2×2 | 111,8 ms | 5,5 ms | 20 | 0,82 GB | 0,18 GB |
| 3×3 | 554,2 ms | 9,0 ms | 62 | 1,73 GB | 0,19 GB |
| 4×4 | 2'026,6 ms | 14,1 ms | 144 | 3,82 GB | 0,20 GB |

Das Original wächst mit einem Exponenten von rund 2,5 in der Zeit und 2 im Speicher. Hochgerechnet aus den 14,4 GiB, die das README für 3×3 mit 101 Frequenzen nennt, bräuchte ein 4×4-Gitter rund 40 GiB (nicht gemessen). Auf der Referenzmaschine mit 32 GB liefe es also nicht.

Einordnung:

- Für die Rangliste zählt das nicht direkt: Gewertet wird ein ähnliches Setup, und bei 3×3 sind rund 420 von 445 ms fixe Kosten (Imports, Kompilation).
- Für die Robustheit ist es abgedeckt: `test_variants.py` vergleicht 1×1 bis 4×4 mit dem Original-JAX. Für 5×5 bis 8×8 ist hier nur geprüft, dass Loss und Gradient endlich sind.
- Für den Pitch ist es ein Argument: Der Ansatz skaliert, das Original nicht.
- Ab Grösse 10 sind die Setups des Generators unbrauchbar: Er setzt die Knotennamen aus den Ziffern der Gitterposition zusammen (`m110` steht für zwei verschiedene Spiegel).

## Was in Runde 10 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 24 | Die Detektor-Basis `Z` entfällt (offene Idee 7, bei Philip seine Nr. 14). Der Vorwärtslauf löst pro Frequenz nur noch `S^T conj(Y) = conj(w)` mit einer rechten Seite statt zwei (Detektor) oder vier (Balanced Homodyne). Im Rückwärtslauf ist die Signal-Adjungierte `alpha_f * conj(Y_f)`, wenn der Kotangent des Signals bei jeder Frequenz ein Vielfaches von `conj(w)` ist (`_signal_adjoint`). Das Residuum `Z` (2,4 bzw. 4,9 MB) wandert nicht mehr durch JAX | `fast.py` | Docker: Erstlauf ohne Imports 202,6 → 199,1 ms, Forward 29,7 → 26,6 ms, warm 36,7 → 33,5 ms. Mac: Forward 23,7 → 21,5 ms, warm 30,0 → 27,6 ms. Homodyne-Varianten in Docker 8 bis 11 ms |

Zu Nr. 24:

- **Wann die Abkürzung gilt:** Eine Sensitivität teilt das Rauschen durch das Signal desselben Detektors. Signal und Rauschen werden durch dieselben Gewichte `w` gelesen, darum ist der Kotangent des Signals `alpha_f * conj(w)`. Geprüft wird das an den Zahlen: Der Kotangent darf nur auf den Detektorzeilen liegen, und der Rest nach der Projektion auf `conj(w)` muss bei jeder Frequenz unter 1e-13 des Kotangenten liegen (`ADJOINT_TOLERANCE`). Gemessen: öffentliches Fixture 4,4e-16, UIFO mit Homodyne 1,3e-16 und 3,4e-16, Voyager 2,4e-16, aLIGO 1,9e-16.
- **Jeder andere Kotangent** (anderes Readout, Homodyne-Phase ungleich 180°, Verlust auf beliebigen Signalzeilen) nimmt `general_adjoint`: eine zweite Zerlegung im Rückwärtslauf, wie bisher für Kotangenten ausserhalb der Detektorzeilen. Das kostet dort rund 10 ms mehr als mit der Basis, das Ergebnis ist dasselbe. `test_adjoint.py` prüft diesen Weg.
- **Zahlen:** Carrier, Signal und Powers des Benchmarks sind bitgleich. Noise ändert sich um 9,4e-16, Sensitivity um 4,1e-16, Gradient um 3,5e-12 (relativ zum Maximum), `ln s` um höchstens 7e-15. Alle Prüfskripte mit denselben Werten wie vorher (Voyager Sensitivity 7,30e-13 statt 7,27e-13).
- `runs/noel/profil.py` nimmt für den Rückwärtslauf jetzt einen Kotangenten wie das Benchmark-Readout. Vorher war er zufällig, das wäre seit dieser Runde der langsame Weg.
- Zurücknehmen: `git revert 9a55e69`.

In Runde 10 ausserdem gemessen (nichts geändert):

- **Build-Phase:** Von den 23 bis 25 ms auf dem Mac gehören 5,0 ms uns (`run_build_step`). 18,5 ms sind drei XLA-Kompilationen von `convert_element_type`, die `benchmark.py` mit `jnp.asarray(..., dtype=jnp.float64)` auslöst: 11 bis 12 ms für die 193 Parameter (die erste Kompilation des Prozesses), 3,4 bis 3,9 ms für die Frequenzen und 3,5 bis 3,9 ms für die Konstante 1.0. `Setup.from_data`, die Gruppierung der Längen und `jnp.asarray(ports)` brauchen zusammen unter 0,2 ms. Diese Kompilationen liegen ausserhalb von `differometor/`. Philip umgeht sie mit einem Eingriff in JAX (`_tuning.FAST_ASARRAY`), daher seine 17 statt 53 ms Build auf dem Xeon.
- **Rückwärts-Callback** (Mac, warm, 7,4 ms): Rückwärtsformeln 2,8 ms, Einträge mit Ableitungen 2,8 ms, Pullback 1,4 ms, `EntryPlan` und `SystemPlan` je 0,4 ms, Projektion des Kotangenten 0,2 ms.

## Runde 9: Setup-Varianten (Feyers Zusatz vom 21.11.)

Feyer im Discord (von Noel am 21.11. gegen 14:35 weitergegeben): «The goal is a simulator speedup that works beyond a specific setup. So the final evaluation will use a similar optical setup like the one included in the GitHub repo, but with slightly different parameters and components. Your simulator submissions should still work with that slight setup variation». Dazu Jonathan Klimesch um 14:30: Die Schlussbewertung nutzt eine logische CPU wie `--cpus 1`, und die JIT-Kompilation zählt zur Gesamtzeit (kompletter Erstlauf).

**Was das für uns heisst (am Code nachgeprüft):**

- Das öffentliche Fixture enthält die Schlüssel `centers` und `boundaries`. Das sind die Rückgaben von `uifo(3, random=True, verbose=True)`. Das Fixture ist also ein Zufalls-UIFO aus dem Generator, dessen Werte danach optimiert wurden.
- Der Generator setzt als Auslese mit je 50 % einen einfachen Detektor oder einen Balanced-Homodyne-Detektor (`rng.choice(["detector", "balanced_homodyne"])`). `benchmark.py` kann beides auslesen (`signals[0] - signals[1]` bei zwei Ports). Von acht Zufalls-Setups (Seeds 0 bis 7) hatten fünf Balanced Homodyne.
- Bis `c15f97e` schloss `fast.supports` Homodyne-Detektoren aus. Ein solches Setup lief über den dichten JAX-Weg: auf dem Mac 10'059 ms statt 452 ms im Erstlauf (gemessen mit `variant_run.py`), mit einer CPU auf dem Xeon noch deutlich mehr.

**Geändert (`differometor/fast.py`, dazu Docstrings in `simulate.py`):**

| Nr. | Änderung | Wirkung |
| --- | --- | --- |
| 22 | Balanced-Homodyne-Detektor (`qhd`) im schnellen Weg. Der Auswahlvektor des Rausch-Detektors bekommt auf dem Host den Phasenfaktor `exp(1j * phase)` an den Einträgen des zweiten Ports (`EntryPlan.selection`, wie `lax.scatter_mul` im Original). Die Detektor-Basis `Z` hatte dann vier statt zwei Spalten (seit Runde 10 gibt es sie nicht mehr). Die Ableitung nach der Phase ist eingebaut, falls sie eine Designvariable ist (`selection_cotangent`) | Homodyne-Variante des Fixtures: 10'059 → 452 ms auf dem Mac |
| 23 | Verknüpfte Parameter (Amplitudenmodulation: Signalamplitude = `sqrt(Laserleistung)`) im schnellen Weg, für die Funktionen in `fast.LINK_RULES` (heute `jnp.sqrt`). Vorwärts auf der Parameterzeile, rückwärts mit der Kettenregel. Unbekannte Funktionen gehen weiter über JAX | aLIGO, Voyager und UIFO mit Amplitudenmodulation laufen schnell. In einem JSON-Fixture kann eine solche Verknüpfung nicht stehen (eine Python-Funktion lässt sich nicht speichern), das ist also Vorsorge |

Für das öffentliche Fixture ändert sich nichts: Benchmark-Ausgaben und alle Prüfwerte sind gleich, die Zeit auch (je 9 bzw. 8 Läufe im Wechsel, Median / Minimum): Mac 449,6 / 446,9 → 449,4 / 444,1 ms, Docker `--cpus 1` 518,5 / 514,9 → 518,6 / 513,2 ms.

**Nachgestellte Schlussbewertung** (`runs/noel/variant_fixture.py` und `variant_run.py`): Varianten-Fixture samt Referenz-Sensitivitäten aus dem unveränderten Original-Code (`5eece1d`), darauf der echte `benchmark.py` (in einer Kopie mit den Hashes der Variante). Alle `PASS`:

| Variante | Auslese | Mac: Erstlauf / warm | Docker `--cpus 1`: Erstlauf / warm | Fehler gegen die Referenz |
| --- | --- | ---: | ---: | ---: |
| Fixture, alle Werte leicht verändert (2 Seeds) | Detektor | 454,1 / 30,2 und 459,9 / 30,7 ms | 519,0 / 35,6 ms | 8,5e-10 und 6,2e-08 |
| Fixture mit Balanced Homodyne statt Detektor | Homodyne | 451,7 / 32,9 ms | 605,3 / 46,0 ms (Imports 367 statt 320 ms, Ausreisser) | 8,3e-08 |
| Fixture mit einem getauschten Zentrum und einer getauschten Quelle (3 Seeds) | Detektor | 450,9 bis 468,1 ms | 538,7 / 35,7 ms | höchstens 1,6e-08 |
| Zufalls-UIFO mit Zufallswerten, Seeds 2, 3, 5 | Detektor | 459,8 bis 480,6 ms | 519,4 und 530,8 ms | höchstens 6,6e-11 |
| Zufalls-UIFO mit Zufallswerten, Seeds 0, 1, 4, 6, 7 | Homodyne | 460,4 bis 489,7 ms | 538,9 bis 551,0 ms | höchstens 7,6e-10 |

Stand Runde 9 (seit Runde 10 sind es noch rund 8 ms, siehe Messwerte): Homodyne-Setups sind in Docker rund 20 ms langsamer als Detektor-Setups: Forward 33 bis 34 statt 29 bis 30 ms (vier statt zwei Spalten der Detektor-Basis, grösseres System), dazu 5 ms in der Kompilation des Readouts und 4 ms in der des Gradienten (das ist der Readout von `benchmark.py`, nicht unser Code).

**Tests:** `runs/noel/test_variants.py` vergleicht 21 Varianten mit `simulate_jax` (Zufalls-UIFOs 1×1 bis 4×4, alle drei Modulationsarten, beide Auslesen, Homodyne-Phase als Designvariable). Alle laufen über den schnellen Weg. Grösste Abweichung von `ln s`: 4,9e-07 (Mac) bzw. 7,7e-07 (Docker), Gradient 2,4e-08 bzw. 2,8e-07. `runs/check_setups.py` erwartet jetzt für aLIGO mit Amplitudenmodulation und für Voyager den schnellen Weg.

**Befunde:**

- Zufallswerte ergeben ab und zu ein schlecht konditioniertes Signalsystem (Seed 4: Konditionszahl 1e12 nach Äquilibrierung). Dort unterscheiden sich schon JAX, NumPy und SuperLU mit voller Pivotisierung untereinander um 3e-08 bis 1e-07 an den Detektorzeilen. Unsere Lösung hat dort den kleinsten komponentenweisen Rückwärtsfehler (1e-15). Das ist kein Fehler des schnellen Wegs, aber bei einem solchen Setup schrumpft der Abstand zur Toleranz 1e-5.
- Setups ohne brauchbare Auslese (kein Licht am Detektor, oder das Signal löscht sich bis auf Rundungsrauschen aus, z. B. Frequenzmodulation mit Homodyne) liefern in beiden Wegen dasselbe Rauschen. Der Test erkennt sie und zählt sie nicht als Fehler.
- Weiter über JAX laufen: mehr als ein Rausch-Detektor, mehrere Carrier-Arbeitspunkte, unbekannte Verknüpfungsfunktionen, komplexe Designvariablen, dichte Systemmatrizen. `benchmark.py` kann die ersten beiden gar nicht auslesen.

```bash
# Varianten gegen simulate_jax (rund 1 Minute, --quick für zwei Fälle)
python runs/noel/test_variants.py .
# Schlussbewertung nachstellen: Original-Code exportieren, Variante erzeugen, Benchmark darauf
git archive 5eece1d | tar -x -C <ordner>/upstream
python runs/noel/variant_fixture.py <ordner>/upstream <ordner>/homodyne homodyne      # auch: perturbed, swapped, random [seed]
python runs/noel/variant_run.py . <ordner>/homodyne                                    # lokal
python runs/noel/variant_run.py . <ordner>/homodyne --image differometor-bench:runde9  # Docker, --cpus 1
```

## Was in Runde 8 geändert wurde

Noel am 21.11. um 13:14: «Optimiere jetzt wirklich nochmals alles was geht, werde kreativ». Eine Änderung ist geblieben, alles andere ist gemessen und verworfen (Tabelle darunter).

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 21 | Index-Arrays sind statisch. Der Build liefert seine ganzzahligen Index-Arrays als `differometor.fast.StaticArray`: schreibgeschützte NumPy-Arrays, die für JAX ein Pytree-Knoten ohne Blätter sind. JAX verfolgt sie nicht, sie sind Konstanten der kompilierten Programme, und die Host-Callbacks bekommen sie über den Callback selbst statt als Argumente. Andere Index-Arrays ergeben eine neue Kompilation (Vergleich nach Inhalt). Die sieben übrigen Float-Arrays reisen als ein gepacktes Residuum. Das Forward-Programm hat 8 statt 36 Eingabe-Arrays, sein Callback 8 statt 29 Argumente | `fast.py`, `build.py`, `_simulate_dicht.py` | Erstlauf ohne Imports: Docker 204,7 → 201,4 ms, Mac 131,5 → 130,5 ms. Forward-Lowering und -Kompilation: Docker 8,2 → 6,5 ms, Mac 7,7 → 6,8 ms |

Zu Nr. 21:

- Benchmark-Ausgaben bitgleich, Build-Arrays von 27 Setups identisch (`test_build.py --compare`), alle Prüfskripte mit denselben Zahlen.
- Wer `simulate` gewöhnliche Integer-Arrays übergibt, landet weiter im schnellen Weg: Sie werden dann verfolgt und gepackt wie vorher. `test_pack.py` prüft beide Wege gegeneinander (identischer Loss und Gradient) und ein verändertes Index-Array gegen das Original-JAX (2,3e-12).
- Der JAX-Weg (`simulate_jax`) macht aus statischen Arrays am Eingang gewöhnliche JAX-Arrays. Dort sind sie jetzt Konstanten des Programms. Für zwei Setups mit gleichen Formen, aber anderen Indizes wird darum neu kompiliert.
- Die Sensitivity-Lowering-Phase wurde um 0,2 bis 0,5 ms langsamer. Erklärung, nicht bewiesen: Einmalige Vorarbeit von JAX für Integer-Argumente fällt jetzt dort an statt im Forward-Programm. Im Gewinn oben ist das enthalten.
- Zurücknehmen: `git revert b6b1119`.

In Runde 8 gemessen und nicht übernommen:

| Versuch | Ergebnis |
| --- | --- |
| Residuum `Y` weglassen und im Rückwärtslauf aus `Z` berechnen (bitgleich) | Mac: Forward −0,5 ms, Gradient +0,35 ms. Docker: Forward −0,8 ms, Gradient +0,4 ms. Netto innerhalb der Streuung. Zurückgenommen |
| Blockfolge der Zerlegung mit grösserem erstem Block (26 + 5×15, 21 + 5×16), Faktoren vor dem nächsten Block freigeben | Seitenfehler der Zerlegung im ersten Aufruf 3'060 → 2'390 bis 2'740, aber kein Zeitgewinn (Forward 28,2 bis 29,5 gegen 29,0 bis 29,6 ms, Einzelläufe). Warm teils schlechter |
| glibc-Schwellen mit einer sofort wieder freigegebenen 31-MB-Anforderung anheben (nur Versuch, Regelfrage B6) | Erstlauf ohne klaren Gewinn. Warm im besten Fall 23,6 ms Forward und 9,6 ms Gradient ohne Seitenfehler |
| Vorab-Elimination: 345 der 751 Unbekannten für alle Frequenzen in NumPy herausrechnen, SuperLU zerlegt nur 406 (Prototyp, Mac) | Zerlegung 9,3 → 7,1 ms inklusive Aufbau des reduzierten Systems (1,0 ms), Faktoreinträge 6'118 → 4'388. Aber jede Lösung mit Rücksubstitution 5,7 statt 3,4 ms. Insgesamt 2,4 ms langsamer. Abweichung der Lösungen 3e-10 |
| Detektoren in XLA statt als Host-Callback (Docker, Summe aus Sensitivity und Gradient im Erstlauf) | Heute 129 bis 141 ms. `power_detector` mit der billigsten Formel in XLA: 137 bis 163 ms. `signal_detector` in XLA: 200 ms. Beide: 209 ms |
| Kommentare und Docstrings aus den Quelltexten entfernen, nie aufgerufene Funktionen auslagern | Übersetzen aller Module auf dem Mac 9,1 → 8,5 ms. Funktionen, die der Benchmark nie aufruft, machen zusammen 0,9 ms aus. Lohnt nicht |
| Speicherbereinigung (Garbage Collector) im Build abschalten | Im Build fallen nur 0,12 ms an (siehe Erkenntnisse) |

## x86-64, emuliert (21.11., 13:04, Stand `cb902d7`, auf Noels Wunsch)

`noel` war bis dahin nur auf Apple Silicon und in Docker auf aarch64 gelaufen. Als x86-64-Image unter Rosetta (`VirtualApple`, Linux x86_64, glibc 2.41, Python 3.12.16, dieselben gepinnten Pakete):

| Prüfung | Ergebnis auf x86-64 |
| --- | --- |
| `benchmark.py --cpus 1 --repeats 3` | `PASS`, Fehler gegen Feyers Referenz 5,3743e-06 (wie auf aarch64), alle Ausgaben und Gradienten endlich, Gradient-Norm 70'517,1007 |
| Ausgaben gegen den nativen Mac-Lauf (relativ zum Maximum) | Carrier 2,9e-13, Signal 8,0e-10, Noise 3,8e-10, Sensitivity 2,4e-12, Powers 5,8e-13, Gradient 1,2e-11. Grösste Abweichung von `ln s`: 1,2e-11 |
| `check_fast.py runs/noel/orig.npz --fallback` | Carrier 9,5e-12, Signal 1,168e-07, Noise 3,817e-08, Sensitivity 5,735e-09, Powers 1,336e-11, Gradient 7,986e-08 |
| `check_setups.py` | aLIGO (schnell) höchstens 7,9e-09, aLIGO verknüpft und Voyager 0 |
| `test_detectors.py`, `test_adjoint.py`, `test_perturbed.py` | WORST 2,2e-15, 1,8e-07 (Blockgrössen: Abweichung 0), 3,9e-08 |
| `test_pack.py`, `test_sparse.py`, `test_build.py` | alle «OK» |

Am 21.11. um 14:21 mit dem Stand `b6b1119` wiederholt: `PASS`, dieselben Abweichungen gegen aarch64, `test_pack.py` und `test_sparse.py` «OK».

Am 21.11. um 15:12 mit dem Stand `8cb5eff` (Runde 9, Image `differometor-bench:x86-9`): Original-Fixture `PASS` (Fehler 5,374e-06), Homodyne-Variante des Fixtures `PASS` (8,3e-08 gegen die Referenz aus dem Original-Code), Zufalls-UIFO Seed 0 mit Homodyne `PASS` (7,4e-11), leicht verändertes Fixture `PASS` (8,5e-10), `test_variants.py --quick` «OK» (`|dln s|` 4,2e-11).

Am 21.11. um 15:23 mit dem Stand der Runde 10 (Image `differometor-bench:x86-10`): Original-Fixture `PASS` (5,374e-06), Homodyne-Variante `PASS` (8,3e-08), Zufalls-UIFO Seed 0 mit Homodyne `PASS` (7,4e-11), `test_adjoint.py` WORST 1,8e-07 mit Abweichung 0 zwischen den Blockgrössen. Die warmen Läufe sind auch dort kürzer geworden (Fixture 50,2 → 46,3 ms, Homodyne 54,1 → 46,9 ms).

Die Zeiten aus der Emulation sagen nichts aus (Erstlauf 2,2 s, davon 0,77 s Sensitivity-Kompilation). Rosetta bietet auch nicht denselben Befehlssatz wie ein Xeon (kein AVX-512). Auf echter Hardware können darum andere Rechenkerne von OpenBLAS und XLA laufen. Zu erwarten sind Unterschiede auf Rundungsniveau, bei einer Toleranz von 1e-5 und einem Abstand von 4,6e-6.

```bash
docker build --platform linux/amd64 -t differometor-bench:x86 .
docker run --rm --platform linux/amd64 differometor-bench:x86 --repeats 3
```

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

## Wo die restliche Zeit steckt (Docker, Stand Runde 3, Build Runde 7, Forward und Gradient Runde 8)

| Phase | ms | Unser Anteil |
| --- | ---: | --- |
| Imports | 309 | 13 ms für `import differometor`, fast alles Übersetzen der Quelltexte, weil das Image keine `__pycache__`-Ordner hat: `fast.py` 4,8, `build.py` 4,4, `components.py` 2,0, `setups.py` 0,9, `simulate.py` 0,6. Mit Cache: 1,4 ms |
| Initialisierung, Laden | 3 | nichts |
| Build | 23 | 5,9 ms (`run_build_step`, vor Runde 7 waren es 8,5 ms). Der Rest gehört zu `benchmark.py`: Der erste `jnp.asarray`-Aufruf startet das JAX-Backend (ca. 9 ms), und jede neue Form in `jnp.asarray(liste, dtype=...)` kompiliert ein kleines XLA-Programm (je rund 3,5 ms, dreimal) |
| Forward: Lowering, Kompilation | 6,5 | alles, aber fast nur JAX-Grundaufwand: In `df.simulate` selbst vergehen beim Tracing 0,6 ms. Ein Programm ganz ohne Rechenoperationen kostet rund 3 ms Kompilation |
| Forward | 29,5 | alles. Callback im Erstlauf 26,8 bis 27,4 ms (Aufteilung aus Runde 3: Zerlegung 11, Lösen 4,6, Schleife in `forward` 3,6, Einträge 2,1, Plan 1,4, Matrixwerte 1,2). Ausserhalb des Callbacks ca. 2 ms: JAX kopiert 6 MB Ergebnisse heraus. Vor Runde 6 waren es 5 bis 6 ms, weil 19 MB Argumente hineinkopiert wurden |
| Sensitivity: Lowering | 11 | 0,4 ms in unseren zwei Detektor-Funktionen selbst, dazu ihr Anteil an JAX' Ableitungs-Maschinerie (zusammen geschätzt 2 bis 3 ms). Der Rest ist das Tracing des Readouts |
| Sensitivity: Kompilation | 66 | nichts. Das ist der Readout aus `benchmark.py` in XLA |
| Gradient: Lowering | 5 | wenig |
| Gradient: Kompilation | 36 | nichts |
| Gradient | 12,3 | fast alles. Callback im Erstlauf 10,3 bis 10,6 ms: Funktionen mit Ableitung, Schleife in `backward`, Rückrechnung auf die Parameter. Ausserhalb ca. 2 ms Übergabe (12 Argumente, darunter rund 7 MB Residuen und Kotangenten) |

Rund 430 von 505 ms sind fix, solange `benchmark.py` unverändert bleibt und keine JAX-Schalter gesetzt werden dürfen.

## Erkenntnisse aus Runde 8 (alle gemessen, Docker wenn nicht anders vermerkt)

| Thema | Befund |
| --- | --- |
| Threads | Seit Runde 6 laufen alle Host-Callbacks im Haupt-Thread, auch der Forward-Callback |
| Erster gegen warmer Forward-Aufruf | 28 bis 29,7 ms gegen 24 bis 26 ms. Der Unterschied sind Seitenfehler: 6'900 bis 7'700 im ersten Aufruf, 1'300 bis 1'700 danach |
| Seitenfehler im ersten Forward-Callback | Zerlegung 3'060 bis 3'150 (pro Block 997, 711, 453, 430, 298, 173, 2), übrige Arbeit in `forward` 1'150, Matrixeinträge 500 bis 1'015, `signal_values` 410 bis 450, Lösen 80 |
| Warum die Zerlegung so viele Seitenfehler hat | SuperLU reserviert für die Faktoren das 20-Fache der Matrixeinträge: für einen Block von 16 Frequenzen zweimal 17,4 MB und einmal 4,4 MB, wovon rund 1,5 MB benutzt werden. glibc holt so grosse Blöcke frisch vom System und gibt sie nach jedem Block zurück. Die Reserve lässt sich über SciPy nicht einstellen |
| Tracing unserer Funktionen | Forward-Lowering 3,2 ms, davon in `df.simulate` 0,62 ms. Sensitivity-Lowering 10,7 ms, davon in `signal_detector` 0,25 ms und in `power_detector` 0,15 ms. Ein `jax.vjp` nur um `signal_detector` braucht 2,7 ms Lowering, um `power_detector` 1,9 ms, um den Rest des Readouts 5,7 ms (je inklusive Grundaufwand pro Programm) |
| XLA-Operationen sind teuer | Selbst `real² + imag²` auf 286 Werten kostet als XLA-Operation 4 bis 6 ms Kompilation und 2,4 ms Lowering mehr als der Host-Callback |
| Speicherbereinigung von Python | Ein Benchmark-Lauf löst 182 Sammelläufe aus, zusammen 19,7 ms. Davon fallen 15,7 ms während `import jax` an, also bevor unser Paket geladen wird, und nur 1,6 ms zwischen dem Import von `differometor` und dem Ende des Erstlaufs (Build 0,12 ms, Sensitivity-Lowering 0,55 ms). Gesammelt wird dabei fast nichts (rund 500 Objekte). Mit ab Prozessstart abgeschalteter Bereinigung (nur Versuch, über ein Startskript um `benchmark.py`) liegt der Erstlauf bei 478 bis 491 statt 500 bis 511 ms. Aus `differometor/` heraus ist das nicht erreichbar |
| JAX-Schalter, neu gemessen | `jax_disable_most_optimizations` auf dem Stand `e311d5c`, 8 Läufe im Wechsel: Erstlauf 515,8 → 475,4 ms (Sensitivity-Kompilation 68,1 → 40,9 ms, Gradient-Kompilation 39,3 → 23,5 ms). Fehler gegen die Referenz unverändert, Gradient-Norm gleich bis zur 13. Stelle. Weiterhin Regelfrage B6 |
| Index-Arrays an JAX vorbei (Obergrenze vor dem Bau von Nr. 21) | Mit einem prozessweiten Wörterbuch als Wegwerf-Versuch: Forward-Lowering −0,5 ms, Forward-Kompilation −0,7 ms, Gradient −0,2 ms, warm −0,9 ms |
| Deckel zu | Von 13:24 bis 13:58 war der Laptop zugeklappt. Messungen aus dieser Zeit sind unbrauchbar (Kompilationszeiten doppelt so hoch) und nicht verwendet |

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
docker build -q -t differometor-bench:neu . && python runs/noel/ab_docker.py 10 vorher=differometor-bench:runde8 neu=differometor-bench:neu
# Profil der Host-Callbacks, lokal oder im Image
source runs/env.sh && python runs/noel/profil.py . forward
docker run --rm --cpuset-cpus 0 --entrypoint python differometor-bench:neu /app/runs/noel/profil.py /app backward
```

Einzelne Läufe streuen um bis zu 40 ms. Darum immer abwechselnd messen und Median und Minimum ansehen. Für kleine Änderungen die Phasen einzeln vergleichen: Die Berichte der Wechselvergleiche liegen in `runs/noel/out/` (`ab-<Name>-<Nr>.json`) und `runs/noel/out/docker/`. Der erste Lauf eines frisch gebauten Images ist ein Ausreisser (562 statt 520 ms) und zählt nicht.

Vorhandene Docker-Images: `differometor-bench:base` (Philip + Max), `:vorher` (`c261b63`), `:kandidat3` (`49d386b`), `:runde3` (`1442ef3`), `:runde4` (`ba8f1f2`), `:runde5` (`45a2328`), `:runde6` (`7d12ba7`), `:runde7` (`e311d5c`), `:runde8` (`b6b1119`), `:x86` (`cb902d7`, linux/amd64, emuliert), `:joel` (`76e799f`).

## Korrektheit prüfen (nach jeder Änderung)

```bash
source .venv/bin/activate && source runs/env.sh
python runs/check_fast.py runs/noel/orig.npz --fallback
python runs/check_setups.py
python runs/noel/test_detectors.py .
python runs/noel/test_adjoint.py .      # allgemeiner Rückwärtslauf, Blockgrösse, Residuen-Grenze
python runs/noel/test_perturbed.py .    # veränderte Designvariablen gegen das Original-JAX
python runs/noel/test_pack.py .         # gepackte Residuen und statische Index-Arrays
python runs/noel/test_sparse.py .       # Sparse-Matrizen: Form, dichte Eingaben, von Hand veränderte Matrizen
python runs/noel/test_build.py .        # Seitenband-Erweiterung im Build gegen eine Referenz
python runs/noel/test_variants.py .     # Zufalls-UIFOs (Detektor und Balanced Homodyne) gegen das Original-JAX
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

Sollwerte (Stand Runde 11, auf Noels M4, unverändert seit Runde 2):

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
| `check_setups.py` | Seit Runde 9 alle vier Setups über den schnellen Weg: aLIGO höchstens 1,1e-8, aLIGO verknüpft 1,8e-8 (Sensitivity) und 1,6e-9 (Gradient), Voyager 1,7e-9 (Signal), 7,3e-13 (Sensitivity), 1,1e-10 (Gradient). Alle ohne zweite Zerlegung im Rückwärtslauf |
| `test_detectors.py` | WORST 2,9e-14 |
| `test_adjoint.py` | WORST 4,0e-08, Blockgrössen 1 und 2 gegen 16: Abweichung 0. Die Prüfung der Residuen-Grenze läuft nur noch bei Ständen vor Runde 6 |
| `test_perturbed.py` | WORST 2,1e-08 |
| `test_pack.py` | «7 arrays restored exactly», zweimal «28 arrays», «23 static index arrays, 7 leaves for JAX», «identical loss and gradient», veränderter Index gegen JAX 2,3e-12, dann «OK» |
| `test_build.py` | «20000 random cases identical to the reference», dann «OK» |
| `test_variants.py` | 21 Zeilen mit `fast=True`, keine mit «FAILED», WORST `|dln s|` 4,95e-07 und Gradient 2,41e-08 (Mac), dann «OK» |
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
| XLA-Flags, Kompilier-Cache, `jax.config`-Schalter, `mallopt`, Allocator mit einer grossen Anforderung «vorwärmen» | Regelfrage, siehe `FRAGEN_AN_FEYER.md` (B6) |
| Rückgabe von Stellvertreter-Objekten statt Arrays, um den Readout zu umgehen | Wäre Schummeln |
| Prüfung der dichten Matrizen in XLA statt im Callback | Jede echte XLA-Operation kostet 5 bis 15 ms Kompilierzeit, mehr als die Kopie. Seit Runde 6 gegenstandslos: Es gibt keine dichten Matrizen mehr |

## Offene Ideen

| Nr. | Idee | Schätzung | Haken |
| --- | --- | --- | --- |
| 1 | Ergebnisse `MU` und `Z` in der Reihenfolge der Elimination lassen statt pro Block umzusortieren. Der Rückwärtslauf greift dann mit umgerechneten Indizes zu | geschätzt 0,4 ms im Forward | Die Reihenfolge müsste als weiteres Residuum mit. Unter der Messgrenze, wie der Versuch mit `Y` gezeigt hat |
| 2 | `EntryPlan` nur einmal bauen: Er hängt nur von statischen Index-Arrays ab und könnte beim Tracing entstehen und beiden Callbacks mitgegeben werden | geschätzt 0,3 ms | Die Callbacks hielten dann ein gemeinsames Objekt. Es dürfte nach dem Bau nicht mehr verändert werden |
| 3 | `build.py` weiter straffen: Aufrufstellen von `set_instructions` (rund 2'400 `np.array`-Aufrufe), Schleife über Bauteile | geschätzt 0,5 bis 1 ms. Der grössere Teil (2,5 bis 3 ms) ist seit Runde 7 erledigt | Jede Aufrufstelle einzeln anfassen. Prüfbar mit `test_build.py --dump` und `--compare` |
| 4 | Falls Feyer oder Noel Einstellungen im Code freigibt (B6): eine Zeile `jax.config.update("jax_disable_most_optimizations", True)` in `differometor/__init__.py` | gemessen in Runde 8: 40,4 ms in Docker (515,8 → 475,4 ms). Auf dem Mac 13 ms (Runde 3). Angehobene glibc-Schwellen bringen im Erstlauf nichts Verlässliches (Runde 8) | Nur nach ausdrücklicher Zusage |
| 4b | Speicherbereinigung von Python vor `import jax` abschalten | gemessen: rund 20 ms (478 bis 491 statt 500 bis 511 ms) | Geht nur ausserhalb von `differometor/` (Startskript, Umgebung oder `benchmark.py`). Das ist Feyers Entscheid, nicht unserer |
| 4c | Zwischenergebnisse des Forward-Laufs im Prozess behalten statt als Residuen durch JAX zu schicken (LU-Faktoren, Pläne) | geschätzt 3 ms in Docker, nicht gemessen | Widerspricht der Abmachung «keine Caches». Bräuchte Noels Ja und einen zustandslosen Rückfall |
| 5 | `noel` auf Joels x86-Maschine messen (AMD EPYC, `taskset -c 14 python benchmark.py --cpus 1`), im Wechsel mit `6b67cfd` (Stand vor Runde 5). Die Bündelung aus Runde 5 lässt sich nach Runde 6 nicht mehr einzeln zurücknehmen. Sie ist jetzt Teil des Entwurfs, und ihr Nachteil im Warmlauf ist weg | – | Joel fragen |
| 8 | Vorab-Elimination der einfachen Unbekannten nach Philips Muster (`_CondensedLU`: 751 → 200 Unbekannte in drei Stufen). In Runde 8 auf dem Mac verworfen (2,4 ms langsamer). Auf x86 ist SuperLU laut Philip 3,5-mal langsamer als auf dem M4, mit rund 0,5 µs festem Aufwand pro Spalte, und dort hat es bei ihm den Vorwärtslauf von 105,8 auf 82,1 ms gebracht | unbekannt für unseren Stand (balancierte Zerlegung mit 6'100 Faktoreinträgen). Philips Stand ohne eigene C-Zerlegung liegt auf dem Xeon im Forward bei 57 ms, `noel` bei 63 ms | Lässt sich nur auf x86 beurteilen. Auf dem Mac und in Docker (aarch64) sieht man den Effekt nicht |
| 6 | Falls ohne `__pycache__` gewertet wird (Frage B4): Code, den der Benchmark nie ausführt, in Module auslagern, die erst bei Bedarf geladen werden (Teile von `build.py` und `components.py`) | unbekannt, höchstens einige der 13 ms | Mit `__pycache__` bringt es nichts. Erst nach Feyers Antwort |

## Stand der anderen Branches

Vor jeder Runde: `git fetch --all --prune`, dann alle fremden Branches auf neue Commits prüfen.

| Branch | Zuletzt gesehen | Stand in `noel` |
| --- | --- | --- |
| `origin/philip` | `6f26dec` (21.11. 22:31): dazu die Aufträge `v2-2209`, `v2-arrays-2210`, `tuning-2220` und `schalter-2226` (siehe «Runde 19»). Davor `3983044` (21.11. 21:59): Ergebnisdateien der VM-Aufträge `konkurrenz-2050`, `main-2122` und `patches-2145`. Davor `8435944` (21.11. 15:28). Neu seit `ab82e42`: `b3676de`, `5934330` und `8435944` (nur die Ergebnisdateien der drei VM-Messreihen, kein Code) | Am Code des Branches selbst ändert sich nichts. Neu ist der Ordner `_nacht/` mit Philips Nachtlauf `philip-nacht` als Git-Bundles, seinen Notizen und Messungen auf einer VM mit Xeon Platinum 8468 (siehe unten). Nichts übernommen: Sein Stand nutzt Mittel, die bei uns unter die Regelfrage B6 und die Abmachung «keine Caches» fallen. Das entscheidet Noel |
| `origin/max/lu-once` | `4d5b43d` (21.11. 13:54), unverändert am 21.11. 18:36. Neu seit `79ae580`: `f5a176d` (statischer Bauplan für den JAX-Weg, `differometor/plan.py`), `4d5b43d` (Nachtbericht und Messdaten) | nicht übernommen. Max' Nachtlauf nennt `noel@6b67cfd` als besten Stand und findet «keinen Hebel ab 5 ms ohne Regelfrage». `plan.py` beschleunigt den dichten JAX-Weg, den wir nach Runde 9 für UIFO-Setups nicht mehr brauchen |
| `origin/max/nacht` | `f1000e9` (21.11. 03:25), am 21.11. 15:06 auf dem Server gelöscht (der Commit steckt in `origin/max/lu-once`) | zusammengeführt in `7279114`. Übernommen: nachgeladene Module, `_EntryBlock.back`, In-place-Produkte, `signal_values` ohne Zwischenkopie. Ersetzt durch den Stand von `noel`: Spaltenordnung `MMD_ATA` (7'916 Faktoreinträge gegen 6'100) |
| `origin/Joel` | `9f79a5f` (21.11. 17:53), neu seit `76e799f`: nur das README (Verweis auf das neue Tag `joels-fastest` = `ac7aedc`). Baut auf Philips `ab82e42` auf | nichts zu übernehmen. Joels drei Code-Änderungen (Host-Detektoren, Plotly bei Bedarf, SuperLU ohne SciPy-Import) hat `noel` seit Runde 1 in eigener Fassung |
| `origin/sx`, `origin/sx2`, `origin/sx2-env` | `2c9e8bf`, `036194a`, `bf6bf60` (21.11. 03:33 bis 04:00), unverändert am 21.11. 18:36 | nicht übernommen. Das sind Joels vorbereitete Varianten mit XLA-Flags bzw. `jax_disable_most_optimizations`, laut seinem README erst nach Zusage von Feyer zu aktivieren. Gleiche Regelfrage wie unser B6 |
| `origin/main` | `9a31fa1` (seit 21.11. 21:49, gleich `philip-fastest`, also mit Balanced Homodyne und beiden Patches). Vorher `50341f9` (17:30), davor das unveränderte Kit `5eece1d` | nicht übernommen (Regelfragen B4 bis B6, Abmachung «keine Caches»). Das unveränderte Kit ist weiter `5eece1d` |
| `origin/philip-fastest` | `9a31fa1` (21.11. 21:49): `8e7c4f1` plus die beiden Patches aus `runs/noel/patches/` (`13bd58d`, `9a31fa1`). `main` plus weitere Start-Massnahmen (315,3 ms auf dem Xeon), seit `708803d` mit Balanced Homodyne im schnellen Weg, seit `11e96b4` mit den Dokumenten des offiziellen Repos | nicht übernommen. Der Patch aus Runde 12 ist hier überholt |

### Philips Messungen auf dem Xeon Platinum 8468 (aus `origin/philip:_nacht/vm/jobs/`, nicht von uns gemessen)

Das ist dieselbe CPU wie in Feyers Brief. Je frischer Prozess, `--cpus 1`, Median / Minimum in ms, alle `PASS`:

| Messreihe | Stand | Erstlauf | Imports | Build | Lower | Kompilieren | Forward | Gradient | Warm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `alle-1430` (10 Läufe) | `noel` `cb902d7` (Runde 7) | 734,1 / 723,4 | 304,7 | 54,7 | 41,3 | 235,1 | 63,3 | 31,2 | 79,8 |
| | `philip-nacht` `d043e68` | 643,1 / 622,3 | 305,9 | 52,3 | 35,0 | 177,8 | 57,8 | 11,7 | 56,6 |
| | Max `f1000e9` | 841,4 | 302,8 | 75,2 | 42,9 | 234,7 | 113,1 | 67,9 | 159,3 |
| | Joel `76e799f` | 896,9 | 302,7 | 76,1 | 43,6 | 235,5 | 137,1 | 99,3 | 212,7 |
| `neu-1500` (12 Läufe) | `noel` `c15f97e` (Runde 8) | 725,7 / 710,4 | 302,8 | 53,0 | 40,1 | 232,3 | 62,6 | 32,7 | 77,7 |
| | `philip-nacht` `3e78a14` («Tuning-Modul + native Faktorisierung») | **441,3 / 431,6** | 303,5 | 17,4 | 34,3 | 39,5 | 34,3 | 11,2 | 36,1 |
| | dasselbe mit «Start-Massnahmen» (GC-Pause, JAX-Module später laden) | 374,2 / 370,3 | 234,8 | 19,2 | 33,1 | 39,9 | 34,6 | 11,5 | 36,6 |
| `final-1503` (6 Läufe, offizieller Aufruf) | `philip-nacht` `38f670e` | 429,8 bis 461,2 | | | | | | | 36 |

Was daraus folgt:

- Unsere Messungen auf dem Mac und in Docker (aarch64) unterschätzen die Kompilation auf x86: 232 ms auf dem Xeon gegen 131 ms in Docker und 66 ms auf dem Mac. Auch Forward (63 gegen 30 ms) und Gradient (33 gegen 14 ms) sind dort doppelt so teuer.
- Philips Stand ist auf dem Xeon rund 285 ms schneller als `noel`. Der grösste Teil kommt aus `differometor/_tuning.py`: Es stellt beim Import XLA für den ganzen Prozess um (klassischer Kernel-Emitter, ein Codegen-Modul, kleine Programme ohne LLVM-Optimierung). Dazu kommen ein Zwischenspeicher zwischen Vorwärts- und Rückwärts-Callback (`FORWARD_CACHE`), die Port-Auswahl des Readouts auf dem Host über eine eigene Tracer-Klasse und eine eigene LU in C (`differometor/_native/libpattern_lu.so`, vorkompiliert im Repo). Das sind genau die Mittel, die wir wegen der Regelfrage B6 und der Abmachung «keine Caches» nicht eingebaut haben.
- **Philips schnellster Stand fällt bei einem Balanced-Homodyne-Setup auf JAX zurück.** Sein `fast.supports` verlangt weiter `qhd_parameter_indices.shape[0] == 0` und `linked_indices.shape[0] == 0`. Auf Noels Mac mit `variant_run.py` gemessen (Stand `38f670e` aus `final-1503.tar.xz`): Original-Fixture 456,9 ms, Detektor-Varianten 400 bis 403 ms, Homodyne-Variante des Fixtures **9'837 ms**, Zufalls-UIFO Seed 0 (Homodyne) **8'924 ms**, alle `PASS`. `noel` braucht für dieselben Varianten 452 und 476 ms. Auf dem Xeon mit einer CPU wird der Rückfall noch deutlich teurer sein.
- Der Umbau für Homodyne ist in Philips `fast.py` kleiner als bei uns, weil es dort keine Detektor-Basis `Z` mehr gibt: Auswahlvektor mit Phasenfaktor (hier `EntryPlan.selection`), `supports` lockern, Ableitung nach der Phase oder Ausschluss dieses Falls. Prüfen lässt es sich mit `test_variants.py`, `variant_fixture.py` und `variant_run.py` aus diesem Branch.

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
- Seit Runde 8 (21.11., `b6b1119`) sind die Index-Arrays aus `run_build_step` `StaticArray`s. Gebaut im Rahmen von Noels Auftrag vom 21.11. um 13:14 («alles was geht, werde kreativ»).
- Verworfene Versuche werden nicht committet.
- Eine Änderung bleibt nur, wenn sie im Wechselvergleich schneller ist und alle Prüfungen unverändert gut ausfallen.
- `benchmark.py`, Fixtures und `requirements.txt` bleiben unangetastet. Keine Caches, nichts vorberechnen.
- Discord (Noel am 21.11. gegen 14:35): Feyer schreibt dort laufend neue Infos. Mitlesen, wenn es geht, aber **im Discord strikt nichts schreiben und keine Fragen stellen**. Stand 15:13: Der eingebaute Browser ist bei Discord nicht angemeldet, den Zugriff auf die Discord-App hat Noel abgelehnt. Neue Nachrichten kommen also über Noel.
- Noel am 21.11. um 20:54: caffeinate wieder starten, Schleife alle 30 Minuten (:13 und :43).
- Noel am 21.11. um 21:02: Die Schleife baut Patches für Philips Stand (`philip-fastest`) statt `noel` zu optimieren (Auftrag `01bbf197`). Gepusht wird weiter nur `noel`. Keine eigenen neuen Eingriffe in JAX/XLA-Interna, den Zeitnehmer oder die Imports von `benchmark.py` ohne Noels ausdrückliches Ja.
- Seit Runde 9 (21.11.) deckt der schnelle Weg Balanced-Homodyne-Detektoren und Verknüpfungen mit `jnp.sqrt` ab. Auftrag: Feyers Zusatz zu den Setup-Varianten, von Noel weitergegeben.
