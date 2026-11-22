# Optimierung auf Branch `noel`: Stand, Messmethode, offene Ideen

Stand: So 22.11.2026, 09:16, nach Runde 34 (letzte Runde vor dem Feature-Freeze um 10:00, nichts gebaut); davor Runde 32 (nur Abgleich: Max hat Carrier-Commit und den Patch `lazy-late` in `origin/max/next` übernommen, siehe unten). **Die Schleife läuft seit 05:16 wieder, jetzt stündlich** (Noel: «Iterier jede Stunde drüber»; davor um 04:10 beendet, Auftrag `01bbf197` gelöscht). Ab So 10:00 gilt der Feature-Freeze. `main` steht auf `a2799ba` (Code wie `73d2075`, mit Carrier-Patch). **Offen als Patch auf `noel`: `patches/philip-fastest-lazy-late.patch`** (auf Noels Ja gebaut, nicht auf `main`). Vorschlag C (Heap) liegt nur im Scratchpad, Vorschlag A baut Max auf `origin/max/next`. Xeon-Messungen für `73d2075` und den neuen Patch fehlen.

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

| [`patches/philip-fastest-rows.patch`](patches/philip-fastest-rows.patch) | Vorwärts- und Rückwärtslauf (`fast.py`): rechte Seiten und Lösungen im Layout des C-Kerns (m, F), keine Umkopier-Schritte zwischen (F, m, k) und (m, F) mehr. Basis `9a31fa1` | x86-64 emuliert mit C-Kern, 6 Läufe im Wechsel: warm Forward 18,8 → 17,1 ms, warm Gradient 4,8 → 4,5 ms, warm gesamt 24,7 → 22,6 ms. Seitenfehler im ersten Aufruf: Vorwärts 5'703 → 5'036, Rückwärts 1'997 → 1'516. Ohne C-Kern (Mac, Docker arm64) Forward gleich, Gradient-Phase 0,6 bis 1,2 ms schneller. **Xeon: noch nicht gemessen**, Schätzung 3 ms im Erstlauf | **auf `main` seit 22.11. 02:31 (`68915ee`)** |

| [`patches/philip-fastest-import.patch`](patches/philip-fastest-import.patch) | Import von `differometor` (`_tuning.py`, `fast.py`): die zwei Prüfungen von JAX-Funktionen in `_tuning.apply()` lesen den kompilierten Code statt den Quelltext (`inspect.getsource`), `_sparsetools` wird erst bei Bedarf geladen. **Nach dem Rows-Patch anwenden** (beide ändern `fast.py` und seinen Bytecode) | Docker arm64 `--cpus 1` mit Bytecode, zweimal 25 frische Prozesse je Stand: `import differometor` **3,86 → 2,68 ms**. Die beiden `getsource`-Aufrufe allein: 2,57 ms → 0,02 ms (M4). Im x86-Lauf wird `_sparsetools` nie geladen. **Xeon: noch nicht gemessen**, Schätzung 2 bis 2,5 ms | **auf `main` seit 22.11. 02:31 (`0dd3d4b`)** |

| [`patches/philip-fastest-memory.patch`](patches/philip-fastest-memory.patch) | Speicher (`fast.py`): Carrier-System ohne dichte Felder, `signal_values` und der Rückwärtslauf bilden ihre Produkte in Blöcken von 128 Zeilen statt über ganze Zwischenfelder. **Nach Rows und Import anwenden** | Seitenfehler im ersten Aufruf auf x86-64 mit C-Kern (alle drei Patches gegen `9a31fa1`): Vorwärts **5'703 → 4'028**, Rückwärts **1'996 → 1'313**. Host-Zeiten M4 warm: `signal_values` 0,52 → 0,45 ms, Rückwärts-Callback 1,68 → 1,50 ms. **Xeon: noch nicht gemessen** | **auf `main` seit 22.11. 02:31 (`712043f`)** |

| [`patches/philip-fastest-store.patch`](patches/philip-fastest-store.patch) | Vorwärtslauf (`fast.py`): Die Signal-Einträge werden direkt in die Zeilen des Wertefelds der Signalmatrizen geschrieben statt erst in ein eigenes Feld (3 MB) und von dort hinüberkopiert. **Nach Rows, Import und Memory anwenden** | Seitenfehler im ersten Vorwärts-Aufruf (x86-64, C-Kern): **4'028 → 3'562**. Zeiten an der Messgrenze: x86 emuliert, 8 Läufe, warm Forward 16,1 → 15,4 ms, Forward-Phase im Erstlauf 50,6 → 49,9 ms. Auf arm64 kein Unterschied. **Xeon: noch nicht gemessen**, Schätzung 0,5 ms | **auf `main` seit 22.11. 02:31 (`be2eb21`)** |
| [`patches/philip-fastest-carrier.patch`](patches/philip-fastest-carrier.patch) | Carrier-System (`fast.py`): mit `NATIVE_LU` zerlegt der C-Kern auch die eine Carrier-Matrix (`_SingleLU`), SuperLU wird erst bei Bedarf geladen und auf x86 gar nicht mehr. Basis `main` `e770ba2` | x86-64 emuliert, je 8 Läufe im Wechsel: Gitter-Erstlauf **470,2 → 464,5 ms** (Forward 38,9 → 34,6, Imports 331,6 → 330,4), zweite Reihe 486,2 → 478,1; aLIGO **462,8 → 455,1 ms**. Laden von SuperLU allein: 0,7 ms (arm64-Linux), 1,8 ms emuliert. Ohne C-Kern (arm64, Mac) neutral, Arrays dort bitgleich. **Xeon: nicht gemessen**, Schätzung 1,5 bis 2,5 ms | **auf `main` seit 22.11. 03:57 (`73d2075`)**, siehe «Runde 29» |
| [`patches/philip-fastest-lazy-late.patch`](patches/philip-fastest-lazy-late.patch) | Start (`_startup.py`): `jax.experimental.multihost_utils` wird vor dem Entfernen des Finders lazy angelegt (`LAZY_LATE_MODULES`), statt beim ersten `jnp.asarray` ganz ausgeführt zu werden. Basis `main` `a2799ba` | arm64-Linux: erstes `asarray` 1,29 → 0,18 ms, Importzeit gleich (88,4 → 88,2 ms, je 30 Prozesse); Build-Phase Gitter 9,0 → 7,7 ms, aLIGO 4,6 → 3,3 ms (je 14 Läufe); Erstlauf 147,0 → 145,8 und 122,7 → 120,5 ms. x86 emuliert: Build 63,0 → 60,8 und 55,0 → 52,5 ms, Gesamtzeit dort in der Streuung. **Xeon: nicht gemessen**, Schätzung 2 bis 3 ms | **von Max in `origin/max/next` übernommen (`e5d9f10`, 05:27)**, noch nicht auf `main`, siehe «Runde 32» |
| [`patches/philip-fastest-dockerfile.patch`](patches/philip-fastest-dockerfile.patch) | Nur Messumgebung (`Dockerfile`, `.dockerignore`): kein `PYTHONDONTWRITEBYTECODE` beim Installieren, `__pycache__` bleibt im Image. Unabhängig von den anderen Patches | Derselbe Code (`9a31fa1`), 6 Läufe im Wechsel, Docker arm64 `--cpus 1`: Imports **248,3 → 113,2 ms**, Erstlauf 320,0 → 187,0 ms. Ausserhalb von Docker ändert sich nichts | **offen**, siehe «Runde 19», Punkt 2 |

| Verknüpfte Parameter (Max, `origin/max/links`) | Abdeckung: Amplitudenmodulation bei aLIGO, Voyager und UIFO über den schnellen Weg | aLIGO mit Amplitudenmodulation, Erstlauf M4: 9'723 → 435 ms. Philips Xeon-Zahl: 233 ms | **auf `main` seit 22.11. 01:58 (`50c8008`)**, von Max selbst gepusht. Mein Patch dazu ist entfernt |

**Alles ausser dem Dockerfile-Patch ist auf `main`:** Build und Forward seit 21.11. 21:49 (Xeon-Zahlen unter «Runde 18»), Astras Build-Änderung seit 22.11. 01:24 (`1faf6eb`), Max' Verknüpfungen seit 01:58 (`50c8008`), `rows`, `import`, `memory` und `store` seit 02:31 (`68915ee`, `0dd3d4b`, `712043f`, `be2eb21`), `carrier` seit 03:57 (`73d2075`). Die Patch-Dateien bleiben zur Nachvollziehbarkeit liegen; sie entsprechen genau diesen Commits. Offen ist nur noch:

```bash
git show origin/noel:runs/noel/patches/philip-fastest-dockerfile.patch | git am   # nur Messumgebung: Docker-Image mit Bytecode
```

Zurücknehmen liesse sich jeder der vier Commits einzeln mit `git revert`, danach einmal `bash runs/philip/bytecode.sh python3.12` (jeder bringt den Bytecode von `fast.py` mit, `import` auch den von `_tuning.py`).

**Alle drei zusammen gegen `9a31fa1`** (frische Prozesse im Wechsel, Median / Minimum in ms, alle `PASS`, Fehler gegen die Referenz unverändert 5,3743e-06):

| Umgebung | Erstlauf | Imports | Forward | Gradient | Warm gesamt |
| --- | --- | --- | --- | --- | --- |
| x86-64 emuliert (Rosetta, C-Kern), 8 Läufe | 1'054,2 / 1'045,3 → **1'047,8 / 1'042,7** | 362,5 → 360,8 | 51,1 → 48,1 | 14,6 → 14,4 | 22,5 / 21,8 → 21,0 / 20,2 (Forward 17,3 → 15,6) |
| Docker arm64 `--cpus 1` (ohne C-Kern, mit Bytecode), 12 Läufe | 184,7 / 182,7 → **181,9 / 179,2** | 111,6 → 110,6 | 26,2 → 25,2 | 5,6 → 4,7 | 24,7 → 23,5 |
| Mac (ohne C-Kern), 12 Läufe | 185,3 / 182,7 → **183,6 / 181,3** | 118,5 → 117,1 | 21,2 → 21,8 | 3,5 → 3,3 | 21,0 → 20,6 |

Auf dem Mac steigt die Forward-Phase um 0,6 ms, weil `_sparsetools` dort jetzt im ersten Vorwärtslauf geladen wird statt beim Import (mit dem C-Kern entfällt das Laden ganz). Unter Rosetta sind die absoluten Erstlauf-Zeiten dreimal so hoch wie auf dem Xeon, weil der x86-Code erst übersetzt wird. Die Differenzen sind darum nur ein Anhaltspunkt. **Schätzung für den Xeon: 5 bis 7 ms im Erstlauf** (Lösungen ohne Kopien rund 2,4, Import rund 2,4, rund 2'400 Seitenfehler weniger rund 2,4, abzüglich Überschneidungen). Das sind rund 2 % von 318 ms.

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

**Runde 34 (22.11., 09:14 bis 09:16): Abgleich, nichts gebaut.** `main` unverändert `a2799ba`. `origin/max/next` neu auf `67f0055` (Commit von 06:54, erst jetzt gepusht): «the first use of a put-off module is safe from several threads», eine Absicherung in Max' `_startup.py`, kein neuer Hebel. Alle anderen Branches unverändert. Kein neuer Hebel im eigenen Code. Ab 10:00 gilt der Feature-Freeze: Die Schleife prüft danach nur noch den Stand. Akku 83 %.

**Runde 33 (22.11., 08:13 bis 08:15): Abgleich, nichts gebaut.** Alle Branches unverändert seit 07:13 (`main` `a2799ba`, `max/next` `0b28453`). Kein neuer Hebel im eigenen Code. Akku 87 %.

**Runde 32 (22.11., 07:13 bis 07:20): Abgleich, nichts gebaut.**

- *Stand der Branches um 07:13:* `main` unverändert `a2799ba`. **`origin/max/next` neu auf `0b28453`** (Max, 06:35), jetzt auf `main` aufgesetzt (0 Commits zurück), sonst alles unverändert. Neu darin seit `8fb61ed`:

| Commit | Inhalt |
| --- | --- |
| `37b46f6` (05:26) | Merge von `main` mit dem Carrier-Commit |
| `e5d9f10` (05:27) | **mein Patch `lazy-late`**, unverändert übernommen (Autor Noel). `LAZY_LATE_MODULES` steht damit in seinem `_startup.py` |
| `a72f0e5` (06:16) | sieben Module der Standardbibliothek lazy (`LAZY_STDLIB_MODULES`: bisect, difflib, glob, gzip, ipaddress, random, socketserver), mit `runs/max/test_lazy_stdlib.py`. Docker arm64, Erstlauf Gitter `main` 174,4 → 158,3 ms, aLIGO 140,5 → 126,8 ms (beide Commits zusammen mit `lazy-late`) |
| `0b28453` (06:35) | `_startup.py` nur einmal pro Prozess, `_mlirHlo` erst bei Bedarf (`_tuning.py`), mit `runs/max/test_package_import.py`. `import differometor` 6,67 → 5,76 ms |

- Max' Hinweis zur Messumgebung: Das Image `python:3.12-slim` hat keinen Bytecode für die Standardbibliothek, das kostet in Docker rund 150 ms pro Lauf (Gitter 320,5 gegen 174,4 ms auf `main`). Mein Dockerfile-Patch behebt nur den fehlenden Bytecode des Repos, nicht diesen Teil.
- Laut Max scheitern `runs/noel/test_pack.py` und `test_sparse.py` auf `main`. Das sind alte Skripte für den Stand von `noel`, nicht für Philips Linie; nichts zu tun.
- Der Patch `lazy-late` passt weiter auf `main`; da Max ihn in `max/next` hat, ist für `main` nichts weiter nötig, sobald `max/next` dort landet. Vorschlag C bleibt ungebaut.
- Kein neuer Hebel im eigenen Code (wie Runde 31). Akku 88 %, der Mac hängt nicht am Strom.

**Runde 31 (22.11., 06:13 bis 06:20): Abgleich, nichts gebaut.**

- *Stand der Branches um 06:13:* alles unverändert seit 04:00. `main` `a2799ba`, `max/next` `8fb61ed` (nicht auf `main`), `philip-fastest` `3c0828f`, `philip` `c29332c`, `Joel`, `sx*`, übrige `max/*` wie zuvor.
- Die offenen Patches `lazy-late` und `dockerfile` lassen sich weiter auf `main` anwenden (`git apply --check`).
- Hebel geprüft: die beiden Pläne, die jeder Vorwärtslauf baut. `EntryPlan` 0,37 ms, `SystemPlan` 0,37 ms (M4, Median aus 300), verteilt auf rund 90 kleine NumPy-Aufrufe (`unique`, `nonzero`, `searchsorted`, `sort`), kein einzelner Posten über 0,06 ms. Ein Umbau brächte höchstens 0,2 bis 0,4 ms auf dem M4, das liegt unter der Streuung der Forward-Phase im Wechselvergleich (±0,3 ms). Nicht gebaut.
- Damit kein Hebel im eigenen Code mit messbarem Gewinn. Offen bleiben die Xeon-Messungen (`73d2075`, Patch `lazy-late`) und Vorschlag C (braucht Noels Ja).
- Der Mac läuft seit einiger Zeit auf Akku (89 % um 06:13).

**Runde 30 (22.11., 04:00 bis 04:15): Patch «lazy-late», Schleife beendet.** Noel um 04:00: «Ja, B als Patch». Das ist sein Ja für diese Start-Massnahme als Patch auf `noel`, nicht für `main`.

- *Stand der Branches um 04:00:* `main` auf `a2799ba` (Max, 03:59, nur `main-pitch/SCRIPT_EN.md`), `max/next` unverändert `8fb61ed`, sonst nichts Neues.
- Anwenden: `git show origin/noel:runs/noel/patches/philip-fastest-lazy-late.patch | git am` (lokal `4ab25c8`, bringt den Bytecode von `_startup.py` mit).
- Geprüft: beide Benchmarks `PASS`, alle Arrays bitgleich gegen `main` (Mac und x86-Image). Mac: alle 15 Prüfskripte, `test_carrier.py` und neu `runs/noel/test_lazy_late.py` grün. x86-Image: alle 17 Skripte grün (Nachtrag 05:16, der Lauf war beim ersten Eintrag noch nicht fertig).
- `test_lazy_late.py`: Modul ist nach dem Kopf von `benchmark.py`, nach `jnp.asarray`, `jit` und `device_put` lazy, lädt bei Benutzung normal (`broadcast_one_to_all` rechnet); mit `DIFFEROMETOR_STARTUP` ohne `lazy` oder `DIFFEROMETOR_STARTUP_SHADOW=0` wird nichts angelegt.
- **Mit `origin/max/next`:** Max ändert `_startup.py` stark (+398 Zeilen), `_Watcher.remove` aber nicht. Nach dessen Übernahme passt der Bytecode im Patch nicht mehr: dann nur den Quelltext nehmen (`git apply --3way --include=differometor/_startup.py`) und `bash runs/philip/bytecode.sh python3.12` laufen lassen. Nicht ausprobiert.
- Vorschlag C (Heap in grossen Schritten) bleibt ungebaut im Scratchpad (`r27/vorschlag/`), Regelfrage B6 und ohne Xeon-Messung.

**Runde 29 (22.11., 03:56 bis 04:00): Carrier-Patch auf `main` gepusht.** Noel um 03:56 auf meine Frage, ob der Carrier-Patch auf `main` soll: «Ja, auf main».

- `main` stand nach `git fetch` unverändert auf `e770ba2`. Patch mit `git am` in einer sauberen Arbeitskopie darauf gesetzt: `73d2075`. Der Baum ist derselbe wie beim in Runde 27 geprüften lokalen Stand (Tree `da78f299`), alle 16 mitgelieferten `.pyc` sind gültig.
- Auf genau diesem Commit noch einmal gelaufen: `benchmark.py` und `benchmark_aligo.py` `PASS` auf dem Mac und im x86-Image mit C-Kern (dort dieselben Dateien, per Prüfsumme verglichen).
- Gepusht um 03:57 als Fast-Forward `e770ba2..73d2075`, erst nachdem ein zweites `git fetch` bestätigt hatte, dass `main` sich nicht bewegt hat. Kein Force-Push, kein Merge.
- Zurücknehmen: `git revert 73d2075`, danach `bash runs/philip/bytecode.sh python3.12` (der Commit bringt den Bytecode von `fast.py` mit).
- Für `origin/max/next` (Max, 03:36): Der Commit ändert nur `fast.py` und dessen Bytecode, die dort niemand anfasst. Ein Rebase von `max/next` auf das neue `main` sollte ohne Konflikt gehen.
- Offen: Philips Xeon-Messung für `73d2075`. Erwartung: Imports rund 1,3 ms kürzer (SuperLU wird nicht mehr geladen), Forward etwas kürzer, Ergebnisse auf x86 in den letzten Stellen anders (Sensitivity 4e-12), Fehler gegen die Referenz unverändert 5,3743e-06.
- Die Vorschläge B und C aus Runde 27 sind von dieser Freigabe nicht gedeckt und bleiben ungepusht.

**Runde 28 (22.11., 03:44 bis 03:50): Abgleich, nichts Neues gebaut.**

- *Stand der Branches um 03:44:* `main` unverändert `e770ba2`, `philip-fastest` `3c0828f`, `philip`, `Joel`, `sx*`, `max/links`, `max/interpreter`, `max/tpu-probe` unverändert. **Neu: `origin/max/next`** (drei Commits von 03:36 auf `e770ba2`): `2155055` billigere TPU-Suche beim Import (`FAST_TPU_PROBE`), `5a37c22` NumPy-Docstrings und die zwei HLO-Dialekt-Module erst bei Bedarf (`_startup.py`, +398 Zeilen, mit `runs/max/test_hlo_later.py` und `test_numpy_docs.py`), `8fb61ed` Interpreter mit `pad`, `add_any`, `sign`, `rev`, `clamp` und reellen Winkelfunktionen. Max misst in Docker arm64 z. B. Voyager mit zwei Ports 307,1 → 287,8 ms, auf x86 hat er nicht gemessen.
- **Vorschlag A aus Runde 27 ist damit zurückgezogen:** Max hat dieselbe Lücke (Gradient-Programm bei zwei Detektor-Ports) selbst geschlossen und deckt mehr ab. Sein `8fb61ed` geht vor, mein Patch dazu bleibt ungenutzt im Scratchpad. Meine Messung der Lücke (Homodyne-Variante, arm64-Linux: 159,2 → 147,4 ms; emuliert x86 rund 450 ms) bestätigt seinen Befund unabhängig.
- Vorschlag B (`multihost_utils` bleibt lazy) und C (Heap in grossen Schritten) berühren ebenfalls `_startup.py`. Sie überschneiden sich inhaltlich nicht mit `max/next`, die Patches müssten aber nach dessen Übernahme neu aufgesetzt werden. Beide warten weiter auf Noels Ja.
- Nachtrag zu Runde 27: Der Versuchsstand mit allen drei Vorschlägen hat im emulierten x86-Image alle 16 Prüfskripte bestanden (15 von Philip und Max, dazu `test_carrier.py`).
- Der Carrier-Patch (`patches/philip-fastest-carrier.patch`) berührt nur `fast.py` und passt unverändert auf `main` und auf `max/next` (dort ändert niemand `fast.py`).
- Kein weiterer Hebel im eigenen Code mit messbarem Gewinn: Was bleibt (Pläne pro Aufruf, Pivot-Suche im C-Kern, Rausch-Summen), liegt je unter 0,5 bis 1 ms und damit unter der Streuung des Wechselvergleichs.

**Runde 27 (22.11., 02:49 bis 03:45): Carrier-Patch, drei Vorschläge mit Zahlen.** Noel um 02:40: «kannst du noch was weiter verbesser». Basis `main` `e770ba2`, andere Branches um 03:29 unverändert.

*Behalten: `patches/philip-fastest-carrier.patch`* (lokal `cdfdd44`, Anwenden: `git show origin/noel:runs/noel/patches/philip-fastest-carrier.patch | git am`).

- Der C-Kern nimmt alle 18 Test-Carrier-Systeme an (Fixture, aLIGO in drei Modi, Voyager, 13 Zufallsgitter). Residuum höchstens 4e-16 (SuperLU 1,5e-16), Abstand zur dichten Lösung wie bei SuperLU (Fixture 1,0e-11 bei Kondition 3,7e5).
- Arrays auf x86 gegen `main`: Gitter Carrier 5,0e-13, Signal 1,3e-09, Noise 1,2e-09, Sensitivity 3,5e-12, Gradient 3,1e-11, Fehler gegen die Referenz 5,374278e-06 → 5,374282e-06. aLIGO höchstens 1,2e-11, Fehler 2,88e-11 → 2,99e-11. Auf dem Mac alles bitgleich.
- Geprüft: beide Benchmarks `PASS` (Mac, x86-Image), alle 15 Prüfskripte auf dem Mac und im x86-Image, neu `runs/noel/test_carrier.py` (Genauigkeit, beide Rückfälle, SuperLU bleibt ungeladen), zehn Setup-Varianten im x86-Image `PASS` mit unveränderten Fehlern. Bytecode von `fast.py` neu, alle 16 `.pyc` gültig.
- Nebenwirkung: Auf dem gewerteten Weg ist die Regelfrage A8 (SuperLU direkt aus SciPy laden) gegenstandslos.

*Verworfen (mit Zahlen):*

| Versuch | Ergebnis |
| --- | --- |
| Früher Abbruch der Pivot-Suche im C-Kern (gleiche Pivotfolge) | Pivotfolgen und Ergebnisse auf 18 Setups identisch, aber nur 279'486 → 270'038 geprüfte Einträge (3 %). Die Suchzeit (1,9 von 14 ms warm) steckt in Schritten, deren billige Kandidaten die Stabilitätsprüfung nicht bestehen. Philips `libpattern_lu.so` lässt sich mit gcc 13.3.0 (Ubuntu 24.04) bitgleich nachbauen |
| Rausch-Summen ohne Zwischenkopien | bitgleich, im Einzelprofil 3'533 → 3'027 Seitenfehler, im Wechselvergleich kein Gewinn (Forward schneller, Gradient-Phase 1,3 ms langsamer) |
| `cos` + `sin` statt komplexem `exp` in den Bauteil-Formeln (66'862 Aufrufe pro Vorwärtslauf) | ohne AVX-512 gleich schnell und bitgleich (12,1 gegen 11,7 ns, emuliert 31,7 gegen 30,8). NumPy hat für AVX-512 eigene Schleifen (`DOUBLE_sin_X86_V4`), auf dem Xeon wären es vielleicht 1,5 ms. Hier nicht messbar, darum nicht gebaut |

*Drei Vorschläge, nur gemessen, nicht ausgeliefert* (Regel: Eingriffe in Start, Imports und JAX-nahe Teile nur mit Noels Ja). Lokale Commits auf dem Carrier-Stand, Patches und `test_vorschlag.py` im Scratchpad (`r27/vorschlag/`), alle 16 Prüfskripte auf dem Mac grün:

| Vorschlag | Datei | Gemessen | Haken |
| --- | --- | --- | --- |
| A. Interpreter kennt `pad` und `add_any` | `_interpret.py` (Max) | **Fund:** Bei zwei Detektor-Ports (Balanced Homodyne) fällt das Gradient-Programm aus dem Interpreter und geht durch MLIR und XLA. Homodyne-Variante, arm64-Linux, je 8 Läufe: Erstlauf 159,2 → **147,4 ms** (Gradient-Lower 6,1 → 1,3, Kompilieren 7,3 → 0). Emuliert kostet die Lücke rund 450 ms. Schätzung Xeon: 15 bis 20 ms | wirkt nur bei Homodyne-Setups, die zwei offiziellen Benchmarks lesen einen Port |
| B. `jax.experimental.multihost_utils` bleibt lazy | `_startup.py` (Philip) | Das Modul steht in `LAZY_MODULES`, wird aber beim ersten `jnp.asarray` doch ausgeführt, weil der Finder dann schon weg ist. Erstes `asarray` 1,8 bis 2,1 → 0,3 ms, Build-Phase 9,8 → 8,3 ms (arm64-Linux). Schätzung Xeon: 3 ms | Import-Massnahme |
| C. Heap in grossen Schritten (`mallopt`, dazu `MADV_HUGEPAGE`) | `_startup.py` (Philip) | Seitenfehler eines Laufs (x86 emuliert): Imports 15'072 → 11'053, Forward 3'702 → 99, Gradient 1'227 → 22. Erstlauf arm64-Linux 159,7 → **151,1 ms**, ohne Huge Pages 158,0 → 152,4; emuliert 482,3 → 467,9 | Regelfrage B6 (`mallopt`). Wirkung hängt von der Huge-Page-Einstellung des Xeon ab, **braucht eine Messung dort**. Philips Fehlversuch 5 hatte nur die Schwellen, ohne grosse Schritte |

Weiter offen und nicht gebaut: MLIR-Bindungen beim Import (rund 9 ms in Docker, tiefer Eingriff), JAX-eigene jit-Funktionen beim Tracing direkt aufrufen (0,7 ms auf dem M4).

**Runde 26 (22.11., 02:28 bis 02:33): die vier Patches auf `main` gepusht.** Noel um 02:27 auf meine Frage, ob ich sie pushen soll: «Ja, auf main pushen».

- `main` stand unverändert auf `c97def5` (gegenüber `50c8008` nur Philips Pitch-Dokument). Die vier Patches mit `git am` darauf gesetzt: `68915ee` (rows), `0dd3d4b` (import), `712043f` (memory), `be2eb21` (store). Der Code ist bis auf das Pitch-Dokument identisch mit dem in Runde 25 vollständig geprüften Stand, alle 16 mitgelieferten `.pyc` sind gültig.
- Auf genau diesem Commit noch einmal gelaufen: `benchmark.py` und `benchmark_aligo.py` `PASS` auf dem Mac (Arrays gegen `main` davor: Gradient 4,7e-17 und 1,3e-17, sonst bitgleich) und im x86-Image mit C-Kern (**alle Arrays bitgleich**), dazu `test_reverse`, `test_forward_cache`, `test_singular`, `test_startup`, `test_homodyne`, `test_tuning`, `test_links`, `test_interpret` und `check_setups.py`.
- Gepusht um 02:31 als Fast-Forward `c97def5..be2eb21`, nur nachdem `git fetch` bestätigt hatte, dass `main` sich nicht bewegt hat. Kein Force-Push, kein Merge.
- Offen: Philips Messung auf dem Xeon für `be2eb21`. Erwartung aus den Messungen hier: Forward und Gradient zusammen rund 3 ms schneller unter Linux arm64, 2'459 Seitenfehler weniger auf x86 (rund 1 µs pro Stück auf dem Xeon), `import differometor` rund 1,2 ms kürzer in Docker. Bei aLIGO neutral.

**Runde 25 (22.11., 02:13 bis 02:24): `main` hat Interpreter und Verknüpfungen, die vier Patches sind darauf umgestellt.**

*Stand der Branches um 02:13* (seit `1faf6eb`):

| Commit auf `main` | Wer, wann | Inhalt |
| --- | --- | --- |
| `5f06f5a` | Max, 01:27 | Pitch-Skript, kein Code |
| `1c2e618` | Max, 01:40 | **`INTERPRET_SMALL_PROGRAMS`**: `differometor/_interpret.py` ersetzt `jax.stages.Traced.lower` für kleine Programme. Die drei Programme des Benchmarks (1, 18 und 14 Gleichungen) werden mit NumPy ausgeführt statt über MLIR und XLA kompiliert. Alles andere nimmt JAX' Weg. Bei Max in Docker arm64: Gitter 369,2 → 342,0 ms, Lower 16,8 → 8,8, Kompilieren 20,7 → 0. Offene Regelfrage laut seinem Commit: Der Report zeigt 0 ms Kompilieren |
| `50c8008` | Max, 01:58 | seine Verknüpfungen (`origin/max/links`) auf `main`. Quelltext identisch zu dem, worauf meine Serie schon aufsetzte |
| `c97def5` | Philip, 02:14 | Pitch-Dokument mit den Xeon-Zahlen des Stands `50c8008`: Erstlauf mit warmem Datei-Cache **Gitter 250 ms, aLIGO 215 ms**, mit verknüpften Parametern aLIGO 233 ms und Voyager 253 ms, kalt 1,30 s, jeder weitere Schritt 34,4 ms. «Die eigentliche Physik braucht im Erstlauf noch etwa 42 ms» |

`origin/max/interpreter` (`1c2e618`) ist in `main` enthalten. `origin/philip-fastest` steht auf `3c0828f` (01:55, «draft»): `_prefetch.bin`, `runs/philip/prefetch_map.py` und 87 Zeilen in `_startup.py`, vermutlich ein Vorab-Einlesen von Dateien für den kalten Start. Das ist Philips Baustelle und nicht auf `main`. `origin/philip`, `max/tpu-probe`, `Joel`, `sx*` unverändert. Auf meine Frage von 01:28, ob die Serie auf `main` soll, gibt es noch keine Antwort; gepusht habe ich darum nur `noel`.

*Serie umgestellt.* Max' Verknüpfungen sind auf `main`, `patches/philip-fastest-links-max.patch` ist darum entfernt. Die vier Patches sind auf `50c8008` neu gebaut (Quelltext wieder ohne Konflikt, auch `_tuning.py`, das Max für den Interpreter geändert hat; Bytecode pro Schritt neu) und lassen sich mit `git am` auf `c97def5` anwenden, der Bytecode ist danach gültig.

*Geprüft auf `50c8008` plus den vier Patches:* alle fünfzehn Skripte (elf unter `runs/philip`, `test_tuning`, `check_setups`, `runs/max/test_links.py`, `runs/max/test_interpret.py` mit 35 Fällen), `test_variants.py` komplett (21 von 21 über den schnellen Weg, WORST 4,95e-07 und 2,41e-08). Beide Benchmarks `PASS` auf dem Mac, in Docker arm64 und im x86-Image mit C-Kern. Arrays gegen `main`: auf x86 **alles bitgleich, auch der Gradient**, ohne C-Kern Gradient 1,0e-16 (Gitter) und 7,4e-18 (aLIGO).

*Gemessen, `50c8008` gegen `50c8008` plus vier Patches* (frische Prozesse im Wechsel, Median / Minimum in ms; der Rechner war ruhiger als in den Runden 22 bis 24, Last um 4):

| Umgebung | Erstlauf | Imports | Forward | Gradient | warm |
| --- | --- | --- | --- | --- | --- |
| Docker arm64 `--cpus 1`, Gitter, 14 Läufe | 174,9 / 149,0 → **168,5 / 148,5** | 118,8 → 116,4 | 29,4 / 26,8 → 27,3 / 24,9 | 5,3 / 4,8 → 4,1 / 3,6 | 26,5 → 26,1 |
| Docker arm64, aLIGO, 14 Läufe | 137,9 / 129,6 → 135,5 / 121,6 | 117,2 → 115,1 | 5,2 / 4,8 → 5,5 / 4,9 | 1,7 / 1,5 → 1,5 / 1,3 | 4,8 → 4,8 |
| Mac, Gitter, 16 Läufe | 153,1 / 141,7 → 150,0 / 141,6 | 111,9 → 107,3 | 21,4 / 20,6 → 22,2 / 20,9 | 3,0 / 2,8 → 2,9 / 2,5 | 21,06 → 20,50 |
| Mac, aLIGO, 16 Läufe | 138,5 / 123,7 → 132,9 / 121,8 | 117,6 → 112,0 | 4,9 / 4,3 → 5,8 / 5,2 | 1,2 → 1,2 | 3,97 → 3,93 |
| x86-64 emuliert (C-Kern), Gitter, 10 Läufe, unruhig | 732,9 / 569,5 → 720,0 / 550,5 | 522,8 → 503,0 | 62,7 / 50,3 → 57,1 / 43,8 | 19,9 / 14,7 → 19,4 / 14,0 | 36,1 / 26,2 → 34,5 / 23,1 |

Seitenfehler im ersten Aufruf auf x86 mit C-Kern: Vorwärts **5'233 → 3'549**, Rückwärts **2'087 → 1'312**. Ohne C-Kern (Mac, arm64) lädt `_sparsetools` jetzt im ersten Vorwärtslauf statt beim Import, darum dort 0,3 bis 0,9 ms mehr im Forward und entsprechend weniger im Import; mit C-Kern wird es gar nicht geladen. Seit dem Interpreter ist der Erstlauf auf dem Mac nur noch rund 145 ms lang (Imports 105, Lower 8, Kompilieren 0), der Anteil der eigenen Rechnung ist damit gewachsen: Die geschätzten 5 bis 7 ms der Serie auf dem Xeon sind jetzt 2 bis 3 % von 250 ms.

**Runde 24 (22.11., 01:13 bis 01:28): Astras Build-Änderung auf `main` gepusht, Serie auf `main` umgestellt.**

*Stand der Branches um 01:13:* `main` war auf `19132db` (01:09, Philip): «Fast path: reflectivity exactly 1 or 0 no longer spoils other gradients». `sqrt` und `arcsin` haben bei Reflektivität 1 oder 0 eine unendliche Steigung, über 0 · ∞ wurde auch das `tuning` desselben Spiegels `nan`. Neu ist `fast._chain` und `runs/philip/test_singular.py`, Fund des Teams SudeepJoshi21. Xeon (`chain-0056`): Gitter 297,6 → 296,9 ms, aLIGO 259,1 → 258,7 ms. Um 01:17 kam `74587ea` dazu (Max, `main-pitch/SCRIPT_EN.md`, kein Code). `philip-fastest` steht weiter auf `e22821d`: **Philip arbeitet jetzt direkt auf `main`**, die Patches setzen darum auf `main` auf. `origin/max/links`, `max/tpu-probe`, `philip`, `Joel`, `sx*` unverändert. Beim Pushen meldet GitHub, das Repo sei nach `mpb700/feyer-muc` umgezogen (der alte Name leitet weiter).

*Astra.* Das ist die zweite Sitzung (Codex) auf dem lokalen Branch `astranuelle` (21 Commits auf `noel@3dae6c2`, nicht auf dem Server). Sie hat auf dem **`noel`-Stand** 14 Versuche gemessen (`runs/astranuelle/NOTIZEN.md` dort) und zwei behalten. Noel um 01:18: «Astra hat nochmals gecooked, probier seine Ansätze auf den Main Branch zu pushen/integrieren». Was das für `main` heisst:

| Astras Ansatz | Auf `noel` gemessen | Auf `main` |
| --- | --- | --- |
| Nr. 4, Dualzahl-Regeln für konstante Operanden (behalten) | Erstlauf 339,7 → 334,6 ms, Gradient 10,2 → 9,5 ms | **Schon da.** Das sind die Regeln aus meinem Vorwärts-Patch, die Philip am 21.11. um 21:49 als `9a31fa1` übernommen hat. Astra hat sie von dort nach `noel` zurückgetragen |
| Nr. 7, Ausgabezeilen der Carrier- und Rausch-Komponenten in einem Schritt (behalten) | `pairs_to_arrays` 1,19 → 1,06 ms | **Übertragen und gepusht** (`1faf6eb`). Auf `main` ist die Funktion schon schneller, der Gewinn dort: 0,58 → 0,53 ms |
| Nr. 13, Detektor-Pullback nur auf belegten Zeilen (von Astra selbst wieder herausgenommen) | Pullback 1,00 → 0,66 ms | Nicht nötig: `main` rechnet den Pullback des Readouts schon nur für die gewählten Ports (`HOST_DETECTOR_PORTS`) |
| Elf weitere Versuche | von Astra verworfen | nichts zu übertragen |

Astras eigenes Fazit für seine Auswahl: 440,9 → 438,0 ms auf dem M4, «der Zeitgewinn für den ganzen Prozess liegt an der Messgrenze», unter Linux arm64 kein bestätigter Gewinn. Für `main` bleibt davon die Build-Änderung.

*Gepusht auf `main`: `1faf6eb` «Build step: output rows of carrier and noise components in one step»* (Fast-Forward von `74587ea`, 01:24). `append_information` merkt sich pro Komponente nur noch die Zahl der Ausgabespalten, `combine_arrays` baut alle Zeilennummern mit einem `np.repeat`. Gemessen und geprüft:

- M4, fünf Prozesse im Wechsel mit je 400 Aufrufen: `pairs_to_arrays` 0,577 bis 0,588 ms → **0,528 bis 0,537 ms** (in jedem Paar schneller), `run_build_step` 3,42 → 3,40 ms. Im ganzen Erstlauf ist das nicht sichtbar (0,05 ms von rund 190).
- `runs/noel/test_build.py`: alle 946 Arrays aus 27 Setups identisch zu `main` davor. Alle dreizehn Skripte unter `runs/philip` und `runs/flags` plus `check_setups.py` laufen durch. `benchmark.py` und `benchmark_aligo.py`: `PASS`, alle Arrays bitgleich, auf dem Mac, in Docker arm64 und im x86-Image mit C-Kern. Bytecode von `build.py` neu erzeugt.

*Serie auf `main`.* Die fünf offenen Patches sind auf `19132db` neu gebaut (Max' Quelltext und meine vier per 3-Wege-Anwendung ohne Konflikt, Bytecode pro Schritt neu, nur der der geänderten Module) und lassen sich mit `git am` auf `1faf6eb` anwenden. Endstand geprüft: alle vierzehn Skripte (elf unter `runs/philip` samt `test_singular.py`, `test_tuning`, `check_setups`, `runs/max/test_links.py`), `test_variants.py` komplett (21 von 21 über den schnellen Weg, WORST 4,95e-07 und 2,41e-08), beide Benchmarks `PASS`, Arrays gegen `main`: Carrier, Signal, Noise, Sensitivity, Powers bitgleich, Gradient 3,8e-16 (Gitter) und 1,3e-17 (aLIGO) auf dem Mac. Nachgeholt um 01:27 für genau diesen Stand (`1faf6eb` plus die fünf Patches): in Docker arm64 und im x86-Image mit C-Kern beide Benchmarks `PASS`, auf x86 **alle Arrays bitgleich zu `main`, auch der Gradient** (arm64: Gradient 1,0e-16 und 6,5e-18). Im x86-Image laufen ausserdem aLIGO mit Amplitudenmodulation über den schnellen Weg, `test_singular.py` und `test_reverse.py` (WORST 3,6e-12).

**Runde 23 (22.11., 00:55 bis 01:08): Max hat die Verknüpfungen zeitgleich portiert, Serie auf seinen Commit umgestellt.**

*Stand der Branches um 00:55:* `philip-fastest` und `main` auf `e22821d` (00:49, Philip): nur `technical-pitch/PITCH.md` gegenüber `6a05d3f`, kein Code. **Neu: `origin/max/links` (`8877134`, 00:48, auf `6a05d3f`)**, «Fast path covers parameter links with jnp.sqrt (amplitude modulation)», laut Commit «ported from Noel's round twelve». `origin/philip`, `max/tpu-probe`, `Joel`, `max/lu-once`, `sx*` unverändert. Im gemeinsamen Checkout arbeitet weiter die zweite Sitzung auf `astranuelle` (dort uncommittete Änderungen an `components.py` und `fast.py`, nicht meine).

*Doppelte Arbeit.* Max und ich haben in derselben halben Stunde dieselbe Lücke geschlossen (er 00:48, ich 00:53), beide mit meiner Lösung aus `noel`. Verglichen (`git diff origin/max/links` gegen meinen Commit): bis auf Kommentare gleich, mit drei Unterschieden.

| Punkt | Max (`8877134`) | mein Port (Runde 22) |
| --- | --- | --- |
| Homodyne-Phase, die über eine Verknüpfung an einer Designvariable hängt | abgedeckt: `phase_is_design_variable` wird dann gesetzt, der Kotangent des Auswahlvektors fliesst | **fehlte**: der Gradient über die Phase wäre in diesem Fall stillschweigend weggefallen. Kein öffentliches Setup verknüpft eine Phase, trotzdem ein echter Fehler |
| Erkennen der Verknüpfungsfunktion | nur über die Identität (`function is jnp.sqrt`) | zusätzlich über die Funktion hinter dem jit-Objekt. Unter Philips `_LazyJit` reicht die Identität, beide Seiten greifen auf dasselbe Objekt zu |
| Eigener Test | `runs/max/test_links.py`: aLIGO und Voyager in Amplituden- und Space-Modulation gegen `simulate_jax`, dazu laut Commit ein Angriffsversuch durch ein zweites Modell | keiner, nur `check_setups.py` und `test_variants.py` |

Darum gilt Max' Fassung. `patches/philip-fastest-links.patch` ist aus `noel` entfernt (im Verlauf: `f227f11`).

*Max' Commit nachgeprüft* (auf `e22821d` gesetzt): sein Bytecode ist gültig, `runs/max/test_links.py` läuft durch (WORST ln s 1,14e-06 bei Voyager mit Amplitudenmodulation an den Konstruktorwerten, wo sich die zwei Detektorports auf 1e-5 aufheben, sonst 1e-12 bis 1e-14; Gradient 9,1e-08), `benchmark.py` und `benchmark_aligo.py` `PASS` und bitgleich zu `6a05d3f`.

*Serie umgestellt.* Die vier Geschwindigkeits-Patches sind auf `e22821d` plus Max' Commit neu gebaut (Quelltext per 3-Wege-Anwendung ohne Konflikt, Bytecode pro Schritt neu, nur der der geänderten Module). Sie lassen sich mit `git am` der Reihe nach auf `origin/max/links` anwenden und ergeben dieselben Dateien wie meine Arbeitskopie.

Endstand (`e22821d`, Max' Verknüpfungen, `rows`, `import`, `memory`, `store`), geprüft:

- Mac: alle dreizehn Skripte (elf unter `runs/philip` und `runs/flags`, `check_setups.py`, `runs/max/test_links.py`) mit unveränderten Kennzahlen. `runs/noel/test_variants.py` komplett: alle 21 Varianten über den schnellen Weg, WORST 4,95e-07 und 2,41e-08, die zwei mit Amplitudenmodulation auf 1,1e-14 und 3,6e-15.
- Beide Benchmarks `PASS` auf Mac, Docker arm64 und x86-64 emuliert mit C-Kern. Arrays gegen Philips Stand: auf x86 **alles bitgleich, auch der Gradient**, ohne C-Kern Gradient 3,8e-16 (Gitter) und 1,3e-17 (aLIGO).
- aLIGO mit Amplitudenmodulation im x86-Image mit C-Kern: schneller Weg, gegen `simulate_jax` ln s 1,2e-07 und Gradient 1,7e-07 (im Rauschen der dichten Referenz, siehe Runde 22).
- Zeiten: Der Rechner war die ganze Runde fremd belastet (Last 5 bis 9). Docker arm64, 14 Läufe im Wechsel, Philips Stand gegen den Endstand, nur die Minima sind brauchbar: Gitter Erstlauf 218,7 → 195,6 ms, Forward 32,5 → 29,3 ms, Gradient 7,0 → 6,1 ms. aLIGO Erstlauf 192,5 → 185,8 ms, Forward 6,5 → 6,7 ms, warm 5,6 → 5,7 ms. Das passt zu den sauberen Messungen der Runden 19 und 20 (Gitter schneller, aLIGO neutral).

*Lehre:* Vor einem Port erst `git fetch` und die Branches der anderen ansehen reicht nicht, wenn zwei gleichzeitig arbeiten. Max' Branch entstand, während ich schon baute. Nächstes Mal vor dem Export des Patches noch einmal holen.

**Runde 22 (22.11., 00:25 bis 00:54): verknüpfte Parameter im schnellen Weg, Serie auf `6a05d3f` neu aufgesetzt.** (Mein Port von hier ist seit Runde 23 durch Max' Branch ersetzt.)

*Stand der Branches um 00:25:* `philip-fastest` und `main` stehen auf `6a05d3f` (00:21): Philip hat die sieben offiziellen Dateien aus `83f8d2f` selbst übernommen (bytegleich nachgeprüft) und auf dem Xeon gemessen (Auftrag `aligo-0017`, 12 Läufe): **aLIGO 258,9 ms im Erstlauf, warm 7,2 ms, Fehler 2,9e-11** (das offizielle Kit dort 24,28 s und 128,5 ms), Gitter 298,5 ms mit dem neuen Harness. Damit ist auch meine Frage nach dem Abgleich erledigt. `origin/philip`, `max/tpu-probe`, `Joel`, `max/lu-once`, `sx*` unverändert.

*Die Lücke.* Mit aLIGO als zweitem öffentlichem Beispiel liegt eine Variante nahe, die bisher niemand im Blick hatte: `aligo(mode="amplitude_modulation")`. Dort hängt die Signalamplitude über eine Verknüpfung an der Laserleistung (`amplitude=("L0_power", jnp.sqrt)`). Dasselbe gilt für `uifo(..., mode="amplitude_modulation")`. `fast.supports` hat jedes Setup mit einer Verknüpfung abgelehnt (bei Philip unter «Offen» vermerkt, `check_setups.py` erwartete dort ausdrücklich den JAX-Weg). Folge: Der Lauf geht durch den dichten JAX-Weg. Gemessen mit der Pipeline von `benchmark.py` (Skript im Scratchpad, `linkrun.py`), aLIGO mit Amplitudenmodulation, Stand `6a05d3f`: **Erstlauf 9'723 ms auf dem M4** (davon 9'402 ms Kompilieren), warm 192 ms. Auf dem Xeon wären das rund 25 s wie bei der Baseline. Frequenzmodulation und Space-Modulation liefen schon über den schnellen Weg.

*Behalten: `patches/philip-fastest-links.patch`.* Das ist die Lösung aus `noel` (Runde 9, `8cb5eff`), in Philips `fast.py` übertragen:

- `LINK_RULES` nennt die Verknüpfungsfunktionen, die der schnelle Weg auf dem Host auswertet: `jnp.sqrt` mit Wert `np.sqrt` und Ableitung `0,5 / y`. Erkannt wird die Funktion über ihre Identität oder über die Python-Funktion hinter dem jit-Objekt (wichtig wegen Philips `_LazyJit`).
- `EntryPlan._row` wendet die Verknüpfungen auf die Parameterzeile an, nach den Designvariablen und vor den Sweeps. Alle Verknüpfungen lesen die Zeile, bevor eine schreibt, wie `simulate_jax`. Die Phase eines Homodyne-Detektors wird danach gelesen.
- Ein verknüpfter Platz reicht Ableitungen an seinen Quellplatz weiter (`differentiable`), und `pullback` verschiebt seinen Kotangenten mit der Ableitung der Verknüpfungsfunktion dorthin.
- `supports` lässt Verknüpfungen zu, wenn alle ihre Funktionen in `LINK_RULES` stehen und die Funktionsindizes beim Tracen bekannt sind (`_links_supported`). Jede andere Verknüpfungsfunktion läuft weiter über `simulate_jax`.
- `HOST_KEYS` bekommt die drei Index-Arrays der Verknüpfungen (bei Philip `Constant`, also keine zusätzlichen Operanden). `runs/check_setups.py` erwartet für aLIGO mit Amplitudenmodulation jetzt den schnellen Weg.

| Messung | `6a05d3f` | mit dem Patch |
| --- | ---: | ---: |
| aLIGO, `amplitude_modulation`, Erstlauf M4 (Weg) | 9'723 ms (JAX) | **435 ms** (schnell) |
| dort warm | 192 ms | 10 ms |
| dasselbe in Docker arm64 `--cpus 1` (Rechner fremd belastet, alle Zeiten rund doppelt so hoch wie üblich) | 84'770 ms | 1'253 ms |
| Zufalls-UIFO 3×3 mit Amplitudenmodulation, ein Auswertungslauf (`test_variants.py`, M4) | 8,4 s (JAX) | 0,07 s |
| Gitter-Benchmark, Docker arm64, 14 Läufe im Wechsel, Erstlauf Median / Minimum (fremd belastet) | 478,9 / 441,1 ms | 477,2 / 418,1 ms |

Geprüft:

- `runs/check_setups.py`, aLIGO mit Amplitudenmodulation, alle 46 Parameter als Designvariablen, gegen `simulate_jax`: Carrier 6,7e-13, Signal 5,4e-12, Noise 3,1e-12, Sensitivity 1,8e-08, Gradient 1,7e-09. Die anderen drei Setups dort unverändert.
- `runs/noel/test_variants.py` komplett: **alle 21 Varianten über den schnellen Weg** (vorher 19), die zwei mit Amplitudenmodulation mit ln s auf 1,1e-14 und 3,6e-15, Gradient 2,3e-15 und 1,9e-15 gegen `simulate_jax`. WORST unverändert 4,95e-07 und 2,41e-08.
- Mit der Benchmark-Pipeline gegen `simulate_jax` (aLIGO, Amplitudenmodulation): M4 ln s 7,4e-08, Gradient 1,6e-08. x86-64 emuliert mit dem C-Kern 1,2e-07 und 1,7e-07, Docker arm64 1,5e-07 und 1,4e-07. Zur Einordnung: Zwei Auswertungen desselben JAX-Codes (mit und ohne `jit`) weichen bei diesem Setup auf dem M4 selbst um 1,0e-07 in ln s voneinander ab. Die Unterschiede liegen also im Rauschen der dichten Referenz.
- Alle elf Skripte unter `runs/philip` und `runs/flags` laufen durch, mit denselben Kennzahlen wie vorher. `benchmark.py` und `benchmark_aligo.py`: `PASS`, alle Arrays **bitgleich zu `6a05d3f`** (Setups ohne Verknüpfung nehmen denselben Weg wie vorher).
- Voyager mit Amplitudenmodulation (zwei Verknüpfungen plus Homodyne-Detektor, nicht in `check_setups.py`): schneller Weg, Carrier 4,9e-13, Signal 6,1e-12, Noise 1,2e-12 gegen `simulate_jax`. Sensitivity und Gradient weichen um 5,7e-06 und 1,0e-06 ab, weil sich das Amplitudenrauschen im balancierten Readout fast aufhebt (Differenz zweier fast gleicher Zahlen). Bei Voyager mit Frequenzmodulation, das schon vorher über den schnellen Weg lief, hebt es sich ganz auf: Die Felder stimmen auf 1e-09, die Sensitivity ist in beiden Implementierungen Rundungsrauschen. Solche Setups taugen nicht als Referenz, das hat mit dem Patch nichts zu tun.
- Nicht prüfbar war ein UIFO mit Amplitudenmodulation gegen `simulate_jax` in Docker: Die dichte Referenz braucht mehr als die 8 GB der Docker-VM. Im x86-Image mit dem C-Kern läuft dieses Setup über den schnellen Weg durch (683 Unbekannte, 197 Variablen, endlich), der Vergleich mit JAX stammt vom Mac.

*Serie neu aufgesetzt.* Weil jeder Patch den Bytecode von `fast.py` mitbringt, hängt die Reihenfolge fest. Ich habe darum die ganze Serie auf `6a05d3f` neu gebaut, mit dem Verknüpfungs-Patch zuerst (der wichtigste, und allein anwendbar): `links`, `rows`, `import`, `memory`, `store`. Die Quelltext-Änderungen sind dieselben wie in den Runden 19 und 20 (per 3-Wege-Anwendung ohne Konflikt übernommen), der Bytecode ist pro Schritt neu erzeugt (`runs/philip/bytecode.sh`, Python 3.12.15). `_startup.cpython-312.pyc` und `argparse.cpython-312.pyc` erzeugt das Skript hier mit anderem Inhalt als Philips eingecheckte Dateien (andere 3.12-Version); diese beiden habe ich nicht angefasst, die Patches enthalten nur den Bytecode der geänderten Module. Nachgeprüft: Alle sechs Patches lassen sich mit `git am` der Reihe nach auf `6a05d3f` anwenden und ergeben denselben Baum wie meine Arbeitskopie, der mitgelieferte Bytecode ist danach gültig.

Endstand der Serie (`6a05d3f` plus `links`, `rows`, `import`, `memory`, `store`):

- Mac: alle zwölf Skripte (elf plus `check_setups.py`), beide Benchmarks `PASS`, Arrays gegen `6a05d3f` bitgleich bis auf den Gradienten (3,8e-16 beim Gitter, 1,3e-17 bei aLIGO, ohne C-Kern).
- Docker arm64 und x86-64 emuliert mit C-Kern: beide Benchmarks `PASS` (Fehler 5,374e-06 und 2,9e-11), auf x86 **alles bitgleich zu `6a05d3f`, auch der Gradient**.
- Gitter-Benchmark, Docker arm64, 14 Läufe im Wechsel unter Fremdlast: `6a05d3f` 478,9 / 441,1 ms, nur `links` 477,2 / 418,1 ms, alle fünf **459,4 / 430,3 ms** (Forward 69,8 → 66,3 ms, warm 66,1 → 61,0 ms). Die absoluten Werte sind wegen der zweiten Sitzung rund 2,3-mal zu hoch, verlässliche Zeiten stehen in den Runden 19 und 20.

*Für Philip, falls er nur eines übernimmt:* den Verknüpfungs-Patch. Er ändert nichts am Tempo der beiden öffentlichen Benchmarks, schliesst aber den einzigen mir bekannten Fall, in dem eine naheliegende Variante der öffentlichen Beispiele um den Faktor 20 bis 100 langsamer wird. Weiter offen bleiben mehrere Carrier-Arbeitspunkte, mehr als ein Rauschdetektor und `df.run(...)` ohne Designvariablen. Die öffentlichen Konstruktoren (`aligo`, `voyager`, `uifo` in allen drei Modi) erzeugen keinen dieser Fälle.

**Runde 21 (22.11., 00:14 bis 00:25): neuer offizieller Benchmark aLIGO, auf Noels Hinweis aus dem Discord.** Nichts am Code geändert, nur gelesen und gemessen.

*Was neu ist.* Das offizielle Repo `feyerlabs/start-hack-munich-2026` steht seit 21.11. 23:40 auf `83f8d2f` («Add aLIGO benchmark and single-CPU timing baseline», vorher `2bb437a`). Geändert: `benchmark.py`, `plot_timings.py`, `README.md`, `UPSTREAM.json`. Neu: `benchmark_aligo.py`, `benchmarks/aligo-cpu-baseline.json` und `.png`. Fixtures, `requirements.txt`, `start_here.py` und der Simulator sind unverändert.

| Punkt | Gitter (bisher) | aLIGO (neu) |
| --- | --- | --- |
| Aufruf | `python benchmark.py --cpus 1` | `python benchmark_aligo.py --cpus 1 --output benchmark-aligo.json` |
| Setup | gespeichertes UIFO `uifo_800_3000.json` | `differometor.setups.aligo(mode="space_modulation")`, vereinfachtes Advanced LIGO mit Spiegelmechanik und gequetschtem Licht |
| Frequenzen | 101, 800 bis 3000 Hz | 101, 20 bis 5000 Hz (`examples/data/aligo_test_data.json`, lag schon im Kit) |
| Designvariablen | 193, gekoppelte Gitterlängen | 29: Reflektivität, Tuning, `db` und `angle` des Squeezers, Laserleistung, Massen, unabhängige Längen |
| Toleranz für ln(Sensitivity) | 1e-5 | **1e-6** |
| Gradient-Check | `mb11.tuning` | `sq1.db` |
| Baseline Xeon 8468, Erstlauf / warm | 44,2 s / 17,7 s | 25,3 s / 0,124 s |
| Systemgrösse | Carrier 286, Signal 751 | Carrier 36, Signal 82 |

`benchmark_aligo.py` ist nur ein Einstieg: Es ruft `main(setup_name="aligo")` aus `benchmark.py` auf und misst als Import-Phase `from benchmark import main`. `benchmark.py` importiert im Kopf jetzt zusätzlich `from copy import deepcopy` und `aligo` aus `differometor.setups`. Das gebaute aLIGO-Setup ist per SHA-256 festgenagelt (`ALIGO_SETUP_SHA256`). Das README nennt aLIGO «a second setup to check whether your improvements generalize». Der Satz zur Schlussbewertung ist unverändert («a slight variation of the supplied example»).

*Unser Stand damit* (offizielle `benchmark.py`, `benchmark_aligo.py` und `plot_timings.py` in eine Arbeitskopie gelegt, sonst nichts geändert):

| Stand | Umgebung | Gitter | aLIGO | max. Fehler aLIGO (Grenze 1e-6) |
| --- | --- | --- | --- | ---: |
| `9a31fa1` | Mac | `PASS` | `PASS` | 2,88e-11 |
| `9a31fa1` plus vier Patches | Mac, Docker arm64, x86-64 emuliert (C-Kern) | `PASS` | `PASS` | 2,86e-11 bis 2,88e-11 |
| `aeb1925` (Philips neuer Stand) | Mac, Docker arm64, x86-64 emuliert (C-Kern) | `PASS` | `PASS` | 2,86e-11 bis 2,88e-11 |
| `aeb1925` plus vier Patches | Mac, Docker arm64, x86-64 emuliert (C-Kern) | `PASS` | `PASS` | 2,86e-11 bis 2,88e-11 |

- aLIGO läuft über den schnellen Weg (auf x86 mit dem C-Kern, `_sparsetools` wird nicht geladen). Das Setup-Hash stimmt, der Gradient-Check an `sq1.db` passt (Autodiff −1,290661026e-02, Differenzen −1,290661025e-02). Reserve zur Toleranz: mehr als vier Grössenordnungen.
- Zeiten aLIGO auf dem Mac (`9a31fa1` plus Patches, warmer Datei-Cache): Erstlauf 181 ms, davon Imports 134, Build 5,2, Lower und Kompilieren 31, Forward 6,1, Gradient 1,0. Warm 3,7 ms (Baseline auf dem Xeon 124 ms). Der Erstlauf besteht also zu über 90 % aus Imports und Kompilieren, die bei beiden Benchmarks gleich sind.
- Meine vier Patches auf aLIGO, Docker arm64 `--cpus 1`, 12 Läufe im Wechsel: Erstlauf 187,5 → 186,3 ms, Forward 5,4 → 5,9 ms, Gradient 2,7 → 2,6 ms, warm 4,9 → 5,1 ms. Auf arm64 lädt `_sparsetools` jetzt im ersten Vorwärtslauf statt beim Import, daher die 0,5 ms mehr im Forward. Auf x86-64 emuliert mit C-Kern, 10 Läufe: Forward 35,6 → 35,1 ms, warm gesamt 6,41 → 6,25 ms. Also neutral, wie bei einem so kleinen System zu erwarten. Arrays gegen den Stand ohne Patches: auf x86 bitgleich, auf arm64 Gradient 6,5e-18.
- Gitter mit dem neuen offiziellen `benchmark.py`, Docker arm64, 10 Läufe: Erstlauf 234,6 / 200,1 → 223,9 / 196,1 ms, Forward 33,3 → 28,8 ms, Gradient 7,7 → 5,9 ms (der Rechner war dabei zeitweise fremd belastet, siehe unten).

*Was das Team tun muss.* `main` und `philip-fastest` haben noch die alten offiziellen Dateien (bytegleich zu `2bb437a`), `benchmark_aligo.py` fehlt dort. Für die Abgabe müssen sieben Dateien unverändert aus `83f8d2f` kommen: `benchmark.py`, `benchmark_aligo.py`, `plot_timings.py`, `README.md`, `UPSTREAM.json`, `benchmarks/aligo-cpu-baseline.json`, `benchmarks/aligo-cpu-baseline.png`. Den letzten Abgleich hat Philip gemacht (`11e96b4`). Ich fasse `benchmark.py` laut Abmachung nicht an und habe darum keinen Patch dafür gebaut. Dazu passend: `_BENCHMARK_HEADER` in `_startup.py` (nur für `PREIMPORT`, das aus ist) kennt den neuen Kopf noch nicht, und `runs/philip/test_startup.py` prüft die Kopf-Module am alten Kopf.

*Philips neuer Stand `aeb1925`* (22.11. 00:10, `philip-fastest` und `main`, enthält auch Joels Pitch-Dokument): `DEFER_REPORT_MODULES`. `importlib.metadata`, `statistics`, `hashlib` und `subprocess` aus dem Kopf von `benchmark.py` laufen erst bei der ersten echten Nutzung, also hinter den Zeitnehmern (bei ihm «Art (C)»). JAX' Frage nach Plugins (`entry_points(group="jax_plugins")`) wird ohne das Modul beantwortet. Der Träger der Start-Massnahmen ist jetzt `argparse.py` statt `statistics.py`, weil `argparse` der erste Import im Kopf ist. Xeon, Auftrag `report-0003`, 16 Läufe: **Erstlauf 317,3 → 299,1 ms**, Imports 190,9 → 174,1 ms. Damit ist meine Regelfrage aus Runde 19 (Punkt 5) erledigt: Philip hat es gebaut, mit Schalter. Mit dem neuen offiziellen Kopf funktioniert es (beide Benchmarks `PASS`, auch über `benchmark_aligo.py`, wo `benchmark` als Modul importiert wird).

*Meine Patches und `aeb1925`.* Alle fünf (`rows`, `import`, `memory`, `store`, `dockerfile`) lassen sich mit `git am` unverändert auf `aeb1925` anwenden, der mitgelieferte Bytecode ist danach gültig. Geprüft wie oben: beide Benchmarks `PASS` in allen drei Umgebungen, Arrays gegen `aeb1925` auf x86 bitgleich.

*Ein Risiko, das der neue Benchmark sichtbar macht.* Feyer baut seine Läufe als Varianten des Harness. Die Start-Massnahmen hängen an der Datei `argparse.py` neben `benchmark.py`: Sie wirken nur, wenn der Lauf aus dem Wurzelverzeichnis unseres Repos startet. Nimmt Feyer für die Schlussbewertung nur unser `differometor/` in ein eigenes Verzeichnis oder startet ein eigenes Skript von woanders, laufen die Massnahmen nicht. Die Ergebnisse bleiben richtig, der Erstlauf wäre aber rund 110 ms langsamer (Philips Reihe `tuning-2220`: 593,9 ms ohne alles, 489,5 ms ohne `_tuning`, 315,8 ms mit allem; `_tuning` hängt nicht am Träger, die Import-Massnahmen schon). Die `.pth`-Datei aus `pip install -e .` wäre der Träger, der vom Startverzeichnis unabhängig ist, sie ist bei Philip bewusst aus. Das ist eine Entscheidung für Philip und Noel, nicht für mich.

*Zustand des Rechners.* Seit 00:20 arbeitet eine zweite Sitzung (Codex) im selben Checkout, hat dort den Branch `astranuelle` angelegt (ein Commit auf `3dae6c2`) und lässt Prüfungen laufen (`runs/noel/test_variants.py`, Last 8). Meine Mac-Zeiten ab 00:20 sind darum unbrauchbar, die `PASS`-Prüfungen nicht. Ich committe `noel` seitdem über eine eigene Arbeitskopie im Scratchpad (`noel-wt`) und lasse den Checkout in Ruhe. Der Laptop hängt seit 00:15 wieder am Netz.

**Runde 20 (22.11., 00:05 bis 00:14): ein weiterer, kleiner Patch.** Basis unverändert `9a31fa1`, gearbeitet auf den drei Patches aus Runde 19.

*Stand der Branches um 00:05:* `philip-fastest` unverändert `9a31fa1`, die Patches aus Runde 19 sind dort noch nicht angewendet. `main` steht auf `218dc22` (22:48): `9a31fa1` plus zwei Commits von Joel, die nur `technical-pitch/PITCH.md` anlegen, kein Code. Neu ist `origin/max/tpu-probe` (`99a000e`, 23:18, auf `9a31fa1`): Max ersetzt in `_startup.py` die Suche nach einem Google-TPU, die `import jax` über `/sys/bus/pci/devices` macht (`FAST_TPU_PROBE`). Gemessen hat er auf einer RunPod-Maschine mit 780 PCI-Geräten (Imports 209,5 → 184,3 ms), auf dem Xeon 8468 noch nicht, dort hängt der Gewinn an der Zahl der PCI-Geräte. Das berührt nur `_startup.py` und dessen Bytecode, also keinen meiner Patches. `origin/philip` steht auf `c29332c` (22:44) mit dem Auftrag `teams-2240`: drei öffentliche Repos anderer Teams auf dem Xeon mit demselben `benchmark.py`, Erstlauf im Median **702,9 ms** (`sudeep`, mit eigenen XLA-Flags), **747,0 ms** (`vv-v2`, eigener C++-Löser) und **861,4 ms** (`akshat`) gegen **316,3 ms** für unseren Stand. Alle `PASS`, Ergebnis-Arrays untereinander auf 1e-9 gleich. `Joel`, `max/lu-once` und `sx*` unverändert.

*Behalten: `patches/philip-fastest-store.patch`.* `_EntryBlock.run` hat die Signal-Einträge in ein eigenes Feld (1'893 Einträge mal 101 Frequenzen, 3 MB) geschrieben, das `signal_values` gleich danach in das Wertefeld der Signalmatrizen kopiert hat. Jetzt legt `_run_forward` das Wertefeld zuerst an (`SystemPlan.signal_store`, mit den konstanten Einträgen), und der Eintragsblock schreibt direkt in dessen Zeilen (`run(store=...)`). Das geht, wenn jeder geschriebene Eintrag eine eigene Zeile hat (`update_live.all()`, der Normalfall), sonst bleibt der alte Weg. `forward` bekommt statt des Feldes ein `_StoredEntries`. Braucht jemand das Wertefeld ein zweites Mal (die native Zerlegung nach einer instabilen Pivotfolge), wird es aus den Einträgen neu aufgebaut. Wer `forward` oder `signal_values` mit einem Feld von Einträgen aufruft (alle Tests, der zustandslose Rückwärts-Callback), bekommt das alte Verhalten.

| Messung | drei Patches | plus dieser |
| --- | ---: | ---: |
| Seitenfehler im ersten Vorwärts-Aufruf, x86-64 emuliert mit C-Kern (zweimal gezählt, identisch) | 4'028 | **3'562** |
| x86-64 emuliert, 8 Läufe im Wechsel: Forward-Phase im Erstlauf, Median / Minimum | 50,6 / 48,9 ms | 49,9 / 48,4 ms |
| dort warm Forward | 16,06 / 15,51 ms | 15,44 / 14,77 ms |
| dort warm gesamt | 21,9 / 21,6 ms | 21,2 / 20,0 ms |
| Docker arm64 `--cpus 1`, 14 Läufe: Forward-Phase | 26,6 / 25,8 ms | 26,7 / 25,5 ms |
| Mac, 14 Läufe: Forward-Phase | 22,4 / 21,7 ms | 22,3 / 21,6 ms |

Auf arm64 ist nichts messbar, auf x86 unter Emulation rund 0,6 ms im warmen Vorwärtslauf. Das ist der kleinste der vier Patches und der einzige, dessen Zeitgewinn an der Messgrenze liegt. Behalten, weil die Seitenfehler-Zählung eindeutig ist und der Vergleich auf x86 in Median und Minimum besser ausfällt. Geprüft: alle zwölf Skripte unter `runs/philip` und `runs/flags` sowie `check_setups.py` mit unveränderten Kennzahlen (darunter `test_forward_cache.py` mit verdrängten Einträgen), `test_variants.py --quick` (WORST 1,99e-11 und 3,02e-12), `variant_run.py` `PASS` im x86-Image (Homodyne-Fixture, Zufalls-UIFO 1) und im arm64-Image (Homodyne-Fixture). Arrays auf dem Mac bitgleich zum Stand davor, im x86-Image bitgleich zu `9a31fa1`. Der Patch enthält den neu erzeugten Bytecode von `fast.py`.

*Was jetzt noch bleibt* (erster Vorwärts-Aufruf, x86 mit C-Kern, 3'562 Seitenfehler): Ergebnisfelder der drei Lösungen 874 (nötig), Dual-Produkte 606 (das Band für den Rückwärtslauf), Wertefeld und Platzierung 762 (nötig bis auf Reste), Rausch-Teil 660 (davon rund 440 durch umsortierte Kopien und ein Zwischenfeld vermeidbar, je rund 0,2 ms), C-Kern 255. Darüber hinaus sehe ich im eigenen Code keinen Hebel mehr über 0,3 ms auf dem Xeon. Die grösseren Posten sind Regel- oder Hardwarefragen: `importlib.metadata` (rund 9 ms, Noels Entscheid steht aus), die Pivotsuche im C-Kern (nur auf x86 messbar) und Max' TPU-Suche (seine Baustelle).

**Runde 19 (21.11., 22:20 bis 23:52): Noels Auftrag «nochmal alles geben, innovative Ansätze, testen und verwerfen».** Basis `9a31fa1` (`philip-fastest` und `main`, unverändert). Drei Patches behalten, zwei Mess-Erkenntnisse, eine Regelfrage.

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

*2. Messfehler in Docker gefunden.* Das `Dockerfile` setzt `PYTHONDONTWRITEBYTECODE=1`, bevor `pip install` läuft. pip übersetzt dann keine `.pyc`, und jeder Lauf im Container kompiliert JAX, NumPy und alles andere neu. Dazu schliesst `.dockerignore` `**/__pycache__` aus, der mitgelieferte Bytecode von `differometor` fehlt im Image also auch (18 ms pro Lauf). Gemessen (6 Läufe im Wechsel, `9a31fa1`): Imports **258,3 ms ohne, 116,0 ms mit Bytecode**, Erstlauf 336,9 gegen 189,4 ms. Alle bisherigen Docker-Zahlen haben also einen um rund 150 ms zu hohen Import-Anteil. Die Vergleiche zwischen zwei Ständen stimmen trotzdem, weil beide gleich betroffen waren. Auf Philips VM (Git-Checkout, pip mit Bytecode) gibt es das Problem nicht. Für Messungen ab jetzt: Dockerfile ohne die Variable beim Installieren und ohne den Ausschluss von `__pycache__` (so gebaut: `differometor-bench:pf5-*`). Das ist als `patches/philip-fastest-dockerfile.patch` umgesetzt: im `Dockerfile` fällt `PYTHONDONTWRITEBYTECODE` weg, in `.dockerignore` die Zeile `**/__pycache__`. Nachgemessen mit dem geänderten Dockerfile des Repos: 15 gültige mitgelieferte `.pyc` im Image, JAX mit Bytecode, Imports 248,3 → 113,2 ms. Wichtig auch für den Fall, dass jemand die Abgabe über das Dockerfile nachmisst.

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

*7. Behalten: `patches/philip-fastest-memory.patch`* (auf Rows und Import). Die drei grössten vermeidbaren Posten aus der Tabelle unter 4:

- **Carrier-System.** Bisher: dichtes Feld (n, n+1) mit den Basiswerten, eine Kopie davon pro Vorwärtslauf, Einträge hineinschreiben, an den Musterpositionen wieder herauslesen (zweimal 1,3 MB für 286 Unbekannte, dazu die Maske). Jetzt hält `SystemPlan` das Muster (Schlüssel `Spalte * n + Zeile`, wie beim Signal-System), die Basiswerte in CSC-Reihenfolge und die Plätze der geschriebenen Einträge als kurze Arrays. `carrier_matrix` kopiert zwei kurze Vektoren und schreibt die Einträge hinein. Muster und Werte sind dieselben wie vorher, die Zerlegung also bitgleich.
- **`signal_values`.** Die Platzierungs-Einträge werden `-pre * carrier`. Das Vorzeichen sitzt jetzt auf dem kurzen Carrier-Faktor, und die Zeilen werden in Blöcken von 128 geschrieben. Das Zwischenfeld eines Blocks (200 kB) wird vom nächsten wiederverwendet, statt dreimal 1,6 MB frisch anzufassen. `(-a)·c` und `a·(-c)` sind bitgleich.
- **Rückwärtslauf.** `np.take(right, cols)` für alle besuchten Einträge auf einmal (2,1 MB) und `signal * alpha` als Ganzes (1,2 MB) laufen jetzt in Blöcken von 128 Zeilen (`_ROW_BLOCK`). Dieselben Operationen, nur zeilenweise gruppiert.

Seitenfehler im ersten Aufruf (x86-64 emuliert, C-Kern, je ein Prozess): Vorwärts 5'703 (`9a31fa1`) → 5'036 (Rows) → **4'028** (alle drei), Rückwärts 1'996 → 1'516 → **1'313**. Die Zählung schwankt je nach Zustand des Allokators um einige Hundert (mit dem Import-Patch allein waren es vorwärts 5'460). Docker läuft mit THP `always`, Ubuntu auf einer VM üblicherweise mit `madvise`: Dort dürften es eher mehr Seitenfehler sein, die Einsparung also eher grösser. Geprüft: alle zwölf Skripte unter `runs/philip` und `runs/flags` sowie `check_setups.py` mit unveränderten Kennzahlen, `test_variants.py --quick` (WORST 1,99e-11 und 3,02e-12), `variant_run.py` `PASS` im x86-Image (Homodyne-Fixture, Zufalls-UIFO 0, veränderte Werte) und im arm64-Image (Homodyne-Fixture). Arrays: auf dem Mac bitgleich zum Stand davor, im x86-Image **bitgleich zu `9a31fa1`**. Zusätzlich im emulierten x86-Image mit dem C-Kern auf dem Endstand gelaufen (22.11., 00:04): `test_reverse.py` (WORST 3,6e-12, Grenze 1e-9, darin die allgemeine Adjungierte über `solve_rows` und die Varianten ohne `kept`) und `test_forward_cache.py` (auch der zustandslose Weg unter `vmap`). `test_homodyne.py` und die übrigen Skripte habe ich dort nach 8 Minuten abgebrochen: Sie vergleichen mit dem dichten JAX-Weg, dessen Kompilierung unter Rosetta sehr lange dauert, und der Laptop lief auf Akku (22 %). Auf dem Mac sind sie alle durchgelaufen.

*8. Sonst gemessen.* Im C-Kern kostet die Analyse (Pivotfolge) 3,0 ms gegen 2,3 ms Numerik (emuliert): Die Markowitz-Suche besucht 279'000 Einträge für 379 Pivots, eine auf wenige Zeilen und Spalten begrenzte Suche wäre der übliche Ausweg (Philips Code, hier nicht messbar).

*Stand der Branches um 23:13:* `philip-fastest` und `main` unverändert `9a31fa1`. `origin/philip` neu auf `6f26dec` (22:31), nur Ergebnisdateien: `v2-2209` (das andere Team mit neuem Stand `final/combined-v2` und eigenem C++-Löser: 763,9 ms gegen unsere 321,1 ms auf dem Xeon, Arrays auf 1e-9 gleich), `tuning-2220` (Schalter einzeln: nur bitgleiche Kompilieroptionen 377,6 ms, ohne XLA-Optionen 455,3, `_tuning` aus 489,5, auch Start-Massnahmen aus 593,9 gegen 315,8 ms mit allem) und `schalter-2226` (`INLINE_DISPATCH` und `SHORT_LOCATIONS` bringen nichts, dazu eine Aufschlüsselung der Import-Phase). Philip misst also gerade selbst an den Imports. `Joel`, `max/lu-once` und `sx*` unverändert.

**Nächste Hebel für Philips Linie:** (a) Einträge mit Ableitungen (`_EntryBlock.run` und die Dual-Produkte, zusammen noch rund 1'450 Seitenfehler im ersten Vorwärtslauf), (b) die Rausch-Summen ohne umsortierte Kopien (`_group_sum`, rund 220), (c) begrenzte Pivotsuche im C-Kern (Analyse 3,0 ms emuliert, nur auf x86 messbar), (d) die Regelfrage zu `importlib.metadata` (rund 9 ms auf dem Xeon, siehe 5). Was nach den drei Patches im ersten Vorwärtslauf an Seitenfehlern bleibt: Ergebnisfelder der Lösungen 875, `run` 844, Dual-Produkte 606, `signal_values` 484, Rausch-Teil 660, C-Kern 257.

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
| `origin/max/interpreter` | `1c2e618` (22.11. 01:40), in `main` enthalten: kleine Programme werden mit NumPy ausgeführt statt kompiliert | auf `main` |
| `origin/max/links` | seit 01:58 als `50c8008` auf `main`. `8877134` (22.11. 00:48), neu, auf `6a05d3f`: verknüpfte Parameter (`jnp.sqrt`) im schnellen Weg, Max' Port meiner Lösung aus `noel`, mit `runs/max/test_links.py`. Siehe «Runde 23» | nicht übernommen (es ist Philips Linie). Meine vier Geschwindigkeits-Patches setzen darauf auf |
| `origin/max/tpu-probe` | `99a000e` (21.11. 23:18), neu, auf `9a31fa1`: billigere TPU-Suche beim Import von JAX (`FAST_TPU_PROBE` in `_startup.py`), siehe «Runde 20» | nicht übernommen (Philips Linie, Import-Massnahme). Verträgt sich mit meinen Patches |
| `origin/philip` | `c29332c` (21.11. 22:44): dazu `teams-2240` (drei andere Teams, siehe «Runde 20»). Davor `6f26dec` (21.11. 22:31): dazu die Aufträge `v2-2209`, `v2-arrays-2210`, `tuning-2220` und `schalter-2226` (siehe «Runde 19»). Davor `3983044` (21.11. 21:59): Ergebnisdateien der VM-Aufträge `konkurrenz-2050`, `main-2122` und `patches-2145`. Davor `8435944` (21.11. 15:28). Neu seit `ab82e42`: `b3676de`, `5934330` und `8435944` (nur die Ergebnisdateien der drei VM-Messreihen, kein Code) | Am Code des Branches selbst ändert sich nichts. Neu ist der Ordner `_nacht/` mit Philips Nachtlauf `philip-nacht` als Git-Bundles, seinen Notizen und Messungen auf einer VM mit Xeon Platinum 8468 (siehe unten). Nichts übernommen: Sein Stand nutzt Mittel, die bei uns unter die Regelfrage B6 und die Abmachung «keine Caches» fallen. Das entscheidet Noel |
| `origin/max/lu-once` | `4d5b43d` (21.11. 13:54), unverändert am 21.11. 18:36. Neu seit `79ae580`: `f5a176d` (statischer Bauplan für den JAX-Weg, `differometor/plan.py`), `4d5b43d` (Nachtbericht und Messdaten) | nicht übernommen. Max' Nachtlauf nennt `noel@6b67cfd` als besten Stand und findet «keinen Hebel ab 5 ms ohne Regelfrage». `plan.py` beschleunigt den dichten JAX-Weg, den wir nach Runde 9 für UIFO-Setups nicht mehr brauchen |
| `origin/max/nacht` | `f1000e9` (21.11. 03:25), am 21.11. 15:06 auf dem Server gelöscht (der Commit steckt in `origin/max/lu-once`) | zusammengeführt in `7279114`. Übernommen: nachgeladene Module, `_EntryBlock.back`, In-place-Produkte, `signal_values` ohne Zwischenkopie. Ersetzt durch den Stand von `noel`: Spaltenordnung `MMD_ATA` (7'916 Faktoreinträge gegen 6'100) |
| `origin/Joel` | `9f79a5f` (21.11. 17:53), neu seit `76e799f`: nur das README (Verweis auf das neue Tag `joels-fastest` = `ac7aedc`). Baut auf Philips `ab82e42` auf | nichts zu übernehmen. Joels drei Code-Änderungen (Host-Detektoren, Plotly bei Bedarf, SuperLU ohne SciPy-Import) hat `noel` seit Runde 1 in eigener Fassung |
| `origin/sx`, `origin/sx2`, `origin/sx2-env` | `2c9e8bf`, `036194a`, `bf6bf60` (21.11. 03:33 bis 04:00), unverändert am 21.11. 18:36 | nicht übernommen. Das sind Joels vorbereitete Varianten mit XLA-Flags bzw. `jax_disable_most_optimizations`, laut seinem README erst nach Zusage von Feyer zu aktivieren. Gleiche Regelfrage wie unser B6 |
| `origin/main` | `73d2075` (22.11. 03:57): dazu der Carrier-Patch, auf Noels Ja gepusht, siehe «Runde 29». Davor `e770ba2` (22.11. 02:39): Philips Pitch-Zahlen für `be2eb21` (Gitter 236 ms, aLIGO 210 ms). Davor `be2eb21` (22.11. 02:31): dazu meine vier Geschwindigkeits-Patches, auf Noels Ja gepusht, siehe «Runde 26». Davor `c97def5` (22.11. 02:14): dazu Max' Interpreter für kleine Programme (`1c2e618`), Max' Verknüpfungen (`50c8008`) und Philips Pitch-Zahlen, siehe «Runde 25». Davor `1faf6eb` (22.11. 01:24): Astras Build-Änderung, von mir auf Noels Anweisung gepusht. Davor `74587ea` (01:17, Max, Pitch-Skript) und `19132db` (01:09, Philip, `fast._chain` für Reflektivität 0 und 1). Siehe «Runde 24». Davor `e22821d` (22.11. 00:49, gleich `philip-fastest`): nur das Pitch-Dokument. Davor `6a05d3f` (22.11. 00:21): dazu der Abgleich mit dem offiziellen Repo `83f8d2f` (aLIGO-Benchmark), siehe «Runde 22». Davor `aeb1925` (22.11. 00:10): dazu Philips `DEFER_REPORT_MODULES`, siehe «Runde 21». Davor `218dc22` (21.11. 22:48): `9a31fa1` plus Joels `technical-pitch/PITCH.md`, kein Code. Davor `9a31fa1` (seit 21.11. 21:49, gleich `philip-fastest`, also mit Balanced Homodyne und beiden Patches). Vorher `50341f9` (17:30), davor das unveränderte Kit `5eece1d` | nicht übernommen (Regelfragen B4 bis B6, Abmachung «keine Caches»). Das unveränderte Kit ist weiter `5eece1d` |
| `origin/philip-fastest` | `3c0828f` (22.11. 01:55, «draft»): Philips `19132db` plus ein Entwurf zum Vorab-Einlesen von Dateien (`_prefetch.bin`, `_startup.py`). Nicht auf `main`. Davor `e22821d` (22.11. 00:49): nur `technical-pitch/PITCH.md`. Davor `6a05d3f` (22.11. 00:21): offizielle Dateien aus `83f8d2f` übernommen, Xeon: aLIGO 258,9 ms, Gitter 298,5 ms. Davor `aeb1925` (22.11. 00:10): vier Kopf-Module von `benchmark.py` laufen erst bei Bedarf, Träger `argparse.py`, Xeon 317,3 → 299,1 ms. Davor `9a31fa1` (21.11. 21:49): `8e7c4f1` plus die beiden Patches aus `runs/noel/patches/` (`13bd58d`, `9a31fa1`). `main` plus weitere Start-Massnahmen (315,3 ms auf dem Xeon), seit `708803d` mit Balanced Homodyne im schnellen Weg, seit `11e96b4` mit den Dokumenten des offiziellen Repos | nicht übernommen. Der Patch aus Runde 12 ist hier überholt |

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
- Noel am 22.11. um 05:15: «Iterier jede Stunde drüber». Neue stündliche Schleife (Minute :07), `caffeinate` läuft wieder. Regeln wie bisher, Basis ist `origin/main`.
- Noel am 22.11. um 04:10: «Hör danach auf mit der Schleife, es reicht». Auftrag `01bbf197` gelöscht, keine weiteren Runden.
- Noel am 22.11. um 04:00: «Ja, B als Patch» (ein JAX-Modul bleibt lazy). Freigabe für den Patch auf `noel`, nicht für `main`.
- Noel am 22.11. um 03:56: «Ja, auf main», auf die Frage nach dem geprüften Carrier-Patch. Ausgeführt um 03:57 (`e770ba2..73d2075`). Das war eine Freigabe für genau diesen Commit, nicht für die Vorschläge B und C aus Runde 27.
- Noel am 22.11. um 02:27: «Ja, auf main pushen», auf die Frage nach den vier geprüften Geschwindigkeits-Patches. Ausgeführt um 02:31 (`c97def5..be2eb21`). Das war eine Freigabe für genau diese vier Commits. Für alles Weitere gilt wieder: nur `noel` pushen, `main` nur auf ausdrückliche Anweisung.
- Noel am 22.11. um 01:18: «Astra hat nochmals gecooked, probier seine Ansätze auf den Main Branch zu pushen/integrieren». Das ist eine Ausnahme von «nur `noel` pushen» für genau diese Sache. Ausgeführt als ein Fast-Forward-Commit auf `main` (`1faf6eb`). Für Max' Verknüpfungen und meine vier Patches gilt sie nicht, die bleiben Patches, bis Noel oder Philip etwas anderes sagt.
- Noel am 21.11. um 21:02: Die Schleife baut Patches für Philips Stand (`philip-fastest`) statt `noel` zu optimieren (Auftrag `01bbf197`). Gepusht wird weiter nur `noel`. Keine eigenen neuen Eingriffe in JAX/XLA-Interna, den Zeitnehmer oder die Imports von `benchmark.py` ohne Noels ausdrückliches Ja.
- Seit Runde 9 (21.11.) deckt der schnelle Weg Balanced-Homodyne-Detektoren und Verknüpfungen mit `jnp.sqrt` ab. Auftrag: Feyers Zusatz zu den Setup-Varianten, von Noel weitergegeben.
