# Faster Physics: technischer Nutzen, durchgerechnet

Grundlage für den Pitch vor der Jury. Stand: 22.11.2026.

Die Zeiten in den Abschnitten 1 bis 3 stammen von der Nebius-VM (Xeon Platinum 8468, ein Kern, Stand `main` `be2eb21`, Auftrag `noel-0232` vom 22.11.; die Zeiten von Feyers Kit auf unserer VM und die der verknüpften Setups stammen aus `kalt-0203` und `main-0201`, Stand `50c8008`) und aus Feyers Baseline-Berichten (`benchmarks/cpu-baseline.json`, `benchmarks/aligo-cpu-baseline.json`), gemessen auf demselben CPU-Modell mit demselben Kernel, Python und denselben Paketversionen. Die übrigen Zahlen stammen aus Philips Bericht (`philip-nacht` / `philip-fastest`). Abschnitt 8 nennt, was noch offen ist.

**Kalt und warm.** Der erste Lauf eines Programms hängt davon ab, ob Python und JAX schon im Arbeitsspeicher liegen (Datei-Cache warm) oder erst von der Platte gelesen werden (kalt). Feyers Baseline-Berichte sehen nach kalten Läufen aus: Die Imports brauchen dort 2,6 s und 1,3 s; dasselbe Kit braucht auf unserer VM für die Imports kalt 1,4 s und warm 0,30 s. Das ist unser Schluss aus den Zahlen, bestätigt hat Feyer es nicht. Verglichen wird deshalb kalt gegen kalt; der warme Wert steht daneben.

---

## Die drei Kernbotschaften

1. **Rund 35-mal schneller im gewerteten Erstlauf, kalt gegen kalt.** Er dauert 1,25 s statt 44,2 s; liegen die Bibliotheken schon im Speicher, sind es 0,24 s. Das zweite öffentliche Setup (aLIGO) läuft in 1,24 s statt 25,3 s.
2. **Eine Optimierung mit 10.000 Schritten dauert knapp 5 Minuten statt 2 Tagen.** Jeder weitere Schritt ist rund 630-mal schneller, der Gradient kostet nur 26 % eines Vorwärtslaufs.
3. **Genauer als das Original.** In einem schlecht konditionierten Testfall liegt das Original doppelt über der Toleranz. Unsere Rechnung ist dort rund 670.000-mal genauer.

---

## 1. Speedups

| Kennzahl | Baseline (Feyer) | Unsere Lösung | Faktor |
| --- | ---: | ---: | ---: |
| Erstlauf UIFO, gewertet, Datei-Cache kalt | 44.204 ms | 1.246 ms | **≈ 35×** |
| Erstlauf aLIGO (zweites Setup), Datei-Cache kalt | 25.309 ms | 1.241 ms | ≈ 20× |
| Jeder weitere Schritt, UIFO (Simulation + Gradient) | 17.679 ms | 28,0 ms | **≈ 631×** |
| Jeder weitere Schritt, aLIGO | 124,4 ms | 6,6 ms | ≈ 19× |
| Tracing + Kompilieren, UIFO, kalt | 22.918 ms | 19 ms | ≈ 1.200× |
| Erstlauf mit Bibliotheken im Speicher (Cache warm) | nicht gemessen | UIFO 236 ms, aLIGO 210 ms | – |
| Erstlauf mit verknüpften Parametern (Amplitudenmodulation, Cache warm, Stand `50c8008`) | – | aLIGO 233 ms, Voyager 253 ms | – |
| Erstlauf auf 7 Setup-Varianten (Cache warm, älterer Stand) | – | 311,9–324,7 ms | – |

- **Maschinen:** Die Baseline hat Feyer gemessen, unsere Werte stammen von der Nebius-VM mit demselben CPU-Modell (Xeon Platinum 8468), jeweils auf einem Kern. Kalt heißt bei uns: Datei-Cache vor jedem Prozess geleert, 6 Läufe, Median. Zwischen zwei Aufträgen schwankt der kalte Wert desselben Codes um etwa 3 %. Feyers unverändertes Kit braucht auf unserer VM kalt 42,6 s (UIFO) und 25,7 s (aLIGO), je ein Lauf; auf derselben Maschine gerechnet sind die Faktoren also ≈ 34× und ≈ 21×.
- **Jeder weitere Schritt** hängt nicht am Datei-Cache. Das ist die Zahl, die in einer Optimierung zählt.
- **Tracing + Kompilieren:** Die drei kleinen Programme des Benchmarks (unsere Host-Aufrufe plus Auslese und deren Ableitung) werden seit dem 22.11. nicht mehr von XLA kompiliert, sondern mit NumPy ausgeführt. Der Report zeigt deshalb 0 ms Kompilieren. Am Stand `6a05d3f`, der noch kompiliert hat, waren es kalt 282 ms, also ≈ 81×.
- **Warmer Erstlauf:** Für ihn gibt es keine passende Baseline von Feyer. Ein Faktor gegen die 44,2 s würde warm mit kalt vergleichen (das wären die früher genannten ≈ 145×).
- **Im Team-Vergleich:** Auf den Varianten ist die Lösung 2,2-mal schneller als der Branch `noel`.

---

## 2. Wo die Zeit im Erstlauf steckt (UIFO)

| Phase | kalt, ms | Anteil | warm, ms | Anteil |
| --- | ---: | ---: | ---: | ---: |
| Imports (überwiegend `import jax`) | 1.117 | 90 % | 171 | 73 % |
| Build (mit Backend-Start und Laden) | 51 | 4 % | 16 | 7 % |
| Tracing / Lowering | 19 | 1 % | 15 | 7 % |
| Kompilieren | 0 | 0 % | 0 | 0 % |
| Vorwärtslauf (Physik) | 50 | 4 % | 26 | 11 % |
| Gradient | 9 | 1 % | 7 | 3 % |
| Sensitivität und Übergabe | 4 | < 1 % | 1 | < 1 % |
| Summe | 1.247 | | 236 | |

Mediane je Phase aus dem Auftrag `noel-0232`; der Median des Gesamtlaufs liegt kalt bei 1.246 ms.

**Botschaft:** Die eigentliche Physik, also Vorwärtslauf plus Gradient, braucht im Erstlauf noch etwa 33 ms und in jedem weiteren Schritt 28 ms. Der Rest ist der Start von Python und JAX. Kalt liest der Lauf dafür 160 MB von der Platte, davon 120 MB aus einer einzigen JAX-Bibliothek (gemessen am Stand `6a05d3f`, am neuen Stand nicht wiederholt). Das zahlt jedes Programm, das JAX lädt.

---

## 3. Was das für Feyers Forschung bedeutet

Bei Feyer sucht eine KI nach neuen Detektor-Designs. Jeder Suchschritt ist eine Simulation plus Gradient. Entscheidend ist deshalb vor allem der Warmlauf.

**Annahmen:** ein CPU-Kern; Baseline mit 44,2 s im ersten und 17,7 s in jedem weiteren Schritt; unsere Lösung mit 1,25 s im ersten (kalt) und 28,0 ms in jedem weiteren Schritt.

### Optimierungslauf mit N Schritten

| Schritte | Baseline | Unsere Lösung | Kernstunden (Baseline → wir) |
| ---: | ---: | ---: | ---: |
| 100 | 30 min | 4 s | 0,50 → 0,001 |
| 1.000 | 4,9 h | 29 s | 4,9 → 0,008 |
| 10.000 | 2,0 Tage | 4,7 min | 49 → 0,08 |
| 100.000 | 20,5 Tage | 47 min | 491 → 0,78 |

### Größere Szenarien

| Szenario | Baseline | Unsere Lösung |
| --- | ---: | ---: |
| Auswertungen pro Kern und Tag | 4.887 | 3,09 Mio. |
| 100 Optimierungen à 10.000 Schritte | 205 Tage (4.912 Kernstunden) | 7,8 h (7,8 Kernstunden) |
| 1.000 verschiedene Setups, je 1 Auswertung inkl. Kompilieren | ≈ 12 h | 3,9 min |
| 1.000 Setups à 100 Schritte | 20,8 Tage | 50 min |

Bei vielen Prozessen nacheinander ist nur der erste kalt. Die beiden letzten Zeilen rechnen bei uns deshalb mit dem warmen Erstlauf von 0,24 s; die Baseline wäre warm je Prozess rund 2 s schneller als ihre 44,2 s, das ändert ihre Werte um höchstens 5 %.

**Botschaft:** Was bisher einen Rechencluster oder Wochen gebraucht hat, läuft jetzt auf einem einzelnen Kern über Nacht oder in der Mittagspause. Bei gleichem Budget kann die KI etwa 630-mal mehr Designs prüfen.

---

## 4. Genauigkeit als eigener Gewinn

| Prüfung | Ergebnis | Anteil an der Toleranz (1e-5) |
| --- | --- | ---: |
| Benchmark-Setup gegen Feyers Referenz | 5,37e-6, `PASS` | 54 %* |
| aLIGO-Benchmark gegen Feyers Referenz (Toleranz dort 1e-6) | 2,9e-11, `PASS` | 0,003 % von 1e-6 |
| 7 Setup-Varianten gegen den Original-Code | höchstens 3,6e-8, alle `PASS` | 0,36 % |
| Schlecht konditioniertes Setup (Konditionszahl 1e15), Abweichung von der exakt nachgerechneten Lösung | Original: 2e-5, unsere Lösung: 3e-11 | Original 200 %, wir ≈ 0 % |
| Zeilenweiser Rückwärtsfehler bei zufälligen rechten Seiten | SuperLU allein: 1e-3 bis 1e-1, mit unserer Vorab-Elimination: < 1e-7 | – |

\* Die 5,37e-6 stammen nicht von unseren Änderungen: Feyers eigener Baseline-Bericht nennt für den unveränderten Code denselben Wert gegen die Referenz (5,374e-6, `benchmarks/cpu-baseline.json`).

**Warum das für Feyer zählt:** Eine KI, die nach exotischen Designs sucht, landet genau in schlecht konditionierten Bereichen. Ist der Simulator dort ungenau, optimiert sie auf Rechenfehler statt auf echte Physik. Sie folgt dann gewissermaßen einem Navi mit falscher Karte. Ein genauerer Simulator verhindert solche Scheinlösungen.

---

## 5. Argumente für den Sonderpreis „überraschendste Lösung"

1. **Schneller durch weniger JAX.** Ausgerechnet in einem JAX-Projekt kam der größte Gewinn daher, die Lineare Algebra *aus* XLA herauszunehmen. Statt dichter Matrizen in JAX rechnen wir mit dünn besetzten Matrizen und einem direkten Löser, dazu eine selbst geschriebene Rückwärtsrechnung.
2. **Die Physik hat die Abkürzung geliefert.** Die Matrizen sind zu über 99 % leer, ihre Diagonale ist exakt 1 und alle anderen Einträge haben höchstens Betrag 1. Damit lassen sich die meisten Unbekannten exakt und nachweislich stabil eliminieren: beim Benchmark 751 → 200 (−73 %), bei aLIGO 82 → 20 (−76 %).
3. **Den Gradienten gibt es fast gratis.** Der Rückwärtslauf kostet 26 % eines Vorwärtslaufs (5,6 ms gegen 21,4 ms auf dem Xeon; vor der nativen Zerlegung, die den Vorwärtslauf schneller gemacht hat, waren es 16 %). Üblich ist das Ein- bis Dreifache. Der Grund: Signal und Rauschen werden über dieselben Detektor-Gewichte gelesen. Deshalb fällt die nötige Adjungierte schon im Vorwärtslauf mit ab.
4. **Eine Pivotfolge für alle 101 Frequenzen.** Alle Frequenzen haben dasselbe Besetzungsmuster. Ein eigener C-Kern zerlegt sie mit einer gemeinsamen Pivotfolge und prüft jeden Pivot an jeder Frequenz. Fällt eine Prüfung durch, wird wie bisher gerechnet.
5. **Schneller *und* genauer.** Normalerweise tauscht man Genauigkeit gegen Tempo. Hier wurde beides besser (Abschnitt 4).
6. **Saubere Methodik.**
   - Gemessen wurde im Wechsel in frischen Prozessen.
   - Fehlversuche sind dokumentiert.
   - Fast jede Maßnahme hat einen eigenen Schalter.
   - Die Build-Ausgaben sind für 27 Setups identisch (946 Arrays).
   - Getestet wurde auch mit aLIGO, Voyager und Zufalls-UIFOs; Feyers zweiter Benchmark (aLIGO, seit 21.11.) besteht ohne Anpassung.

---

## 6. Ehrlich trennen: Benchmark-Gewinn und Forschungs-Gewinn

| Bereich | Gewinn | Nutzen für die Forschung |
| --- | --- | --- |
| Numerischer Kern (Vorwärtslauf, Gradient) | jeder weitere Schritt ≈ 631× | **hoch:** zählt in jedem Optimierungsschritt |
| Kompilieren | ≈ 1.200× (kalt gegen kalt; die kleinen Programme werden interpretiert statt kompiliert) | **mittel:** zählt bei jedem neuen Setup, also bei der Suche nach neuen Topologien |
| Start-Maßnahmen (Imports, Bytecode usw.) | ≈ 144 ms, einmal pro Prozess (warm gemessen) | **gering:** 0,05 % eines Laufs mit 10.000 Schritten |

**Empfehlung:** Die Start-Maßnahmen offen als Benchmark-Optimierung benennen. Das Feyer-Team erkennt das ohnehin, und die klare Trennung macht den Rest glaubwürdiger.

---

## 7. Erwartbare Fragen der Jury

| Frage | Antwort |
| --- | --- |
| Hält das auch auf dem finalen Setup? | Auf 7 veränderten Setups alle `PASS`, Fehler höchstens 3,6e-8. Feyers zweiter Benchmark (aLIGO mit Spiegelmechanik und gequetschtem Licht) besteht mit 2,9e-11 bei Toleranz 1e-6. Die Struktur wird zur Laufzeit aus dem Setup abgeleitet, nichts ist fest einprogrammiert. |
| Läuft das auf der GPU? | Der schnelle Weg ist für die CPU gebaut, so wie der Benchmark. Der ursprüngliche JAX-Weg bleibt erhalten und lässt sich jederzeit nutzen. |
| Was wird nicht abgedeckt? | Parameter-Verknüpfungen mit einer anderen Funktion als der Wurzel, mehrere Carrier-Arbeitspunkte und mehr als ein Rauschdetektor. Diese Fälle laufen über den alten Weg: korrekt, aber langsamer. Amplitudenmodulation (Verknüpfung über die Wurzel) läuft seit dem 22.11. auf dem schnellen Weg: aLIGO 233 ms statt 21,5 s, Voyager 253 ms statt 22,8 s (Cache warm). |
| Warum zeigt der Report 0 ms Kompilieren? | Die drei kleinen Programme des Benchmarks werden mit NumPy ausgeführt statt von XLA kompiliert. Es wird nichts ausgelassen und nichts zwischengespeichert; dieselben Operationen laufen auf denselben Eingaben. Ein Schalter stellt den kompilierten Weg wieder her (`INTERPRET_SMALL_PROGRAMS=0`), dann dauerte der warme Erstlauf am Stand `1c2e618` 297 statt 246 ms. Die Ergebnisse stimmen auf 1e-14 überein. |
| Was passiert bei einer anderen JAX-Version? | Die Eingriffe sind auf jax 0.9.0 abgestimmt. Fast jede Maßnahme hat einen eigenen Schalter, der mitgelieferte Bytecode lässt sich entfernen. |
| Wie viel Speicher braucht es? | Rund 300 MB in der Spitze (UIFO, ganzer Prozess mit Validierung), gemessen in Docker auf einem Apple M4; auf dem Xeon noch nicht gemessen (siehe Abschnitt 8). Die Baseline braucht 14,4 GiB. |

---

## 8. Vor dem Pitch noch messen

| Messung | Warum | Wie |
| --- | --- | --- |
| Ob Feyer mit kaltem Datei-Cache misst | Entscheidet, ob 1,25 s oder 0,24 s die gewertete Zahl ist. Bisher ist es nur ein Schluss aus ihren Berichten | bei Feyer nachfragen |
| Ob Feyer das Interpretieren der kleinen Programme akzeptiert | Der Report zeigt 0 ms Kompilieren. Der letzte Stand, der noch kompiliert hat (`6a05d3f`), liegt kalt bei 1,56 s statt 1,25 s | bei Feyer nachfragen; Schalter `INTERPRET_SMALL_PROGRAMS` |
| Spitzenspeicher auf dem Xeon | Weniger Speicher heißt mehr parallele Läufe pro Maschine. Bei 14,4 GiB passen auf 32 GB nur 2 Läufe; bei uns sind es auf dem M4 rund 300 MB | `/usr/bin/time -v python benchmark.py --cpus 1 --repeats 3`, Zeile „Maximum resident set size" |

---

## Sätze für den Pitch

- „Eine Detektor-Optimierung mit 10.000 Schritten dauert bei uns knapp 5 Minuten statt 2 Tagen."
- „Vom kalten Start bis zum Ergebnis vergehen 1,25 Sekunden statt 44. Gut 1,1 Sekunden davon ist das Laden von JAX von der Platte."
- „Die Physik selbst braucht pro Schritt noch 28 Millisekunden. Den Rest des ersten Laufs verbringt Python mit `import jax`."
- „Der Gradient kostet uns gut ein Viertel einer Simulation. Wir haben die Struktur der Physik genutzt, nicht nur den Code beschleunigt."
- „Wir sind nicht nur schneller, sondern auch genauer: Im getesteten Fall, in dem das Original die Toleranz reißt, liegen wir bei 3e-11."
