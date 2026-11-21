# Fragen an Feyer

Stand: Sa 21.11.2026, 15:25, Branch `noel` (Philips Branch plus Max' `lu-once` und `nacht` plus zehn Runden vom 21.11.).
Zu stellen am Sa 21.11. bei der Challenge-Präsentation (11–12 Uhr) oder direkt beim Mentor.

## Unser Stand in drei Sätzen (zum Erklären vor der Frage)

- Der Rechenkern läuft nicht mehr in JAX/XLA, sondern in NumPy/SciPy: Sparse-LU mit SuperLU, eingebunden über `jax.pure_callback`. Den Gradienten liefert eine handgeschriebene Rückwärtsregel (`jax.custom_vjp`).
- Das gilt für Setups mit einem Rausch-Detektor, einfach oder Balanced Homodyne (seit Runde 9). Alle anderen laufen weiter über den Original-JAX-Code.
- Gemessen auf einem Apple M4: Original 12,8 s → 0,43 s auf macOS und 0,50 s in Docker (Linux, `--cpus 1`, aarch64). Überall `PASS`, auch auf x86-64 (dort bisher nur emuliert geprüft). Eine Zeitmessung auf echter x86-Hardware fehlt noch.

## Schon beantwortet (Discord, 21.11., von Noel weitergegeben)

Im Discord nur mitlesen: Noel hat festgelegt, dass wir dort nichts schreiben und keine Fragen stellen.

| Frage | Antwort von Feyer | Folge für uns |
| --- | --- | --- |
| C3: andere Setups? | «The final evaluation will use a similar optical setup like the one included in the GitHub repo, but with slightly different parameters and components. Your simulator submissions should still work with that slight setup variation.» | Der schnelle Weg deckt seit Runde 9 auch Balanced-Homodyne-Detektoren ab. Der Zufallsgenerator, aus dem das Fixture stammt, wählt sie in der Hälfte der Fälle. Nachgestellt mit 14 Varianten, alle `PASS` bei 0,45 bis 0,49 s (Mac). Siehe `runs/noel/NOTIZEN.md`, Runde 9 |
| C3, im offiziellen README (feyerlabs `2bb437a`, 21.11. 14:26) | «Final evaluation uses a different optical setup, a slight variation of the supplied example.» Die Preise gelten für den «complete first-run total on the final optical setup», korrekt heisst «on both the development example and the final setup» | Wie oben. Auf `noel` und seit 19:51 auf `philip-fastest` abgedeckt, auf `main` noch nicht |
| Zweiter Benchmark (offizielles Repo `83f8d2f`, 21.11. 23:40, im Discord angekündigt) | `benchmark_aligo.py`: vereinfachtes Advanced LIGO mit Mechanik und gequetschtem Licht, 101 Frequenzen von 20 bis 5000 Hz, 29 Designvariablen, Toleranz 1e-6. Gedacht, «to check whether your improvements generalize». Baseline auf dem Xeon 25,3 s | Unser Stand besteht ihn über den schnellen Weg (`PASS`, Fehler 2,9e-11). Die offiziellen Dateien müssen noch in `main` nachgezogen werden. Siehe `runs/noel/NOTIZEN.md`, Runde 21 |
| B1, B2 (Teil): wie viele CPUs? | Jonathan Klimesch, 14:30: eine logische CPU, wie `--cpus 1` im Benchmark | Mehr Kerne nützen nichts. Unser Weg ist ohnehin einfädig |
| B2 (Teil): zählt die Kompilation? | Ja. Gemessen wird der komplette Erstlauf: Setup, Kompilation, Forward, Sensitivity und Gradient | Wie bisher angenommen |

## Die sechs wichtigsten Fragen

Wenn nur wenig Zeit ist, diese zuerst. Von den ersten fünf hängt ab, ob unsere Lösung überhaupt zählt. Die sechste ist die mit dem grössten Zeitgewinn.

1. Zählt eine Lösung, deren Rechenkern in SciPy statt in JAX läuft? (A1)
2. Wir haben schon vor dem offiziellen Start am öffentlichen Kit gearbeitet. Ist das erlaubt? (D1)
3. Auf welcher Maschine und mit welchem Befehl wird die Wertung gemessen? (B1, B2)
4. Wie geht ihr mit der Streuung um, wenn die Gesamtzeit unter 1,5 s liegt? (B3)
5. Beantwortet: Die Schlussbewertung nutzt ein ähnliches Setup mit leicht anderen Parametern und Komponenten (C3, siehe oben).
6. Darf unser Code eine JAX-Einstellung setzen, die den Compiler weniger optimieren lässt? Das spart in Docker 40 ms, also 8 % der Gesamtzeit. (B6)

---

## A. Ist unser Ansatz zulässig?

| Nr. | Frage | Warum wir fragen |
| --- | --- | --- |
| A1 | Zählt eine Lösung, bei der die Gleichungssysteme ausserhalb von JAX in SciPy (SuperLU) gelöst werden, eingebunden über `jax.pure_callback`? `df.simulate` bleibt aufrufbar und unter `jax.jit` und `jax.vjp` nutzbar. | Daran hängt fast der ganze Gewinn. Bei Nein müssten wir den Sparse-Ansatz in reinem JAX nachbauen. |
| A2 | Ist ein handgeschriebener Reverse-Mode-Gradient (`jax.custom_vjp`) in Ordnung, wenn er mit dem JAX-Gradienten übereinstimmt? | JAX kann nicht durch SciPy hindurch ableiten. |
| A3 | Reicht es, wenn der schnelle Weg nur für bestimmte Setups greift (kein Parameter-Linking, kein Homodyn-Detektor, genau ein Rauschdetektor) und alle anderen den unveränderten JAX-Code nutzen? | Unser Dispatcher fällt sonst auf `simulate_jax` zurück. |
| A4 | Das Rauschen rechnen wir als `Re(yᴴ·N·y)` mit `y = S⁻ᴴ·w`. Das ist algebraisch dasselbe wie im Original, spart aber eine Lösung. Gilt das als «weggelassene Berechnung»? | Im Regeltext steht, dass ausgelassene Berechnungen nicht zählen. |
| A5 | Komponenten, die nicht von der Frequenz abhängen, werten wir einmal aus und kopieren das Resultat auf alle 101 Frequenzen. In Ordnung? | Gleiche Sorge wie bei A4. |
| A6 | Muss die Lösung auf der GPU lauffähig bleiben? | `pure_callback` rechnet immer auf der CPU. |
| A7 | Braucht ihr `vmap` über viele Parametersätze (`run_with_parameter_sets`), Forward-Mode (`jvp`) oder höhere Ableitungen? | Der schnelle Weg kann nur Reverse-Mode nach den Designvariablen. Alles andere läuft über den Rückfall. |
| A8 | Wir laden den kompilierten SuperLU-Löser direkt aus dem SciPy-Paket (`scipy/sparse/linalg/_dsolve/_superlu`), ohne `import scipy.sparse.linalg`. Das spart rund 90 ms Import. Ist das in Ordnung? | Es ist ein internes Modul von SciPy. Falls nein, kostet uns der normale Import etwa 15 % der Gesamtzeit. |
| A9 | `df.signal_detector` und `df.power_detector` rechnen bei uns in NumPy (Host-Callback) statt in JAX, mit eigener Rückwärtsregel. Das spart rund 30 ms Kompilierzeit im Readout. Zählt das? | Die Formeln sind unverändert, Forward-Mode geht damit aber nur noch über einen Schalter (`HOST_DETECTORS = False`). |
| A10 | Dürfen wir Einstellungen des Lösers ändern? Wir nutzen SuperLU mit `Relax=1`, `PanelSize=1`, skalieren das System vorher (`D⁻¹·S·D` mit Zweierpotenzen) und lassen die Diagonale als Pivot zu, solange sie 10 % des grössten Spalteneintrags erreicht. Je 16 Frequenzen übergeben wir dem Löser gemeinsam als ein Block-Diagonalsystem. | Die Resultate ändern sich im Rahmen der Rundung (Signal 1e-7). Die Zerlegung wird insgesamt etwa doppelt so schnell, und bei veränderten Designvariablen ist sie genauer als ohne Skalierung. Das Bündeln ändert das Ergebnis nicht (bitgleich). |
| A11 | Der Build übergibt die Systemmatrizen bei uns in Koordinatenform (Zeile, Spalte, Wert) statt als dichte Arrays. Die Einträge sind dieselben, die dichte Signal- und Rauschmatrix (je 9 MB, fast nur Nullen) entstehen nicht mehr. `arrays` aus `run_build_step` enthält dafür drei Objekte eines eigenen Typs. Die Index-Arrays sind ausserdem statische NumPy-Arrays, die JAX als Konstanten behandelt statt als Eingaben. In Ordnung? | `benchmark.py` reicht `arrays` nur durch und läuft unverändert. Wer die Matrizen dicht braucht, bekommt sie mit `np.asarray(...)`. Die Resultate sind bitgleich. Gewinn: rund 5 ms im Erstlauf. |

## B. Wie wird gemessen?

| Nr. | Frage | Warum wir fragen |
| --- | --- | --- |
| B1 | Auf welcher Maschine läuft die Wertung: derselbe Xeon Platinum 8468, x86-64, wie viel RAM? Messt ihr oder wir? | Wir haben bisher nur Apple Silicon und Docker auf aarch64. |
| B2 | Welcher Befehl genau, mit welchen Wiederholungen? Zählt ein einzelner Erstlauf oder der beste bzw. mittlere aus mehreren frischen Prozessen? Zählt die warme Laufzeit für die Rangliste gar nicht, auch nicht bei Gleichstand? | Der «first-run total» entsteht nur einmal pro Prozess. Wir optimieren auf den Erstlauf. Die warme Laufzeit liegt bei 30 ms (macOS) bzw. 35 ms (Docker). |
| B3 | Wie geht ihr mit Streuung um? Bei uns schwankt der Erstlauf je nach Zustand der Maschine um bis zu 40 ms, in älteren Docker-Messungen zwischen 0,87 s und 1,57 s. | Bei so kleinen Zeiten entscheidet das Rauschen über die Rangliste. |
| B4 | Ist der Zustand der Maschine definiert: Dateien schon im Page-Cache, `__pycache__` vorhanden oder frischer Checkout? | Die Imports machen bei uns fast 60 % der Gesamtzeit aus, und davon gehören rund 290 ms zu `benchmark.py` selbst (`import jax`, `importlib.metadata`). Ohne `__pycache__` kostet der Import unserer Module 12 bis 13 ms (fast alles Übersetzen der Quelltexte), mit Cache 1,4 bis 1,9 ms. Unsere eigenen Messungen laufen alle ohne Cache. |
| B5 | Dürfen wir alle Pakete aus `requirements.txt` nutzen (insbesondere SciPy)? Dürfen wir neue Abhängigkeiten hinzufügen, z. B. einen anderen Sparse-Löser, Numba oder eine C-Erweiterung? | SciPy steht in der gepinnten Liste, wird vom Original aber nicht direkt verwendet. |
| B6 | Dürfen wir Paketversionen, die Python-Version, Umgebungsvariablen oder XLA-Flags ändern? Und dürfte unser Code selbst Einstellungen setzen (`jax.config`, Speicherverwaltung über `mallopt`)? Konkret: `jax.config.update("jax_disable_most_optimizations", True)` in `differometor/`. | Die Baseline nennt ausdrücklich «keine XLA_FLAGS». Wir setzen nichts davon. In Docker sind aber rund 100 ms reine XLA-Kompilation des Readouts aus `benchmark.py`. Gemessen am 21.11.: Der eine Schalter spart dort 42 ms (Sensitivity-Kompilation 64 → 36,5 ms, Gradient-Kompilation 35,3 → 20,6 ms), auf dem Mac 13 ms. Loss und Gradient stimmen danach auf 11 Stellen überein. Joel hat dieselbe Variante vorbereitet (Branches `sx2`, `sx2-env`). Neu gemessen am 21.11. nachmittags: 515,8 → 475,4 ms. Zwei verwandte Punkte: Pythons Speicherbereinigung kostet während `import jax` rund 16 ms. Abschalten liesse sie sich nur vor diesem Import, also in `benchmark.py` oder der Umgebung. Dürfte man das? Und angehobene glibc-Schwellen helfen nur den warmen Wiederholungen. |
| B7 | Sind der persistente Kompilier-Cache und vorab exportierte Programme (`jax.export`) wirklich ausgeschlossen? | Wir nutzen beides nicht und wollen sicher sein, dass es auch andere nicht dürfen. |
| B8 | Wir laden Plotly erst bei Bedarf (`df.visualize_setup` funktioniert weiter). In Ordnung? | Der Benchmark braucht `visualize_setup` nicht, der Import lief aber in der gewerteten Zeit. |
| B9 | Dürfen wir nur in `differometor/` ändern und dort auch neue Dateien anlegen? `benchmark.py` und die Fixtures lassen wir unverändert. | Der Report speichert die Hashes dieser Dateien. |

## C. Korrektheit

| Nr. | Frage | Warum wir fragen |
| --- | --- | --- |
| C1 | Die unveränderte Baseline weicht schon um 5,37e-6 von eurer Referenz ab, erlaubt sind 1e-5. Woher stammt die Referenz (andere Maschine, andere Version, anderes Programm)? | Uns bleiben nur 4,6e-6 Spielraum, obwohl wir vom Original nur um 5,8e-9 abweichen. |
| C2 | Prüft ihr über `benchmark.py` hinaus, z. B. alle 193 Gradienten oder die Arrays carrier, signal und noise gegen die Baseline? Mit welcher Toleranz? | `benchmark.py` prüft nur einen der 193 Gradientenwerte. Unsere Abweichung zum Original: Gradient 8,0e-8, Signal 1,2e-7 (relativ zum Maximum). |
| C3 | Testet ihr weitere Setups, öffentliche (aLIGO, Voyager) oder nicht veröffentlichte? | Der schnelle Weg deckt nicht jedes Setup ab (siehe A3). |
| C4 | Einzelne Gradienteneinträge liegen bei 1e-13, das Maximum bei 5,8e4. Diese Einträge schwanken schon im Original. Ist das akzeptiert? | Ein elementweiser relativer Vergleich würde dort fehlschlagen. |

## D. Regeln und Abgabe

| Nr. | Frage | Warum wir fragen |
| --- | --- | --- |
| D1 | Das Kit ist seit Mi 18.11. öffentlich. Ist Arbeit daran vor Sa 13:00 erlaubt, und müssen wir sie deklarieren? | Wir haben bereits eine lauffähige Lösung und wollen das offen sagen. |
| D2 | In welcher Form wollt ihr die Erklärung zum KI-Einsatz? | Wir haben mit KI-Coding-Tools gearbeitet. |
| D3 | Was gehört genau in die Abgabe, und in welcher Form: Repo-Link, ZIP, Reports von welcher Maschine? | Die `.gitignore` des Kits schliesst `benchmark*.json` und `*.npz` aus. |
| D4 | Wonach vergebt ihr den Spezialpreis? Zählen Allgemeingültigkeit, Verbleib in JAX oder GPU-Tauglichkeit? | Wir entscheiden damit, ob wir zusätzlich den JAX-Rückfall beschleunigen. |
| D5 | Was würde die Lösung für euch im Alltag brauchbar machen? Würdet ihr so etwas upstream übernehmen? | Gibt uns die Richtung für die restliche Zeit und für den Pitch. |

---

## Was wir je nach Antwort tun

| Antwort | Folge |
| --- | --- |
| A1 Nein (kein SciPy) | Sparse-Ansatz in reinem JAX nachbauen. Max' `lu-once` und die Rausch-Abkürzung im JAX-Pfad werden dann zur Hauptsache. |
| C3: ähnliche Setups zählen (beantwortet) | Erledigt in Runde 9: Balanced Homodyne und verknüpfte Parameter laufen über den schnellen Weg. Offen bleibt der JAX-Rückfall für Setups mit mehreren Rausch-Detektoren, die `benchmark.py` aber nicht auslesen kann. |
| B5 Ja (neue Pakete erlaubt) | Schnelleren Sparse-Löser prüfen, Import-Zeit dabei mitmessen. |
| B6 Ja (JAX-Einstellung im Code erlaubt) | Eine Zeile in `differometor/__init__.py`: `jax.config.update("jax_disable_most_optimizations", True)`. Danach neu messen und alle Prüfungen laufen lassen. Die Änderung in der Abgabe offenlegen. |
| A8 Nein | SuperLU wieder über `scipy.sparse.linalg` importieren (in `fast.py` reicht dafür der Rückfall in `_load_superlu`). |
| A9 Nein | `HOST_DETECTORS = False` in `components.py` setzen. |
| B8 Nein | Plotly-Import in `__init__.py` wieder direkt machen. |
| D1 Nein (keine Vorarbeit) | Mit Feyer klären, wie wir damit umgehen, bevor wir etwas abgeben. |

## Noch selbst zu erledigen (keine Frage an Feyer)

- Philip hat `noel` auf einem Xeon Platinum 8468 gemessen (`origin/philip:_nacht/vm/jobs/`): 726 ms im Erstlauf für `c15f97e`, sein eigener Stand `philip-nacht` 441 ms. Sein Stand fällt bei Balanced Homodyne aber auf JAX zurück (auf Noels Mac 9,8 s statt 0,45 s). Zahlen in `runs/noel/NOTIZEN.md`. Früherer Eintrag: Zeitmessung von `noel` auf echter x86-Hardware mit `--cpus 1`, mehrfach in frischen Prozessen. Die Korrektheit auf x86-64 ist am 21.11. emuliert geprüft (`PASS`, alle Prüfskripte). Joel hat eine passende Maschine (AMD EPYC). Sein eigener Stand braucht dort laut seinem README 0,76 s, auf Noels Rechner ist `noel` rund 9 % schneller als Joels Stand.
- Erledigt am 21.11.: `runs/check_fast.py --fallback` und `runs/check_setups.py` laufen auf diesem Branch durch, auch mit Max' `lu-once` im JAX-Rückfall. Die Werte stehen in `runs/noel/NOTIZEN.md`.
