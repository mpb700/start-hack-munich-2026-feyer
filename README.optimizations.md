# Optimierungen: Was wir verbessert haben und was es bringt

Differometor ist auf dem Hackathon-Workload (uifo 3×3, 101 Frequenzen, 193 Designvariablen) mit Kompilierung **ca. 27× schneller** und im warmen Lauf **ca. 77× schneller**. Physikalische Outputs und Gradienten bleiben dabei erhalten. Alle Frequenzen werden voll in double precision gerechnet, nichts ist vorberechnet und keine Berechnung wird ausgelassen.

## Gesamtbild

macOS, auf einen Thread begrenzt (`runs/env.sh`), gleiche Maschine:

| Stand | First-Run gesamt | Kompilieren | Warm-Median | Forward | Gradient |
|---|---:|---:|---:|---:|---:|
| Original (unverändertes Differometor), `runs/baseline.json` | 11,04 s | 3,68 s | 5,072 s | 3,509 s | 1,633 s |
| Sparse-Fast-Path, `runs/step2.json` | 0,79 s | 0,09 s | 0,111 s | 0,053 s | 0,057 s |
| + Build in NumPy, `runs/step4.json` | 0,45 s | 0,09 s | 0,108 s | 0,052 s | 0,056 s |
| **+ Solver-/Adjoint-Optimierungen**, `runs/opt/after-*.json` | **0,39–0,44 s** | 0,09 s | **0,066 s** | 0,043 s | **0,022 s** |

Linux in Docker, `--cpus 1`, aarch64. Vorher und nachher liefen im selben Image, abwechselnd gemessen (`runs/opt/docker/`):

| Stand | First-Run gesamt | Warm-Median | Forward | Gradient |
|---|---:|---:|---:|---:|
| Vor der letzten Runde | 1,34–1,57 s | 0,196–0,202 s | 90 ms | 105–111 ms |
| **Nach der letzten Runde** | **1,19–1,28 s** | **0,126–0,131 s** | 76 ms | **49–55 ms** |

Absolute Docker-Zeiten hängen stark vom Zustand der Docker-VM ab. Ein früherer Lauf (`runs/docker/benchmark-docker.json`) maß mit der Vorversion 0,87 s First-Run. Aussagekräftig ist deshalb nur der direkte Vergleich, der im selben Durchgang gemessen wurde.

Zur Einordnung: Die offizielle Baseline im Brief (Xeon Platinum 8468, 1 CPU) liegt bei 44,2 s First-Run und 17,7 s warm.

## Was geändert wurde

### 1. Sparse-Fast-Path statt dichter JAX-Solves (`differometor/fast.py`)

Das Original löst pro Frequenz dichte 751×751-Systeme mit `jnp.linalg.solve`. Für die Rauschpropagation macht es das sogar pro Rauschquelle. Alles läuft durch XLA, und allein das Kompilieren kostet mehrere Sekunden. Die Signalmatrix hat aber nur 2668 Nichtnullen (0,5 %).

- **Sparse LU:** Carrier- und Signalsysteme werden als CSC-Matrizen gebaut und mit SuperLU (`scipy.sparse.linalg.splu`) zerlegt.
- **Rauschen mit einem Solve:** Das Quantenrauschen des Detektors ist `Re(yᴴ N y)` mit `y = S⁻ᴴ w`. Das braucht einen Solve pro Frequenz statt einem pro Rauschquelle.
- **Matrixeinträge über Dual-Zahlen:** Die Einträge entstehen aus den unveränderten Komponentenfunktionen, ausgewertet auf Dual-Zahlen in NumPy. Dadurch muss weder XLA kompilieren, noch muss man die Physikformeln von Hand ableiten.
- **Handgeschriebener Reverse Mode:** Die Ableitung ist ein Adjoint mit JAX-Konvention für komplexe Kotangenten. Eingebunden ist er als `jax.custom_vjp` um zustandslose `jax.pure_callback`s.
- **Fallback auf JAX:** Der Dispatcher nimmt den Fast-Path nur, wenn das Setup passt. Er greift nicht bei Parameter-Linking, Homodyn-Detektoren, mehreren Rauschdetektoren, Carrier-Sweeps oder Ableitungen nach anderen Eingaben. Dann läuft der Original-JAX-Code (`simulate_jax`).

### 2. Build-Schritt in NumPy (`differometor/build.py`)

Der Build hat konstante Blöcke mit eager `jax.numpy` ausgewertet. Das hat beim ersten Lauf ca. 64 kleine XLA-Programme kompiliert (0,45 s statt 0,04 s). Jetzt läuft der Build in NumPy, die Arrays stimmen bis auf 1 ulp mit dem alten Build überein.

### 3. Solver- und Adjoint-Optimierungen (letzte Runde)

Zuerst wurde das Benchmark-Setup analysiert:

- Von den 2668 Nichtnullen der Signalmatrix hängen nur 1052 von der Frequenz ab, 217 der 751 Zeilen sind komplett frequenzunabhängig.
- **Zeilen streichen ist trotzdem nicht möglich.** Jede Unbekannte wirkt auf die Signal-Lösung oder auf den Detektor-Adjoint.
- Der Benchmark liest das Signal nur an einem Port, also an 2 Zeilen (oberes und unteres Seitenband). Der Rauschdetektor liest exakt dieselben 2 Zeilen.

Darauf bauen vier Änderungen auf:

| Änderung | Vorher | Nachher |
|---|---|---|
| **a) Detektor-Basis statt zweiter LU im Backward.** Der Forward löst pro Frequenz einmal `Z = S⁻ᵀ [e_d₁, e_d₂]`, ein Solve mit 2 rechten Seiten. Daraus folgen `Y = conj(Z · conj(w_d))` im Forward und `λ = Z · signal_bar` im Backward. | Backward zerlegt alle 101 Signalmatrizen neu (ca. 24 ms) | Backward braucht für die Signalsysteme **keine** LU mehr |
| **b) Frequenzunabhängige Komponenten nur einmal auswerten.** Komponenten, deren Eingaben keinen Sweep-Parameter enthalten, werden für eine Zeile ausgewertet und dann auf alle Frequenzen kopiert. | 679 Signal- und 66 Rausch-Komponenten, je 101-mal | 179 Signal- und alle 66 Rausch-Komponenten nur 1-mal |
| **c) Forward ohne Dual-Zahlen.** Der Forward braucht nur Werte. Ableitungen werden nur noch im Backward einmal mitgeschrieben. | Dual-Auswertung im Forward **und** im Backward (je ca. 17 ms) | Werte-Auswertung im Forward, Dual-Auswertung nur im Backward |
| **d) Feste Spaltenordnung für die LU.** Alle Frequenzen haben dasselbe Sparsity-Muster. COLAMD läuft deshalb einmal, danach zerlegt SuperLU mit `permc_spec="NATURAL"` die vorab umsortierte Matrix. | COLAMD bei jeder der 101 Zerlegungen (27 ms) | 22 ms bei identischem Fill-in (8238 Einträge in L+U) |

Der Backward rechnet `V_bar`, `w_bar` und die Rückstreuung auf die Parameter jetzt außerdem vektorisiert statt in einer Python-Schleife über die Frequenzen.

**Fallback:** Wenn der Kotangent des Signals außerhalb der Detektorzeilen liegt, rechnet der Backward wie vorher mit einer LU pro Frequenz. Dasselbe gilt, wenn der Detektor mehr als zwei Zeilen liest. Die Ergebnisse sind dann identisch, nur langsamer.

**Ersparnis dieser Runde:**

| | macOS | Docker |
|---|---:|---:|
| Gradient | −61 % (57 → 22 ms) | −52 % |
| Forward | −19 % (53 → 43 ms) | −16 % |
| Warm gesamt | −40 % (0,110 → 0,066 s) | −35 % |

Die Ersparnis wurde nur gesamt gemessen und nicht pro Einzelmaßnahme a–d aufgeschlüsselt.

## Runde vom 21.11. (Branch `noel`)

Gemessen im Wechsel gegen den Stand davor (Philips Branch plus Max' `lu-once`), je frischer Prozess, Erstlauf gesamt als Median:

| Umgebung | Vorher | Nachher |
|---|---:|---:|
| Apple M4, macOS, Threads auf 1 begrenzt (12 Läufe) | 0,673 s | **0,508 s** |
| Docker, Linux aarch64, `--cpus 1` (10 Läufe) | 0,734 s | **0,560 s** |

Was geändert wurde:

| Änderung | Wirkung |
|---|---|
| SuperLU-Modul direkt laden statt `import scipy.sparse.linalg` | ca. 85 ms weniger Import |
| `signal_detector` und `power_detector` als Host-Callback (`components.HOST_DETECTORS`) | ca. 30 ms weniger Kompilieren im Readout |
| Ableitungen nur für Eingänge, die Designvariablen sind | Rückwärtslauf 24 → 15 ms |
| SuperLU mit `Relax=1`, `PanelSize=1`, zwei Löser-Aufrufe pro Frequenz statt drei | Vorwärtslauf 43 → 30 ms |
| Plotly erst bei Bedarf laden | ca. 7 ms weniger Import |

Die Abweichungen zum Original-JAX sind unverändert (Sensitivität 5,8e-9, Gradient 8,0e-8). Details, Messmethode und offene Ideen stehen in [`runs/noel/NOTIZEN.md`](runs/noel/NOTIZEN.md). Die Tabellen weiter oben zeigen den Stand vor dieser Runde.

## Zweite Runde vom 21.11. (Branch `noel`)

Drei Stände im Wechsel gemessen, je frischer Prozess, Erstlauf gesamt als Median. Die absoluten Werte hängen vom Zustand der Maschine ab und liegen an diesem Tag tiefer als in der Tabelle oben. Vergleichbar sind nur Werte aus derselben Messreihe.

| Umgebung | Philip + Max | Nach der ersten Runde | Nach der zweiten Runde |
|---|---:|---:|---:|
| Apple M4, macOS, Threads auf 1 begrenzt (12 Läufe) | 0,602 s | 0,459 s | **0,453 s** |
| Docker, Linux aarch64, `--cpus 1` (10 Läufe) | 0,711 s | 0,539 s | **0,529 s** |

Was geändert wurde:

| Änderung | Wirkung (Docker / Mac) |
|---|---|
| Rückwärtslauf ohne dichte Matrizen: Der Plan entsteht nur aus den Index-Arrays, die Rauschmatrix wird nicht mehr übergeben | Gradient 26,9 → 21,4 ms / 16,2 → 13,3 ms |
| Signal-Systeme skaliert (`D⁻¹·S·D`), symmetrisch angeordnet und mit Schwellen-Pivotsuche zerlegt | Forward 42,3 → 39,3 ms / 32,8 → 30,6 ms |
| Dichte Matrizen im Build ausgerichtet anlegen, damit JAX sie ohne Kopie übernimmt | netto 1,1 ms / 0,6 ms |

Zum zweiten Punkt: Jede Gleichung des Signal-Systems bestimmt ihre eigene Unbekannte, die Diagonale ist 1. Darum eliminiert der Löser jetzt in einer Minimum-Degree-Reihenfolge von `S + Sᵀ` und nimmt die Diagonale als Pivot, solange sie mindestens 10 % des grössten Eintrags ihrer Spalte erreicht. Sonst nimmt SuperLU wie bisher den grössten Eintrag. Die L- und U-Faktoren haben damit rund 6'100 statt 8'330 Einträge. Vor der Zerlegung wird das System mit Zweierpotenzen skaliert, weil die Unbekannten sehr verschiedene Einheiten haben (Einträge von 1e-11 bis 1e13).

Die Skalierung macht den Löser auch genauer. Bei zufällig veränderten Designvariablen (`runs/noel/test_perturbed.py`) stimmt die Sensitivität jetzt in allen sechs Versuchen auf 2e-8 (im Logarithmus) mit dem Original-JAX überein. Die unskalierte Zerlegung lag in einem der Versuche um 8e-6 daneben, also knapp unter der Benchmark-Toleranz von 1e-5.

Am gespeicherten Design bleiben die Abweichungen zum Original-JAX auf dem bisherigen Niveau (Sensitivität 5,7e-9, Gradient 8,0e-8, Signal 1,2e-7). Der neue Test `runs/noel/test_adjoint.py` prüft zusätzlich den allgemeinen Rückwärtslauf mit Gewichten auf allen Ausgängen.

## Dritte Runde vom 21.11. (Branch `noel`)

Wieder im Wechsel gemessen, je frischer Prozess, Erstlauf gesamt als Median. Alle Läufe `PASS`.

| Umgebung | Philip + Max | Runde 1 | Runde 2 | Runde 3 |
|---|---:|---:|---:|---:|
| Apple M4, macOS, Threads auf 1 begrenzt (12 Läufe) | 0,607 s | 0,460 s | 0,454 s | **0,446 s** |
| Docker, Linux aarch64, `--cpus 1` (10 Läufe) | 0,717 s | 0,543 s | 0,532 s | **0,520 s** |

Warm (Forward, Sensitivity und Gradient ohne Kompilation) sind es jetzt 33 ms auf dem Mac und 39 ms in Docker, nach 38 und 52 ms in Runde 2.

Was geändert wurde:

| Änderung | Wirkung |
|---|---|
| Max' Branch `max/nacht` zusammengeführt: Beispiel-Setups und der dichte JAX-Weg werden erst bei Bedarf geladen | `import differometor` ohne `.pyc` in Docker 15,5 → 13,0 ms |
| 16 Frequenzen pro SuperLU-Aufruf als Block-Diagonalsystem (`_BlockLU`), Ergebnis bitgleich | Zerlegung und Lösen im ersten Aufruf in Docker 18,5 → 16,8 ms |
| Die dichte Signalmatrix (9 MB) geht nicht mehr an den Rückwärtslauf, nur noch ihre konstanten Einträge in kompakter Form | zusammen mit den zwei Zeilen darüber: Forward 42,0 → 38,1 ms, Gradient 21,3 → 19,2 ms in Docker |
| Forward- und Rückwärtslauf verarbeiten 16 Frequenzen am Stück statt alle 101 auf einmal | Forward 37,5 → 36,0 ms, Gradient 19,3 → 18,8 ms in Docker |
| Komponentenfunktionen schreiben direkt in die Matrixeinträge, frequenzunabhängige Eingänge werden nur einmal ausgewertet | Callbacks auf dem Mac: Forward 21,6 → 20,6 ms, Rückwärts 9,1 → 7,8 ms |

Der gemeinsame Nenner ist Speicher. JAX kopiert jedes Array, das in einen Callback hineingeht, und frisch angeforderter Speicher kostet unter Linux pro 4 kB einen Seitenfehler (in Docker rund 0,6 µs). Im Erstlauf fielen in der Forward-Phase rund 16'000 Seitenfehler an, jetzt sind es rund 11'000. In der Gradient-Phase sank die Zahl von rund 9'000 auf rund 6'400.

Am gespeicherten Design sind alle Abweichungen zum Original-JAX unverändert (Sensitivität 5,7e-9, Gradient 8,0e-8, Signal 1,2e-7). `runs/noel/test_adjoint.py` prüft neu auch, dass die Blockgrösse das Ergebnis nicht ändert.

Nicht eingebaut, weil es eine Regelfrage ist: Der Schalter `jax_disable_most_optimizations` würde die Kompilation des Readouts aus `benchmark.py` in Docker um rund 42 ms verkürzen (Mac: 13 ms). Die Frage steht als B6 in `FRAGEN_AN_FEYER.md`.

## Vierte Runde vom 21.11. (Branch `noel`)

Eine Änderung: Der Rückwärtslauf speichert Ableitungen der Komponentenfunktionen nur noch für die Ausgänge, die wirklich von einer Designvariable abhängen. Vorher füllte er pro Funktionsgruppe ein dichtes Array, das grösstenteils aus Nullen bestand.

| Umgebung | Erstlauf Runde 3 | Erstlauf Runde 4 | Gradient-Phase | Warm |
|---|---:|---:|---|---|
| Apple M4, macOS (16 Läufe im Wechsel) | 0,438 s | **0,436 s** | 10,7 → 9,8 ms | 32,4 → 31,3 ms |
| Docker, Linux aarch64, `--cpus 1` (12 Läufe im Wechsel) | 0,519 s | **0,515 s** | 16,4 → 14,8 ms | 40,3 → 38,4 ms |

Alle Prüfungen fallen aus wie zuvor. Gemessen und in dieser Runde noch nicht übernommen wurde eine Bündelung der kleinen Arrays für den Rückwärtslauf: Sie macht den Erstlauf in Docker 3 ms schneller, die warmen Wiederholungen dort aber 6 ms langsamer. Die Einzelheiten stehen in `runs/noel/NOTIZEN.md`.

## Fünfte Runde vom 21.11. (Branch `noel`)

Eine Änderung, auf Noels Entscheid: Die 22 kleinen Arrays, die der Rückwärtslauf braucht (Index-Arrays, Parameter, Frequenzen), reisen als zwei gepackte Residuen statt einzeln. Der Rückwärts-Callback bekommt 16 statt 36 Argumente.

| Umgebung | Erstlauf ohne Imports | Gradient-Phase | Warm |
|---|---|---|---|
| Apple M4, macOS (20 Läufe im Wechsel) | 139,5 → **138,4 ms** | 11,1 → 11,2 ms | 32,8 → 32,5 ms |
| Docker, Linux aarch64, `--cpus 1` (12 Läufe im Wechsel) | 219,4 → **216,1 ms** | 17,1 → 14,6 ms | 41,6 → 44,8 ms |
| Docker, zweite Reihe (16 Läufe, Reihenfolge getauscht) | 221,1 → **214,6 ms** | 17,7 → 14,5 ms | 39,7 → 46,6 ms |

Was man dazu wissen muss:

- **In der Gesamtzeit ist der Gewinn nicht nachweisbar.** Die Import-Phase (rund 320 ms, fast alles `import jax`) streut um ±3 ms. Die Gesamtzeiten der drei Reihen: 461,2 → 461,6 ms, 534,7 → 535,3 ms und 548,5 → 535,5 ms. Die Messung lief auf Akku, darum liegen alle Werte rund 5 % höher als in den Runden davor.
- **Die warmen Wiederholungen werden in Docker 3 bis 7 ms langsamer.** Das ist der bewusst in Kauf genommene Preis: Für die Wertung zählt der Erstlauf.
- **Der grössere Teil des Gewinns hängt am Speicherlayout.** In Docker fallen im Erstlauf rund 6'000 Seitenfehler weniger an (rund 3,5 ms), dafür in jedem warmen Aufruf mehr. Beides hängt davon ab, wann glibc Speicher ans System zurückgibt, und kann auf einer x86-Maschine anders ausfallen. Sicher übertragbar ist rund 1 ms (weniger Argumente beim Lowering und Kompilieren). Darum sollten beide Stände (`6b67cfd` ohne, `45a2328` mit Bündelung) auf x86 im Wechsel gemessen werden. Nachtrag: Seit der sechsten Runde ist die Bündelung fester Teil des Entwurfs, und ihr Nachteil im Warmlauf ist weg.

Die Ausgaben aller Prüfskripte sind Zeile für Zeile gleich wie zuvor. Neu ist `runs/noel/test_pack.py`: Es prüft, dass das Packen jedes Array exakt wiederherstellt.

## Sechste Runde vom 21.11. (Branch `noel`)

Eine Änderung: Der Build übergibt die drei Systemmatrizen (Träger, Signal, Rauschen) in Koordinatenform statt als dichte Arrays. Dafür gibt es den Typ `differometor.fast.SparseMatrix` (Zeilen und Spalten, Werte, feste Form). Die dichte Signal- und Rauschmatrix belegten im Benchmark-Setup je 9 MB, für 1'539 bzw. 0 konstante Einträge. JAX kopierte beide bei jedem Forward-Aufruf in den Callback.

| Umgebung | Erstlauf gesamt | Forward-Phase | Gradient-Phase | Warm |
|---|---|---|---|---|
| Apple M4, macOS (16 Läufe im Wechsel, in der siebten Runde nachgemessen) | 0,446 → **0,445 s** | 27,0 → 24,0 ms | 10,4 → 10,0 ms | 31,6 → 29,8 ms |
| Docker, Linux aarch64, `--cpus 1` (16 Läufe im Wechsel) | 0,515 → **0,512 s** | 35,1 → 29,7 ms | 13,6 → 13,3 ms | 43,3 → 35,2 ms |
| Docker, zweite Reihe (12 Läufe, Reihenfolge getauscht) | 0,518 → **0,507 s** | 34,8 → 29,4 ms | 14,0 → 13,1 ms | 43,4 → 35,4 ms |

- **Korrektheit:** Die Benchmark-Ausgaben (Carrier, Signal, Noise, Sensitivity, Powers, Gradient) sind bitgleich zum Stand davor. Die Build-Ausgaben von fünf Setups sind als dichte Arrays identisch. Alle Prüfskripte geben dieselben Zahlen aus. Neu ist `runs/noel/test_sparse.py`.
- **Was sich an der Schnittstelle ändert:** `arrays["carrier_matrix"]`, `arrays["signal_matrix"]` und `arrays["noise_matrix"]` aus `run_build_step` sind `SparseMatrix`-Objekte. `np.asarray(matrix)` und `matrix.dense()` liefern die dichte Matrix wie bisher. Der Original-JAX-Weg (`simulate_jax`) wandelt sie am Eingang selbst um. Wer `simulate` dichte Matrizen übergibt, bekommt dieselben Resultate über den JAX-Weg.
- **Nebenwirkung:** Die warmen Wiederholungen in Docker sind jetzt schneller als vor der fünften Runde. Die feste Obergrenze für konstante Einträge im Rückwärtslauf gibt es nicht mehr.
- **Zurücknehmen:** `git revert 7d12ba7`.
- **Korrektur:** Die erste Mac-Messung dieser Runde (0,441 → 0,433 s gesamt) war in der Import-Phase um rund 3 ms zugunsten des neuen Stands verfälscht, weil nur im Arbeitsbaum eine `.pyc`-Datei lag. Die Tabelle zeigt die Nachmessung. Auf dem Mac geht der Gewinn von rund 3,5 ms in der Streuung der Import-Phase unter. Die Phasen und die Docker-Reihen waren nicht betroffen.

## Siebte Runde vom 21.11. (Branch `noel`)

Eine Änderung: Der Build-Schritt (`run_build_step`) ist schlanker, bei identischer Ausgabe. Die Funktion `set_instructions` brauchte mehr als die Hälfte der Build-Zeit für die Erweiterung der Matrixpositionen auf zwei Seitenbänder, mit rund 35 kleinen NumPy-Aufrufen pro Aufruf. Diese Erweiterung läuft jetzt auf Python-Listen. Dazu kommen fünf kleinere Stellen.

| Umgebung | Erstlauf gesamt | Build-Phase | Warm |
|---|---|---|---|
| Apple M4, macOS (16 Läufe im Wechsel) | 0,438 → **0,438 s** | 25,8 → 23,6 ms | 29,3 → 29,7 ms |
| Docker, Linux aarch64, `--cpus 1` (16 Läufe im Wechsel) | 0,511 → **0,508 s** | 26,2 → 23,0 ms | 35,8 → 35,5 ms |

- **Korrektheit:** Die Arrays aus `run_build_step` sind für 27 Setups identisch (946 Arrays). `set_instructions` liefert in 20'000 zufälligen Aufrufen dasselbe wie vorher. Die Benchmark-Ausgaben sind bitgleich, alle Prüfskripte geben dieselben Zahlen aus. Neu ist `runs/noel/test_build.py`.
- **Einordnung:** `run_build_step` allein braucht im Docker-Ablauf 5,9 statt 8,5 ms. Von den 23 ms der Build-Phase gehören die übrigen rund 17 ms zu `benchmark.py` (Start des JAX-Backends, drei kleine XLA-Programme für `jnp.asarray`).
- **Reports:** `runs/noel/mac-neu.json`, `mac-neu.png` und `docker-neu.json` zeigen diesen Stand: 0,432 s auf macOS und 0,499 s in Docker, je der mittlere von fünf Läufen.

## Achte Runde vom 21.11. (Branch `noel`)

Eine Änderung: Die Index-Arrays eines Setups sind statisch. Der Build liefert sie als `differometor.fast.StaticArray`, das sind schreibgeschützte NumPy-Arrays, die JAX nicht verfolgt. Sie sind damit Konstanten der kompilierten Programme und müssen nicht mehr bei jedem Aufruf in die Host-Callbacks kopiert werden. Das Forward-Programm hat 8 statt 36 Eingabe-Arrays.

| Umgebung | Erstlauf gesamt | Erstlauf ohne Imports | Forward: Lowering + Kompilation | Warm |
|---|---|---|---|---|
| Apple M4, macOS (16 Läufe im Wechsel) | 0,455 → **0,453 s** | 131,5 → 130,5 ms | 7,7 → 6,8 ms | 30,8 → 30,4 ms |
| Docker, Linux aarch64, `--cpus 1` (16 Läufe im Wechsel) | 0,521 → **0,516 s** | 204,7 → 201,4 ms | 8,2 → 6,5 ms | 37,1 → 37,4 ms |

Die absoluten Werte liegen höher als in der siebten Runde, weil der Rechner am Nachmittag rund 3 % langsamer war. Vergleichbar sind nur die zwei Spalten einer Messreihe.

- **Korrektheit:** Benchmark-Ausgaben bitgleich, Build-Arrays von 27 Setups identisch, alle Prüfskripte mit denselben Zahlen. `runs/noel/test_pack.py` prüft zusätzlich, dass der schnelle Weg mit gewöhnlichen Integer-Arrays dasselbe liefert und dass ein verändertes Index-Array zu einer neuen Kompilation mit dem richtigen Ergebnis führt.
- **Schnittstelle:** `arrays` aus `run_build_step` enthält die Index-Arrays als `StaticArray` (verhält sich wie ein NumPy-Array). Zurücknehmen: `git revert b6b1119`.
- **Auf x86-64 geprüft:** Der Stand vor dieser Runde läuft in einem emulierten x86-64-Image und meldet `PASS` (Abweichung zu aarch64 höchstens 8e-10 relativ, alle Prüfskripte bestanden).

In dieser Runde ausserdem gemessen und verworfen: ein Residuum weniger (netto null), andere Blockfolgen für die Zerlegung (kein Gewinn im Erstlauf), Vorab-Elimination der Hälfte der Unbekannten in NumPy (2,4 ms langsamer), Detektoren in XLA statt als Callback (8 bis 80 ms langsamer), schlankere Quelltexte (unter 1 ms). Die Zahlen stehen in `runs/noel/NOTIZEN.md`.

Zwei Befunde ausserhalb unserer Reichweite: Der JAX-Schalter `jax_disable_most_optimizations` würde 40 ms sparen (Regelfrage an Feyer), und Pythons Speicherbereinigung kostet während `import jax` rund 16 ms, bevor unser Paket überhaupt geladen ist.

## Neunte Runde vom 21.11. (Branch `noel`): Setup-Varianten

Feyer hat am 21.11. ergänzt, dass die Schlussbewertung ein ähnliches Setup mit leicht anderen Parametern und Komponenten nutzt. Das öffentliche Fixture stammt aus dem Zufallsgenerator `uifo(3, random=True)`, und der wählt als Auslese in der Hälfte der Fälle einen Balanced-Homodyne-Detektor. Solche Setups liefen bisher über den dichten JAX-Weg.

Der schnelle Weg deckt jetzt ab:

- **Balanced-Homodyne-Detektoren** (`qhd`): zwei Ports, der zweite mit dem Phasenfaktor des Detektors.
- **Verknüpfte Parameter** mit `jnp.sqrt` (Amplitudenmodulation in aLIGO, Voyager und UIFO).

| Setup | vorher | nachher |
|---|---|---|
| Öffentliches Fixture, Mac (9 Läufe im Wechsel) | 0,450 s | 0,449 s |
| Öffentliches Fixture, Docker `--cpus 1` (8 Läufe im Wechsel) | 0,519 s | 0,519 s |
| Fixture mit Balanced Homodyne statt Detektor, Mac | 10,06 s | **0,452 s** |

- **Nachgestellte Schlussbewertung:** `runs/noel/variant_fixture.py` erzeugt eine Variante des Fixtures (leicht veränderte Werte, getauschte Komponenten, Homodyne-Auslese oder ein neues Zufalls-UIFO) und rechnet die Referenz-Sensitivitäten mit dem unveränderten Original-Code. `runs/noel/variant_run.py` lässt den echten `benchmark.py` darauf laufen. 14 Varianten, alle `PASS`: 0,45 bis 0,49 s auf dem Mac, 0,52 bis 0,55 s in Docker mit einer CPU, Fehler gegen die Referenz höchstens 8,3e-8.
- **Gegen das Original-JAX:** `runs/noel/test_variants.py` vergleicht 21 Zufalls-Setups (1×1 bis 4×4, alle drei Modulationsarten, beide Auslesen). Grösste Abweichung von `ln s`: 4,9e-7, Gradient 2,4e-8.
- Für das öffentliche Fixture ändert sich nichts: gleiche Ausgaben, gleiche Prüfwerte.

## Wo die verbleibende Zeit steckt

Im First-Run (ca. 0,4 s auf macOS) steckt kaum noch Physik:

- **Imports:** ca. 0,27–0,4 s, davon `import jax` allein ca. 0,22 s und `scipy.sparse.linalg` ca. 0,04 s.
- **JAX-Kompilieren:** ca. 0,09 s, für Readout/Sensitivität und die Gradienten-Verkettung im Benchmark.
- **Forward und Gradient:** zusammen ca. 0,065 s.

Weitere Optimierungen an der Physik bringen für den Gesamtwert also nur noch wenig.

## Korrektheit

- **Benchmark-Checks:** `benchmark.py` meldet in allen Läufen **PASS**. Der maximale Log-Fehler der Sensitivität zur Referenz liegt bei 5,4e-6, erlaubt sind 1e-5. Den gleichen Wert erreicht die Vorversion.
- **Vorher gegen nachher (letzte Runde):**
  - Carrier und Powers sind bitidentisch.
  - Signal weicht um 5,5e-20 ab.
  - Sensitivität weicht um 1,1e-10 ab, Gradient um 7,6e-10 (relativ zum Maximum).
- **Gegen das Original (dichtes JAX):**
  - Sensitivität 5,8e-9, Gradient 8,0e-8, Signal 1,9e-7.
  - Einzelne Gradienteneinträge in der Größenordnung 1e-13 bei einem Maximum von 5,8e4 sind numerisches Rauschen. Sie weichen schon zwischen dem Original und der ersten Fast-Path-Version um bis zu 100 % ab.
- **Andere Setups:** `runs/check_setups.py` vergleicht gegen JAX für aLIGO, aLIGO mit Linking und Voyager mit Homodyne-Detektor. Seit der neunten Runde laufen alle über den Fast-Path, die Abweichungen liegen bei höchstens ca. 2e-8. `runs/noel/test_variants.py` prüft dazu 21 Zufalls-UIFOs.
- **Ableitung nach Frequenzen:** Wird nach etwas anderem als den Designvariablen abgeleitet, fällt der Dispatcher korrekt auf JAX zurück (`runs/check_fast.py --fallback`).

## Reproduzieren

```bash
# macOS / lokal (ohne CPU-Affinität: Threads begrenzen)
source runs/env.sh
python benchmark.py --repeats 10 --output runs/opt/after.json --save-arrays runs/opt/after.npz
python runs/compare_npz.py runs/opt/before.npz runs/opt/after.npz   # Arrays vergleichen
python runs/check_fast.py runs/opt/before.npz --fallback             # inkl. JAX-Fallback
python runs/check_setups.py                                          # aLIGO / Voyager

# Linux in Docker, eine CPU
runs/docker_bench.sh --repeats 10
```

Für den Vorher/Nachher-Vergleich in Docker lief die alte `fast.py` im selben Image, per `-v <alte fast.py>:/app/differometor/fast.py:ro` gemountet.
