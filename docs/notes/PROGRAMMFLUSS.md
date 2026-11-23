# Programmfluss und Matrizen, ganz einfach

Stand: Sa 21.11.2026. Alle Zahlen stammen aus einem echten Lauf des Benchmark-Setups (`examples/data/uifo_800_3000.json`) auf diesem Branch.

## 1. Der Ablauf in einem Bild

```
benchmark.py
   │
   ├─ 1. Bauplan lesen            JSON: 196 Bauteile, 71 Strecken
   ├─ 2. build                    Bauplan → Nummern, leere Matrizen, Formel-Listen
   ├─ 3. simulate   (vorwärts)    Matrizen füllen und lösen
   │        ├─ Carrier            1 System:   wo ist wie viel Laserlicht?
   │        ├─ Signal             101 Systeme: was macht eine Gravitationswelle?
   │        └─ Noise              wie viel Rauschen kommt am Detektor an?
   ├─ 4. readout                  Sensitivity = Rauschen / Signal, daraus der Loss
   └─ 5. backward   (rückwärts)   Gradient: 193 Zahlen, eine pro Drehknopf
```

| Schritt | Code | Was passiert | Bild dazu |
| --- | --- | --- | --- |
| 1 | `benchmark.py`, `load_inputs` | Der Bauplan wird als JSON gelesen | Rezept aufschlagen |
| 2 | `differometor/build.py` | Jede Stelle im Aufbau bekommt eine Nummer. Es entstehen leere Matrizen und Listen der Art «Formel X schreibt in Zelle Y» | Zutaten abwiegen und bereitstellen |
| 3 | `differometor/simulate.py` → `fast.py` | Die Formeln füllen die Matrizen, dann werden die Gleichungssysteme gelöst | Kochen |
| 4 | `benchmark.py`, `readout` | Aus Signal und Rauschen wird die Sensitivity, aus 101 Sensitivities eine einzige Zahl (Loss) | Abschmecken, eine Note geben |
| 5 | `benchmark.py`, `backward` | Für jeden Drehknopf: Wie ändert sich die Note, wenn ich daran drehe? | Welche Zutat ändern? |

`simulate()` ist eine Weiche: Einfache Setups (wie dieses) gehen auf den schnellen Weg in `fast.py`. Alle anderen laufen über den alten Code `simulate_jax`.

## 2. Wie die Stellen nummeriert sind

Jede Unbekannte ist «das Licht an einer Stelle». Der Build gibt jeder Stelle eine Nummer.

**Carrier: 286 Unbekannte**

| Nummern | Was | Beispiel |
| --- | --- | --- |
| 0–21 | Enden der Strecken bei Lasern, Squeezern und dem Detektor (11 × 2) | Nr. 1: hier speist der Laser `boundary10` ein |
| 22–285 | Anschlüsse der Bauteile, je «in» und «out» | Nr. 30 `ml11.left.in`, Nr. 31 `ml11.left.out` |

Ein Spiegel hat 4 Nummern (links/rechts, je in/out), ein Strahlteiler 8.

**Signal: 751 Unbekannte**

| Nummern | Was | Beispiel |
| --- | --- | --- |
| 0–285 | Oberes Seitenband, gleiche Reihenfolge wie beim Carrier | Nr. 31: oberes Seitenband an `ml11.left.out` |
| 286–571 | Unteres Seitenband (Carrier-Nummer + 286) | Nr. 317: unteres Seitenband an `ml11.left.out` |
| 572–750 | Mechanik und Signalquellen (179 Stück) | Nr. 572: Kraft auf Spiegel `ml11`, Nr. 573: seine Auslenkung |

Die 179 setzen sich zusammen aus 54 aufgehängten Spiegeln mit je 2 Plätzen (Kraft, Auslenkung) und 71 Signalquellen (eine pro Strecke).

## 3. Alle Matrizen mit echten Werten

### 3.1 Parameter-Zeile (`parameters`, 1 × 532)

Keine Matrix im eigentlichen Sinn, sondern eine Liste aller Zahlen des Aufbaus.

| Platz | Name | Wert | Bedeutung |
| ---: | --- | ---: | --- |
| 145 | `f_frequency` | 800 … 3000 | Signalfrequenz, wird 101-mal ersetzt |
| 3 | `center11_ml11_length` | 1 | Länge der Strecke in Metern |
| 147 | `ml11_reflectivity` | 0,9997463 | Spiegel wirft 99,97 % der Leistung zurück |
| 148 | `ml11_tuning` | 359,99998 | Feinverschiebung des Spiegels in Grad |
| 146 | `ml11_loss` | 0,000005 | Verlust am Spiegel |
| 377 | `boundary10_power` | 200 | Laserleistung in Watt |
| 395 | `boundary20_db` | 0,00249 | Squeezing-Stärke |
| 478 | `ml11sus_mass` | 0,1361 | Spiegelmasse in kg |

Die 193 Drehknöpfe (`optimized_parameters`) werden in 201 dieser Plätze geschrieben. Es sind mehr Plätze als Knöpfe, weil gekoppelte Längen sich einen Knopf teilen.

### 3.2 Carrier-Matrix (`carrier_matrix`, 286 × 287)

Frage: Wie viel Laserlicht ist wo, wenn nichts wackelt?

- 680 Einträge sind nicht null (286 Einsen auf der Diagonale, 394 Kopplungen). Der Rest sind Nullen.
- Die letzte Spalte (Nr. 286) ist die rechte Seite: wo Licht eingespeist wird.

Jede Zeile ist eine Gleichung. Beispiel Zeile 31 (`ml11.left.out`):

| Spalte | Stelle | Wert | Bedeutung |
| ---: | --- | --- | --- |
| 31 | `ml11.left.out` | 1 | die Unbekannte selbst |
| 30 | `ml11.left.in` | −0,9999 | Reflexion: √0,99975 der Amplitude kommt zurück |
| 32 | `ml11.right.in` | −0,01593j | Transmission: 1,6 % der Amplitude kommt von der anderen Seite durch |

Gelesen: «Was links aus dem Spiegel kommt = 0,9999 × was links hineingeht + 0,0159j × was von rechts durchkommt.» Die Minuszeichen entstehen, weil alles auf die linke Seite der Gleichung gebracht ist.

Weitere Beispiele:

| Zeile | Eintrag | Wert | Bedeutung |
| --- | --- | --- | --- |
| 30 (`ml11.left.in`) | Spalte 23 (`center11.left.out`) | −1 | Strecke: was am einen Ende hinausgeht, kommt am anderen an (mal Phasenfaktor) |
| 1 (Strecken-Ende) | rechte Seite | 20 | Laser `boundary10`: Feld = √(2 · 200 W) = 20 |
| 3 (Strecken-Ende) | rechte Seite | 0,1246 | schwacher Laser |

Die Lösung ist der Vektor `carrier` (286 komplexe Zahlen). Beispiele: grösster Betrag 2633 an `mr12.right.out` (dort staut sich Licht in einer Kavität). Am Detektor (`m03.left.out`, Nr. 279) steht −0,0058 − 0,1201j, das sind 7,2 mW.

### 3.3 Signal-Matrix (`signal_matrix`, 101 × 751 × 752)

Frage: Was passiert, wenn eine Gravitationswelle mit Frequenz f an den Strecken zieht?

- Pro Frequenz eine Matrix. 2597 Einträge sind nicht null, das sind 0,5 %. Dazu 71 Einträge in der rechten Seite.
- Auf der Diagonale steht überall exakt 1.
- 1052 Einträge hängen von der Frequenz ab, die anderen sind für alle 101 Frequenzen gleich.
- Die Werte reichen von 1e-11 bis 1,5e13, weil Licht (Feldstärken) und Mechanik (Meter, Newton) in einer Matrix stehen.

Die Matrix besteht aus Blöcken:

| Block | Einträge | Inhalt |
| --- | ---: | --- |
| oben × oben | 680 | Dieselben Regeln wie beim Carrier, für das obere Seitenband |
| unten × unten | 680 | Dasselbe für das untere Seitenband (konjugiert) |
| Licht ← Mechanik | 2 × 262 | Ein bewegter Spiegel erzeugt Seitenband-Licht |
| Mechanik ← Licht | 2 × 240 | Licht drückt auf den Spiegel (Strahlungsdruck) |
| Mechanik × Mechanik | 233 | Kraft → Auslenkung |

Beispiel Zeile 31 (oberes Seitenband an `ml11.left.out`, 800 Hz):

| Spalte | Stelle | Wert | Bedeutung |
| ---: | --- | --- | --- |
| 31 | selbst | 1 | |
| 30 | oben `ml11.left.in` | −0,9999 | Reflexion wie beim Carrier |
| 32 | oben `ml11.right.in` | −0,01593j | Transmission wie beim Carrier |
| 573 | Auslenkung von `ml11` | 0,0083 − 0,072j | Neu: Bewegt sich der Spiegel, entsteht hier Seitenband-Licht |

Beispiel Zeile 573 (Auslenkung von `ml11`):

| Spalte | Stelle | Wert | Bedeutung |
| ---: | --- | --- | --- |
| 573 | selbst | 1 | |
| 572 | Kraft auf `ml11` | 2,908e-07 | Auslenkung = Kraft / (Masse · (2πf)²) = 1 / (0,1361 · (2π · 800)²) |

Beispiel Zeile 572 (Kraft auf `ml11`): acht Einträge zwischen 40 und 6458 an den vier Anschlüssen des Spiegels, je für oberes und unteres Seitenband. Sie sind gross, weil sie mit dem Carrier-Licht am Spiegel multipliziert werden: Viel Licht drückt stark.

Rechte Seite: 71 Zeilen mit +1 oder −1, eine pro Strecke. Das ist die Gravitationswelle, die an jeder Strecke zieht.

Die Lösung ist `signal` (751 × 101). Am Detektor bei 800 Hz: oberes Seitenband 1,03e16 + 5,2e15j, unteres −1,23e16 − 1,0e15j.

### 3.4 Rausch-Matrix (`noise_matrix`, 751 × 751)

Kein Gleichungssystem. Sie ist eine Tabelle: Wo kommt wie viel Vakuum-Rauschen ins System? Sie ist für alle Frequenzen gleich.

- 272 Einträge sind nicht null, davon 264 auf der Diagonale.

| Wert | Anzahl | Bedeutung |
| ---: | ---: | --- |
| 0,0000025 | 240 | Verlust an Spiegel-Anschlüssen (halber Verlust 5e-6) |
| 0,5 | 18 | Offener Anschluss: volles Vakuum-Rauschen |
| 0,5005 bis 0,5029 | 6 | Squeezer-Eingänge |
| ausserhalb der Diagonale | 8 | Squeezer koppeln oberes und unteres Seitenband, z. B. 0,0517 + 0,0145j |

### 3.5 Auswahl-Vektor (`noise_selection_vectors`, 1 × 1 × 751)

Sagt, welche Stellen der Detektor liest. Nur zwei Einträge sind nicht null:

| Platz | Stelle | Wert |
| ---: | --- | ---: |
| 279 | oberes Seitenband an `m03.left.out` | 1,4142 |
| 565 | unteres Seitenband an `m03.left.out` | 1,4142 |

### 3.6 Hilfslisten (keine Physik, nur Verdrahtung)

| Name | Grösse | Inhalt |
| --- | --- | --- |
| `*_function_indices` | 77 / 679 / 66 | Welche Formel (Spiegel, Strecke, Laser …) wird aufgerufen? Für Carrier, Signal, Noise |
| `*_function_input_indices` | je × 7 | Welche Plätze der Parameter-Zeile bekommt die Formel? |
| `*_output_indices`, `*_matrix_indices` | 279 / 1893 / 272 | Welches Resultat der Formel landet in welcher Matrix-Zelle? |
| `signal_carrier_indices` | 1004 | Diese Signal-Einträge werden mit dem Carrier-Licht multipliziert |

Im Benchmark-Setup kommen diese Formeln vor: 54 Spiegel-Flächen, 71 Strecken, 7 Laser, 4 Squeezer, 54 Massen.

## 4. Was am Ende herauskommt

| Grösse | Form | Beispielwert |
| --- | --- | --- |
| `carrier` | 286 × 1 | −0,0058 − 0,1201j am Detektor |
| `signal` | 751 × 101 | 1,03e16 + 5,2e15j am Detektor bei 800 Hz |
| `noise` | 101 | 3,726e-10 bei 800 Hz |
| Detektor-Signal | 101 | Betrag 2,781e15 bei 800 Hz |
| Sensitivity | 101 | 3,726e-10 / 2,781e15 = **1,33986e-25** bei 800 Hz |
| Referenz von Feyer | 101 | 1,33986e-25 bei 800 Hz |
| Loss | 1 Zahl | 9,7e-08 |
| Gradient | 193 | grösster Wert 57'586 (`mr13.reflectivity`), `mb11.tuning` = 7,616e-05 |

Die Sensitivity ist bei 800 Hz am besten (1,34e-25) und bei 3000 Hz am schlechtesten (1,80e-24).

Der grösste Gradient gehört zur Reflektivität von `mr13`. Das passt zum Bild: An den Spiegeln der Kavität mit dem meisten Licht hängt das Ergebnis am stärksten.

## 5. Vom Modell zur Zahl: eine Kette zum Nachverfolgen

1. Im Bauplan steht: Spiegel `ml11`, Reflektivität 0,9997463, Masse 0,1361 kg.
2. Der Build schreibt 0,9997463 auf Platz 147 der Parameter-Zeile und merkt sich: Formel «Spiegel-Fläche» liest Platz 147 und schreibt in Zeile 31.
3. Im Carrier steht darum in Zeile 31 der Wert −0,9999 (Reflexion) und −0,01593j (Transmission).
4. Im Signal steht dasselbe in Zeile 31 und Zeile 317, dazu die Kopplung an die Spiegelbewegung in Spalte 573.
5. Die Masse 0,1361 landet als 2,908e-07 in Zeile 573, Spalte 572.
6. Nach dem Lösen liest der Detektor die Zeilen 279 und 565.
7. Der Gradient sagt am Ende, wie stark die Sensitivity auf `ml11.reflectivity` reagiert.
