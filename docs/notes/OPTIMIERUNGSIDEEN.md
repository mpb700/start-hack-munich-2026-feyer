# Optimierungsideen: Analyse und Abgleich mit `philip`

Stand: 20.11.2026, Branch `max/lu-once`. Diese Datei hält fest, wo das Original Zeit verliert, welche Ideen wir dazu haben und welche davon auf Philips Branch `philip` schon umgesetzt sind.

Die Angaben zum Stand von `philip` stammen aus [`README.optimizations.md` am Stand `ab82e42`](https://github.com/mpb700/feyer-muc/blob/ab82e42/README.optimizations.md) (heute [`OPTIMIERUNGEN.md`](OPTIMIERUNGEN.md)). Diese Messungen habe ich nicht selbst wiederholt.

## Kurzfassung

- Gewertet wird die **First-Run-Gesamtzeit auf einer CPU** (`--cpus 1`), inklusive Imports und Kompilieren. Mehr Kerne helfen nicht, es zählt nur weniger Arbeit pro Kern.
- Im Original geht über die Hälfte der Zeit ins **Kompilieren** (22,9 von 44,2 s), der Rest in dichte 751×751-Lösungen und den Gradienten.
- Die großen Hebel sind **auf `philip` umgesetzt**: Sparse LU, Löser per `pure_callback` außerhalb von XLA, eigene Rückwärtsregel und die Rausch-Abkürzung. Laut Philips Messung fällt die First-Run-Zeit auf macOS damit von 11,0 s auf ca. 0,4 s.
- Jetzt dominieren die **Imports**. `plotly` wird auf `max/lu-once` inzwischen erst bei Bedarf geladen (spart ca. 15 ms), der Rest ist vor allem `import jax`.

## Rahmenbedingungen

- Gemessen wird die komplette First-Run-Zeit: Imports, Initialisierung, Inputs, alles Kompilieren, Forward, Sensitivität, Gradient, Transfer.
- Pflicht: alle 101 Frequenzen, alle 193 Designvariablen, double precision.
- Die Sensitivität muss pro Frequenz `abs(log(s) - log(ref)) <= 1e-5` erfüllen. Der Gradient muss den Sanity-Check bestehen.
- Vorberechnete Ergebnisse und ausgelassene Berechnungen zählen nicht. Den Persistent Compilation Cache nutzen wir deshalb nicht.

## Wo die Zeit im Original steckt

Offizielle Baseline (Xeon Platinum 8468, 1 CPU, `benchmarks/cpu-baseline.json`):

| Phase | Sekunden |
|---|---:|
| Imports | 2,61 |
| Build (Setup aufbauen) | 1,03 |
| Forward: Tracen + Kompilieren | 2,01 + 14,95 |
| Forward ausführen | 13,29 |
| Gradient: Tracen + Kompilieren | 0,37 + 5,18 |
| Gradient ausführen | 4,34 |
| **First-Run gesamt** | **44,20** |

## Ideen und Status

| # | Idee | Status |
|---|---|---|
| 1 | Rauschen ohne dritte Lösung: `Re(wᴴ N w)` statt `Re(sᴴ M⁻¹ N w)` | umgesetzt im Fast-Path (`philip`), offen im JAX-Fallback |
| 2 | Rauschmatrix N einmal bauen statt 101-mal (frequenzunabhängig) | umgesetzt (`philip`: alle 66 Rausch-Komponenten 1×) |
| 3 | Rückwärtsvektor des Signals aus dem Detektor-Adjoint wiederverwenden | umgesetzt als „Detektor-Basis“ (`philip`) |
| 4 | Frequenzunabhängige Komponenten nur einmal auswerten | umgesetzt (`philip`: 179 von 679 Signal-Komponenten 1×) |
| 5 | `vmap(lax.switch)` ersetzen (heute: alle 19 Funktionen pro Eintrag) | im Fast-Path entfallen (NumPy statt XLA), offen im JAX-Fallback |
| 6 | Matrix in einem Schritt schreiben: keine Block-Kopien, rechte Seite getrennt | im Fast-Path entfallen (CSC wird direkt gebaut) |
| 7 | Dünnbesetztheit ausnutzen (Sparse LU) | umgesetzt (`philip`: SuperLU, feste COLAMD-Ordnung) |
| 8 | Eigene Rückwärtsregel, Kotangenten nur an Nichtnull-Positionen | umgesetzt (`philip`: `custom_vjp`) |
| 9 | Löser aus XLA herausnehmen (`pure_callback`) | umgesetzt (`philip`: Kompilieren nur noch ca. 0,09 s) |
| 10 | LU-Zerlegung pro Frequenz einmal statt dreimal | Commit `3153d6d` hier, wirkt nur noch im JAX-Fallback |
| 11 | `plotly` nicht beim Import laden | umgesetzt auf `max/lu-once` (Commit `3cf5fab`): 26,3 → 11,7 ms, für `philip` noch zu übernehmen |
| 12 | Blockstruktur oben/unten/Mechanik (Schur-Komplement) | offen, neben Sparse LU vermutlich überflüssig |
| 13 | XLA-Flags mit niedrigerer Optimierungsstufe | verworfen: Grauzone, und das Kompilieren ist schon fast weg |

### 1. Rauschen ohne dritte Lösung

Codestelle im Original: `differometor/simulate.py`, Zeilen 371–412.

Heute rechnet der Code pro Frequenz:

```
w = M⁻ᴴ · s              (Rausch-Gewichte)
c = M⁻¹ · (N · w)        (Kovarianz, dritte Lösung)
Rauschen² ∝ Re(sᴴ · c)
```

Wegen `sᴴ · M⁻¹ = (M⁻ᴴ · s)ᴴ = wᴴ` gilt exakt:

```
Rauschen² ∝ Re(wᴴ · N · w)
```

Damit fallen die dritte Lösung, das Produkt `N · w` und deren gespeicherte Zwischenergebnisse weg. N ist dünn besetzt: Diagonale aus Vakuum und Verlusten, dazu Squeezer-Paare zwischen oberem und unterem Seitenband.

Geprüft habe ich das numerisch an zufälligen 6×6-Systemen, inklusive Squeezer-Kopplung. Beide Wege liefern bitgleiche Ergebnisse (Differenz 0,0). Auf `philip` steckt dieselbe Idee in `fast.py` (`Re(yᴴ N y)` mit `y = S⁻ᴴ w`).

### 2. und 4. Frequenzabhängig gegen frequenzunabhängig

Achtung: Auch Spiegel hängen formal von der Frequenz ab. `phase_shift` in `components.py` enthält den Faktor `(1 + frequency / F0)`. Mit F0 ≈ 2,8·10¹⁴ Hz ist das eine Änderung von ca. 10⁻¹¹, aber eben nicht null. Exakt bleibt man nur, wenn man genau die Komponenten **ohne** Sweep-Parameter in den Eingaben einmal auswertet. So macht es `philip`.

Laut Philips Analyse hängen 1052 der 2668 Nichtnullen der Signalmatrix von der Frequenz ab, und 217 der 751 Zeilen sind komplett frequenzunabhängig.

### 5. und 6. Matrix-Aufbau im JAX-Pfad

- `update` (`simulate.py`, Zeile 110) ruft `vmap(lax.switch)` mit gebatchtem Funktionsindex auf. JAX wertet dann **alle 19 Komponentenfunktionen** für jeden Eintrag aus und wählt danach aus. Das bläht Kompilierzeit, Forward und Gradient auf.
  - **Lösung:** Einträge beim Build nach Funktionstyp gruppieren und jede Funktion einmal auf ihre Gruppe anwenden.
- Konjugation des unteren Bands (Zeilen 325–334), Carrier-Skalierung (Zeile 316) und `[:, :, :-1]` erzeugen jeweils eine Kopie von 101 × 751 × 752 komplexen Zahlen (ca. 0,9 GB).
  - **Lösung:** Welche Einträge betroffen sind, steht beim Build fest. Man kann also auf der Werteliste konjugieren und skalieren und dann einmal in die Matrix schreiben.

Für den Benchmark ist das erledigt, weil der Fast-Path greift. Relevant bleibt es für Setups, die in den JAX-Fallback fallen (Linking, Homodyn-Detektoren, mehrere Rauschdetektoren).

### 7. Dünnbesetztheit

Meine Schätzung aus der Topologie (71 Strecken × 4 Feldrichtungen ≈ 286 Carrier-Unbekannte, 2–4 Kopplungen pro Zeile) lag bei ca. 1 % Nichtnullen. Gemessen hat Philip **2668 Nichtnullen, also 0,5 %**. L und U haben zusammen 8238 Einträge Fill-in.

### 11. `plotly` beim Import

`differometor/__init__.py` lädt `differometor.plot`, und das importiert `plotly.graph_objects`. Der Benchmark nutzt `visualize_setup` nicht, der Import läuft aber in der gewerteten Zeit mit. Bei ca. 0,4 s First-Run auf `philip` ist jeder Import-Anteil spürbar.

**Umgesetzt** in Commit `3cf5fab`: `visualize_setup` wird erst beim ersten Zugriff über ein modulweites `__getattr__` in `__init__.py` geladen. `from differometor.plot import visualize_setup`, `dir()` und Stern-Importe funktionieren weiter.

Gemessen auf macOS (Python 3.12, gepinnte Requirements, Median aus 9 frischen Prozessen, nach `import jax, numpy` wie im Benchmark): `import differometor` braucht **26,3 ms vorher, 11,7 ms nachher**. Der Gewinn ist mit ca. 15 ms klein, weil plotly 7 intern selbst schon lazy lädt. Bei ca. 0,4 s First-Run auf `philip` sind das trotzdem ca. 3–4 %. Die Änderung lässt sich per Cherry-Pick auf `philip` übernehmen (`__init__.py` ist dort unverändert).

## Was als Nächstes noch lohnt

1. **Imports weiter aufschlüsseln.** Laut Philip entfallen auf `import jax` ca. 0,22 s und auf `scipy.sparse.linalg` ca. 0,04 s. Der plotly-Anteil (ca. 15 ms) ist erledigt, der Rest sollte per `python -X importtime` geprüft werden.
2. **Offizielle Messung auf Linux x86 mit `--cpus 1`.** Bisher gibt es nur macOS (Threads begrenzt) und Docker auf aarch64.
3. **JAX-Fallback** (Punkte 1, 5, 6, 10): bringt für die Wertung nichts, wäre aber für andere Setups und den Sonderpreis eine saubere Ergänzung.
