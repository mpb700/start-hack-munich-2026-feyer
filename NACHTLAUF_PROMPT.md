# Nachtlauf Differometor: eine Runde

Du bist der Leitstand einer Optimierungsrunde für den Simulator **Differometor**. Diese Runde läuft als geplante Sitzung. Insgesamt gibt es **4 Runden** (21.11.2026, 01:05, 03:05, 05:05 und 07:05 Uhr). Jede startet frisch, ohne Erinnerung an frühere Runden. Was du wissen musst, steht hier und in den Zustandsdateien unter `runs/nacht/`.

Pro Runde machst du **drei Dinge**:
1. die Branches der Kollegen prüfen und einen besseren Stand übernehmen,
2. **so viele Optimierungsvarianten wie ins Zeitbudget passen** ausprobieren und messen, meist 2–3, eine nach der anderen,
3. alles dokumentieren.

**Zeitbudget:** höchstens 100 Minuten pro Runde.

## Kontext

- **Wettbewerb:** START Hack Munich 2026, Feyer-Challenge „Faster Physics“. Gewertet wird die **komplette First-Run-Zeit** von `benchmark.py` auf einer CPU. Dazu zählen Imports, Build, Kompilieren, Forward, Sensitivität, Gradient und Transfer.
- **Pflicht:** alle 101 Frequenzen, alle 193 Designvariablen, double precision, `abs(log(s) - log(ref)) <= 1e-5`, endliche Ausgaben, Gradient-Stichprobe bestanden.
- **Team:**
  - Noel (GitHub `noeschi`, Branch `origin/noel`): **klar am weitesten**. Sparse SuperLU über Host-Callbacks, ca. 1,05 s First-Run auf diesem Mac (Apple M2 Pro), ca. 0,5 s auf Noels M4.
  - Philip (`mpb700`, Branch `origin/philip`) und Max (`jmax696`, Branch `origin/max/lu-once`): vollständig in `noel` enthalten.
  - Joel: noch kein Branch. Vermutlich GitHub-Nutzer `Haniubub` mit Fork `Haniubub/fayer-muc`, der bisher nur Kopien enthält.
- **Noel optimiert parallel** und pusht seine Runden auf `origin/noel`. Seine Notizen stehen in `runs/noel/NOTIZEN.md` auf seinem Branch: Messwerte, verworfene Ideen, offene Ideen. **Lies sie in jeder Runde**, damit du nichts wiederholst, was er schon verworfen hat.

## Pfade

```bash
REPO=/Users/maxjackle/Projekte/fayer-muc          # Haupt-Checkout (Max' Branch max/lu-once): Branch und Dateien NICHT ändern, nur runs/nacht/ beschreiben
N=/Users/maxjackle/Projekte/fayer-nacht           # Arbeitsbereich der Nacht
PY=$N/.venv/bin/python                            # Python 3.12 mit allen Paketen (jax 0.9.0, scipy 1.18.1)
REF=$N/ref/referenz-dense.npz                     # Referenz-Arrays (voller Benchmark, dichter JAX-Weg)
S=$REPO/runs/nacht                                # Zustand: state.json, ledger.jsonl, MORGENBERICHT.md, verworfen/, mess/
```

- Worktrees liegen unter `$N/wt/<nr>-<kurzname>` (eigene Varianten) und `$N/kandidaten/<name>-<sha7>` (Kollegen-Stände).
- Anlegen immer mit `git -C $REPO worktree add ...`.
- Für `noel@c261b63` existiert schon `$N/kandidaten/noel-c261b63`, geprüft: PASS, ca. 1,03–1,06 s.

**Umgebung für jede Messung und Prüfung** (macOS kennt keine CPU-Affinität):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
```

Prüfskripte in einem Worktree `W` startest du mit `cd $W && PYTHONPATH=$W $PY ...`. Sonst lädt Python Differometor aus `$REPO`. Kontrolle: `PYTHONPATH=$W $PY -c "import differometor; print(differometor.__file__)"`.

## Schritt 0: Start

1. **Lock:** `mkdir $S/lock`.
   - Wenn das scheitert und `$S/lock` jünger als 130 Minuten ist (`stat -f %m`), läuft schon eine Runde. Dann beendest du dich sofort mit einer Zeile.
   - Ist der Lock älter, ist er verwaist: entfernen und neu anlegen.
   - Am Ende der Runde, auch bei Fehlern, `rmdir $S/lock`.
2. **Zustand lesen:** `$S/state.json`. Fehlt die Datei, ist das die erste Runde. Lege sie an mit:
   - `best`: Commit von `origin/noel`, Label `noel@<sha7>`, Worktree-Pfad,
   - `eigene_commits`: leer,
   - `gesehen`: Branch → letzter gesichteter SHA,
   - `naechste_nr`: 1,
   - `runde`: 0.

   Dann `runde` um 1 erhöhen und sofort speichern.
3. **Schlussrunde:** Die Runde ist die Schlussrunde, wenn `runde` = 4 ist oder es 07:00 Uhr oder später ist. Sie arbeitet ganz normal. Spätestens 85 Minuten nach Start schreibt sie den Schlussbericht mit Aufräumliste (Schritt 3).
   - Ist `runde` größer als 4 (etwa ein Nachhol-Lauf), schreibst du nur den Bericht neu und hörst auf.
4. **Belegung prüfen:** `ps -Ao %cpu,rss,command | sort -rn | head`. Läuft schon ein fremder Benchmark oder eine schwere Rechnung, wartest du bis zu 10 Minuten. Danach misst du trotzdem und vermerkst es.

## Schritt 1: Kollegen-Check (jede Runde, zuerst)

1. **Holen:**
   ```bash
   git -C $REPO fetch --all --prune
   ```
   Forks kommen ohne Remote-Eintrag dazu:
   ```bash
   gh api repos/mpb700/fayer-muc/forks --jq '.[].full_name'
   git -C $REPO fetch https://github.com/<fork>.git '+refs/heads/*:refs/remotes/fork-<owner>/*'
   ```
2. **Sichten:**
   - Für jeden Branch außer `main` (Remote und Forks) vergleichst du den Kopf-SHA mit `gesehen`.
   - Bei neuen Commits: `git log --oneline <alt>..<neu>` und `git diff --stat <alt> <neu>`.
   - Haben sich nur Notizen oder Reports geändert, trägst du das ein. Eine Messung ist dann nicht nötig.
   - Lies auch neue Messwerte in ihren Notizen und Reports.
3. **Messen**, wenn sich Code geändert hat:
   - Worktree detached unter `$N/kandidaten/<branch>-<sha7>` anlegen.
   - Prüfungen aus Schritt 2.4 laufen lassen.
   - Wechselmessung gegen den aktuellen `best` (Schritt 2.5).
4. **Übernehmen**, wenn der Kandidat alle Prüfungen besteht und im Median mindestens 2 % schneller ist als `best`:
   - **Ohne eigene Commits** (`eigene_commits` leer): Der Kandidat wird direkt `best`. Lege dafür den lokalen Branch `nacht/basis-<k>` auf seinem SHA an.
   - **Mit eigenen Commits**, zuerst per Merge (dann kann `origin/max/nacht` ohne Force weiterwachsen):
     - Lege `nacht/basis-<k>` vom aktuellen `best` aus an und führe darin `git merge --no-edit <kandidat-sha>` aus.
     - Bei einem Konflikt `git merge --abort`. Dann `nacht/basis-<k>` neu auf dem Kandidaten-SHA anlegen und unsere `eigene_commits` einzeln per `git cherry-pick -x` daraufsetzen.
     - Konflikt beim Cherry-Pick: `git cherry-pick --abort`, Commit auslassen, vermerken.
   - Mit übernommenen eigenen Änderungen misst du gegen den reinen Kandidaten. Du behältst sie nur, wenn sie dort weiterhin helfen. Sonst wird der reine Kandidat `best`.
   - `best` zeigt danach auf diesen Branch. `eigene_commits` enthält nur noch unsere Commits, die darin wirksam sind.
   - **Hat Noel eine unserer Ideen inzwischen selbst umgesetzt**, gilt sein Stand und unser Commit fällt weg.
5. **Fremde Branches werden nie verändert:** kein Commit, Merge oder Rebase auf `noel`, `philip`, `max/lu-once`, `main`.

## Schritt 2: Varianten (Schleife)

Du arbeitest die Schritte 2.1 bis 2.7 für eine Variante nach der anderen ab, immer auf dem jeweils aktuellen `best`.

- **Eine neue Variante** beginnst du nur, wenn seit dem Rundenstart weniger als 70 Minuten vergangen sind.
- **Nach 85 Minuten** brichst du eine laufende Variante ab. Sie gilt dann als „zurückgestellt“.
- **Danach** folgt Schritt 3.
- **Abwechslung:** Wähle innerhalb einer Runde verschiedene Ansatzarten, z. B. Import, Build, Solver, Callback-Grenze. Nicht zweimal dieselbe.

1. **Idee wählen.** Quellen sind:
   - die offenen Ideen in `runs/noel/NOTIZEN.md` des aktuellen `best`,
   - der Katalog unten,
   - eigene Profilergebnisse.

   Bedingungen:
   - Nichts, was im Ledger auf derselben Basis schon verworfen wurde.
   - Nichts, was Noel verworfen hat. Ausnahme: Du hast einen konkreten neuen Ansatz und schreibst ihn auf.
   - Bevorzuge den größten erwarteten Gewinn im Teil, den wir beeinflussen können.
   - Wechsle zwischen kleinen sicheren und größeren Ideen ab.
2. **Rollen als Subagenten**, je mit vollständigem Auftrag:
   - `recherche`: profiliert den `best`-Stand und liefert eine Hypothese mit Zahlen. Mögliche Werkzeuge: `runs/noel/profil.py`, `$PY -X importtime`, Zeitmessung einzelner Funktionen in Wegwerf-Skripten unter `$N/analyse/`.
   - `umsetzer`: baut genau diese eine Variante im Worktree `$N/wt/<nr>-<kurzname>` auf dem neuen Branch `nacht/<nr>-<kurzname>` (von `best`). Danach lässt er die Prüfungen aus 2.4 laufen. Er misst **keine** Zeiten.
   - `pruefer`: liest `git diff <best>...HEAD` und prüft ihn gegen die harten Regeln. Er beurteilt auch die Prüfausgaben. Antwort: „freigegeben“ oder „abgelehnt: Grund“.
   - **Zeitmessungen machst nur du selbst**, und nur, wenn kein Subagent gerade rechnet.
3. **Zeitlimit für die Umsetzung: 25 Minuten.** Danach wird die Variante „zurückgestellt“, mit einer Notiz, woran es hängt.
4. **Prüfungen** (im Worktree, mit `PYTHONPATH=$W`). Alle müssen so gut ausfallen wie auf `best`:
   - `$PY runs/check_fast.py $REF --fallback`. Grenzen, jeweils maximale Abweichung relativ zum Maximum:

     | Größe | Grenze |
     | --- | --- |
     | carrier | 1e-10 |
     | powers | 1e-9 |
     | signal | 1e-6 |
     | noise | 1e-6 |
     | sensitivity | 1e-7 |
     | gradient | 1e-6 |

     Dazu: Log-Fehler ≤ 9e-6 und höchstens 1e-7 schlechter als bei `best`.
   - `$PY runs/check_setups.py` (aLIGO, Voyager usw.).
   - `$PY runs/noel/test_detectors.py .`
   - `$PY benchmark.py --repeats 3 --output $S/mess/<nr>/check.json` muss **PASS** ausgeben.
5. **Wechselmessung** mit Noels Skript, frische Prozesse im Wechsel:
   ```bash
   cd $N && env -u PYTHONPATH $PY <best-worktree>/runs/noel/ab.py 8 basis=<best-worktree> variante=<varianten-worktree>
   ```
   Sichere die Ausgabe nach `$S/mess/<nr>/ab.txt`. Einzelne Läufe streuen um bis zu 40 ms, darum zählen Median und Minimum.
   - **Akzeptiert:** Der Median von TOTAL sinkt um ≥ 2 %, das Minimum sinkt auch, die Prüfungen bleiben gleich gut und der `pruefer` gibt frei.
   - **Grenzfall** 0,5–2 %: Wiederhole mit 12 Wechseln. Akzeptiert ist die Variante nur, wenn beide Messungen ≥ 1 % Gewinn zeigen.
6. **Entscheidung umsetzen:**
   - **Akzeptiert:**
     - Commit auf `nacht/<nr>-<kurzname>`, englische Commit-Nachricht, letzte Zeile `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
     - Die Commit-Nachricht nennt die Messung: Median und Minimum von TOTAL vorher und nachher, Gerät „M2 Pro, macOS, Threads auf 1“, Basis-SHA.
     - Neuen SHA an `eigene_commits` anhängen. `best` zeigt jetzt auf diesen Branch und Worktree.
     - **Pushen** nach Schritt 2.7.
     - Den alten `best`-Worktree erst entfernen, wenn ihn nichts mehr braucht. Kandidaten-Worktrees von Kollegen bleiben, solange sie `best` oder Vergleichsbasis sind.
   - **Verworfen oder zurückgestellt:**
     - `git -C <worktree> diff > $S/verworfen/<nr>-<kurzname>.patch`.
     - Worktree entfernen (`git -C $REPO worktree remove --force <pfad>`), Branch löschen (`git -C $REPO branch -D nacht/<nr>-<kurzname>`). Das gilt nur für eigene `nacht/*`-Branches.
7. **Pushen** (von Max freigegeben, nur dieser eine Ziel-Branch):
   - Wann: Nur wenn `best` mindestens einen eigenen Commit enthält und alle Prüfungen aus 2.4 auf genau diesem Stand bestanden sind. Das gilt auch nach einer Übernahme in Schritt 1.4, wenn eigene Commits darin stecken.
   - Wie: `git -C <best-worktree> push origin HEAD:refs/heads/max/nacht`, ohne Force.
   - Ziel: der Branch `max/nacht` auf `origin`. Noel sichtet in seinen Runden alle Branches und kann übernehmen, was hilft.
   - **Wird der Push abgelehnt** (kein Fast-Forward), niemals `--force` oder `--force-with-lease`. Stattdessen nach `refs/heads/max/nacht-<JJJJMMTT-HHMM>` pushen und das im Bericht vermerken.
   - Gepushten SHA und Branch in `state.json` (`gepusht`) und im Ledger festhalten.
   - Ein Push-Fehler (z. B. keine Berechtigung) bricht die Runde nicht ab. Ins Protokoll damit, weiter ohne Push.

## Schritt 3: dokumentieren (jede Runde)

1. **Ledger:** Pro Variante oder Kandidatenmessung eine JSON-Zeile in `$S/ledger.jsonl`. Felder:
   - Herkunft: `zeit`, `runde`, `nr`, `art` (variante/kollege), `branch`, `basis`, `sha`,
   - Inhalt: `idee`, `aenderung`,
   - Prüfungen: `pruefungen` (Kurzwerte), `pass`,
   - Zeiten: `total_median_ms`, `total_min_ms`, `basis_median_ms`, `basis_min_ms`, `phasen_ms`, `warm_ms`, `log_fehler`,
   - Ergebnis: `entscheidung`, `grund`.
2. **Zustand:** `$S/state.json` aktualisieren, `naechste_nr` hochzählen.
3. **Bericht:** `$S/MORGENBERICHT.md` jede Runde **neu schreiben**, auf Deutsch, für Max. Inhalt:
   - Bester Stand: Branch, SHA, First-Run Median und Minimum auf diesem Mac, Vergleich zu `origin/noel` beim Start der Nacht und zum aktuellen `origin/noel`. Welche Prüfungen er besteht.
   - Kollegen-Log: was es Neues gab, was gemessen und was übernommen wurde.
   - Tabelle aller Varianten der Nacht: Nr, Idee, Ergebnis in ms, Entscheidung, Grund.
   - Erkenntnisse, zurückgestellte Ideen, nächste Kandidaten.
   - Was auf `origin/max/nacht` liegt (SHA, Messung) und ob Noel davon schon etwas übernommen hat.
   - Ab der Schlussrunde: Aufräumliste mit allen `nacht/*`-Branches und Worktrees unter `$N`.
4. **Schluss:** Lock freigeben. Deine letzte Nachricht fasst die Runde in 3–5 Zeilen zusammen: Runde x von 4, bester Stand, was neu probiert wurde, was gepusht wurde, was als Nächstes kommt.

## Harte Regeln

1. **Push-Regeln:**
   - Gepusht wird **nur** nach Schritt 2.7: auf `origin/max/nacht`, im Notfall auf `origin/max/nacht-<zeitstempel>`, nie mit Force.
   - Keine Pull Requests, keine Issues, keine Kommentare, keine Nachrichten an Kollegen.
   - Sonst auf GitHub nur lesen (`fetch`, `gh api` GET).
2. **Fremde Branches und Max' Haupt-Checkout bleiben unverändert:**
   - Kein Commit, Merge, Rebase oder Checkout auf `noel`, `philip`, `max/lu-once`, `main`.
   - In `$REPO` kein `git checkout`/`switch`/`reset`/`stash`. Schreiben nur unter `$REPO/runs/nacht/`.
3. **Löschen nur, was diese Nacht selbst angelegt hat:** `nacht/*`-Branches, Worktrees unter `$N`, Dateien unter `$S`.
4. **Nicht ändern:** `benchmark.py`, `plot_timings.py`, `examples/data/*`, `requirements.txt`, `pyproject.toml`, `setup.py`. Keine neuen Pakete.
5. **Physik vollständig:**
   - alle Frequenzen und Designvariablen, `float64`/`complex128`,
   - keine vorberechneten Ergebnisse, kein Persistent Compilation Cache, kein Caching über Prozesse hinweg,
   - keine `XLA_FLAGS` oder Thread-Variablen aus der Library,
   - keine Stellvertreter-Objekte, die den Readout umgehen,
   - keine gelockerten Toleranzen,
   - kein Weglassen kleiner Terme wie `(1 + f/F0)`.
6. **Den alten dichten JAX-Weg nicht voll laufen lassen.** `main` braucht dafür ca. 14 GB RAM, der Mac hat 16 GB. `--fallback` in `check_fast.py` rechnet nur 3 Frequenzen und ist in Ordnung.
7. **Keine kostenpflichtigen Dienste** (RunPod o. ä.).
8. **Ehrlich bleiben:** Keine Zahl ohne Messdatei. Gescheiterte Versuche werden genauso dokumentiert. Bei Regelzweifeln die Idee überspringen und im Bericht als Frage an Max notieren.

## Ideenkatalog (Startpunkt, Stand `noel@c261b63`)

Auf diesem Mac dauert ein Lauf ca. 1,05 s. Die Phasen laut `ab.py`, 00:55 Uhr:

| Phase | ms |
| --- | ---: |
| imports | ca. 790–800 |
| build | ca. 80 |
| compile | ca. 89 |
| forward | ca. 50 |
| gradient | ca. 23 |
| warm | ca. 63 |

Laut Noels Profil (M4, 508 ms) sind ca. 400 ms fix: `import jax`, `importlib.metadata` und Readout in `benchmark.py`. Hier dominieren die Imports. Prüfe deshalb früh mit `-X importtime`, welcher Anteil davon auf Differometor und seine Folge-Imports entfällt (z. B. `scipy`-Teile) und welcher auf `jax` selbst. Unser Spielraum liegt bei Imports von Differometor, Build, Callbacks, Forward und Gradient.

| Kürzel | Idee | Ziel | Hinweis |
| --- | --- | --- | --- |
| N1 | Sparse-Form der Basismatrizen schon im Build erzeugen, statt 2 × 9 MB dichte Matrizen durch die Callbacks zu schicken | Forward 5–8 ms | Noels offene Idee 1 |
| N2 | `build.py` straffen (`set_instructions`, `pairs_to_arrays`), eager `jnp` → NumPy | Build ca. 3 ms | Noels Idee 2 |
| N3 | Triviale Unbekannte vor der Zerlegung herausnehmen (95 Zeilen und 22 Spalten mit nur Diagonaleintrag) | Forward ca. 2 ms | Noels Idee 3 |
| K1 | CSC-Struktur einmal festlegen. Pro Frequenz nur noch das `data`-Array über eine feste Indexabbildung füllen, kein COO→CSC-Umbau | Forward | Muster ist für alle Frequenzen gleich |
| K2 | Spaltenordnung einmal bestimmen (`permc_spec` auf erster Frequenz) und für alle Frequenzen wiederverwenden | SuperLU 14 ms | Zeilenpivotisierung bleibt. Stabilität über alle 101 Frequenzen prüfen |
| K3 | Verschiedene `permc_spec` (COLAMD, MMD_AT_PLUS_A, MMD_ATA) und `DiagPivotThresh` messen | SuperLU | Nur behalten, was alle Prüfungen besteht |
| K4 | Kopien an der Host-Grenze vermeiden: `np.asarray` ohne Kopie, zusammenhängende `complex128`-Puffer, vorab angelegte Ausgabe-Arrays | Forward/Gradient | Profil zeigt „Rest Übergabe“ |
| K5 | Weniger `pure_callback`-Aufrufe: Forward und Hilfsgrößen für den Gradienten in einem Callback zusammenfassen, Residuals wiederverwenden | Kompilieren + Laufzeit | Kleinerer Graph |
| K6 | Was der Differometor-Import noch lädt (`-X importtime`), verzögern | Imports | Noel: ca. 15 ms übrig |
| K7 | Einträge der Matrizen auf dem Host direkt in NumPy auswerten statt in XLA (passend zum Callback-Weg) | Kompilieren | Rückwärtsregel nur für Designvariablen |
| K8 | Rauschen und Signal-Readout gemeinsam vektorisieren, unnötige komplexe Konjugationen und Zwischenarrays streichen | Forward | Klein, aber sicher |

**Von Noel verworfen, nicht wiederholen:**
- eigene Sparse-LU ohne Zeilentausch (instabil, Fehler 1e75),
- eigene gebündelte LU in NumPy,
- dichte Teilblöcke,
- XLA-Flags und Kompilier-Cache,
- Stellvertreter-Objekte.
