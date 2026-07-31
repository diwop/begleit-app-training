# Gemma Finetuning Leichte Sprache

Aktueller Git-Branch: evaluate-separately-jpods

# Next steps (high level)

~~Mit den 800 Texten (- 20 % Testsplit) trainieren (+ die Daten dann über DVC in AWS
ablegen)~~ — **Daten und Splits stehen** (siehe unten, Punkt 0); der erste echte
Trainingslauf darauf steht noch aus.

Inferenzlauf mit
- Testsplit
- 20% des Trainingssplits
- Ganz einfacher LLM as a Judge mit Schulnoten (schon mit Lebenshilfe halb vorbereitet)
  auf
- Gemma 4 mit Reasoning
- Gemma 4 mit Fine Tuning & Reasoning
- Gemma 4 mit Reasoning und Few Shots (ggf. dynamisch aus den Beispielübersetzungen)
- Mistral Small 4 ohne Reasoning
- Mistral Small 4 mit Reasoning
- Schomacker


## AKtueller Stand

- läuft lokal
- ~~Fehler in RunPod beim ersten predict-Schritt des Trainings~~ — **gelöst am 2026-07-31.**
  Zwei verschiedene cuBLAS-Builds im selben Prozess: `libcublas` aus dem pip-Wheel,
  `libcublasLt` aus dem System-CUDA des Images. `scripts/lib/platform.sh` stellt jetzt die
  Wheel-Bibliotheken im `LD_LIBRARY_PATH` voran. Details in `failures-and-fixes.md`,
  Training-Iteration 15. Das ist ein Notnagel, kein Fix — siehe Punkt 1.
- Verifiziert im Lauf vom 2026-07-31 23:02 UTC: 533 Trainings- und 75 Validierungsbeispiele
  geladen, 26B-Modell mit LoRA (37,2 M trainierbare Parameter, 0,14 %), erste Evaluation
  vollständig durchgelaufen — `eval_loss 3.5`, `eval_ppl 33.11`, 44 GiB von 96 GiB belegt.
  Der Lauf wurde danach von Hand beendet; ein vollständiger Lauf steht noch aus.

# Offene Aufgaben

## 1. Reproduzierbare Trainingsumgebung  ← nächster Schritt

`scripts/lib/platform.sh` sortiert heute den `LD_LIBRARY_PATH` so um, dass die
CUDA-Bibliotheken aus den torch-Wheels vor denen des Images gefunden werden. Das behebt das
Symptom von Iteration 15 zuverlässig, aber **die Ursache bleibt**: das Image bringt eine
vollständige CUDA-Installation unter `/usr/local/cuda-13.0` mit, die Wheels bringen eine
zweite unter `site-packages/nvidia/`, beide in unterschiedlichen Versionen, und nichts
verhindert, dass sie im selben Prozess gemischt werden. Ein Pfad-Patch ist die falsche
Abstraktionsebene — er gewinnt ein Wettrennen, statt es abzuschaffen.

Gewollt ist eine Umgebung, die *garantiert* keine inkompatiblen Versionen still
nebeneinander installiert. Zwei Richtungen, beide noch zu bewerten:

1. **Eigene Umgebung mit `pixi`.** Conda-forge liefert CUDA als echte Pakete mit
   Abhängigkeiten, statt als zwei unabhängige Kopien; der Solver kann Konflikte dann
   überhaupt sehen. Preis: wir bauen den kompletten Stack (torch, DeepSpeed,
   FlashAttention, axolotl) selbst und geben das fertige Axolotl-Image auf.
2. **Auf einem bekannten Basis-Image aufsetzen, aber mit harten Garantien.** Das Image
   bleibt die Wahrheit für torch und CUDA; unsere Installation darf dann nichts
   torch-Nahes mehr anfassen. Offen ist, womit sich das *erzwingen* lässt — Kandidaten:
   `uv pip install --no-deps` für die eigenen Pakete, `[tool.uv] constraint-dependencies`
   gegen die im Image vorhandenen Versionen, oder ein Check nach der Installation, der
   doppelte CUDA-Bibliotheken findet und laut abbricht.

Entscheidungsgrundlage sollte sein, welche Variante den Fehler *unmöglich* macht, nicht
welche ihn heute vermeidet. Verwandt: der Lockfile-Punkt unter „Weitere offene Punkte".

## 2. FP8-Basismodell mit Adapter testen

Das eigentliche Deployment-Ziel. Getestet wurde bisher nur die bf16-Basis (51 GB); FP8
wäre 28 GB.

**Der Adapter muss nicht quantisiert werden.** LoRA-Adapter bleiben bf16/fp16, unabhängig
von der Quantisierung der Basis; die Engine rechnet das Delta in höherer Präzision oben
drauf. Quantisierung ändert keine Tensor-Shapes, der vorhandene 71-MiB-Adapter passt
unverändert.

**Was gebraucht wird:** eine GPU mit FP8-Tensor-Cores, also Ada (sm_89) oder neuer.
Die aktuelle A100 ist Ampere (sm_80) und kann es nicht.

`scripts/start_runpod.sh` bietet nur noch FP8-fähige Karten an — die Allow-Liste `FP8_GPUS`
im Skript ist genau diese Anforderung, in ausführbarer Form. Eine Ampere-Karte kann man
weiterhin wiederverwenden, bekommt dann aber eine Warnung.

Verfügbar am 2026-07-31, mit ≥80 GB gesamt, in der Reihenfolge, in der das Skript sie
anbietet — eine Karte unter 2 $/h zuerst, dann mehrere unter 2 $/h, dann bis 3 $/h:

| GPU | VRAM gesamt | $/hr | Anmerkung |
|---|---|---|---|
| 1x RTX PRO 6000 Blackwell WS | 96 GB | 1.89 | Einzelkarte, Verfügbarkeit Low |
| 1x RTX PRO 6000 Blackwell SE | 96 GB | 1.99 | Einzelkarte, **Verfügbarkeit High** |
| 2x RTX 6000 Ada | 96 GB | 1.68 | billiger, aber zwei Karten |
| 2x L40S | 96 GB | 1.98 | bisher genutzt, Low |
| 1x H100 SXM | 80 GB | 2.99 | letzte Wahl im Budget, High |

Für den reinen FP8-Test reichen 28 GB Basis + Adapter, also genügt hier auch eine einzelne
48-GB-Karte (L40S, RTX 6000 Ada) — die 80-GB-Untergrenze im Skript stammt vom Training.
Dafür `MIN_TOTAL_VRAM_GB=48` setzen.

**Durchführung:** nur zwei Env-Variablen, keine Code- oder Adapter-Änderung.

    SMOKE_BASE=RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic
    SMOKE_ADAPTER_S3=20260728_152846_run

Nicht zu verwechseln mit dem gemergten FP8-Modell unter
`models/20260728_152846_run/train-gemma4-fp8/` — das ist der Fallback-Pfad ohne Adapter.

## 3. Kompilierten Pfad verifizieren (`SMOKE_EAGER=0`)

Der erfolgreiche Lauf verwendete `enforce_eager=True` und hat `torch.compile` und das
CUDA-Graph-Capturing **übersprungen**. Der Lauf davor ist genau dort nach ~23 Minuten
gestorben:

    torch._inductor.exc.InductorError: OSError: [Errno 5] Input/output error

Das passierte in `_initialize_kv_caches → profile_run → aot_compile`, also in der normalen
vLLM-Engine-Initialisierung — **nicht** in etwas Smoke-Test-Spezifischem.

Der Fehler ist **nicht behoben, nur umgangen**. Ursache unbekannt. Möglicherweise transient.

**Durchführung:** einmal `SMOKE_EAGER=0` auf einem warmen Pod (~25 min). Sinnvoll zusammen
mit Aufgabe 1 und 2 auf demselben Pod.

## 4. Wirkung des `eot_tokens`-Fixes am bestehenden Adapter prüfen

`config/train-gemma4.yml` benannte `<end_of_turn>` — ein Gemma-3-Token, das es in Gemma 4
nicht gibt (es zerfällt in 7 Text-Tokens). Der echte Turn-Terminator ist `<turn|>` (ID 106,
zugleich zweiter Eintrag in `eos_token_id: [1, 106]`). Ist inzwischen korrigiert; verifiziert
über die Wirkung, nicht nur über die verschwundene Warnung: trainierbare Tokens stiegen von
12.570 auf 12.600, also exakt +1 pro Beispiel — der Terminator liegt jetzt im trainierten
Bereich. Siehe `failures-and-fixes.md`, Training Iteration 10.

**Zu prüfen:** Alle vor diesem Fix trainierten Adapter haben nie gelernt aufzuhören.
Symptom wäre eine Generierung, die über das Ende der Antwort hinausläuft. Beim bestehenden
RunPod-Adapter nachsehen, und bei Bedarf neu trainieren.

---

# Weitere offene Punkte (niedrigere Priorität)

- **Pod-Lifecycle-Skript**: `scripts/start_runpod.sh` deckt jetzt Anlegen, Wiederverwenden
  und Anhängen ab (Punkte 1-13 aus `docs/launcher-hardening.md`). Offen bleiben die Punkte,
  die nicht am Launcher hängen: ein Netzwerk-Volume gegen den 50-GB-Download pro neuem Pod
  (Punkt 20) und `--stop-after` (Punkt 24) — das gibt es in der RunPod-API nicht, an seiner
  Stelle stoppt `scripts/lib/finish.sh` den Pod am Ende des Laufs.
- **`post_training_merge: false`** in `config/train-gemma4.yml` setzen, sobald der
  Adapter-Pfad im Deployment steht. Spart den Merge-Schritt und 28 GB Download pro Eval.
  In den lokalen Configs steht es bereits auf `false`.
- **`wget`** zur Installationsliste in `dockerStartCmd` hinzufügen oder `docs/pipeline.md`
  auf `curl` umstellen — das vLLM-Image hat keines von beiden.
- **Trainingsdaten**: 8 Beispiele. Der Adapter wird angewandt, aber die Wirkung ist
  entsprechend klein. Das ist die eigentliche Begrenzung, nicht die Inferenz.
- **`scripts/setup.sh` durch einen echten Lockfile ersetzen.** Das lokale Training-venv wird
  aktuell in vier Schritten von Hand zusammengebaut: `src-train/[local]`, dann `axolotl`,
  dann ein Reparatur-Schritt, der drei Pins zurückdreht, die axolotl überschreibt
  (`antlr4-python3-runtime==4.9.3` für omegaconf, `boto3`/`botocore` für dvc[s3] und awscli,
  `torchvision` gegen die ABI von axolotls torch). Jeder dieser Punkte war ein
  `ImportError` zur Laufzeit, keiner davon eine Warnung.

  Das ist Reihenfolge-abhängig und damit fragil. Gewollt wäre ein konsistenter,
  plattformübergreifender Lockfile, den `uv sync` einfach auflöst — eine deklarative
  Wahrheit statt einer Abfolge von Kommandos.

  Der Grund, warum es das heute nicht gibt, steht in `src-train/pyproject.toml`: axolotl
  pinnt `antlr4-python3-runtime==4.13.2`, `dvc[s3]` zieht ein `hydra-core`, das dem
  widerspricht, und `uv`s universeller Lock muss alle Plattformen gleichzeitig erfüllen —
  diese Kombination ist unlösbar. Mögliche Auswege, in der Reihenfolge, in der sie
  wahrscheinlich funktionieren:
  1. `dvc[s3]` aus dem Training-venv herausnehmen und den DVC-Pull einem eigenen Tool-venv
     überlassen (`uv tool install dvc[s3]`). Damit verschwindet der hydra-Konflikt komplett,
     und `scripts/train.sh` ruft ohnehin schon eine `dvc`-Binary auf.
  2. `[tool.uv] constraint-dependencies` bzw. `override-dependencies` nutzen, um axolotls
     antlr-Pin einmal zentral zu überstimmen, statt hinterher zu reparieren.
  3. Getrennte Lockfiles pro Plattform (`uv lock --python-platform`), falls ein universeller
     Lock unerreichbar bleibt.

  **Vom antlr-Konflikt ist der RunPod-Pfad nicht betroffen** — dort installiert `setup.sh`
  nur `uv pip install src-train/` auf das fertige Axolotl-Image, ohne axolotl selbst.
  Reproduzierbar ist er deshalb aber nicht: Iteration 15 hat gezeigt, dass genau dieser
  Pfad eine zweite CUDA-Installation neben die des Images stellt. Punkt 1 behandelt das.

- **Frühe Fehler im Container sind in S3 unsichtbar.** `scripts/train.sh` startet den
  Live-Sync (`scripts/lib/s3_sync.sh`) erst kurz vor dem Training, und `finish.sh` — das den
  Log nach S3 hochlädt — läuft nur, wenn das Skript bis zum Ende kommt. Alles davor (DVC-Pull,
  Holdout-Guard, `setup.sh`) stirbt wegen `set -e` lautlos: der Bucket bleibt leer, und der
  einzige Hinweis steht im RunPod-Log des Pods. Genau so lief der 403 am 2026-07-31.

  Sauber wäre, die Ausgabe des ganzen Skripts von der ersten Zeile an in `$LOG_FILE` zu
  spiegeln und den Log auch im Fehlerfall (`trap ... ERR`) hochzuladen. Nicht während eines
  laufenden Incidents umgebaut.

- **`src-eval/evaluation.py`** ist verwaist: die alte Readability-Metrics-Pipeline, wird von
  keinem Skript mehr aufgerufen, importiert vLLM auf Modulebene und hängt an `/app`-Pfaden.
  Entweder reaktivieren oder löschen.
- **MPS ist nicht deterministisch**: derselbe lokale Lauf liefert mal `3/3`, mal `1/3`
  geänderte Ausgaben. Bestanden ist alles über 0, aber `1/3` ist nah an der Grenze. Bei
  Bedarf `max_steps`/`learning_rate` in `train-gemma4-tiny.yml` erhöhen.
