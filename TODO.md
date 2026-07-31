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

# Offene Aufgaben

## 0. Erster Lauf auf dem echten Datensatz  ← nächster Schritt

Der Datenteil ist fertig, der Lauf darauf nicht. Was jetzt steht:

- **780 Paare** in `data/raw`, als *ein* DVC-Verzeichnis (vorher 16 Einzel-Pointer).
  Import und Namens-Normalisierung: `src-train/import_raw.py`.
- **DVC-Remote** ist `s3://diwop-leichte-sprache/dvc` — derselbe Bucket wie `S3_BUCKET`.
  Zuerst lag er neben dem Quellkorpus unter `s3://diwop-analysis/dvc`; der RunPod-Lauf am
  2026-07-31 ist genau daran gescheitert:

      ERROR: failed to connect to s3 (diwop-analysis/dvc/files/md5)
             Forbidden: An error occurred (403) when calling the HeadObject operation

  Die RunPod-Rolle darf `diwop-analysis` nicht lesen. Zurückverlegen nur mit einer
  entsprechenden IAM-Berechtigung (`s3:GetObject` + `s3:ListBucket` auf dem Prefix).
  Persönliche SSO-Credentials in die Pod-Umgebung zu exportieren ist **kein** Ausweg:
  `scripts/start_runpod.sh` kopiert Secrets bewusst nicht in die Pod-Env (Zeilen 108-112).
- **Splits** 70/10/20 → `data/train/dataset.jsonl` (533), `data/train/validation.jsonl`
  (75), `data/eval/holdout.jsonl` (167). Die Zuordnung ist `sha256(salt:id)`, kein
  Shuffle — neue Dokumente verschieben kein einziges altes über die Holdout-Grenze.
- **`data/excluded.json`**: Paare, die aus *allen* Splits fliegen, mit Begründung. Liegt in
  git statt in DVC, damit die Entscheidung im PR review-bar ist. Aktuell `0013` und `0224`
  — beides keine Übersetzungen, sondern andere Texte zum selben Thema. Weil die
  Split-Zuordnung pro ID läuft, verschiebt ein Ausschluss nichts anderes.
- **Holdout-Sperre**: `scripts/train.sh` zieht nur benannte Dateien und bricht im Container
  ab, wenn `data/eval/holdout.jsonl` doch da liegt.
- **Validation im Training**: `test_datasets` → Eval-Loss pro Epoche,
  `load_best_model_at_end` auf `eval_loss`, plus `src-train/validation_metrics.py`
  (generiert auf ein paar Validation-Samples und misst `src-eval/rules.py` +
  Flesch/Wiener als Abstand zur menschlichen Referenz).
- **`run_manifest.json`** liegt jetzt neben dem Adapter: Basismodell, Config-Hash und die
  exakten IDs pro Split.

Offen und noch **nicht** verifiziert:

- **GitHub-Secrets**: Der CI-Job `validate-data` macht `dvc pull`. Das brauchte bisher keine
  Credentials (Remote war das lokale `data/s3-mock`), jetzt schon. `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY` und optional `AWS_DEFAULT_REGION` müssen als Repository-Secrets
  angelegt werden, sonst schlägt jeder PR-Build fehl.
- Generierung im Callback unter **DeepSpeed ZeRO-3** — lokal auf MPS läuft sie, auf
  mehreren Karten ist sie ungetestet. Sie ist gekapselt: ein Fehler loggt eine Warnung und
  killt den Lauf nicht. Notausgang: `VALIDATION_METRICS_OFF=1`.
- Laufzeit dieser Generierung auf dem 26B. Defaults sind bewusst klein
  (`VALIDATION_METRICS_SAMPLES=4`, `VALIDATION_METRICS_MAX_TOKENS=256`); vor dem Hochdrehen
  einmal `eval_ls_seconds` im Log ansehen.
- `num_epochs: 3` steht weiter auf 3. Jetzt gibt es zum ersten Mal eine Eval-Kurve, an der
  man das entscheiden kann.

**Erster echter Lauf auf E2B (120 Schritte, 279 Trainingsdokumente, 2026-07-31):**

| Schritt | `eval_loss` | `eval_ls_distance` | kurze Sätze | Bindestrich-Komposita/100w |
|---|---|---|---|---|
| 0 (Basis) | 4.016 | 0.2625 | 88 % | 3.95 |
| 60 | 1.264 | **0.0706** | 78 % | **2.05** |
| 120 | **1.204** | 0.2082 | 63 % | 3.85 |
| Mensch | — | 0 | 72 % | 1.40 |

Zwei Dinge daraus:

1. **Die Daten lehren den Stil.** Nach 60 Schritten schreibt das Modell
   `Boccia-Kugel` statt `Boccia Kugel`, einen Satz pro Zeile (100 %, exakt auf
   Referenzniveau) und Aufzählungen als Listen. Das ist die Typografie der Leichten
   Sprache, gelernt aus 60 Beispielen.
2. **`eval_loss` und `eval_ls_distance` laufen auseinander.** Zwischen Schritt 60 und 120
   sinkt der Loss weiter (1.264 → 1.204), der Stilabstand verdreifacht sich aber
   (0.0706 → 0.2082) — die Bindestriche gehen fast vollständig wieder verloren.
   `load_best_model_at_end` auf `eval_loss` hat deshalb **checkpoint-120 gewählt, also den
   schlechter formatierten**. Vor dem 26B-Lauf zu entscheiden, ob
   `metric_for_best_model` bleibt oder auf eine Kombination umgestellt wird.

   ⚠️ `eval_ls_distance` steht auf 8 Validation-Samples, `eval_loss` auf 37 — ein Teil des
   Ausschlags kann Rauschen sein. Vor einer Entscheidung `VALIDATION_METRICS_SAMPLES`
   hochdrehen und den Lauf wiederholen.

   `eval_ls_distance` sollte **nicht** allein zum Auswahlkriterium werden: Bei Schritt 60
   behauptet das Modell `Dafür braucht man einen Rollstuhl` — frei erfunden, die Quelle sagt
   „eingeschränkte Mobilität". Der Stil war da am besten, der Inhalt am schlechtesten. Genau
   dafür braucht es P2-1 (Bedeutungserhalt) als Gegengewicht.

**Datenqualität, neu und unangenehm:** Der Median des Wortzahl-Verhältnisses
(Leichte Sprache ÷ Standard) liegt über alle 780 Paare bei **0.61** — die Leichte-Sprache-
Seite ist meistens *kürzer*. Die Annahme in `docs/train-eval-review.md`, Leichte Sprache
expandiere um 1.5–3x, stammt aus den ursprünglichen 8 Dokumenten und gilt für den echten
Korpus nicht. Das Tier-A-Band in `src-eval/rules.py` ist entsprechend neu kalibriert
(`[0.2, 2.2]`, 5./95. Perzentil), flaggt jetzt ~11% statt 89%. Stichproben zeigen aber ein
tieferliegendes Problem: Es gibt Paare, die dasselbe *Thema* behandeln, aber keine
Übersetzungen voneinander sind. Siehe `docs/data.md`.

## 1. Linux-Pfad auf RunPod verifizieren

Die vereinheitlichten Skripte sind **ausschließlich auf macOS getestet**. `src-train/train.py`
hat jetzt eine Geräte-Erkennung (`detect_accelerator()`), und DeepSpeed sowie
FlashAttention-2 werden nur noch bei CUDA injiziert. Das ist so geschrieben, dass sich das
CUDA-Verhalten nicht ändert — belegt ist es nicht.

**Konkret zu prüfen:**
- `scripts/setup.sh` installiert im Container nur das Delta, legt kein venv an
- `detect_accelerator()` liefert `("cuda", N)` mit der tatsächlichen Kartenzahl, DeepSpeed
  ZeRO-3 wird injiziert
- `scripts/eval.sh` behält nvidia-smi-Gate, `pkill VLLM::EngineCore` und die
  Inductor-Cache-Umleitung (MooseFS)
- Artefakte landen weiter unter `/app/output` (`OUTPUT_ROOT`)
- `TP_SIZE` kommt jetzt aus `nvidia-smi --list-gpus` statt aus einer fest verdrahteten `2`,
  und `CUDA_VISIBLE_DEVICES` wird daraus abgeleitet statt auf `0,1` festgenagelt. Auf einer
  Ein-Karten-Instanz (1x RTX PRO 6000) muss also `TP_SIZE=1` herauskommen.

**Durchführung:** ein kurzer Trainingslauf mit gedeckeltem `max_steps`, danach Eval:

    bash scripts/start_runpod.sh train
    bash scripts/start_runpod.sh eval

Das Skript sucht zuerst einen vorhandenen GPU-Pod, legt sonst nach Rückfrage einen neuen an,
setzt Image und Start-Kommando, wartet auf den Pod und hängt sich an das Log. Es ist bisher
nur gegen eine nachgebaute API getestet — der erste echte Pod ist gleichzeitig sein Test.

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

  **Der RunPod-Pfad ist davon nicht betroffen** — dort installiert `setup.sh` nur
  `uv pip install src-train/` auf das fertige Axolotl-Image, ohne axolotl selbst.

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
