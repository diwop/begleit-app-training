# Gemma Finetuning Leichte Sprache

Aktueller Git-Branch: evaluate-separately-jpods

# Next steps (high level)

Mit den 800 Texten (- 20 % Testsplit) trainieren (+ die Daten dann über DVC in AWS ablegen)

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

## 1. Linux-Pfad auf RunPod verifizieren  ← nächster Schritt

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
- **`src-eval/evaluation.py`** ist verwaist: die alte Readability-Metrics-Pipeline, wird von
  keinem Skript mehr aufgerufen, importiert vLLM auf Modulebene und hängt an `/app`-Pfaden.
  Entweder reaktivieren oder löschen.
- **MPS ist nicht deterministisch**: derselbe lokale Lauf liefert mal `3/3`, mal `1/3`
  geänderte Ausgaben. Bestanden ist alles über 0, aber `1/3` ist nah an der Grenze. Bei
  Bedarf `max_steps`/`learning_rate` in `train-gemma4-tiny.yml` erhöhen.
