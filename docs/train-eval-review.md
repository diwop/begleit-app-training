# Train/Eval Pipeline Review

Critique of the current training and evaluation process and the changes that follow from
it, ordered by priority. Every finding below is anchored to a file and line in the repo as
of branch `evaluate-separately-jpods` (2026-07-30).

**Architecture assumption:** evaluation stays in the separate vLLM container, as today.
See [Decision: eval container vs. training container](#decision-eval-container-vs-training-container)
for why that is workable and what it costs.

> **Superseded in part (2026-07-31).** The corpus is now 780 pairs, not 8, which settles
> some of this and refutes one of its premises. See `docs/data.md`.
>
> * **Done:** P0-3 (split + `run_manifest.json`), P1-2 (validation loss during training,
>   plus generated rule metrics — `src-train/validation_metrics.py`). P0-2's ruleset exists
>   as `src-eval/rules.py`.
> * **Refuted:** the claim under Finding 1 that Leichte Sprache expands a text 1.5–3×. That
>   came from the eight documents available at the time, whose median ratio was 2.28. Over
>   780 professional pairs the median is **0.61** — the Leichte Sprache side is usually
>   *shorter*. The Tier A band has been recalibrated from `[1.3, 3.5]` to `[0.2, 2.2]`; the
>   old band flagged 89% of the corpus. The four numbers in the table below, and every
>   argument that rests on n=8, should be read as history.
> * **Still open, and now the main data risk:** spot checks found pairs whose two texts
>   share a topic but are not translations of each other. That is P2-1, and it matters more
>   than the length ratio ever did.

## The four numbers that frame everything

| | value | where |
|---|---|---|
| training pairs | **8** | `data/raw/` — 16 text files |
| trainable parameters | **~37 million** | 71 MiB bf16 adapter, `lora_r: 32` (`config/base.yml:3`) |
| validation samples | **0** | `val_set_size: 0` (`config/base.yml:15`) |
| holdout samples scored today | **0** | see P0-4 below |

Roughly 37 million adjustable numbers are fitted to 8 documents over 3 epochs
(`config/base.yml:25`) with nothing measuring whether the model learns the task or
memorises the examples. Every recommendation below is a consequence of that.

## Findings

| # | finding | evidence | severity |
|---|---|---|---|
| 1 | One of 8 pairs is not a translation | `0002`: 358 → 5148 words (14.4x) | critical |
| 2 | No Leichte Sprache quality is measured | only Flesch + Wiener Sachtextformel | critical |
| 3 | Training and evaluation use the same samples | `evaluation.py` Part 3 reads `dataset.jsonl` | critical |
| 4 | The scored eval never runs | `evaluation_set[:8]` (`evaluation.py:308`) | critical |
| 5 | Reference texts are loaded but never used | `reference_text` (`evaluation.py:304`) is read, never scored | high |
| 6 | Evaluation is non-deterministic | `temperature=1.0` (`evaluation.py:168`) | high |
| 7 | No validation loss during training | `val_set_size: 0` (`config/base.yml:15`) | high |
| 8 | No baseline to beat | only base model vs. fine-tuned; no few-shot | medium |
| 9 | Two identical pipeline entries | `evaluation.py:320` and `:323` are the same call | low |
| 10 | Failures score as 0.0 instead of failing | `evaluation.py:218-220` | low |

Two of these deserve elaboration.

**Finding 1 — the data teaches the opposite of the system prompt.** Leichte Sprache
normally expands a text by roughly 1.5–3x: long sentences are broken up, terms explained,
compounds hyphenated. Word-count ratios across all 8 pairs:

| ID | Standard | Leicht | ratio | assessment |
|---|---|---|---|---|
| 0197 | 275 | 287 | 1.04 | flat — content possibly dropped |
| 0054 | 201 | 221 | 1.10 | flat |
| 0022 | 200 | 228 | 1.14 | flat |
| 0196 | 605 | 1348 | 2.23 | healthy |
| 0202 | 1405 | 3261 | 2.32 | healthy |
| 0007 | 240 | 655 | 2.73 | healthy |
| 0021 | 815 | 4412 | 5.41 | inflated |
| 0002 | 358 | 5148 | 14.38 | inflated — standalone brochure, not a translation |

The median is 2.28, right where proper Leichte Sprache sits, so the three healthy pairs
define the band and the outliers fall away from it on both sides. `0002`'s target text is
a well-written Hamburg brochure containing substantial content absent from its 358-word
source. Training on it teaches the model to invent, contradicting the first rule of
`data/system-prompt.md`: *"Du übersetzt, du erfindest nichts dazu."* The three flat pairs
are also the three shortest sources, so they may simply be short simple texts — they need
a look, not necessarily a fix.

**Finding 2 — the current metrics cannot detect bad Leichte Sprache.** Flesch Reading Ease
and Wiener Sachtextformel are arithmetic over sentence length and word length. A model
that chops the source into fragments and silently drops half the content scores *better*
on both. Neither formula can see rule compliance, meaning preservation, or invention.

---

# Priority 0 — before the next training run

Nothing here needs a GPU. Together these turn the pipeline's output from an anecdote into
a measurement.

### P0-1 Repair the dataset

Apply the length-ratio check to all 8 pairs and review every outlier by hand. **Flag, do
not auto-exclude:** a band of [1.3, 3.5] flags five of eight pairs, and dropping them
leaves three training examples, which is worse than training on flawed data. For `0002`
the fix is to trim the Leichte Sprache text to the sections its source actually covers —
that converts a bad pair into a good one and keeps the sample count. Automatic exclusion
only becomes safe once losing 20% of the data does not hurt.

### P0-2 `src-eval/rules.py` — measure Leichte Sprache directly

`data/system-prompt.md` already contains the rubric; turn it into code. Split it into two
tiers, because they answer different questions and only one of them may filter data:

**Tier A — objective pair errors.** Describe the relationship between source and target.
Safe to filter or fix on.
- word-count ratio outside the calibrated band (both floor and ceiling)
- output statements not supported by the source (see P2-1)

**Tier B — style rules.** Describe one text against the system prompt. **Never delete data
on these.**
- share of sentences ≤ 10 words
- share of lines holding exactly one sentence
- long compounds lacking a hyphen (`Krankenversicherung` vs. `Kranken-Versicherung`)
- subordinate-clause markers per sentence (`weil`, `dass`, `obwohl`, `damit`)
- negations where a positive formulation exists
- Nominalstil markers (`-ung`, `-heit`, `-keit`) — the prompt demands Verbalstil
- Futur II / Plusquamperfekt occurrences — the prompt forbids both
- presence of Zwischenüberschriften

**The ruleset validates itself against the references.** Run Tier B over the 8
human-written Leichte Sprache texts first. They are written by professionals and should
score near-perfect; wherever they do not, either the rule implementation is wrong or the
system prompt is stricter than real Lebenshilfe practice. Both are worth knowing, and
neither is a reason to delete the professionals' work. Do not filter the references by
Tier B — the references are what validate Tier B, and filtering by it makes the check
self-confirming.

The reference scores are also the only sensible target for the model. *"The fine-tuned
model reaches 84% short sentences"* means nothing until you know the human references
reach 91%.

Cheap enough to run in CI on every data change.

### P0-3 Split the data and make the split travel

`prepare_dataset.py` emits `data/train/dataset.jsonl` and `data/eval/holdout.jsonl` from a
deterministic split, plus a `run_manifest.json` written next to the adapter by
`train.py` recording base model, config hash, and **the exact IDs trained on**. The eval
container reads the manifest and scores only what is not in that list. That is what makes
the eval unbiased across a container boundary.

At n=8 a fixed 6/2 split produces a number computed on two documents — an anecdote, not an
estimate. Options, best first:

1. **More data** (see [Standing item](#standing-item-more-data)).
2. **Leave-one-out cross-validation** — 8 runs, each trained on 7 and scored on 1. Gives 8
   scores with a spread instead of one number. Expose the fold as `FOLD=0..7` so the
   launcher can loop. Expensive on RunPod but statistically correct at this size.
3. **Sub-document splitting** — the documents are sectioned (`FC St. Pauli`, `HSV`, …).
   Aligning at section level turns 8 documents into perhaps 40–60 shorter pairs, and it
   matches deployment: the app will translate paragraphs, not 5000-word brochures. Split
   folds at document boundaries so sections of one brochure never straddle train and test.

### P0-4 Fix the eval harness

| change | file |
|---|---|
| remove the truncation that drops every dataset record | `evaluation.py:308` |
| decode greedily (`temperature=0`, fixed seed) for all scored runs | `evaluation.py:168` |
| score the holdout against `reference_text` instead of ignoring it | `evaluation.py:304` |
| move the 17 hardcoded probes to `data/eval/probes.jsonl` | `evaluation.py:260-278` |
| drop or differentiate the duplicate `Plain`/`Reasoning` entries | `evaluation.py:320,323` |
| let engine failures fail the run instead of scoring 0.0 | `evaluation.py:218-220` |

On **temperature**: at each generation step the model scores every possible next word;
temperature controls how sharply it favours its top choice. At 0 it always takes the best
one, so the same input yields the same output. At 1.0 it samples according to its own
probabilities, so every run differs. With a holdout of a few documents, that run-to-run
wobble is larger than the difference between base and fine-tuned model — you would be
measuring the dice. `evaluation.py:168` uses 1.0, which is Google's recommendation for
Gemma in open-ended chat and reasonable *there*, but not for scoring.

The 17 hardcoded prompts (hate speech, `Quadratwurzel aus 16`, the English Genesis text)
are **robustness probes** for the ethical rules in the system prompt. They have no
reference and cannot be scored like parallel pairs. They belong to the demo path, not the
measurement path.

---

# Priority 1 — the next iteration

### P1-1 Rename the modes so they stop borrowing each other's credibility

The current `MODE=eval` is a mix of "inference works" and a manual plausibility check.
That is a useful job, but it is not evaluation.

| now | becomes | runs | purpose |
|---|---|---|---|
| `MODE=train` | `MODE=train` | axolotl container | fit adapter, write `run_manifest.json` |
| `MODE=eval` (probes) | `MODE=inference` | vLLM container | probes + `smoke_adapter.py`; no scores claimed |
| — | `MODE=eval` (new) | vLLM container | scores the holdout, writes `eval_metrics.json` |

Touches `scripts/launch.sh:9,80`, `scripts/eval.sh`, `tests/eval_container_smoke_test.py`,
`docs/pipeline.md`. `src-eval/smoke_adapter.py` already covers the "does it actually
serve" half correctly and needs no change.

The scored eval writes, next to the adapter and synced to S3 as one unit:

```
/app/output/adapter/train-gemma4/
  ├─ adapter_model.safetensors
  ├─ run_manifest.json      # base model, config hash, train IDs, holdout IDs
  ├─ eval_metrics.json      # the scores
  └─ eval_outputs.jsonl     # every generated text, kept for later re-scoring
```

Saving raw outputs matters: it lets you re-score old runs when a metric improves, without
retraining, and it is what makes the offline judge in P2-2 possible.

### P1-2 Validation loss during training

Point axolotl's `test_datasets` at the holdout so an eval-loss curve is produced per epoch,
and reconsider `num_epochs: 3` (`config/base.yml:25`) against it. At 8 samples the moment
the model stops learning and starts memorising arrives early, and right now it is
invisible.

### P1-3 Add a few-shot baseline

Three systems, not two: base + system prompt, base + 2 examples in the prompt, fine-tuned
adapter. With 8 training examples, putting two of them in the prompt may well beat
fine-tuning. That is worth knowing before spending further GPU hours, and it is the
cheapest experiment on this list.

### P1-4 Anchor the readability formulas

`prepare_dataset.py:171` already computes `reference_fre` / `reference_wstf` and the eval
never compares against them. Report distance from the human reference band rather than the
raw score. *"FRE 71"* means nothing; *"FRE 71 where the human reference is 78"* means
something. Keep the formulas as descriptive statistics, not as the headline metric.

---

# Priority 2 — once the basics hold

### P2-1 Meaning preservation

Rule compliance alone rewards deleting content, so it needs a counterweight.

- **No invented facts:** split the output into sentences and check each against the source
  with a German entailment model. Unsupported sentences are hallucination candidates. This
  is the metric that would have caught `0002`.
- **No lost content:** extract key statements from the source and check coverage in the
  output.

Both are Tier A signals and can also be run over the training references as a data filter.

### P2-2 LLM-as-judge, offline

A rubric lifted from `data/system-prompt.md`, scored as a **pairwise comparison** (base vs.
fine-tuned, order randomised) rather than an absolute 1–5 — absolute judge scores drift and
cluster around 4. Runs outside the container, over the saved `eval_outputs.jsonl`, because
it needs network and an API key that the GPU pod should not carry.

### P2-3 Human review

Leichte Sprache has a formal review step: texts are checked by people with learning
difficulties (Prüfgruppe). Lebenshilfe can actually do this. At 3 systems × 10 outputs the
effort is small, and it is the only ground truth that counts. Everything in P0-2 and P2-1
is a proxy that should be periodically checked *against* the human ratings, not trusted
in place of them.

---

# Standing item: more data

8 pairs is the root cause of most of the above, and every other item on this list is a
workaround for it. Sources worth evaluating:

- **Lebenshilfe's own archive.** `0021` shows this is already being drawn on; it is the
  best-matched source and the style is by definition correct.
- **Public German parallel corpora** (DEplain, Simple German Corpus). Licence and register
  both need checking — "Einfache Sprache" and "Leichte Sprache" are different registers,
  and the system prompt targets the latter.

---

## Decision: eval container vs. training container

Running the scored eval inside the training container was considered and rejected for now.

**For the training container:** no S3 round-trip, no second pod, the base weights are
already in the local HF cache and the adapter is already on local disk.

**Against:** the containers were split because axolotl and vLLM do not coexist, and the
literal "reuse the loaded model" version is fragile — under DeepSpeed ZeRO-3 the weights
are split across GPUs and `train.py:127` sets `gather_16bit_weights_on_model_save = False`,
so nothing ever holds the complete model. Generating from that state requires every GPU
process to stay in lockstep with the weights temporarily reassembled. A separate
`transformers` + `peft` process in the same container would avoid that, but it duplicates
inference machinery that the vLLM container already has working.

**Consequence of staying in the vLLM container:** the eval is no longer physically part of
the training run, so the link has to be explicit. `run_manifest.json` (P0-3) is what
carries it — the eval scores the holdout that *that specific run* did not train on. The
estimate is unbiased regardless of which container computes it; what matters is that the
split is recorded and honoured, not where the GPU sits.

**Revisit if:** the S3 round-trip or the second pod becomes the bottleneck in iteration
speed, or the two-pod sequence proves unreliable to automate.

---

## Suggested order of work

1. P0-1 — repair the dataset (no GPU)
2. P0-2 — `src-eval/rules.py`, validated against the references (no GPU)
3. P0-3 — split + `run_manifest.json` (no GPU)
4. P0-4 — fix the eval harness (no GPU)
5. P1-2 — validation loss, revisit `num_epochs`
6. P1-1 — rename the modes
7. P1-3 — few-shot baseline; first honest three-way comparison
8. P2-1 / P2-2 — meaning preservation and the judge
9. P2-3 — human review round

Steps 1–4 need no GPU time and can be developed and tested locally.
