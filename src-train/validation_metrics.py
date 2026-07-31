"""Leichte Sprache quality on the validation split, measured while the training runs.

Eval loss says the model predicts the reference's tokens better. It does not say the output
is Leichte Sprache -- memorising the corpus's phrasing and learning the register both drive
it down, and only one of them is the goal. So at every evaluation this generates from a
handful of validation sources and scores the generated text with src-eval/rules.py and the
readability formulas, next to the loss.

Every number is reported as a **gap to the human reference** for the same documents, per
docs/train-eval-review.md P1-4: "84% short sentences" means nothing until you know the
professionals reach 91%. `eval_ls_distance` collapses those gaps into one scalar so there
is something to watch converge; it is a placeholder with no claim to being a good measure
of translation quality, and is meant to be replaced by an LLM-as-a-judge score.

Turned on by listing the plugin in a training config:

    plugins:
      - validation_metrics.ValidationMetricsPlugin

Cost is the thing to watch. Generation under DeepSpeed ZeRO-3 re-gathers sharded parameters
for every token, so it is far slower per token than training is. The defaults are
deliberately small; raise them only after timing one evaluation on the real pod.

    VALIDATION_METRICS_SAMPLES=4      validation documents to generate from
    VALIDATION_METRICS_MAX_TOKENS=256 cap per generation
    VALIDATION_METRICS_OFF=1          skip generation; eval loss alone
"""
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from transformers import TrainerCallback

try:
    from axolotl.integrations.base import BasePlugin
except ImportError:  # tests and tooling run without axolotl; only the plugin needs it
    BasePlugin = object  # type: ignore[assignment,misc]

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src-eval"))
from rules import StyleMetrics, Thresholds, analyse_style, count_words  # noqa: E402

VALIDATION_PATH = REPO_ROOT / "data" / "train" / "validation.jsonl"

# How far into validation.jsonl to look for documents that fit the context before giving up.
CANDIDATE_POOL = 100

# Of the first generation, echoed to the log so the metrics can be sanity-checked.
SHOWN_CHARS = 300

# Rates already live in 0..1, so a raw gap is comparable across them. Densities are counts
# per 100 words (or per sentence) and are scaled by the reference itself, so "twice as much
# Nominalstil as the humans" weighs the same whether the reference is 2.0 or 8.0.
RATE_METRICS = ("short_sentence_rate", "one_sentence_per_line_rate")
DENSITY_METRICS = ("unhyphenated_compounds_per_100w", "nominalstil_per_100w",
                   "subordinate_markers_per_sentence", "negations_per_sentence")


@dataclass(frozen=True)
class ValidationCase:
    pair_id: str
    prompt_messages: List[Dict[str, str]]
    source: str
    reference: str


@dataclass(frozen=True)
class Scored:
    """Aggregate style of one side (generated or reference) over all cases.

    `flesch` and `wiener` are None when textstat is not importable. They are descriptive
    statistics, not the headline metric (docs/train-eval-review.md P1-4), and the rule
    metrics that carry the actual signal need nothing but the standard library -- so
    losing them must not cost the whole measurement.
    """
    style: Dict[str, float]
    length_ratio: float
    flesch: Optional[float]
    wiener: Optional[float]


def render_prompt(case: ValidationCase, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        case.prompt_messages, tokenize=False, add_generation_prompt=True)


def select_cases(path: Path, tokenizer, samples: int, budget: int) -> List[ValidationCase]:
    """The first `samples` validation documents whose prompt leaves room to generate.

    Prompts longer than the training context are out of distribution for the adapter --
    Axolotl drops exactly those documents from the training set -- so generating from them
    would measure truncation rather than the register.
    """
    chosen: List[ValidationCase] = []
    for case in load_cases(path, limit=CANDIDATE_POOL):
        if len(chosen) >= samples:
            break
        if len(tokenizer(render_prompt(case, tokenizer))["input_ids"]) <= budget:
            chosen.append(case)
    return chosen


def load_cases(path: Path, limit: int) -> List[ValidationCase]:
    """The first `limit` records -- the file's order is already deterministic."""
    cases: List[ValidationCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(cases) >= limit:
                break
            entry = json.loads(line)
            messages = entry["messages"]
            reference = next(m["content"] for m in messages if m["role"] == "assistant")
            cases.append(ValidationCase(
                pair_id=entry["id"],
                prompt_messages=[m for m in messages if m["role"] != "assistant"],
                source=entry["original_text"],
                reference=reference,
            ))
    return cases


def _median_style(texts: Sequence[str], thresholds: Thresholds) -> Dict[str, float]:
    measured: List[StyleMetrics] = [analyse_style(text, thresholds) for text in texts]
    fields = RATE_METRICS + DENSITY_METRICS + ("forbidden_tense_count",)
    return {name: float(statistics.median(getattr(m, name) for m in measured))
            for name in fields}


@lru_cache(maxsize=1)
def _textstat():
    """The configured module, or None if it cannot be imported. Warns exactly once.

    textstat drags in nltk and scipy, so it is the most fragile dependency here and the
    only one that is not the standard library. The rule metrics do not need it.
    """
    try:
        import textstat
    except Exception as e:  # noqa: BLE001 -- an optional metric, not a reason to stop
        print(f"ℹ️  readability formulas unavailable ({type(e).__name__}: {e}); "
              f"reporting rule metrics only.", flush=True)
        return None
    textstat.set_lang("de")
    return textstat


def _readability(texts: Sequence[str]) -> Tuple[Optional[float], Optional[float]]:
    """Median Flesch and Wiener, or (None, None) without textstat."""
    textstat = _textstat()
    if textstat is None:
        return None, None
    return (round(statistics.median(textstat.flesch_reading_ease(t) for t in texts), 2),
            round(statistics.median(textstat.wiener_sachtextformel(t, 1) for t in texts), 2))


def score(texts: Sequence[str], sources: Sequence[str],
          thresholds: Thresholds = Thresholds()) -> Scored:
    """Median style, length ratio and readability of `texts` against their `sources`."""
    ratios = [count_words(t) / count_words(s) for t, s in zip(texts, sources) if count_words(s)]
    flesch, wiener = _readability(texts)
    return Scored(
        style=_median_style(texts, thresholds),
        length_ratio=round(statistics.median(ratios), 3) if ratios else 0.0,
        flesch=flesch,
        wiener=wiener,
    )


def compare(generated: Scored, reference: Scored) -> Dict[str, float]:
    """Flat metric dict: the model's own values, its gap to the humans, and the composite.

    Gaps are signed -- the direction says which way the model is wrong -- while the
    composite uses their magnitudes.
    """
    metrics: Dict[str, float] = {}
    distances: List[float] = []

    for name in RATE_METRICS:
        gap = generated.style[name] - reference.style[name]
        metrics[f"eval_ls_{name}"] = round(generated.style[name], 4)
        metrics[f"eval_ls_gap_{name}"] = round(gap, 4)
        distances.append(min(1.0, abs(gap)))

    for name in DENSITY_METRICS:
        gap = generated.style[name] - reference.style[name]
        metrics[f"eval_ls_{name}"] = round(generated.style[name], 3)
        metrics[f"eval_ls_gap_{name}"] = round(gap, 3)
        distances.append(min(1.0, abs(gap) / (abs(reference.style[name]) + 1.0)))

    metrics["eval_ls_forbidden_tense_count"] = generated.style["forbidden_tense_count"]
    metrics["eval_ls_length_ratio"] = generated.length_ratio
    metrics["eval_ls_gap_length_ratio"] = round(generated.length_ratio - reference.length_ratio, 3)

    # Descriptive only, and absent without textstat. They stay out of the composite either
    # way: a model can score better on both by chopping the text into fragments.
    for name, mine, theirs in (("fre", generated.flesch, reference.flesch),
                               ("wstf", generated.wiener, reference.wiener)):
        if mine is not None and theirs is not None:
            metrics[f"eval_ls_{name}"] = mine
            metrics[f"eval_ls_gap_{name}"] = round(mine - theirs, 2)

    # Each term is clamped to 1, so this is 0 (matches the humans) .. 1 (at least one rule
    # a full scale off). Unclamped, a single runaway component owned the whole number: an
    # untrained model emitting one 400-character non-word scored 16.8. The raw gaps above
    # keep the detail; this is only the thing to watch converge.
    metrics["eval_ls_distance"] = round(statistics.mean(distances), 4)
    return metrics


def generate(trainer, cases: Sequence[ValidationCase], max_new_tokens: int) -> List[str]:
    """Greedy generation, one case at a time.

    One at a time, not batched, on purpose: batching needs left padding and a correct
    attention mask per sequence, and getting that subtly wrong produces plausible garbage
    that would silently poison every metric below. Under ZeRO-3 every rank must run this --
    the parameter gathers are collective, so a rank that skips them deadlocks the rest.
    """
    import torch

    model = trainer.model
    tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer
    was_training = model.training
    model.eval()

    outputs: List[str] = []
    try:
        with torch.no_grad():
            for case in cases:
                encoded = tokenizer(render_prompt(case, tokenizer),
                                    return_tensors="pt").to(model.device)
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    # gradient_checkpointing turns the KV cache off for training; generation
                    # without it is quadratic and, under ZeRO-3, unusably slow.
                    use_cache=True,
                )
                new_tokens = generated[0][encoded["input_ids"].shape[-1]:]
                outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    finally:
        if was_training:
            model.train()
    return outputs


class ValidationMetricsCallback(TrainerCallback):

    def __init__(self, trainer, samples: int, max_new_tokens: int, sequence_len: int) -> None:
        self.trainer = trainer
        self.max_new_tokens = max_new_tokens
        self.reference: Optional[Scored] = None
        self.cases: List[ValidationCase] = []

        if not VALIDATION_PATH.exists():
            print(f"⚠️  {VALIDATION_PATH} missing -- Leichte Sprache validation metrics are "
                  f"off. Run `dvc pull data/train/validation.jsonl`.", flush=True)
            return

        tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer
        self.cases = select_cases(VALIDATION_PATH, tokenizer, samples,
                                  budget=sequence_len - max_new_tokens)
        if not self.cases:
            print(f"⚠️  No validation document fits {sequence_len} tokens minus "
                  f"{max_new_tokens} to generate -- Leichte Sprache metrics are off.",
                  flush=True)

    def on_evaluate(self, args, state, control, **kwargs):
        """Never raises. A broken metric must not cost a training run that is going fine."""
        if not self.cases:
            return control
        try:
            if self.reference is None:
                self.reference = score([c.reference for c in self.cases],
                                       [c.source for c in self.cases])
            started = time.time()
            generated = generate(self.trainer, self.cases, self.max_new_tokens)
            metrics = compare(score(generated, [c.source for c in self.cases]), self.reference)
            metrics["eval_ls_seconds"] = round(time.time() - started, 1)

            self.trainer.log(metrics)
            if state.is_world_process_zero:
                print(f"\n📏 Leichte Sprache on {len(self.cases)} validation samples "
                      f"({metrics['eval_ls_seconds']}s): "
                      f"distance {metrics['eval_ls_distance']}, "
                      f"short sentences {metrics['eval_ls_short_sentence_rate']:.0%} "
                      f"(gap {metrics['eval_ls_gap_short_sentence_rate']:+.0%})", flush=True)
                # A sample of the actual output, every time. Rule metrics are arithmetic
                # over text and stay perfectly plausible when the text is empty, English,
                # or the prompt echoed back -- the numbers alone cannot tell you that.
                head = generated[0].replace("\n", " ⏎ ")[:SHOWN_CHARS]
                print(f"   [{self.cases[0].pair_id}] {head}", flush=True)
        except Exception as e:  # noqa: BLE001 -- diagnostics, never a run-killer
            print(f"⚠️  Leichte Sprache validation metrics failed: {type(e).__name__}: {e}",
                  flush=True)
        return control


class ValidationMetricsPlugin(BasePlugin):
    """Axolotl plugin entry point: `plugins: [validation_metrics.ValidationMetricsPlugin]`."""

    def add_callbacks_post_trainer(self, cfg, trainer) -> List[TrainerCallback]:
        if os.environ.get("VALIDATION_METRICS_OFF") == "1":
            print("ℹ️  VALIDATION_METRICS_OFF=1 -- eval loss only.", flush=True)
            return []

        samples = int(os.environ.get("VALIDATION_METRICS_SAMPLES", "4"))
        max_new_tokens = int(os.environ.get("VALIDATION_METRICS_MAX_TOKENS", "256"))
        callback = ValidationMetricsCallback(trainer, samples, max_new_tokens,
                                             sequence_len=int(cfg.sequence_len))
        if not callback.cases:
            return []
        print(f"✅ Leichte Sprache validation metrics on: {len(callback.cases)} samples "
              f"({', '.join(c.pair_id for c in callback.cases)}), "
              f"{max_new_tokens} tokens each, every evaluation.", flush=True)
        return [callback]
