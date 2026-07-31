"""Compile data/raw into the three splits the pipeline trains, tunes and scores on.

    data/train/dataset.jsonl      70%  what the adapter is fitted to
    data/train/validation.jsonl   10%  convergence signal during training (eval loss +
                                       src-train/validation_metrics.py)
    data/eval/holdout.jsonl       20%  scored in the eval container; the training
                                       container never pulls this file
    data/split_manifest.json           which id went where, and why anything was dropped

**The split is a pure function of the pair id.** A document's split is
`sha256(salt:id)`, not a shuffle with a seed, so adding documents to `data/raw` never moves
an existing one across the train/holdout boundary. That property is what makes the holdout
worth anything: with a seeded shuffle, growing the corpus silently leaks yesterday's
holdout into today's training set.

Two things keep a pair out of all three splits. `data/excluded.json` is the deliberate one:
a hand-maintained list of pairs whose two texts are not translations of each other, kept in
git so the decision is reviewable. Removing a pair there does not move any other pair
between splits, because assignment is per-id.

Over-long pairs are dropped and listed rather than killing the run -- at 780 documents a
handful of 30k-token brochures is expected, and losing the whole dataset preparation over
them helps nobody. A drop rate above MAX_DROP_RATE is still fatal, because that is a
misconfigured `sequence_len`, not an outlier.

    python src-train/prepare_dataset.py
    python src-train/prepare_dataset.py --data-dir data/raw --seq-len 8192
"""
import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import textstat
from transformers import AutoTokenizer

# src-eval/rules.py is plain-stdlib and is the single definition of a "pair" and of the
# Tier A length band. Importing it here keeps one implementation instead of two that drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src-eval"))
from rules import Thresholds, analyse_pair, find_pairs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pinned, not derived from the training config: the token count decides which samples are
# dropped, so a dataset that changes depending on whether HF_TOKEN happens to be set is not
# reproducible. tiny-random/gemma-4-moe carries google/gemma-4-26b-a4b-it's tokenizer and
# chat template verbatim (see config/train-gemma4-tiny.yml) and is ungated, so this is the
# production tokenizer without the production gate. Override only to compile a dataset for
# a different model family -- Mistral tokenizes German quite differently.
DEFAULT_TOKENIZER = "tiny-random/gemma-4-moe"

# Changing this reshuffles every document. Don't, unless you mean to invalidate every
# holdout score ever computed.
SPLIT_SALT = "leichte-sprache-2026"

# Above this share of over-long pairs, `sequence_len` is wrong rather than the data.
MAX_DROP_RATE = 0.05


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    validation: float = 0.10
    holdout: float = 0.20

    def name_for(self, fraction: float) -> str:
        if fraction < self.train:
            return "train"
        if fraction < self.train + self.validation:
            return "validation"
        return "holdout"


@dataclass(frozen=True)
class Sample:
    pair_id: str
    split: str
    messages: List[Dict[str, str]]
    original_text: str
    metrics: Dict[str, float]
    total_tokens: int
    assistant_tokens: int
    length_ratio: float
    ratio_in_band: bool

    def to_record(self) -> Dict[str, object]:
        """The JSONL line. `messages` is what Axolotl reads; the rest is for evaluation."""
        return {
            "id": self.pair_id,
            "messages": self.messages,
            "original_text": self.original_text,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class DroppedPair:
    pair_id: str
    total_tokens: int


@dataclass
class Manifest:
    """Written next to the splits and copied into the adapter's run_manifest.json.

    Deliberately carries no timestamp. It is a DVC output, and a field that changes on
    every run makes the stage non-reproducible: `dvc repro` then always emits a new blob,
    the lock file points at a hash nobody pushed, and the next `dvc pull` fails with
    "Checkout failed ... Is your cache up to date?" -- which is exactly how a RunPod run
    died. When this file was built is recoverable from git and from dvc.lock; the training
    run's own timestamp lives in run_manifest.json, which is not a DVC output.
    """
    split_salt: str
    ratios: Dict[str, float]
    tokenizer: str
    sequence_len: int
    counts: Dict[str, int] = field(default_factory=dict)
    splits: Dict[str, List[str]] = field(default_factory=dict)
    dropped_too_long: List[DroppedPair] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)
    ratio_flagged: Dict[str, List[str]] = field(default_factory=dict)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def read_exclusions(path: Path) -> Dict[str, str]:
    """Pair ids to keep out of every split, mapped to the reason.

    Hand-maintained and reviewed in git rather than tracked by DVC -- see the comment in
    data/excluded.json. Keys starting with '_' are commentary, not ids.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {pair_id: reason for pair_id, reason in raw.items() if not pair_id.startswith("_")}


def calculate_token_count(messages: List[Dict[str, str]], tokenizer) -> int:
    """Sequence length of a whole conversation, over the model's own chat template."""
    try:
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
        return len(tokens)
    except Exception:
        try:
            tokens = tokenizer.apply_chat_template(messages, tokenize=True)
            if isinstance(tokens, dict) and "input_ids" in tokens:
                return len(tokens["input_ids"])
            return len(tokens)
        except Exception:
            raw_text = "\n".join([m["content"] for m in messages])
            return len(tokenizer.encode(raw_text))


def assign_split(pair_id: str, ratios: SplitRatios, salt: str = SPLIT_SALT) -> str:
    """Deterministic and independent of every other document in the corpus."""
    digest = hashlib.sha256(f"{salt}:{pair_id}".encode("utf-8")).hexdigest()
    return ratios.name_for(int(digest[:8], 16) / 0x100000000)


def read_sequence_len(base_yml: Path) -> int:
    """Read `sequence_len` without pulling omegaconf into the DVC stage's environment."""
    for line in base_yml.read_text(encoding="utf-8").splitlines():
        if line.startswith("sequence_len:"):
            return int(line.split(":", 1)[1].split("#")[0].strip())
    sys.exit(f"❌ No 'sequence_len' in {base_yml}")


def build_sample(pair_id: str, source_path: Path, target_path: Path, system_prompt: str,
                 template: str, tokenizer, ratios: SplitRatios,
                 thresholds: Thresholds) -> Sample:
    source = read_text(source_path)
    target = read_text(target_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": template.replace("%INPUT%", source)},
        {"role": "assistant", "content": target},
    ]
    pair = analyse_pair(source, target, thresholds)

    return Sample(
        pair_id=pair_id,
        split=assign_split(pair_id, ratios),
        messages=messages,
        original_text=source,
        metrics={
            "original_fre": round(textstat.flesch_reading_ease(source), 2),
            "original_wstf": round(textstat.wiener_sachtextformel(source, 1), 2),
            "reference_fre": round(textstat.flesch_reading_ease(target), 2),
            "reference_wstf": round(textstat.wiener_sachtextformel(target, 1), 2),
        },
        total_tokens=calculate_token_count(messages, tokenizer),
        assistant_tokens=len(tokenizer.encode(target)),
        length_ratio=pair.length_ratio,
        ratio_in_band=pair.ratio_in_band,
    )


def write_split(samples: Sequence[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_record(), ensure_ascii=False) + "\n")
    print(f"  {path}  ({len(samples)} samples)")


def percentiles(values: Sequence[int]) -> Dict[str, int]:
    ordered = sorted(values)
    if not ordered:
        return {}
    pick = lambda q: ordered[min(int(len(ordered) * q), len(ordered) - 1)]  # noqa: E731
    return {"p50": pick(0.5), "p75": pick(0.75), "p90": pick(0.9),
            "p99": pick(0.99), "max": ordered[-1]}


def print_report(kept: List[Sample], dropped: List[DroppedPair], excluded: Dict[str, str],
                 seq_len: int, thresholds: Thresholds) -> None:
    print("\n" + "=" * 70)
    print("                  DATASET SUMMARY")
    print("=" * 70)
    print(f" Pairs found        : {len(kept) + len(dropped) + len(excluded)}")
    print(f" Excluded by hand   : {len(excluded)}  (data/excluded.json)")
    for pair_id, reason in sorted(excluded.items()):
        print(f"     {pair_id}: {reason[:60]}...")
    print(f" Dropped (too long) : {len(dropped)}  (limit {seq_len} tokens)")
    for drop in dropped:
        print(f"     {drop.pair_id}: {drop.total_tokens} tokens")

    print("\n Split sizes:")
    for split in ("train", "validation", "holdout"):
        members = [s for s in kept if s.split == split]
        flagged = [s for s in members if not s.ratio_in_band]
        print(f"   {split:<11}: {len(members):>4}   "
              f"length-ratio flags: {len(flagged)}")

    for label, values in (("Total tokens", [s.total_tokens for s in kept]),
                          ("Assistant tokens", [s.assistant_tokens for s in kept])):
        stats = percentiles(values)
        print(f"\n {label}:")
        for name, value in stats.items():
            print(f"   {name:<4}: {value}")
        print(f"   mean: {round(statistics.mean(values))}")

    flagged = [s for s in kept if not s.ratio_in_band]
    print(f"\n Tier A length ratio outside "
          f"[{thresholds.min_length_ratio}, {thresholds.max_length_ratio}]: "
          f"{len(flagged)}/{len(kept)} pairs")
    print("   Reported, NOT excluded -- see docs/train-eval-review.md P0-1. Inspect with")
    print("   `python src-eval/rules.py`.")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile data/raw into train/validation/holdout.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="defaults to sequence_len in config/base.yml")
    parser.add_argument("--exclusions", type=Path,
                        default=REPO_ROOT / "data" / "excluded.json")
    args = parser.parse_args()

    seq_len = args.seq_len or read_sequence_len(REPO_ROOT / "config" / "base.yml")
    ratios = SplitRatios()
    thresholds = Thresholds()

    textstat.set_lang("de")
    print(f"Loading tokenizer {args.tokenizer} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    system_prompt = read_text(REPO_ROOT / "data" / "system-prompt.md")
    template = read_text(REPO_ROOT / "data" / "prompt-template.md")

    pairs = find_pairs(args.data_dir)
    if not pairs:
        sys.exit(f"❌ No <id>_Standardsprache / <id>_Leichte_Sprache pairs in {args.data_dir}")
    print(f"Found {len(pairs)} pairs in {args.data_dir}", flush=True)

    excluded = read_exclusions(args.exclusions)
    stale = sorted(set(excluded) - set(pairs))
    if stale:
        # Not fatal, but it always means the file is lying about something: either the pair
        # was renamed and is now silently back in the training set under a new id, or the
        # entry is a leftover that makes the exclusion list look more considered than it is.
        print(f"⚠️  {args.exclusions} names {len(stale)} id(s) that are not in "
              f"{args.data_dir}: {', '.join(stale)}", flush=True)
    for pair_id in excluded:
        pairs.pop(pair_id, None)

    kept: List[Sample] = []
    dropped: List[DroppedPair] = []
    for pair_id, (source_path, target_path) in pairs.items():
        sample = build_sample(pair_id, source_path, target_path, system_prompt, template,
                              tokenizer, ratios, thresholds)
        if sample.total_tokens > seq_len:
            dropped.append(DroppedPair(pair_id, sample.total_tokens))
        else:
            kept.append(sample)

    drop_rate = len(dropped) / len(pairs)
    if drop_rate > MAX_DROP_RATE:
        sys.exit(f"❌ {len(dropped)}/{len(pairs)} pairs ({drop_rate:.0%}) exceed "
                 f"sequence_len={seq_len}. That is a configuration problem, not an "
                 f"outlier. Raise sequence_len in config/base.yml (and expect a higher "
                 f"VRAM bill) or shorten the sources.")

    print("\nWriting splits:")
    by_split = {name: [s for s in kept if s.split == name]
                for name in ("train", "validation", "holdout")}
    write_split(by_split["train"], args.out_dir / "train" / "dataset.jsonl")
    write_split(by_split["validation"], args.out_dir / "train" / "validation.jsonl")
    write_split(by_split["holdout"], args.out_dir / "eval" / "holdout.jsonl")

    manifest = Manifest(
        split_salt=SPLIT_SALT,
        ratios=asdict(ratios),
        tokenizer=args.tokenizer,
        sequence_len=seq_len,
        counts={name: len(members) for name, members in by_split.items()},
        splits={name: [s.pair_id for s in members] for name, members in by_split.items()},
        dropped_too_long=dropped,
        excluded=excluded,
        ratio_flagged={name: [s.pair_id for s in members if not s.ratio_in_band]
                       for name, members in by_split.items()},
    )
    manifest_path = args.out_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  {manifest_path}")

    print_report(kept, dropped, excluded, seq_len, thresholds)

    if not by_split["validation"]:
        sys.exit("❌ The validation split is empty; training has no convergence signal.")


if __name__ == "__main__":
    main()
