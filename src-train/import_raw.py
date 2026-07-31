"""Pull the parallel corpus out of S3 into data/raw/ under canonical names.

The bucket prefix is a dump of a shared drive, so the naming is inconsistent in every way
a human can be inconsistent: `Leichte Sprache.txt` with a space, `Standardsprache .txt`
with a trailing one, `Standartsprache.txt`, `Leichte Spracheprache.txt`, three-digit ids,
`Kopie von …` duplicates of files that already exist under their own name, and a handful
of `.docx` that were never converted.

Everything downstream -- prepare_dataset.py, src-eval/rules.py -- matches the strict
`<4-digit id>_Standardsprache.txt` / `<4-digit id>_Leichte_Sprache.txt` form. This script
is the single place that knows about the mess, so the strictness elsewhere stays honest.

Re-runnable: it overwrites, and reports what it took and what it left behind.

    python src-train/import_raw.py                      # dry run, prints the plan
    python src-train/import_raw.py --write
    python src-train/import_raw.py --write --bucket other --prefix other/
"""
import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_BUCKET = "diwop-analysis"
DEFAULT_PREFIX = "training-data/"

STANDARD = "Standardsprache"
LEICHT = "Leichte_Sprache"

# Longer spellings first so the intended alternative is tried before its prefix.
_NAME = re.compile(
    r"^(?P<copy>Kopie von )?"
    r"(?P<id>\d{3,4})[_ ]?"
    r"(?P<kind>Standardsprache|Standartsprache|Leichte[_ ]?Spracheprache|Leichte[_ ]?Sprache)"
    r"\s*(?P<dup>\(\d+\))?\s*"
    r"\.(?P<ext>txt|md)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    """One remote object that could serve as one side of one pair."""
    key: str
    pair_id: str
    kind: str
    rank: int      # lower wins: 0 canonical, 1 numbered duplicate, 2 'Kopie von'
    size: int


@dataclass
class ImportReport:
    written: List[str] = field(default_factory=list)
    incomplete: Dict[str, str] = field(default_factory=dict)          # pair_id -> missing side
    superseded: List[str] = field(default_factory=list)               # lost a rank tie-break
    unmatched: List[str] = field(default_factory=list)                # no rule applies


def classify(name: str) -> Optional[Candidate]:
    """Map one object name onto a (pair_id, kind) slot, or None if it is not a text pair."""
    match = _NAME.match(name.strip())
    if not match:
        return None
    kind = LEICHT if "leichte" in match["kind"].lower() else STANDARD
    rank = 2 if match["copy"] else (1 if match["dup"] else 0)
    return Candidate(key=name, pair_id=match["id"].zfill(4), kind=kind, rank=rank, size=0)


def choose(candidates: List[Candidate]) -> Tuple[Dict[Tuple[str, str], Candidate], List[str]]:
    """One winner per (pair_id, kind); the rest are recorded as superseded."""
    by_slot: Dict[Tuple[str, str], List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_slot[(candidate.pair_id, candidate.kind)].append(candidate)

    winners: Dict[Tuple[str, str], Candidate] = {}
    superseded: List[str] = []
    for slot, group in by_slot.items():
        group.sort(key=lambda c: (c.rank, c.key))
        winners[slot] = group[0]
        superseded.extend(c.key for c in group[1:])
    return winners, sorted(superseded)


def plan(names: List[str]) -> Tuple[Dict[str, Dict[str, Candidate]], ImportReport]:
    """Resolve the object listing into complete pairs plus everything that fell out."""
    report = ImportReport()
    candidates: List[Candidate] = []
    for name in names:
        candidate = classify(name)
        if candidate:
            candidates.append(candidate)
        elif not name.endswith(("_Analysis.json", "index.json", "mdr_glossar.json")) and name:
            report.unmatched.append(name)

    winners, report.superseded = choose(candidates)

    sides: Dict[str, Dict[str, Candidate]] = defaultdict(dict)
    for (pair_id, kind), candidate in winners.items():
        sides[pair_id][kind] = candidate

    pairs: Dict[str, Dict[str, Candidate]] = {}
    for pair_id, found in sorted(sides.items()):
        missing = {STANDARD, LEICHT} - set(found)
        if missing:
            report.incomplete[pair_id] = missing.pop()
        else:
            pairs[pair_id] = found
    report.unmatched.sort()
    return pairs, report


def list_objects(bucket: str, prefix: str) -> List[str]:
    """Object names relative to the prefix, top level only."""
    import boto3

    paginator = boto3.client("s3").get_paginator("list_objects_v2")
    names: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix):]
            if name and "/" not in name:
                names.append(name)
    return names


def download(bucket: str, prefix: str, pairs: Dict[str, Dict[str, Candidate]],
             out_dir: Path, report: ImportReport) -> None:
    import boto3

    client = boto3.client("s3")
    out_dir.mkdir(parents=True, exist_ok=True)
    for pair_id, sides in pairs.items():
        for kind, candidate in sorted(sides.items()):
            target = out_dir / f"{pair_id}_{kind}.txt"
            client.download_file(bucket, prefix + candidate.key, str(target))
            report.written.append(target.name)
    report.written.sort()


def print_report(pairs: Dict[str, Dict[str, Candidate]], report: ImportReport, wrote: bool) -> None:
    renamed = sorted(
        f"{c.key}  ->  {pid}_{kind}.txt"
        for pid, sides in pairs.items() for kind, c in sides.items()
        if c.key != f"{pid}_{kind}.txt"
    )

    print("\n" + "=" * 78)
    print(f"{'IMPORTED' if wrote else 'PLAN'}: {len(pairs)} complete pairs")
    print("=" * 78)

    if renamed:
        print(f"\nrenamed on the way in ({len(renamed)}):")
        for line in renamed:
            print(f"  {line}")
    if report.superseded:
        print(f"\nduplicates ignored ({len(report.superseded)}): a better-named object "
              f"exists for the same pair")
        for key in report.superseded:
            print(f"  {key}")
    if report.incomplete:
        print(f"\nincomplete pairs ({len(report.incomplete)}): only one side is a text file")
        for pair_id, missing in sorted(report.incomplete.items()):
            print(f"  {pair_id}: no {missing}")
    if report.unmatched:
        print(f"\nnot recognised as a pair ({len(report.unmatched)}):")
        for name in report.unmatched:
            print(f"  {name}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the parallel corpus from S3.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--write", action="store_true",
                        help="actually download; without it only the plan is printed")
    args = parser.parse_args()

    print(f"Listing s3://{args.bucket}/{args.prefix} ...", flush=True)
    names = list_objects(args.bucket, args.prefix)
    if not names:
        sys.exit(f"❌ Nothing under s3://{args.bucket}/{args.prefix}")

    pairs, report = plan(names)
    if not pairs:
        sys.exit(f"❌ {len(names)} objects, but no complete pair among them")

    if args.write:
        print(f"Downloading {2 * len(pairs)} files to {args.out_dir} ...", flush=True)
        download(args.bucket, args.prefix, pairs, args.out_dir, report)

    print_report(pairs, report, wrote=args.write)
    if not args.write:
        print("Dry run. Re-run with --write to download.\n")


if __name__ == "__main__":
    main()
