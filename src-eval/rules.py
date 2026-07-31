"""Does a text follow the Leichte Sprache rules in data/system-prompt.md?

The rubric is split into two tiers, because they answer different questions and only one
of them may be used to filter training data:

  Tier A -- describes the *relationship* between source and target (length ratio). These
           are objective pair errors. Safe to filter or repair on.
  Tier B -- describes a *single text* against the system prompt (sentence length,
           hyphenation, Verbalstil, ...). NEVER filter references on these: the human
           references are what validate the rules, so filtering by them is circular.
           A rule the references fail is a bug in the rule or in the system prompt.

Every Tier B check is a heuristic over plain text -- no dictionary, no parser. The
thresholds in `Thresholds` are guesses until they are calibrated against the references,
which is exactly what the default CLI run prints:

    python src-eval/rules.py               # score every pair in data/raw
    python src-eval/rules.py --json        # machine-readable, for eval_metrics.json
    python src-eval/rules.py --data-dir path/to/pairs
"""
import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# "z. B." must not end a sentence. Single letters and bare numbers cover ordinals and the
# numbered headings that appear throughout data/raw.
ABBREVIATIONS = {
    "bspw", "bzw", "ca", "evtl", "ggf", "inkl", "max", "min", "nr", "sog", "usw", "vgl",
    "dr", "prof", "st", "hr", "fr", "abs", "art", "bzgl", "ggfs", "jhd", "mio", "mrd",
}

SUBORDINATE_MARKERS = {
    "weil", "dass", "obwohl", "damit", "während", "bevor", "nachdem", "falls", "sodass",
    "wenn", "sobald", "solange", "seitdem", "indem", "wobei", "sofern",
}

NEGATIONS = {
    "nicht", "nichts", "kein", "keine", "keinen", "keinem", "keiner", "keines", "nie",
    "niemals", "niemand", "nirgends", "weder", "ohne",
}

NOMINALSTIL_SUFFIXES = ("ung", "heit", "keit", "nis", "tion", "ismus", "ität")

# `\bge\w+(?:t|en)\b` also matches these, and "Das war gestern" is not Plusquamperfekt.
NON_PARTICIPLES = {
    "gegen", "gegenüber", "gerade", "gern", "gerne", "gesamt", "gesund", "genau",
    "gemeinsam", "gestern", "gewiss", "geschwind", "gegenteil", "gebiet", "gerecht",
    "gebäude", "gesicht", "geschichten", "gefühlen", "gedanken", "geschwister",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_WORD = re.compile(r"[\w][\w\-']*", re.UNICODE)
_PARTICIPLE = re.compile(r"\b(?:ge\w{2,}(?:t|en)|\w{3,}iert)\b", re.IGNORECASE)
_PLUSQUAMPERFEKT_AUX = re.compile(r"\b(?:hatte|hattest|hatten|hattet|war|warst|waren|wart)\b", re.IGNORECASE)
_FUTUR_AUX = re.compile(r"\b(?:werde|wirst|wird|werden|werdet)\b", re.IGNORECASE)
_FUTUR_TAIL = re.compile(r"\b(?:haben|sein)\b", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s")
_CLAUSE_BOUNDARY = re.compile(r"[,;:]|\s[-–—]\s")


@dataclass(frozen=True)
class Thresholds:
    """Every value here is provisional until validated against the human references."""
    max_sentence_words: int = 10          # data/system-prompt.md: "Richtwert"
    min_compound_chars: int = 13          # long word without a hyphen -> likely a compound
    max_heading_words: int = 8
    min_length_ratio: float = 1.3         # Leichte Sprache expands; near 1.0 means content was dropped
    max_length_ratio: float = 3.5         # far above means content was invented


@dataclass(frozen=True)
class Violation:
    rule: str
    detail: str


@dataclass
class StyleMetrics:
    """Tier B: one text against the system prompt. Rates are 0..1 unless stated."""
    words: int
    sentences: int
    short_sentence_rate: float            # share of sentences <= max_sentence_words
    one_sentence_per_line_rate: float     # share of non-empty lines holding <= 1 sentence
    unhyphenated_compounds_per_100w: float
    subordinate_markers_per_sentence: float
    negations_per_sentence: float
    nominalstil_per_100w: float
    forbidden_tense_count: int            # Futur II + Plusquamperfekt
    headings: int
    violations: List[Violation] = field(default_factory=list)


@dataclass
class PairMetrics:
    """Tier A: source against target. The only tier that may filter training data."""
    source_words: int
    target_words: int
    length_ratio: float
    ratio_in_band: bool


@dataclass
class PairReport:
    pair_id: str
    pair: PairMetrics
    source_style: StyleMetrics
    target_style: StyleMetrics


def split_sentences(text: str) -> List[str]:
    """Split on terminal punctuation, re-joining fragments cut off at an abbreviation."""
    merged: List[str] = []
    for part in _SENTENCE_BOUNDARY.split(text.replace("\n", " ")):
        part = part.strip()
        if not part:
            continue
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _ends_with_abbreviation(fragment: str) -> bool:
    words = _WORD.findall(fragment)
    if not words:
        return False
    last = words[-1].lower()
    return last in ABBREVIATIONS or len(last) == 1 or last.isdigit()


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def _is_heading(line: str, previous: Optional[str], thresholds: Thresholds) -> bool:
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if _BULLET.match(line) or stripped[-1:] in {".", "!", "?", ",", ";", ":"}:
        return False
    if previous is not None and previous.strip():
        return False
    return 0 < count_words(stripped) <= thresholds.max_heading_words


def _forbidden_tenses(sentence: str) -> List[Violation]:
    """Auxiliary and participle must share a clause.

    Scanning the whole sentence flags "Das war immer gleich: Wir haben Menschen
    getroffen." -- 'war' and 'getroffen' sit in unrelated clauses, and the result is
    Präteritum plus Perfekt, not Plusquamperfekt.
    """
    found: List[Violation] = []
    for clause in _CLAUSE_BOUNDARY.split(sentence):
        has_participle = any(
            match.group(0).lower() not in NON_PARTICIPLES
            for match in _PARTICIPLE.finditer(clause)
        )
        if not has_participle:
            continue
        if _FUTUR_AUX.search(clause) and _FUTUR_TAIL.search(clause):
            found.append(Violation("futur_ii", clause.strip()))
        elif _PLUSQUAMPERFEKT_AUX.search(clause):
            found.append(Violation("plusquamperfekt", clause.strip()))
    return found


def analyse_style(text: str, thresholds: Thresholds = Thresholds()) -> StyleMetrics:
    """Tier B metrics for a single text."""
    sentences = split_sentences(text)
    words = count_words(text)
    violations: List[Violation] = []

    long_sentences = [s for s in sentences if count_words(s) > thresholds.max_sentence_words]
    violations.extend(
        Violation("sentence_too_long", f"{count_words(s)} Wörter: {s}") for s in long_sentences
    )

    lines = [line for line in text.splitlines() if line.strip()]
    crowded = [line for line in lines if len(split_sentences(line)) > 1]
    violations.extend(Violation("multiple_sentences_per_line", line) for line in crowded)

    compounds = [
        word for word in _WORD.findall(text)
        if "-" not in word and word.isalpha() and len(word) >= thresholds.min_compound_chars
    ]
    violations.extend(Violation("unhyphenated_compound", word) for word in sorted(set(compounds)))

    all_words = [word.lower() for word in _WORD.findall(text)]
    subordinates = [word for word in all_words if word in SUBORDINATE_MARKERS]
    negations = [word for word in all_words if word in NEGATIONS]
    nominalstil = [
        word for word in all_words
        if len(word) >= 6 and word.endswith(NOMINALSTIL_SUFFIXES)
    ]
    violations.extend(Violation("nominalstil", word) for word in sorted(set(nominalstil)))

    tense_violations: List[Violation] = []
    for sentence in sentences:
        tense_violations.extend(_forbidden_tenses(sentence))
    violations.extend(tense_violations)

    previous: Optional[str] = None
    headings = 0
    for line in text.splitlines():
        if line.strip() and _is_heading(line, previous, thresholds):
            headings += 1
        previous = line

    sentence_count = len(sentences) or 1
    return StyleMetrics(
        words=words,
        sentences=len(sentences),
        short_sentence_rate=_rate(len(sentences) - len(long_sentences), len(sentences)),
        one_sentence_per_line_rate=_rate(len(lines) - len(crowded), len(lines)),
        unhyphenated_compounds_per_100w=round(100 * len(compounds) / max(words, 1), 1),
        subordinate_markers_per_sentence=round(len(subordinates) / sentence_count, 2),
        negations_per_sentence=round(len(negations) / sentence_count, 2),
        nominalstil_per_100w=round(100 * len(nominalstil) / max(words, 1), 1),
        forbidden_tense_count=len(tense_violations),
        headings=headings,
        violations=violations,
    )


def analyse_pair(source: str, target: str, thresholds: Thresholds = Thresholds()) -> PairMetrics:
    """Tier A metrics: does the target plausibly translate the source?"""
    source_words = count_words(source)
    target_words = count_words(target)
    ratio = round(target_words / source_words, 2) if source_words else 0.0
    return PairMetrics(
        source_words=source_words,
        target_words=target_words,
        length_ratio=ratio,
        ratio_in_band=thresholds.min_length_ratio <= ratio <= thresholds.max_length_ratio,
    )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 1.0


def find_pairs(data_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """Mirrors the pairing in src-train/prepare_dataset.py."""
    pattern = re.compile(r"^(\d+)_(Standardsprache|Leichte_Sprache)\.(md|txt)$")
    standard: Dict[str, Path] = {}
    leicht: Dict[str, Path] = {}

    for path in sorted(data_dir.iterdir()):
        match = pattern.match(path.name)
        if not match:
            continue
        pair_id, kind, _ = match.groups()
        (standard if kind == "Standardsprache" else leicht)[pair_id] = path

    return {
        pair_id: (standard[pair_id], leicht[pair_id])
        for pair_id in sorted(set(standard) & set(leicht))
    }


def build_reports(data_dir: Path, thresholds: Thresholds) -> List[PairReport]:
    reports: List[PairReport] = []
    for pair_id, (source_path, target_path) in find_pairs(data_dir).items():
        source = source_path.read_text(encoding="utf-8").strip()
        target = target_path.read_text(encoding="utf-8").strip()
        reports.append(PairReport(
            pair_id=pair_id,
            pair=analyse_pair(source, target, thresholds),
            source_style=analyse_style(source, thresholds),
            target_style=analyse_style(target, thresholds),
        ))
    return reports


def print_report(reports: List[PairReport], thresholds: Thresholds) -> None:
    print("\n" + "=" * 78)
    print("TIER A -- pair length ratio (may be used to filter training data)")
    print(f"band: {thresholds.min_length_ratio}x .. {thresholds.max_length_ratio}x")
    print("=" * 78)
    print(f"{'ID':<8}{'Standard':>10}{'Leicht':>10}{'ratio':>9}   verdict")
    for report in reports:
        pair = report.pair
        verdict = "ok" if pair.ratio_in_band else (
            "FLAG too flat" if pair.length_ratio < thresholds.min_length_ratio else "FLAG inflated"
        )
        print(f"{report.pair_id:<8}{pair.source_words:>10}{pair.target_words:>10}"
              f"{pair.length_ratio:>8.2f}x   {verdict}")

    flagged = [r.pair_id for r in reports if not r.pair.ratio_in_band]
    if flagged:
        print(f"\n⚠️  {len(flagged)}/{len(reports)} pairs flagged for human review: {', '.join(flagged)}")
        print("   Review and repair -- do NOT auto-exclude at this dataset size.")

    print("\n" + "=" * 78)
    print("TIER B -- style, per text (diagnostic only; NEVER filter references on this)")
    print("=" * 78)
    header = (f"{'ID':<8}{'text':<10}{'<=10w':>8}{'1s/line':>9}{'cmpd':>7}"
              f"{'subord':>8}{'neg':>6}{'nomin':>7}{'tense':>7}{'head':>6}")
    print(header)
    for report in reports:
        for label, style in (("Standard", report.source_style), ("Leicht", report.target_style)):
            print(f"{report.pair_id if label == 'Standard' else '':<8}{label:<10}"
                  f"{style.short_sentence_rate:>8.0%}{style.one_sentence_per_line_rate:>9.0%}"
                  f"{style.unhyphenated_compounds_per_100w:>7.1f}"
                  f"{style.subordinate_markers_per_sentence:>8.2f}"
                  f"{style.negations_per_sentence:>6.2f}{style.nominalstil_per_100w:>7.1f}"
                  f"{style.forbidden_tense_count:>7}{style.headings:>6}")

    print("\n" + "-" * 78)
    print("MEDIAN over all texts -- the Leicht row is the target the model must reach")
    print("-" * 78)
    print(header)
    for label, styles in (("Standard", [r.source_style for r in reports]),
                          ("Leicht", [r.target_style for r in reports])):
        print(f"{'':<8}{label:<10}"
              f"{statistics.median(s.short_sentence_rate for s in styles):>8.0%}"
              f"{statistics.median(s.one_sentence_per_line_rate for s in styles):>9.0%}"
              f"{statistics.median(s.unhyphenated_compounds_per_100w for s in styles):>7.1f}"
              f"{statistics.median(s.subordinate_markers_per_sentence for s in styles):>8.2f}"
              f"{statistics.median(s.negations_per_sentence for s in styles):>6.2f}"
              f"{statistics.median(s.nominalstil_per_100w for s in styles):>7.1f}"
              f"{statistics.median(s.forbidden_tense_count for s in styles):>7.0f}"
              f"{statistics.median(s.headings for s in styles):>6.0f}")
    print("\nA Tier B rule the human references fail is a bug in the rule or evidence that")
    print("data/system-prompt.md is stricter than real practice -- not a reason to drop data.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Leichte Sprache rule compliance.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--json", action="store_true", help="emit metrics instead of a table")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        sys.exit(f"❌ No such directory: {args.data_dir}")

    thresholds = Thresholds()
    reports = build_reports(args.data_dir, thresholds)
    if not reports:
        sys.exit(f"❌ No <id>_Standardsprache / <id>_Leichte_Sprache pairs in {args.data_dir}")

    if args.json:
        print(json.dumps({
            "thresholds": asdict(thresholds),
            "pairs": [asdict(report) for report in reports],
        }, ensure_ascii=False, indent=2))
    else:
        print_report(reports, thresholds)


if __name__ == "__main__":
    main()
