import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src-train")))
from import_raw import LEICHT, STANDARD, classify, plan

# Every spelling below is a real object name from s3://diwop-analysis/training-data/.
NAMES = [
    "0001_Standardsprache.txt", "0001_Leichte_Sprache.txt",
    "0032_Standardsprache.txt", "0032_Leichte Sprache.txt",        # space, not underscore
    "0114_Standardsprache .txt", "0114_Leichte Sprache .txt",      # trailing spaces
    "0124_Standartsprache.txt", "0124_Leichte_Sprache.txt",        # typo in the source
    "0137_Standardsprache.txt", "0137_Leichte Spracheprache.txt",  # duplicated syllable
    "0165_Standardsprache.txt", "165_Leichte Sprache .txt",        # missing leading zero
    "0001_Analysis.json", "index.json", "mdr_glossar.json",
]


def test_canonical_names_are_recognised():
    candidate = classify("0001_Standardsprache.txt")
    assert candidate.pair_id == "0001" and candidate.kind == STANDARD and candidate.rank == 0


def test_three_digit_ids_are_zero_padded():
    assert classify("165_Leichte Sprache .txt").pair_id == "0165"


def test_every_real_spelling_resolves_to_a_pair():
    pairs, report = plan(NAMES)
    assert sorted(pairs) == ["0001", "0032", "0114", "0124", "0137", "0165"]
    assert not report.incomplete
    assert not report.unmatched  # the json files are known noise, not failures


def test_kopie_von_loses_to_the_file_it_copies():
    """0150 exists under its own name; the 'Kopie von' variant must not win."""
    pairs, report = plan([
        "0150_Standardsprache.txt", "0150_Leichte Sprache.txt",
        "Kopie von 0150_Standardsprache.txt", "Kopie von 0150_Leichte Sprache .txt",
    ])
    assert pairs["0150"][STANDARD].key == "0150_Standardsprache.txt"
    assert pairs["0150"][LEICHT].key == "0150_Leichte Sprache.txt"
    assert len(report.superseded) == 2


def test_numbered_duplicate_loses_to_the_plain_name():
    pairs, _ = plan([
        "0038_Standardsprache.txt", "0038_Leichte_Sprache.txt", "0038_Leichte_Sprache(1).txt",
    ])
    assert pairs["0038"][LEICHT].key == "0038_Leichte_Sprache.txt"


def test_docx_is_not_a_pair_and_is_reported():
    pairs, report = plan(["0115_Standardsprache.txt", "0115_Leichte_Sprache.docx"])
    assert not pairs
    assert report.incomplete == {"0115": LEICHT}
    assert report.unmatched == ["0115_Leichte_Sprache.docx"]


def test_unrelated_files_are_reported_not_silently_dropped():
    _, report = plan(["Untitled document.docx", "0001_Standardsprache.txt",
                      "0001_Leichte_Sprache.txt"])
    assert report.unmatched == ["Untitled document.docx"]
