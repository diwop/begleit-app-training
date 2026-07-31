import json
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src-train")))
from validation_metrics import compare, load_cases, score

# One-sentence-per-line, short sentences, hyphenated compounds -- what the system prompt asks for.
GOOD = ("Die Kranken-Versicherung zahlt.\n"
        "Sie hilft kranken Menschen.\n"
        "Der Arzt hilft Ihnen.\n")
# The same content as one long sentence with an unhyphenated compound and Nominalstil.
BAD = ("Die Krankenversicherung zahlt und hilft kranken Menschen, weil der Arzt Ihnen bei "
       "der Behandlung und der Untersuchung eine Unterstützung anbietet.\n")
SOURCE = "Die Krankenkasse übernimmt die Kosten der ärztlichen Behandlung.\n"


def test_identical_text_has_zero_distance():
    reference = score([GOOD], [SOURCE])
    metrics = compare(score([GOOD], [SOURCE]), reference)
    assert metrics["eval_ls_distance"] == 0.0
    assert metrics["eval_ls_gap_short_sentence_rate"] == 0.0


def test_worse_text_is_further_from_the_reference():
    reference = score([GOOD], [SOURCE])
    good = compare(score([GOOD], [SOURCE]), reference)["eval_ls_distance"]
    bad = compare(score([BAD], [SOURCE]), reference)["eval_ls_distance"]
    assert bad > good


def test_distance_stays_bounded_when_one_metric_runs_away():
    """A single runaway component must not swallow the composite.

    Unclamped, an untrained model emitting one 400-character non-word scored
    unhyphenated_compounds_per_100w = 100 against a reference of 0, and the composite came
    out at 16.8 -- a number with no scale to read it against, driven entirely by that one
    term. Every raw gap is still logged separately, so clamping loses nothing.
    """
    reference = score([GOOD], [SOURCE])
    garbage = compare(score(["Xqzwkjhgfdsayxcvbnmqwertzuiopasdfghjkl"], [SOURCE]), reference)
    assert 0.0 <= garbage["eval_ls_distance"] <= 1.0
    # ...while the underlying gap is still reported in full.
    assert garbage["eval_ls_gap_unhyphenated_compounds_per_100w"] > 1.0


def test_gaps_are_signed_so_the_direction_is_readable():
    """Fewer short sentences than the humans must read as negative, not just 'different'."""
    reference = score([GOOD], [SOURCE])
    metrics = compare(score([BAD], [SOURCE]), reference)
    assert metrics["eval_ls_gap_short_sentence_rate"] < 0
    assert metrics["eval_ls_gap_unhyphenated_compounds_per_100w"] > 0


def test_load_cases_drops_the_answer_from_the_prompt(tmp_path):
    """The assistant turn is the thing being predicted; leaving it in would leak it."""
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps({
        "id": "0042",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "antwort"},
        ],
        "original_text": "quelle",
        "metrics": {},
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    case = load_cases(path, limit=5)[0]
    assert case.pair_id == "0042"
    assert case.reference == "antwort"
    assert [m["role"] for m in case.prompt_messages] == ["system", "user"]


def test_load_cases_respects_the_limit(tmp_path):
    path = tmp_path / "validation.jsonl"
    lines = [json.dumps({
        "id": f"{i:04d}",
        "messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
        "original_text": "q",
        "metrics": {},
    }) for i in range(10)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert len(load_cases(path, limit=3)) == 3
