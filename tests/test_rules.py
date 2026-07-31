import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src-eval")))
from rules import Thresholds, analyse_pair, analyse_style, count_words, split_sentences


def test_split_sentences_keeps_abbreviations_together():
    # "z. B." must not end a sentence, or every example in the system prompt splits wrongly
    assert split_sentences("Nimm Obst. Zum Beispiel: z. B. Äpfel und Birnen.") == [
        "Nimm Obst.", "Zum Beispiel: z. B. Äpfel und Birnen."
    ]


def test_split_sentences_joins_lines_of_one_sentence():
    # Leichte Sprache breaks a single sentence across lines; that is one sentence, not two
    assert split_sentences("Wir möchten allen zeigen,\nwie toll Hamburg ist.") == [
        "Wir möchten allen zeigen, wie toll Hamburg ist."
    ]


def test_count_words_treats_hyphenated_compound_as_one_word():
    assert count_words("Die Kranken-Versicherung zahlt.") == 3
    assert count_words("Ich bin 20 Jahre alt.") == 5


def test_short_sentence_rate():
    text = "Das ist gut.\nDieser Satz hat deutlich mehr als zehn Wörter und ist damit zu lang."
    assert analyse_style(text).short_sentence_rate == 0.5


def test_unhyphenated_compound_is_flagged():
    flagged = analyse_style("Die Krankenversicherung zahlt.").unhyphenated_compounds_per_100w
    hyphenated = analyse_style("Die Kranken-Versicherung zahlt.").unhyphenated_compounds_per_100w
    assert flagged > 0 and hyphenated == 0


def test_multiple_sentences_per_line_is_flagged():
    assert analyse_style("Das ist gut.\nDas ist schön.").one_sentence_per_line_rate == 1.0
    assert analyse_style("Das ist gut. Das ist schön.").one_sentence_per_line_rate == 0.0


def test_plusquamperfekt_needs_aux_and_participle_in_one_clause():
    assert analyse_style("Er war weit gegangen.").forbidden_tense_count == 1
    # Regression: 'war' and 'getroffen' sit in unrelated clauses -- Präteritum plus Perfekt
    assert analyse_style("Das war immer gleich: Wir haben Menschen getroffen.").forbidden_tense_count == 0


def test_non_participles_do_not_trigger_the_tense_rule():
    assert analyse_style("Das Wetter war gestern schön.").forbidden_tense_count == 0


def test_headings_are_counted_but_bullets_and_continuations_are_not():
    text = "FC St. Pauli\n\nDer Verein ist bekannt.\n\n- FC bedeutet: Fußball-Club.\n"
    assert analyse_style(text).headings == 1


def test_pair_ratio_band():
    thresholds = Thresholds()
    inflated = analyse_pair("ein Wort", " ".join(["Wort"] * 40), thresholds)
    assert inflated.length_ratio == 20.0 and not inflated.ratio_in_band

    # The corpus median is 0.61: a Leichte Sprache text of roughly the source's length is
    # the normal case, not a flag. Only the tails are flagged.
    flat = analyse_pair(" ".join(["Wort"] * 10), " ".join(["Wort"] * 11), thresholds)
    assert flat.length_ratio == 1.1 and flat.ratio_in_band

    gutted = analyse_pair(" ".join(["Wort"] * 100), " ".join(["Wort"] * 10), thresholds)
    assert not gutted.ratio_in_band


def test_empty_source_does_not_divide_by_zero():
    assert analyse_pair("", "Ein Text.").length_ratio == 0.0
