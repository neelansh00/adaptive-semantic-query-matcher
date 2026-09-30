import numpy as np
import pandas as pd
import pytest

from src.evaluation.error_analysis import error_tags
from src.features.lexical import FEATURE_NAMES, lexical_features, pair_features
from src.preprocessing.text import normalize, preprocess, tokenize


# ---------------------------------------------------------------- preprocessing

@pytest.mark.parametrize("raw, expected", [
    ("Why can't I sleep?", ["why", "can", "not", "i", "sleep"]),
    ("I don’t know", ["i", "do", "not", "know"]),          # curly apostrophe
    ("He won't cannot", ["he", "will", "not", "can", "not"]),
])
def test_negation_is_preserved(raw, expected):
    assert preprocess(raw) == expected


def test_numbers_are_preserved():
    assert preprocess("Lose 3.5 kg by 2016 for 1,000 USD") == ["lose", "3.5", "kg", "by", "2016", "for", "1,000", "usd"]


def test_normalize_handles_missing_and_whitespace():
    assert normalize(None) == ""
    assert normalize(float("nan")) == ""
    assert normalize("  What   is\nAI? ") == "what is ai?"


def test_math_tags_removed_but_formula_kept():
    assert "x^2" in normalize("Solve [math]x^2=4[/math]")
    assert "[math]" not in normalize("Solve [math]x^2=4[/math]")


def test_punctuation_optional():
    assert tokenize("what is ai?", keep_punct=True)[-1] == "?"
    assert "?" not in tokenize("what is ai?")


# ---------------------------------------------------------------- lexical features

def test_features_are_symmetric():
    a, b = "How do I learn Python quickly?", "What is the fastest way to learn Python?"
    assert pair_features(a, b) == pair_features(b, a)


def test_identical_questions():
    f = dict(zip(FEATURE_NAMES, pair_features("Who founded Microsoft?", "who founded  microsoft?")))
    assert f["word_jaccard"] == 1.0 and f["char3_jaccard"] == 1.0
    assert f["word_count_absdiff"] == 0 and f["first_word_equal"] == 1.0


def test_entity_swap_keeps_high_overlap():
    # Documents WHY lexical features alone fail: one-entity swaps still look similar.
    f = dict(zip(FEATURE_NAMES, pair_features("Who founded Microsoft?", "Who founded Apple?")))
    assert f["word_jaccard"] == pytest.approx(0.5)
    assert f["common_ratio_min"] == pytest.approx(2 / 3)


def test_empty_input_does_not_crash():
    f = pair_features("", "")
    assert len(f) == len(FEATURE_NAMES)
    assert all(np.isfinite(f))


def test_lexical_features_frame():
    df = pd.DataFrame({"question1": ["a b", "c"], "question2": ["a", "d"]}, index=[5, 7])
    out = lexical_features(df)
    assert list(out.columns) == FEATURE_NAMES
    assert list(out.index) == [5, 7]


# ---------------------------------------------------------------- error tags (analysis only)

def test_error_tags():
    assert error_tags("How can I lose 10 kg?", "How can I lose 30 kg?")["number_diff"]
    assert error_tags("Who founded Microsoft?", "Who founded Apple?")["entity_diff"]
    assert error_tags("examples of non-movable joints", "examples of movable joints")["negation_diff"]
    assert not error_tags("Why is the sky blue?", "Why is the sky blue?")["entity_diff"]
