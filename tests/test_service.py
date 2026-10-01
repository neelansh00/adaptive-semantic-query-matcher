"""Phase 9: service layer and Streamlit app. These load the frozen models (~15 s), once per module."""
import numpy as np
import pandas as pd
import pytest

from src.utils.data import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "artifacts" / "phase6" / "meta_model.joblib").exists(),
                                reason="frozen models not available")


@pytest.fixture(scope="module")
def matcher():
    from src.service.matcher import DuplicateMatcher
    return DuplicateMatcher()


def test_result_fields_and_frozen_policy(matcher):
    r = matcher.match("How can I lose weight quickly?", "What is the fastest way to lose weight?")
    assert 0.0 <= r.final_probability <= 1.0 and 0.0 <= r.base_probability <= 1.0
    assert r.global_threshold == pytest.approx(0.35) and r.base_threshold == pytest.approx(0.32)
    assert r.decision == (r.final_probability >= r.global_threshold)
    assert r.cluster_threshold_applied is False and "Phase 7" in r.cluster_threshold_note
    assert 0 <= r.cluster_id < 12 and r.cluster_name
    assert r.rationale and r.rationale[-1].startswith("Final probability")


def test_symmetric_and_deterministic(matcher):
    a, b = "Who won the 2012 US presidential election?", "Who won the 2016 US presidential election?"
    r1, r2, r3 = matcher.match(a, b), matcher.match(b, a), matcher.match(a, b)
    assert r1.final_probability == pytest.approx(r2.final_probability, abs=1e-6)
    assert r1.cluster_id == r2.cluster_id
    assert r1.final_probability == r3.final_probability and r1.rationale == r3.rationale


@pytest.mark.parametrize("a,b,expected_dup,signal", [
    ("How can I lose weight quickly?", "What is the fastest way to lose weight?", True, None),
    ("How can I lose 5 kg in a month?", "How can I lose 20 kg in a month?", False, "numbers mismatch"),
    ("Why do people believe in God?", "Why do people not believe in God?", False, "negation mismatch"),
    ("What is the temperament of a Doberman/Lab mix?", "What is the temperament of a Lab/Pitbull mix?", False,
     "named entities mismatch"),
])
def test_demo_examples_behave_as_documented(matcher, a, b, expected_dup, signal):
    r = matcher.match(a, b)
    assert r.decision is expected_dup
    if signal:
        assert signal in [s["type"] for s in r.signals]


def test_what_if_is_coherent(matcher):
    r = matcher.match("How can I lose 5 kg in a month?", "How can I lose 20 kg in a month?")
    groups = {w["group"] for w in r.what_if}
    assert "numbers" in groups and "scope / specificity" not in groups      # specificity has no coherent counterfactual
    num = next(w for w in r.what_if if w["group"] == "numbers")
    assert num["probability_if_no_difference"] > r.final_probability       # matching numbers would raise the score
    assert num["effect"] == pytest.approx(r.final_probability - num["probability_if_no_difference"])
    identical = matcher.match("Who founded Microsoft?", "Who founded Microsoft?")
    assert identical.signals == [] and identical.what_if == [] and identical.decision


def test_extraction_reported(matcher):
    r = matcher.match("What is the best phone under 10000 rupees in India?", "What is the best phone under 20000 rupees?")
    assert r.extracted["question_a"]["numbers"] == ["10000"] and r.extracted["question_b"]["numbers"] == ["20000"]
    assert "india" in r.extracted["question_a"]["entities"]


def test_empty_input_rejected(matcher):
    with pytest.raises(ValueError):
        matcher.match("   ", "What is AI?")


def test_demo_reproduces_evaluated_scores(matcher):
    """The demo must give the same decisions as the system evaluated in Phases 6-8."""
    if not (PROJECT_ROOT / "data" / "processed" / "splits" / "val.csv").exists():
        pytest.skip("split CSVs not regenerated (python scripts/reproduce.py --run extract splits)")
    val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "splits" / "val.csv",
                      keep_default_na=False, na_values=[""]).sample(25, random_state=7)
    stored = pd.read_csv(PROJECT_ROOT / "artifacts" / "phase6" / "val_scores.csv").set_index("id").loc[val["id"], "C_no_spacy_hgb"]
    demo = np.array([matcher.match(a, b).final_probability for a, b in zip(val.question1, val.question2)])
    assert np.abs(demo - stored.to_numpy()).max() < 1e-5
    assert np.array_equal(demo >= 0.35, stored.to_numpy() >= 0.35)


def test_streamlit_app_runs_and_renders_decision():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"), default_timeout=240)
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Duplicate" in s.value for s in at.success)                  # default example is a paraphrase
    assert {m.label for m in at.metric} >= {"Final probability", "Semantic similarity", "Global threshold"}
    at.sidebar.radio[0].set_value("Negation mismatch").run()
    at.button[0].click().run()
    assert any("Not a duplicate" in e.value for e in at.error)
