import json
import re

import numpy as np
import pandas as pd
import pytest

from src.features.constraints import (FEATURE_GROUPS, _canon, _numbers, _undot, annotate,
                                      pair_constraint_features)
from src.models.meta import VARIANTS, MetaModel, logit, variant_by_name
from src.utils.data import PROJECT_ROOT


def feats(a, b):
    return pair_constraint_features(annotate(a), annotate(b), use_spacy=False)


# ---------------------------------------------------------------- extraction

def test_number_extraction():
    assert _numbers("lose 5 kg in 15 days") == {"5", "15"}
    assert _numbers("costs 1,000 or 2.5k") == {"1000", "2.5"}
    assert _numbers("how does one apply") == set()            # 'one' is not treated as a number
    assert _numbers("two thousand people") == {"2", "1000"}


def test_acronym_and_case_normalisation():
    assert _undot("the U.S. army") == "the US army"
    assert _canon("U.S.") == "us"
    f = feats("How do I apply to the U.S. army?", "How do I apply to the US Army?")
    assert f["enth_mismatch"] == 0 and f["enth_one_sided"] == 0
    f = feats("Best places to visit in india?", "Best places to visit in India?")
    assert f["enth_one_sided"] == 0                           # casing alone is not a mismatch


# ---------------------------------------------------------------- pair features

@pytest.mark.parametrize("a,b,key", [
    ("Who founded Microsoft?", "Who founded Apple?", "enth_mismatch"),
    ("How can I lose 5 kg in a month?", "How can I lose 20 kg in a month?", "num_mismatch"),
    ("Who won the election in 2012?", "Who won the election in 2016?", "date_mismatch"),
    ("Why do people believe in God?", "Why do people not believe in God?", "neg_xor"),
    ("Examples of movable joints?", "Examples of non-movable joints?", "neg_xor"),
    ("How do I learn programming?", "How do I learn Python programming for data science?", "content_subset"),
])
def test_mismatch_signals_fire(a, b, key):
    assert feats(a, b)[key] == 1.0


def test_paraphrase_has_no_constraint_mismatch():
    f = feats("How can I lose weight quickly?", "What is the fastest way to lose weight?")
    assert all(f[k] == 0 for k in ("num_mismatch", "date_mismatch", "enth_mismatch", "neg_xor"))


@pytest.mark.parametrize("a,b", [
    ("Who founded Microsoft in 1975?", "Who founded Apple in 1976?"),
    ("Why do people not like 5 apples?", "What is the best phone under 20000 in India?"),
    ("", "What is AI?"),
])
def test_pair_features_are_exactly_symmetric(a, b):
    assert feats(a, b) == feats(b, a)


def test_identical_questions_have_zero_mismatch():
    q = "Who founded Microsoft in 1975 and why not Apple?"
    f = feats(q, q)
    mism = [k for k in f if k.endswith(("_mismatch", "_one_sided", "_unmatched_total", "_xor"))]
    assert all(f[k] == 0 for k in mism)
    assert f["extra_content_max"] == 0 and f["content_subset"] == 0


def test_spacy_features_present_only_with_spacy():
    a, b = annotate("Best hotels in Delhi?"), annotate("Best hotels in Mumbai?")
    assert "loc_mismatch" not in pair_constraint_features(a, b, use_spacy=False)
    assert "loc_mismatch" in pair_constraint_features(a, b, use_spacy=True)


# ---------------------------------------------------------------- meta-model

def _toy_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    cols = {c: rng.integers(0, 2, n).astype(float) for g in FEATURE_GROUPS.values() for c in g}
    df = pd.DataFrame(cols)
    y = rng.integers(0, 2, n)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, n), 0.01, 0.99)
    df["base_logit"] = logit(p)
    df["cluster"] = rng.integers(0, 12, n)
    for c in ["word_jaccard", "content_jaccard", "char3_jaccard", "common_ratio_min", "common_ratio_max"]:
        df[c] = rng.random(n)
    return df, y


@pytest.mark.parametrize("variant", [v.name for v in VARIANTS])
def test_meta_variants_fit_and_are_deterministic(variant):
    df, y = _toy_frame()
    v = variant_by_name(variant)
    p1 = MetaModel(v).fit(df, y).predict_proba(df)
    p2 = MetaModel(v).fit(df, y).predict_proba(df)
    np.testing.assert_allclose(p1, p2)
    assert ((p1 >= 0) & (p1 <= 1)).all()


def test_meta_base_only_is_monotone_in_base_score():
    df, y = _toy_frame()
    m = MetaModel(variant_by_name("B_recal")).fit(df, y)
    order = np.argsort(df["base_logit"].to_numpy())
    assert np.all(np.diff(m.predict_proba(df)[order]) >= -1e-12)   # recalibration preserves ranking


def test_logit_is_finite_at_extremes():
    assert np.isfinite(logit(np.array([0.0, 1.0, 0.5]))).all()


# ---------------------------------------------------------------- no test-split access

@pytest.mark.parametrize("script", ["build_constraint_features.py", "crossfit_base.py", "train_meta.py"])
def test_phase6_scripts_never_reference_test_split(script):
    src = (PROJECT_ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert not re.search(r"test\.csv|[\"']test[\"']", src)


def test_saved_meta_model_matches_chosen_variant():
    chosen = PROJECT_ROOT / "artifacts" / "phase6" / "chosen.json"
    if not chosen.exists():
        pytest.skip("Phase 6 meta-model not trained yet")
    import joblib
    bundle = joblib.load(PROJECT_ROOT / "artifacts" / "phase6" / "meta_model.joblib")
    meta = json.loads(chosen.read_text())
    assert bundle["variant"] == meta["chosen"]
    assert bundle["threshold"] == pytest.approx(meta["threshold"])


def test_train_serve_feature_parity_without_spacy():
    """Features recomputed through the inference path (no spaCy) must equal the stored training features
    for every column a spaCy-free variant uses (guards against train/serve skew)."""
    from src.models.meta import variant_by_name
    from src.models.sentence_encoder import clean
    stored = PROJECT_ROOT / "data" / "processed" / "features" / "constraints_val.csv"
    if not stored.exists():
        pytest.skip("constraint features not built")
    val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "splits" / "val.csv",
                      keep_default_na=False, na_values=[""]).head(300)
    feats_stored = pd.read_csv(stored).set_index("id").loc[val["id"]]
    cols = [c for c in variant_by_name("C_no_spacy_hgb").columns() if c != "base_logit"]
    for (_, row), (_, s) in zip(val.iterrows(), feats_stored.iterrows()):
        f = pair_constraint_features(annotate(clean(row.question1)), annotate(clean(row.question2)), use_spacy=False)
        for c in cols:
            assert f[c] == pytest.approx(s[c]), (row.id, c)
