import json
import re

import numpy as np
import pandas as pd
import pytest

from src.clustering.core import CentroidModel
from src.clustering.pairs import UNASSIGNED, assign_pairs, centroid_geometry, pair_embedding
from src.evaluation.cluster_analysis import (bootstrap_clusters, cluster_table, f1_curve, optimal_threshold,
                                             random_partition_null, robustness_summary)
from src.evaluation.metrics import best_f1_threshold, classification_metrics, threshold_sweep
from src.utils.data import PROJECT_ROOT

FROZEN = PROJECT_ROOT / "artifacts" / "phase4" / "cluster_model"


def unit_rows(n, d=16, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def toy_model():
    return CentroidModel(unit_rows(5, seed=1) * np.array([[0.3], [0.4], [0.35], [0.45], [0.3]], dtype=np.float32))


# ---------------------------------------------------------------- pair assignment

def test_pair_average_assignment_is_symmetric(toy_model):
    u, v = unit_rows(500, seed=2), unit_rows(500, seed=3)
    assert (assign_pairs(u, v, toy_model, "pair_avg") == assign_pairs(v, u, toy_model, "pair_avg")).all()
    pe = pair_embedding(u, v)
    np.testing.assert_allclose(np.linalg.norm(pe, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(pe, pair_embedding(v, u), atol=1e-6)


def test_q1_rule_is_order_dependent_and_same_only_marks_unassigned(toy_model):
    u, v = unit_rows(500, seed=4), unit_rows(500, seed=5)
    q1_uv, q1_vu = assign_pairs(u, v, toy_model, "q1"), assign_pairs(v, u, toy_model, "q1")
    assert (q1_uv != q1_vu).any()                       # asymmetric by construction
    same = assign_pairs(u, v, toy_model, "same_only")
    a, b = toy_model.predict(u), toy_model.predict(v)
    assert ((same == UNASSIGNED) == (a != b)).all()
    assert (same[a == b] == a[a == b]).all()
    with pytest.raises(ValueError):
        assign_pairs(u, v, toy_model, "nope")


def test_identical_questions_assigned_like_the_question(toy_model):
    u = unit_rows(200, seed=6)
    assert (assign_pairs(u, u, toy_model, "pair_avg") == toy_model.predict(u)).all()


def test_geometry_is_consistent_with_assignment(toy_model):
    X = unit_rows(1000, seed=7)
    g = centroid_geometry(X, toy_model)
    assert (g["cluster"] == toy_model.predict(X)).all()   # margin is measured in the assignment geometry
    assert (g["margin"] >= 0).all()
    assert (g["second_cluster"] != g["cluster"]).all()


def test_frozen_centroids_load_and_assign_deterministically():
    if not (FROZEN / "centroids.npy").exists():
        pytest.skip("frozen Phase 4 centroids not present")
    m1, m2 = CentroidModel.load(FROZEN), CentroidModel.load(FROZEN)
    assert m1.k == 12 and m1.centroids.shape == (12, 384)
    u, v = unit_rows(300, d=384, seed=8), unit_rows(300, d=384, seed=9)
    for rule in ("q1", "pair_avg", "same_only"):
        np.testing.assert_array_equal(assign_pairs(u, v, m1, rule), assign_pairs(u, v, m2, rule))


# ---------------------------------------------------------------- metrics and thresholds

@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(10)
    y = rng.integers(0, 2, 3000)
    s = np.clip(rng.normal(0.3 + 0.35 * y, 0.2), 0, 1)
    groups = rng.integers(0, 4, 3000)
    return y, s, groups


def test_f1_curve_matches_reference_sweep(scored):
    y, s, _ = scored
    np.testing.assert_allclose(f1_curve(y, s), threshold_sweep(y, s)["f1"].to_numpy(), atol=1e-12)
    assert optimal_threshold(y, s)[0] == pytest.approx(best_f1_threshold(y, s))


def test_cluster_table_matches_per_group_metrics(scored):
    y, s, g = scored
    t = cluster_table(y, s, g, global_threshold=0.4)
    for c in range(4):
        m = classification_metrics(y[g == c], s[g == c], 0.4)
        assert t.loc[c, "f1"] == pytest.approx(m["f1"])
        assert t.loc[c, "fpr"] == pytest.approx(m["fpr"])
        assert t.loc[c, "n"] == (g == c).sum()
        assert t.loc[c, "opt_threshold"] == pytest.approx(best_f1_threshold(y[g == c], s[g == c]))
        assert t.loc[c, "f1_at_opt_in_sample"] >= t.loc[c, "f1"] - 1e-12   # optimum can only be >= any fixed threshold
    summ = robustness_summary(t, overall_f1=0.7, global_threshold=0.4)
    assert summ["worst_cluster_f1"] == pytest.approx(t["f1"].min())
    assert summ["macro_cluster_f1"] == pytest.approx(t["f1"].mean())


def test_bootstrap_and_null_behave_on_random_groups(scored):
    y, s, g = scored
    boot = bootstrap_clusters(y, s, g, 0.4, n_boot=100)
    t = cluster_table(y, s, g, 0.4)
    assert ((boot["f1_ci_low"] <= t["f1"]) & (t["f1"] <= boot["f1_ci_high"])).all()
    null = random_partition_null(y, s, g, 0.4, n_perm=50)
    # random groups: the observed spread should look like the null (not extreme)
    assert t["f1"].std(ddof=0) < null["f1_std"]["max"] * 1.5


# ---------------------------------------------------------------- validation-only / no test access

PHASE5_SCRIPTS = ["analyze_clusters.py", "threshold_reliability.py", "summarize_manual_errors.py"]


@pytest.mark.parametrize("script", PHASE5_SCRIPTS)
def test_phase5_scripts_never_reference_test_split(script):
    src = (PROJECT_ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert not re.search(r"test\.csv|[\"']test[\"']", src), f"{script} references the test split"


def test_phase5_outputs_cover_exactly_the_validation_split():
    pairs = PROJECT_ROOT / "artifacts" / "phase5" / "pair_assignments.csv"
    if not pairs.exists():
        pytest.skip("Phase 5 analysis not run")
    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    val_ids = set(pd.read_csv(split_dir / "val.csv", usecols=["id"])["id"])
    out_ids = set(pd.read_csv(pairs, usecols=["id"])["id"])
    assert out_ids == val_ids
    meta = json.loads((split_dir / "split_metadata.json").read_text())
    assert meta["counts"]["val"] == len(out_ids)
