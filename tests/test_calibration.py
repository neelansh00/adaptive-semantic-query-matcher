import re

import numpy as np
import pytest

from src.calibration.thresholds import (ThresholdPolicy, cross_fitted_predictions, decision_metrics, group_folds)
from src.evaluation.metrics import best_f1_threshold
from src.utils.data import PROJECT_ROOT


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 6000
    clusters = rng.integers(0, 4, n)
    y = rng.integers(0, 2, n)
    # cluster 0 scores are shifted up, so its optimal threshold should be higher than the others
    s = np.clip(rng.normal(0.3 + 0.35 * y + 0.2 * (clusters == 0), 0.15), 0, 1)
    return y, s, clusters


def test_global_policy_matches_reference_threshold(data):
    y, s, c = data
    pol = ThresholdPolicy("global").fit(y, s, c)
    assert pol.global_threshold == pytest.approx(best_f1_threshold(y, s))
    assert pol.cluster_thresholds == {}
    np.testing.assert_array_equal(pol.predict(s, c), (s >= pol.global_threshold).astype(int))


def test_cluster_policy_learns_shifted_threshold(data):
    y, s, c = data
    pol = ThresholdPolicy("cluster", min_pairs=500, min_positives=50).fit(y, s, c)
    assert set(pol.cluster_thresholds) == {0, 1, 2, 3}
    assert pol.cluster_thresholds[0] > max(pol.cluster_thresholds[k] for k in (1, 2, 3))


def test_small_clusters_fall_back_to_global(data):
    y, s, c = data
    pol = ThresholdPolicy("cluster", min_pairs=10**6).fit(y, s, c)   # nobody qualifies
    assert pol.cluster_thresholds == {} and sorted(pol.fallback_clusters) == [0, 1, 2, 3]
    np.testing.assert_array_equal(pol.predict(s, c), ThresholdPolicy("global").fit(y, s, c).predict(s, c))
    # a cluster absent at fit time also gets the global threshold
    assert pol.thresholds_for(np.array([7]))[0] == pol.global_threshold


def test_min_positives_is_enforced(data):
    y, s, c = data
    y2 = y.copy()
    y2[c == 3] = 0                                                       # cluster 3 has no duplicates
    pol = ThresholdPolicy("cluster", min_pairs=100, min_positives=1).fit(y2, s, c)
    assert 3 in pol.fallback_clusters


def test_policy_roundtrip(tmp_path, data):
    y, s, c = data
    pol = ThresholdPolicy("cluster", min_pairs=500, min_positives=50).fit(y, s, c)
    pol.save(tmp_path / "p.json")
    loaded = ThresholdPolicy.load(tmp_path / "p.json")
    np.testing.assert_array_equal(loaded.predict(s, c), pol.predict(s, c))


def test_group_folds_keep_groups_together_and_are_deterministic():
    groups = np.repeat(np.arange(300), np.random.default_rng(1).integers(1, 6, 300))
    f1, f2 = group_folds(groups, 5, seed=3), group_folds(groups, 5, seed=3)
    np.testing.assert_array_equal(f1, f2)
    for g in np.unique(groups):
        assert len(set(f1[groups == g])) == 1                           # no group split across folds
    sizes = np.bincount(f1)
    assert sizes.max() - sizes.min() <= 5                                # greedy balancing


def test_cross_fitting_uses_only_other_folds(data):
    """Thresholds applied to a fold must not depend on that fold's labels."""
    y, s, c = data
    folds = np.arange(len(y)) % 5
    p1, _ = cross_fitted_predictions(y, s, c, folds, "cluster", 500, 50)
    y_flip = y.copy()
    y_flip[folds == 0] = 1 - y_flip[folds == 0]                          # change labels of fold 0 only
    p2, _ = cross_fitted_predictions(y_flip, s, c, folds, "cluster", 500, 50)
    np.testing.assert_array_equal(p1[folds == 0], p2[folds == 0])


def test_decision_metrics_match_definitions():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([1, 0, 1, 0, 1, 0])
    c = np.array([0, 0, 0, 1, 1, 1])
    m = decision_metrics(y, p, c, k=2)
    assert m["f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    np.testing.assert_allclose(m["per_cluster_f1"], [2 * 1 / (2 + 1 + 1), 1.0])
    assert m["worst_cluster_f1"] == pytest.approx(0.5) and m["worst_cluster"] == 0


def test_phase7_script_never_references_test_split():
    src = (PROJECT_ROOT / "scripts" / "calibration_experiments.py").read_text(encoding="utf-8")
    assert not re.search(r"test\.csv|[\"']test[\"']", src)


def test_frozen_phase8_policies():
    """Phase 7 freezes the thresholds used in Phase 8; both must be global (cluster thresholds rejected)."""
    import json
    art = PROJECT_ROOT / "artifacts" / "phase7"
    if not (art / "frozen_policy_final.json").exists():
        pytest.skip("Phase 7 experiments not run")
    res = json.loads((art / "results.json").read_text())
    final = ThresholdPolicy.load(art / "frozen_policy_final.json")
    base = ThresholdPolicy.load(art / "frozen_policy_baseline.json")
    assert final.mode == res["final_threshold_mode"]
    assert base.mode == "global" and base.global_threshold == pytest.approx(0.32)   # Phase 3 threshold reproduced
    if final.mode == "global":
        assert final.cluster_thresholds == {} and final.global_threshold == pytest.approx(0.38)  # Phase 6 threshold
