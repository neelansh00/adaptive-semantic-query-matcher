import numpy as np
import pytest

from src.evaluation.metrics import (best_f1_threshold, classification_metrics,
                                    expected_calibration_error, group_summary, per_group_metrics)

Y = np.array([1, 1, 0, 0, 1, 0])
S = np.array([0.9, 0.6, 0.55, 0.2, 0.4, 0.1])


def test_classification_metrics_counts():
    m = classification_metrics(Y, S, threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 2, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(8 / 9)


def test_best_threshold_beats_default():
    t = best_f1_threshold(Y, S)
    assert classification_metrics(Y, S, t)["f1"] >= classification_metrics(Y, S, 0.5)["f1"]
    # every threshold in (0.2, 0.4] yields F1 = 6/7; ties resolve to the lowest one
    assert 0.2 < t <= 0.4
    assert classification_metrics(Y, S, t)["f1"] == pytest.approx(6 / 7)


def test_ece_measures_confidence_gap():
    assert expected_calibration_error([0, 1, 0, 1], [0.05, 0.95, 0.05, 0.95]) == pytest.approx(0.05)
    assert expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 0.99, 0.99]) == pytest.approx(0.005)


def test_per_group_uses_global_fallback():
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    pg = per_group_metrics(Y, S, groups, threshold={"a": 0.95}, fallback_threshold=0.5)
    assert pg.loc["a", "threshold"] == 0.95
    assert pg.loc["b", "threshold"] == 0.5
    summary = group_summary(pg)
    assert summary["worst_group_f1"] <= summary["macro_group_f1"]
