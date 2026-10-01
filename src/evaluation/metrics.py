"""Evaluation metrics shared by every phase (see docs/evaluation_plan.md).

All functions take y_true in {0,1} and a continuous score (probability or similarity).
Thresholds are always chosen on validation data by the caller, never on test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def classification_metrics(y_true, score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    y_pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": float((y_pred == y_true).mean()),
        "roc_auc": roc_auc_score(y_true, score) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y_true, score) if both_classes else float("nan"),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def threshold_sweep(y_true, score, thresholds=None) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    pos = max(int(y_true.sum()), 1)
    for t in thresholds:
        pred = score >= t
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / pos
        rows.append({"threshold": float(t), "precision": p, "recall": r,
                     "f1": 2 * p * r / (p + r) if p + r else 0.0})
    return pd.DataFrame(rows)


def best_f1_threshold(y_true, score, thresholds=None) -> float:
    """Validation-optimal threshold (ties -> lowest threshold, i.e. first max)."""
    sweep = threshold_sweep(y_true, score, thresholds)
    return float(sweep.loc[sweep["f1"].idxmax(), "threshold"])


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(prob[m].mean() - y_true[m].mean())
    return float(ece)


def calibration_metrics(y_true, prob, n_bins: int = 10) -> dict:
    return {"brier": brier_score_loss(y_true, prob),
            "ece": expected_calibration_error(y_true, prob, n_bins)}


def per_group_metrics(y_true, score, groups, threshold=0.5, fallback_threshold: float = 0.5,
                      min_size: int = 1) -> pd.DataFrame:
    """Per-cluster metrics. `threshold` may be a float or a {group: threshold} mapping;
    groups missing from the mapping use `fallback_threshold` (the global threshold)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(score, float), "g": np.asarray(groups)})
    rows = []
    for g, part in df.groupby("g"):
        if len(part) < min_size:
            continue
        t = threshold.get(g, fallback_threshold) if isinstance(threshold, dict) else threshold
        m = classification_metrics(part["y"], part["s"], t)
        rows.append({"group": g, "n": len(part), "prevalence": part["y"].mean(),
                     "mean_score": part["s"].mean(), **m})
    return pd.DataFrame(rows).set_index("group")


def group_summary(per_group: pd.DataFrame) -> dict:
    return {"macro_group_f1": float(per_group["f1"].mean()),
            "worst_group_f1": float(per_group["f1"].min()),
            "worst_group": per_group["f1"].idxmin()}


def threshold_transfer(y_true, score, seed: int = 42) -> dict:
    """Optimism check: tune the threshold on one random half of the data, evaluate on the other half."""
    y_true, score = np.asarray(y_true), np.asarray(score, dtype=float)
    idx = np.random.default_rng(seed).permutation(len(y_true))
    halves = idx[: len(idx) // 2], idx[len(idx) // 2:]
    f1s = []
    for tune, ev in (halves, halves[::-1]):
        t = best_f1_threshold(y_true[tune], score[tune])
        f1s.append(classification_metrics(y_true[ev], score[ev], t)["f1"])
    return {"cross_half_f1_mean": float(np.mean(f1s)), "cross_half_f1": [float(f) for f in f1s]}


def validation_report(y_true, score, is_probability: bool = True) -> dict:
    """Standard per-model report: metrics at 0.5 and at the validation-F1-optimal threshold."""
    thr = best_f1_threshold(y_true, score)
    report = {"tuned_threshold": thr,
              "val_at_tuned_threshold": classification_metrics(y_true, score, thr),
              "val_at_0.5": classification_metrics(y_true, score, 0.5),
              "threshold_transfer": threshold_transfer(y_true, score)}
    if is_probability:
        report["calibration"] = calibration_metrics(y_true, score)
    return report
