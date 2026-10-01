"""Per-cluster evaluation, bootstrap uncertainty and a random-partition null (Phase 5).

The null answers "how much cluster-to-cluster spread would we see if the clusters were random groups
of the same sizes?". Spread beyond the null is evidence that the semantic regions genuinely differ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.metrics import classification_metrics

THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)  # same grid as best_f1_threshold


def f1_curve(y: np.ndarray, s: np.ndarray, thresholds: np.ndarray = THRESHOLDS) -> np.ndarray:
    """F1 of the rule (s >= t) for every t, vectorised (identical to metrics.threshold_sweep)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s)
    s_sorted, y_sorted = s[order], y[order]
    pos_from = np.concatenate([np.cumsum(y_sorted[::-1])[::-1], [0]])  # positives at index >= i
    first = np.searchsorted(s_sorted, thresholds, side="left")         # first index with s >= t
    pred_pos = len(s) - first
    tp = pos_from[first]
    total_pos = y.sum()
    denom = pred_pos + total_pos
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def optimal_threshold(y, s, thresholds: np.ndarray = THRESHOLDS) -> tuple[float, float]:
    """(threshold, F1) maximising F1; ties resolve to the lowest threshold (as best_f1_threshold)."""
    curve = f1_curve(y, s, thresholds)
    i = int(np.argmax(curve))
    return float(thresholds[i]), float(curve[i])


def cluster_table(y, s, groups, global_threshold: float, plateau_tol: float = 0.01) -> pd.DataFrame:
    y, s, groups = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    rows = []
    for g in np.unique(groups):
        m = groups == g
        yg, sg = y[m], s[m]
        met = classification_metrics(yg, sg, global_threshold)
        t_opt, f1_opt = optimal_threshold(yg, sg)
        curve = f1_curve(yg, sg)
        plateau = THRESHOLDS[curve >= curve.max() - plateau_tol]
        rows.append({
            "cluster": g, "n": int(m.sum()), "prevalence": float(yg.mean()),
            "precision": met["precision"], "recall": met["recall"], "f1": met["f1"],
            "fpr": met["fpr"], "fnr": met["fnr"], "roc_auc": met["roc_auc"], "pr_auc": met["pr_auc"],
            "pr_auc_lift": met["pr_auc"] / yg.mean(),  # PR-AUC relative to the chance level (= prevalence)
            "mean_score": float(sg.mean()), "mean_score_pos": float(sg[yg == 1].mean()),
            "mean_score_neg": float(sg[yg == 0].mean()),
            "opt_threshold": t_opt,
            "f1_at_opt_in_sample": f1_opt,  # optimistic upper bound, NOT an evaluation result
            "f1_plateau_low": float(plateau.min()), "f1_plateau_high": float(plateau.max()),
            "tp": met["tp"], "fp": met["fp"], "tn": met["tn"], "fn": met["fn"],
        })
    return pd.DataFrame(rows).set_index("cluster")


def robustness_summary(table: pd.DataFrame, overall_f1: float, global_threshold: float) -> dict:
    f1, thr = table["f1"], table["opt_threshold"]
    return {"overall_f1": float(overall_f1), "macro_cluster_f1": float(f1.mean()),
            "worst_cluster_f1": float(f1.min()), "worst_cluster": int(f1.idxmin()),
            "best_cluster_f1": float(f1.max()), "best_cluster": int(f1.idxmax()),
            "f1_range": float(f1.max() - f1.min()), "f1_std": float(f1.std(ddof=0)),
            "global_threshold": global_threshold,
            "opt_threshold_min": float(thr.min()), "opt_threshold_max": float(thr.max()),
            "opt_threshold_range": float(thr.max() - thr.min()), "opt_threshold_std": float(thr.std(ddof=0)),
            "opt_threshold_median": float(thr.median())}


def bootstrap_clusters(y, s, groups, global_threshold: float, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Per-cluster 95% intervals (resampling pairs within each cluster) for F1 at the global threshold,
    the cluster-optimal threshold and PR-AUC."""
    y, s, groups = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    rng = np.random.default_rng(seed)
    gi = int(np.argmin(np.abs(THRESHOLDS - global_threshold)))
    rows = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        f1g, topt, prauc = [], [], []
        for _ in range(n_boot):
            b = rng.choice(idx, len(idx), replace=True)
            curve = f1_curve(y[b], s[b])
            f1g.append(curve[gi])
            topt.append(THRESHOLDS[int(np.argmax(curve))])
            prauc.append(average_precision_score(y[b], s[b]))
        q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
        rows.append({"cluster": g, "f1_ci_low": q(f1g)[0], "f1_ci_high": q(f1g)[1],
                     "opt_threshold_ci_low": q(topt)[0], "opt_threshold_ci_high": q(topt)[1],
                     "opt_threshold_boot_std": float(np.std(topt)),
                     "pr_auc_ci_low": q(prauc)[0], "pr_auc_ci_high": q(prauc)[1]})
    return pd.DataFrame(rows).set_index("cluster")


def random_partition_null(y, s, groups, global_threshold: float, n_perm: int = 500, seed: int = 42) -> dict:
    """Spread of per-group F1 / optimal threshold when the same group SIZES are filled at random."""
    y, s, groups = np.asarray(y), np.asarray(s, float), np.asarray(groups)
    rng = np.random.default_rng(seed)
    gi = int(np.argmin(np.abs(THRESHOLDS - global_threshold)))
    f1_range, f1_std, thr_range, thr_std, roc_range = [], [], [], [], []
    for _ in range(n_perm):
        perm = rng.permutation(groups)
        f1s, thrs, rocs = [], [], []
        for g in np.unique(groups):
            m = perm == g
            curve = f1_curve(y[m], s[m])
            f1s.append(curve[gi]); thrs.append(THRESHOLDS[int(np.argmax(curve))])
            rocs.append(roc_auc_score(y[m], s[m]))
        f1_range.append(np.ptp(f1s)); f1_std.append(np.std(f1s))
        thr_range.append(np.ptp(thrs)); thr_std.append(np.std(thrs)); roc_range.append(np.ptp(rocs))
    pct = lambda a: {"mean": float(np.mean(a)), "p95": float(np.percentile(a, 95)), "max": float(np.max(a))}  # noqa: E731
    return {"n_perm": n_perm, "f1_range": pct(f1_range), "f1_std": pct(f1_std),
            "opt_threshold_range": pct(thr_range), "opt_threshold_std": pct(thr_std), "roc_auc_range": pct(roc_range),
            "_raw": {"f1_std": f1_std, "opt_threshold_std": thr_std, "f1_range": f1_range,
                     "opt_threshold_range": thr_range, "roc_auc_range": roc_range}}
