"""Global vs cluster-specific decision thresholds (Phase 7).

A ThresholdPolicy is fitted on (y, score, cluster) of the data it is given and nothing else:
  - global threshold   : F1-optimal threshold over all fitting pairs (0.01 grid, ties -> lowest)
  - cluster thresholds : F1-optimal threshold within a cluster, ONLY if the cluster has at least
                         `min_pairs` fitting pairs and `min_positives` duplicates; otherwise the cluster
                         falls back to the global threshold.
No threshold is ever set by hand.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.evaluation.cluster_analysis import optimal_threshold

MIN_PAIRS = 1000      # Phase 5: below ~1,000 pairs threshold std > ~0.05 and noise rivals the between-cluster spread
MIN_POSITIVES = 100   # an F1-optimal threshold is meaningless without enough duplicates


@dataclass
class ThresholdPolicy:
    mode: str = "global"                    # "global" or "cluster"
    min_pairs: int = MIN_PAIRS
    min_positives: int = MIN_POSITIVES
    global_threshold: float | None = None
    cluster_thresholds: dict = field(default_factory=dict)   # {cluster: threshold} for eligible clusters only
    fallback_clusters: list = field(default_factory=list)

    def fit(self, y, score, clusters) -> "ThresholdPolicy":
        y, score, clusters = np.asarray(y), np.asarray(score, float), np.asarray(clusters)
        self.global_threshold = optimal_threshold(y, score)[0]
        self.cluster_thresholds, self.fallback_clusters = {}, []
        if self.mode == "cluster":
            for c in np.unique(clusters):
                m = clusters == c
                if m.sum() >= self.min_pairs and y[m].sum() >= self.min_positives:
                    self.cluster_thresholds[int(c)] = optimal_threshold(y[m], score[m])[0]
                else:
                    self.fallback_clusters.append(int(c))
        return self

    def thresholds_for(self, clusters) -> np.ndarray:
        clusters = np.asarray(clusters)
        t = np.full(len(clusters), self.global_threshold, dtype=float)
        for c, thr in self.cluster_thresholds.items():
            t[clusters == c] = thr
        return t

    def predict(self, score, clusters) -> np.ndarray:
        return (np.asarray(score, float) >= self.thresholds_for(clusters)).astype(int)

    def save(self, path: Path, extra: dict | None = None) -> None:
        d = asdict(self)
        d["cluster_thresholds"] = {str(k): v for k, v in self.cluster_thresholds.items()}
        Path(path).write_text(json.dumps({**d, **(extra or {})}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ThresholdPolicy":
        d = json.loads(Path(path).read_text())
        p = cls(mode=d["mode"], min_pairs=d["min_pairs"], min_positives=d["min_positives"],
                global_threshold=d["global_threshold"],
                cluster_thresholds={int(k): v for k, v in d["cluster_thresholds"].items()},
                fallback_clusters=d["fallback_clusters"])
        return p


def group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign whole groups (question-graph components) to folds at random, balancing fold sizes greedily."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]  # big components first, random among equals
    load = np.zeros(n_folds)
    fold_of_group = {}
    for i in order:
        f = int(np.argmin(load + rng.random(n_folds) * 1e-6))
        fold_of_group[uniq[i]] = f
        load[f] += counts[i]
    return np.array([fold_of_group[g] for g in groups])


def cross_fitted_predictions(y, score, clusters, folds, mode: str, min_pairs: int = MIN_PAIRS,
                             min_positives: int = MIN_POSITIVES) -> tuple[np.ndarray, list]:
    """Out-of-fold decisions: thresholds fitted on the other folds, applied to the held-out fold."""
    y, score, clusters = np.asarray(y), np.asarray(score, float), np.asarray(clusters)
    pred = np.zeros(len(y), dtype=int)
    policies = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        pol = ThresholdPolicy(mode, min_pairs, min_positives).fit(y[tr], score[tr], clusters[tr])
        pred[te] = pol.predict(score[te], clusters[te])
        policies.append(pol)
    return pred, policies


def decision_metrics(y: np.ndarray, pred: np.ndarray, clusters: np.ndarray, k: int = 12) -> dict:
    """Overall and per-cluster metrics from hard decisions (fast; used inside bootstraps)."""
    y, pred, clusters = np.asarray(y), np.asarray(pred), np.asarray(clusters)
    tp = np.bincount(clusters, weights=(pred == 1) & (y == 1), minlength=k)
    fp = np.bincount(clusters, weights=(pred == 1) & (y == 0), minlength=k)
    fn = np.bincount(clusters, weights=(pred == 0) & (y == 1), minlength=k)
    tn = np.bincount(clusters, weights=(pred == 0) & (y == 0), minlength=k)
    f1_c = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    TP, FP, FN, TN = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    return {"f1": 2 * TP / max(2 * TP + FP + FN, 1), "precision": TP / max(TP + FP, 1), "recall": TP / max(TP + FN, 1),
            "fpr": FP / max(FP + TN, 1), "fnr": FN / max(FN + TP, 1),
            "macro_cluster_f1": float(f1_c.mean()), "worst_cluster_f1": float(f1_c.min()),
            "worst_cluster": int(f1_c.argmin()), "per_cluster_f1": f1_c}
