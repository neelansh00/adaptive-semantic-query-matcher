"""Assigning a question PAIR to one of the frozen Phase 4 clusters.

Phase 4 clusters are soft regions of a continuous embedding space, so a pair's cluster is a
coordinate in that space, not a discrete "query type". Rules:
  A  q1        - cluster of question 1 (asymmetric: depends on pair order)
  B  pair_avg  - normalise(u + v) assigned to the nearest centroid (symmetric; the pair's midpoint)
  C  same_only - cluster if both questions share it, else UNASSIGNED (diagnostic only)
"""
from __future__ import annotations

import numpy as np

from src.clustering.core import CentroidModel

UNASSIGNED = -1
RULES = ("q1", "pair_avg", "same_only")


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def pair_embedding(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Re-normalised mean of two L2-normalised question embeddings (symmetric in u, v)."""
    return _unit(_unit(u) + _unit(v))


def assign_pairs(u: np.ndarray, v: np.ndarray, model: CentroidModel, rule: str) -> np.ndarray:
    if rule == "q1":
        return model.predict(u)
    if rule == "pair_avg":
        return model.predict(pair_embedding(u, v))
    if rule == "same_only":
        a, b = model.predict(u), model.predict(v)
        return np.where(a == b, a, UNASSIGNED)
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


def centroid_geometry(X: np.ndarray, model: CentroidModel) -> dict:
    """Assignment confidence in the SAME geometry used for assignment (nearest raw centroid).

    margin = score(assigned) - score(runner-up), where score = x.c - |c|^2/2 is the nearest-centroid
    criterion; always >= 0. Euclidean assignment and cosine-to-unit-centroid disagree for ~6% of
    questions (centroid norms differ, 0.26-0.46), so a cosine margin would be inconsistent with the labels.
    cos_assigned is the plain cosine to the assigned centroid, reported for intuition only."""
    X = _unit(X)
    scores = model.scores(X)
    order = np.argsort(scores, axis=1)
    assigned, second = order[:, -1], order[:, -2]
    rows = np.arange(len(X))
    unit_c = _unit(model.centroids)
    return {"cluster": assigned,
            "margin": scores[rows, assigned] - scores[rows, second],
            "second_cluster": second,
            "cos_assigned": np.einsum("ij,ij->i", X, unit_c[assigned])}
