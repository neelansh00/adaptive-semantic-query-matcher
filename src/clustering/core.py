"""Unsupervised query clustering on sentence embeddings (Phase 4).

Embeddings are L2-normalised, so Euclidean K-Means is equivalent to clustering by cosine geometry.
Clusters are discovered from TRAIN questions only; nothing here looks at duplicate labels.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from src.preprocessing.text import preprocess


# --------------------------------------------------------------------------- model

class CentroidModel:
    """Frozen cluster model = centroids only. Assignment = nearest centroid (same rule as KMeans.predict)."""

    def __init__(self, centroids: np.ndarray):
        self.centroids = np.asarray(centroids, dtype=np.float32)
        self._half_sq_norm = 0.5 * np.sum(self.centroids ** 2, axis=1)

    @property
    def k(self) -> int:
        return len(self.centroids)

    def scores(self, X: np.ndarray) -> np.ndarray:
        # argmin ||x - c||^2  ==  argmax (x.c - ||c||^2 / 2)
        return np.asarray(X, dtype=np.float32) @ self.centroids.T - self._half_sq_norm

    def predict(self, X: np.ndarray, batch: int = 65536) -> np.ndarray:
        return np.concatenate([self.scores(X[i:i + batch]).argmax(axis=1) for i in range(0, len(X), batch)])

    def cosine_to_centroid(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unit = self.centroids / np.linalg.norm(self.centroids, axis=1, keepdims=True)
        return np.einsum("ij,ij->i", np.asarray(X, np.float32), unit[labels])

    def save(self, directory: Path, meta: dict) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "centroids.npy", self.centroids)
        (directory / "cluster_model.json").write_text(json.dumps({"k": self.k, **meta}, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "CentroidModel":
        return cls(np.load(Path(directory) / "centroids.npy"))


def fit_kmeans(X: np.ndarray, k: int, seed: int, n_init: int = 5) -> tuple[CentroidModel, np.ndarray, float]:
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
    return CentroidModel(km.cluster_centers_), km.labels_, float(km.inertia_)


# --------------------------------------------------------------------------- evaluation

def size_stats(labels: np.ndarray, k: int) -> dict:
    counts = np.bincount(labels, minlength=k)
    share = counts / counts.sum()
    nz = share[share > 0]
    return {"min_share": float(share.min()), "max_share": float(share.max()),
            "max_to_min_ratio": float(share.max() / max(share.min(), 1e-12)),
            "size_entropy_normalised": float(-(nz * np.log(nz)).sum() / np.log(k)),  # 1.0 = perfectly balanced
            "sizes": counts.tolist()}


def internal_metrics(X: np.ndarray, labels: np.ndarray, model: CentroidModel, silhouette_sample: int,
                     seed: int) -> dict:
    cos = model.cosine_to_centroid(X, labels)
    per_cluster = np.array([cos[labels == c].mean() for c in range(model.k)])
    return {
        # Silhouette is O(n^2): computed on a fixed random sample.
        "silhouette_sample": float(silhouette_score(X, labels, sample_size=silhouette_sample, random_state=seed)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "coherence_mean_cos_to_centroid": float(cos.mean()),
        "coherence_worst_cluster": float(per_cluster.min()),
    }


def stability(X: np.ndarray, k: int, seed: int, eval_size: int = 50_000, n_init: int = 3,
              include_seed: bool = True) -> dict:
    """(a) Split-half: fit on two disjoint random halves, label a common held-out sample with both,
    compare with ARI. (b) Seed: two full fits with different seeds, ARI of their labels.
    ARI = 1 means identical partitions (up to renaming); ~0 means chance agreement."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    ev, rest = idx[:eval_size], idx[eval_size:]
    half_a, half_b = rest[: len(rest) // 2], rest[len(rest) // 2:]
    ma, _, _ = fit_kmeans(X[half_a], k, seed, n_init)
    mb, _, _ = fit_kmeans(X[half_b], k, seed + 1, n_init)
    split_half = adjusted_rand_score(ma.predict(X[ev]), mb.predict(X[ev]))
    if not include_seed:
        return {"split_half_ari": float(split_half)}
    s1, l1, _ = fit_kmeans(X, k, seed + 2, n_init)
    s2, l2, _ = fit_kmeans(X, k, seed + 3, n_init)
    return {"split_half_ari": float(split_half), "seed_ari": float(adjusted_rand_score(l1, l2))}


# --------------------------------------------------------------------------- description (after the fact)

_STOP = frozenset(ENGLISH_STOP_WORDS) | {"best", "good", "way", "ways", "does", "did", "use", "make", "know",
                                         "people", "like", "think", "really", "possible", "difference"}
_WORD = re.compile(r"^[a-z][a-z0-9+#.'-]*$")


def ctfidf_terms(texts: list[str], labels: np.ndarray, k: int, top_n: int = 12, min_df: int = 20) -> dict:
    """Class-based TF-IDF: terms frequent in a cluster but rare across clusters.
    weight(t, c) = tf(t, c) / |c| * log(1 + mean_class_size / total_freq(t))"""
    per_class = [Counter() for _ in range(k)]
    for text, c in zip(texts, labels):
        per_class[c].update(t for t in set(preprocess(text)) if t not in _STOP and _WORD.match(t))
    total = Counter()
    for cnt in per_class:
        total.update(cnt)
    sizes = np.bincount(labels, minlength=k)
    avg = sizes.mean()
    out = {}
    for c in range(k):
        scored = [(t, (f / sizes[c]) * np.log(1 + avg / total[t])) for t, f in per_class[c].items() if total[t] >= min_df]
        out[c] = [t for t, _ in sorted(scored, key=lambda x: -x[1])[:top_n]]
    return out


def representatives(texts: list[str], X: np.ndarray, labels: np.ndarray, model: CentroidModel,
                    n_central: int = 10, n_random: int = 8, seed: int = 42) -> dict:
    cos = model.cosine_to_centroid(X, labels)
    rng = np.random.default_rng(seed)
    out = {}
    for c in range(model.k):
        members = np.flatnonzero(labels == c)
        central = members[np.argsort(-cos[members])[:n_central]]
        rand = rng.choice(members, min(n_random, len(members)), replace=False)
        out[c] = {"nearest_centroid": [texts[i] for i in central], "random_members": [texts[i] for i in rand]}
    return out
