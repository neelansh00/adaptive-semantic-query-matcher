import json

import numpy as np
import pytest
import yaml
from sklearn.cluster import KMeans

from src.clustering.core import (CentroidModel, ctfidf_terms, fit_kmeans, internal_metrics, size_stats,
                                 stability)
from src.utils.data import PROJECT_ROOT


@pytest.fixture(scope="module")
def blobs():
    """Three well-separated groups of unit vectors."""
    rng = np.random.default_rng(0)
    centers = np.eye(8)[:3]
    X = np.vstack([c + 0.05 * rng.normal(size=(300, 8)) for c in centers])
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def test_centroid_model_matches_sklearn_predict(blobs):
    km = KMeans(3, n_init=3, random_state=0).fit(blobs)
    model = CentroidModel(km.cluster_centers_)
    assert (model.predict(blobs) == km.predict(blobs)).all()
    rng = np.random.default_rng(1)
    new = rng.normal(size=(50, 8)).astype(np.float32)
    assert (model.predict(new) == km.predict(new)).all()  # same rule on unseen points


def test_fit_is_deterministic_and_save_load_roundtrip(blobs, tmp_path):
    m1, l1, _ = fit_kmeans(blobs, 3, seed=7, n_init=2)
    m2, l2, _ = fit_kmeans(blobs, 3, seed=7, n_init=2)
    assert (l1 == l2).all()
    m1.save(tmp_path, {"note": "test"})
    loaded = CentroidModel.load(tmp_path)
    assert (loaded.predict(blobs) == l1).all()
    assert json.loads((tmp_path / "cluster_model.json").read_text())["k"] == 3


def test_metrics_on_clear_structure(blobs):
    model, labels, _ = fit_kmeans(blobs, 3, seed=0, n_init=2)
    m = internal_metrics(blobs, labels, model, silhouette_sample=500, seed=0)
    assert m["silhouette_sample"] > 0.8 and m["davies_bouldin"] < 0.3
    assert stability(blobs, 3, seed=0, eval_size=200, n_init=2)["split_half_ari"] > 0.99


def test_size_stats():
    s = size_stats(np.array([0, 0, 1, 1, 2, 2]), 3)
    assert s["size_entropy_normalised"] == pytest.approx(1.0)
    assert s["max_to_min_ratio"] == pytest.approx(1.0)
    assert size_stats(np.array([0, 0, 0, 0, 1, 2]), 3)["size_entropy_normalised"] < 1.0


def test_ctfidf_finds_distinctive_terms():
    texts = ["how to cook pasta"] * 30 + ["how to cook rice"] * 30 + ["python list sort"] * 30 + ["python dict keys"] * 30
    labels = np.array([0] * 60 + [1] * 60)
    terms = ctfidf_terms(texts, labels, 2, top_n=3, min_df=1)
    assert "cook" in terms[0] and "python" in terms[1]
    assert "python" not in terms[0]


def test_frozen_cluster_model_matches_config():
    model_dir = PROJECT_ROOT / "artifacts" / "phase4" / "cluster_model"
    if not (model_dir / "centroids.npy").exists():
        pytest.skip("cluster model not built yet (scripts/build_clusters.py)")
    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "clustering.yaml").read_text())
    model = CentroidModel.load(model_dir)
    assert model.k == cfg["chosen_k"]
    assert model.centroids.shape[1] == 384


def test_frozen_centroids_hash_and_deterministic_assignment():
    import hashlib
    model_dir = PROJECT_ROOT / "artifacts" / "phase4" / "cluster_model"
    if not (model_dir / "centroids.npy").exists():
        pytest.skip("cluster model not built yet (scripts/build_clusters.py)")
    meta = json.loads((model_dir / "cluster_model.json").read_text())
    model = CentroidModel.load(model_dir)
    assert hashlib.sha256(model.centroids.tobytes()).hexdigest() == meta["centroids_sha256"]
    X = np.random.default_rng(0).normal(size=(500, 384)).astype(np.float32)
    assert (model.predict(X) == CentroidModel.load(model_dir).predict(X)).all()
