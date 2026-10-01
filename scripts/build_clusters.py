"""Phase 4b: fit the final K-Means model (K from configs/clustering.yaml), assign questions, describe clusters.

Interpretation happens only AFTER clusters are formed: distinctive terms (class-based TF-IDF) and
questions nearest each centroid are written out for manual inspection. No duplicate labels are used.

Usage:  python scripts/build_clusters.py
Writes: artifacts/phase4/cluster_model/{centroids.npy, cluster_model.json}
        artifacts/phase4/cluster_descriptions.json
        data/processed/clusters/{train,val}_question_clusters.csv
        docs/figures/phase4_pca_clusters.png
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_clustering import load_cfg  # noqa: E402
from src.clustering.core import (CentroidModel, ctfidf_terms, fit_kmeans, internal_metrics, representatives,  # noqa: E402
                                 size_stats)
from src.models.sentence_encoder import EmbeddingCache  # noqa: E402
from src.utils.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "phase4"
OUT_DATA = PROJECT_ROOT / "data" / "processed" / "clusters"
FIG = PROJECT_ROOT / "docs" / "figures"
INK, INK2, GRID, SURFACE, BLUE, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#2a78d6", "#d9d8d2"


def plot_pca(X: np.ndarray, labels: np.ndarray, k: int, names: dict, seed: int) -> None:
    """Visualisation only: 2-D PCA of a sample, one small multiple per cluster (highlight vs. grey rest)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), 15000, replace=False)
    pca = PCA(2, random_state=seed).fit(X[idx])
    Z, lab = pca.transform(X[idx]), labels[idx]
    cols = 4
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.9 * rows), facecolor=SURFACE, sharex=True, sharey=True)
    for c, ax in enumerate(axes.ravel()):
        if c >= k:
            ax.axis("off")
            continue
        ax.scatter(Z[lab != c, 0], Z[lab != c, 1], s=1, color=MUTED, rasterized=True)
        ax.scatter(Z[lab == c, 0], Z[lab == c, 1], s=1.5, color=BLUE, rasterized=True)
        title = f"C{c}: {names.get(c, '')}"
        if len(title) > 40:
            title = title[:40].rsplit(" ", 1)[0].rstrip(",&/") + "…"
        ax.set_title(title, loc="left", color=INK, fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(SURFACE)
        for s in ax.spines.values():
            s.set_color(GRID)
    ev = pca.explained_variance_ratio_
    fig.suptitle(f"PCA of 15k train questions (PC1+PC2 explain {ev.sum():.1%} of variance). "
                 "Visualisation only, not evidence of cluster quality.", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "phase4_pca_clusters.png", dpi=130); plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true", help="re-fit K-Means and overwrite the frozen centroids")
    args = ap.parse_args()
    cfg = load_cfg()
    k = cfg["chosen_k"]
    if not k:
        raise SystemExit("Set chosen_k in configs/clustering.yaml after running compare_clustering.py")
    train_cache, val_cache = EmbeddingCache(cfg["encoder"], "train"), EmbeddingCache(cfg["encoder"], "val")
    X, tr_index = train_cache.load()
    tr_texts = [None] * len(tr_index)
    for t, i in tr_index.items():
        tr_texts[i] = t

    model_dir = ART / "cluster_model"
    if (model_dir / "centroids.npy").exists() and not args.refit:
        # K-Means refits are not bit-reproducible (multithreaded float32 reductions shift a few
        # boundary questions; refit ARI vs frozen ~0.9997), so the frozen centroids are the source of truth.
        model = CentroidModel.load(model_dir)
        if model.k != k:
            raise SystemExit(f"frozen model has K={model.k} but config says {k}; use --refit deliberately")
        labels = model.predict(X)
        print("[freeze] using frozen centroids", flush=True)
    else:
        model, labels, inertia = fit_kmeans(X, k, cfg["seed"], cfg["n_init_final"])
        # Canonical cluster ids: order by size (largest = C0) so ids are stable and readable.
        order = np.argsort(-np.bincount(labels, minlength=k), kind="stable")
        remap = np.empty(k, dtype=int)
        remap[order] = np.arange(k)
        model = CentroidModel(model.centroids[order])
        labels = remap[labels]
        assert (model.predict(X) == labels).all(), "nearest-centroid rule must reproduce the K-Means labels"
        model.save(model_dir, {"encoder": cfg["encoder"], "seed": cfg["seed"], "n_init": cfg["n_init_final"],
                               "fitted_on": "train unique questions", "n_train_questions": len(X), "inertia": inertia,
                               "assignment": "nearest centroid (Euclidean on L2-normalised embeddings)",
                               "centroids_sha256": hashlib.sha256(model.centroids.tobytes()).hexdigest()})

    Xv, va_index = val_cache.load()
    va_texts = [None] * len(va_index)
    for t, i in va_index.items():
        va_texts[i] = t
    vlabels = model.predict(Xv)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"question": tr_texts, "cluster": labels}).to_csv(OUT_DATA / "train_question_clusters.csv", index=False)
    pd.DataFrame({"question": va_texts, "cluster": vlabels}).to_csv(OUT_DATA / "val_question_clusters.csv", index=False)

    cos = model.cosine_to_centroid(X, labels)
    # Boundary softness: cosine gap between the nearest and second-nearest centroid.
    unit = model.centroids / np.linalg.norm(model.centroids, axis=1, keepdims=True)
    top2 = np.sort(X @ unit.T, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    terms = ctfidf_terms(tr_texts, labels, k)
    reps = representatives(tr_texts, X, labels, model)
    sz = size_stats(labels, k)
    vsz = np.bincount(vlabels, minlength=k)
    desc = {"k": k, "metrics": internal_metrics(X, labels, model, cfg["silhouette_sample"], cfg["seed"]),
            "size_stats": {kk: v for kk, v in sz.items() if kk != "sizes"},
            "assignment_margin": {"median": float(np.median(margin)),
                                  "share_below_0.02": float((margin < 0.02).mean()),
                                  "share_below_0.05": float((margin < 0.05).mean())},
            "clusters": {}}
    for c in range(k):
        desc["clusters"][c] = {"train_questions": int(sz["sizes"][c]), "train_share": sz["sizes"][c] / len(X),
                               "val_questions": int(vsz[c]), "val_share": float(vsz[c] / len(Xv)),
                               "coherence_mean_cos": float(cos[labels == c].mean()),
                               "top_terms": terms[c], **reps[c]}
    (ART / "cluster_descriptions.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")

    names_path = ART / "cluster_names.json"
    names = ({int(c): v["name"] for c, v in json.loads(names_path.read_text()).items() if not c.startswith("_")}
             if names_path.exists() else {})
    plot_pca(X, labels, k, names, cfg["seed"])
    for c in range(k):
        d = desc["clusters"][c]
        print(f"C{c:<2} {d['train_share']:.3f} coh={d['coherence_mean_cos']:.3f} {', '.join(d['top_terms'][:10])}")


if __name__ == "__main__":
    main()
