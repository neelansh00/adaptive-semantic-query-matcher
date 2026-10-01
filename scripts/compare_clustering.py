"""Phase 4a: compare K-Means over a K grid and HDBSCAN on TRAIN question embeddings.

No duplicate labels are used here. Validation questions are only *assigned* (unlabelled) to report how
many validation pairs each cluster would receive, a practical constraint for per-cluster thresholds later.

Usage:  python scripts/compare_clustering.py
Writes: artifacts/phase4/k_sweep.json, artifacts/phase4/hdbscan.json, docs/figures/phase4_k_sweep.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clustering.core import fit_kmeans, internal_metrics, size_stats, stability  # noqa: E402
from src.models.sentence_encoder import EmbeddingCache, clean  # noqa: E402
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "phase4"
FIG = PROJECT_ROOT / "docs" / "figures"
INK, INK2, GRID, SURFACE, BLUE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#2a78d6"


def load_cfg() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / "clustering.yaml").read_text())


def val_pair_counts(model, k: int) -> dict:
    """Unlabelled sizing check: validation pairs per cluster under the simplest association (question-1 cluster),
    plus how often both questions of a pair fall in the same cluster."""
    val = pd.read_csv(resolve(load_config()["split_dir"]) / "val.csv", usecols=["question1", "question2"],
                      keep_default_na=False, na_values=[""])
    matrix, index = EmbeddingCache(load_cfg()["encoder"], "val").load()
    la = model.predict(matrix[[index[clean(t)] for t in val.question1]])
    lb = model.predict(matrix[[index[clean(t)] for t in val.question2]])
    counts = np.bincount(la, minlength=k)
    return {"min_val_pairs_q1_cluster": int(counts.min()), "val_pairs_q1_cluster": counts.tolist(),
            "val_pairs_both_questions_same_cluster": float((la == lb).mean())}


def run_hdbscan(X: np.ndarray, cfg: dict) -> list[dict]:
    h = cfg["hdbscan"]
    rng = np.random.default_rng(cfg["seed"])
    sample = X[rng.choice(len(X), h["sample_size"], replace=False)]
    out, reduced = [], {}
    for dims, mcs, ms in h["settings"]:
        if dims not in reduced:
            reduced[dims] = PCA(dims, random_state=cfg["seed"]).fit_transform(sample)
        t = time.time()
        labels = HDBSCAN(min_cluster_size=mcs, min_samples=ms, copy=True).fit_predict(reduced[dims])
        clusters = labels[labels >= 0]
        sizes = np.bincount(clusters) if len(clusters) else np.array([0])
        res = {"pca_dims": dims, "min_cluster_size": mcs, "min_samples": ms, "sample_size": h["sample_size"],
               "n_clusters": int(len(np.unique(clusters))), "noise_share": float((labels < 0).mean()),
               "largest_cluster_share_of_sample": float(sizes.max() / len(labels)),
               "median_cluster_size": float(np.median(sizes)), "seconds": round(time.time() - t, 1)}
        print(f"[hdbscan] {res}", flush=True)
        out.append(res)
    return out


def plot_sweep(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [r["k"] for r in rows]
    panels = [("silhouette_sample", "Silhouette (20k sample) ↑, all ≈ 0.02"), ("davies_bouldin", "Davies-Bouldin ↓"),
              ("calinski_harabasz", "Calinski-Harabasz ↑"), ("split_half_ari", "Split-half stability (ARI) ↑"),
              ("coherence_mean_cos_to_centroid", "Mean cosine to centroid ↑"),
              ("min_val_pairs_q1_cluster", "Smallest cluster: val pairs ↑")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.ravel(), panels):
        ax.plot(ks, [r[key] for r in rows], color=BLUE, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xticks(ks)
        if key == "silhouette_sample":
            ax.set_ylim(-0.05, 0.25)  # absolute scale: a zoomed axis would exaggerate a 0.006 change
            ax.axhline(0, color=INK2, linewidth=0.8)
        ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2); ax.set_facecolor(SURFACE)
    for ax in axes[1]:
        ax.set_xlabel("K (number of clusters)", color=INK2)
    fig.suptitle("K-Means on 424,012 train question embeddings (MiniLM)", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "phase4_k_sweep.png", dpi=130); plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdbscan-only", action="store_true", help="re-run only the HDBSCAN comparison")
    ap.add_argument("--plot-only", action="store_true", help="redraw the K-sweep figure from k_sweep.json")
    args = ap.parse_args()
    cfg = load_cfg()
    ART.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_sweep(json.loads((ART / "k_sweep.json").read_text()))
        return
    X, _ = EmbeddingCache(cfg["encoder"], "train").load()
    print(f"train questions: {X.shape}", flush=True)
    if args.hdbscan_only:
        (ART / "hdbscan.json").write_text(json.dumps(run_hdbscan(X, cfg), indent=2))
        return

    rows = []
    for k in cfg["k_grid"]:
        t = time.time()
        model, labels, inertia = fit_kmeans(X, k, cfg["seed"], cfg["n_init_sweep"])
        row = {"k": k, "inertia": inertia, **internal_metrics(X, labels, model, cfg["silhouette_sample"], cfg["seed"])}
        sz = size_stats(labels, k)
        row.update({kk: v for kk, v in sz.items() if kk != "sizes"}, train_question_sizes=sz["sizes"])
        row.update(stability(X, k, cfg["seed"]))
        row.update(val_pair_counts(model, k))
        row["seconds"] = round(time.time() - t, 1)
        rows.append(row)
        print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()
                          if not isinstance(v, list)}), flush=True)
        (ART / "k_sweep.json").write_text(json.dumps(rows, indent=2))  # incremental: survives interruption
    plot_sweep(rows)

    hdb = run_hdbscan(X, cfg)
    (ART / "hdbscan.json").write_text(json.dumps(hdb, indent=2))


if __name__ == "__main__":
    main()
