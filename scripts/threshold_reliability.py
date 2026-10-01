"""Phase 5: how reliable is a validation-optimal threshold estimated from n pairs?

Draw random subsamples of size n (from all validation pairs, and within each cluster), tune the F1-optimal
threshold on the subsample, and measure (a) the spread of that threshold and (b) the F1 lost when it is
applied to the full set (pool or cluster) instead of the full-set optimum ("regret").
Diagnostic only: informs the minimum-size rule for Phase 7; no threshold is applied here.

Usage:  python scripts/threshold_reliability.py      (needs artifacts/phase5/pair_assignments.csv)
Writes: artifacts/phase5/threshold_reliability.json, docs/figures/phase5_threshold_reliability.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.cluster_analysis import THRESHOLDS, f1_curve  # noqa: E402
from src.utils.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "phase5"
FIG = PROJECT_ROOT / "docs" / "figures"
SIZES = [100, 250, 500, 1000, 2000, 4000]
REPS = 300
SEED = 42


def reliability(y: np.ndarray, s: np.ndarray, n: int, rng) -> dict:
    full = f1_curve(y, s)
    best = full.max()
    thr, regret = [], []
    for _ in range(REPS):
        idx = rng.choice(len(y), n, replace=False)
        i = int(np.argmax(f1_curve(y[idx], s[idx])))
        thr.append(THRESHOLDS[i]); regret.append(best - full[i])
    return {"n": n, "threshold_std": float(np.std(thr)), "threshold_p5": float(np.percentile(thr, 5)),
            "threshold_p95": float(np.percentile(thr, 95)), "regret_mean": float(np.mean(regret)),
            "regret_p95": float(np.percentile(regret, 95))}


def main() -> None:
    pairs = pd.read_csv(ART / "pair_assignments.csv")
    y, s = pairs["is_duplicate"].to_numpy(), pairs["score"].to_numpy()
    rng = np.random.default_rng(SEED)
    out = {"reps": REPS, "pooled": [reliability(y, s, n, rng) for n in SIZES], "by_cluster": {}}
    for c, g in pairs.groupby("cluster"):
        yc, sc = g["is_duplicate"].to_numpy(), g["score"].to_numpy()
        out["by_cluster"][int(c)] = [reliability(yc, sc, n, rng) for n in SIZES if n < len(g)]
    (ART / "threshold_reliability.json").write_text(json.dumps(out, indent=2))
    for r in out["pooled"]:
        print(f"pooled n={r['n']:5d}: threshold std {r['threshold_std']:.3f} "
              f"[p5 {r['threshold_p5']:.2f}, p95 {r['threshold_p95']:.2f}]  regret mean {r['regret_mean']:.4f} p95 {r['regret_p95']:.4f}")
    agg = pd.DataFrame([{**r, "cluster": c} for c, rows in out["by_cluster"].items() for r in rows])
    by_n = agg.groupby("n")[["threshold_std", "regret_mean", "regret_p95"]].median()
    print("within-cluster (median across clusters):\n", by_n.round(4))
    out["within_cluster_median_by_n"] = by_n.round(5).reset_index().to_dict("records")
    (ART / "threshold_reliability.json").write_text(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID, SURFACE, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), facecolor=SURFACE)
    for ax, key, title in zip(axes, ["threshold_std", "regret_p95"],
                              ["Std of the tuned threshold", "F1 lost vs full-data optimum (95th percentile)"]):
        ax.plot(SIZES, [r[key] for r in out["pooled"]], color=BLUE, linewidth=2, marker="o", label="all val pairs")
        ax.plot(by_n.index, by_n[key], color=ORANGE, linewidth=2, marker="o", label="within a cluster (median)")
        ax.set_xscale("log"); ax.set_xticks(SIZES, [str(n) for n in SIZES])
        ax.set_title(title, loc="left", color=INK, fontsize=10); ax.set_xlabel("pairs used to tune the threshold (log)", color=INK2)
        ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.set_facecolor(SURFACE); ax.tick_params(colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "phase5_threshold_reliability.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
