"""Phase 5: cluster-level error analysis on VALIDATION (frozen Phase 4 centroids, Phase 3 primary model).

Order of operations is deliberate:
  1. Compare pair-to-cluster rules using LABEL-FREE properties only (coverage, symmetry, sizes,
     agreement, robustness to small embedding noise). The primary rule is fixed before any F1 is computed.
  2. Per-cluster metrics, bootstrap intervals and a random-partition null for the chosen rule.
  3. Sensitivity: the same summaries under the other rules (reported, never used to choose).
  4. Threshold variation, boundary sensitivity, error samples for manual inspection.
No clustering is refit, no model is changed, no threshold is applied. The test split is never read.

Usage:  python scripts/analyze_clusters.py
Writes: artifacts/phase5/*, docs/figures/phase5_*.png
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clustering.core import CentroidModel  # noqa: E402
from src.clustering.pairs import UNASSIGNED, assign_pairs, centroid_geometry, pair_embedding  # noqa: E402
from src.evaluation.cluster_analysis import (bootstrap_clusters, cluster_table, f1_curve,  # noqa: E402
                                             random_partition_null, robustness_summary, THRESHOLDS)
from src.evaluation.error_analysis import error_tags  # noqa: E402
from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.models.sentence_encoder import EmbeddingCache, clean  # noqa: E402
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "phase5"
FIG = PROJECT_ROOT / "docs" / "figures"
MODEL_SCORES = PROJECT_ROOT / "artifacts" / "phase3" / "sbert_all-MiniLM-L6-v2"
CLUSTER_DIR = PROJECT_ROOT / "artifacts" / "phase4" / "cluster_model"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
PRIMARY_MODEL = "sbert_mlp"
PRIMARY_RULE = "pair_avg"  # fixed on label-free grounds (section 1); never chosen by F1
SEED = 42
NOISE_SIGMA = 0.0104       # per-dimension Gaussian noise -> mean cos(x, x + noise) ~ 0.98 in 384-d
INK, INK2, GRID, SURFACE, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#2a78d6", "#eb6834"
_DATE = re.compile(r"\b(19\d\d|20\d\d|january|february|march|april|may|june|july|august|september|october|"
                   r"november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)


def load_validation() -> pd.DataFrame:
    path = resolve(load_config()["split_dir"]) / "val.csv"
    assert path.name == "val.csv", "Phase 5 must only read the validation split"
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def size_profile(labels: np.ndarray, k: int) -> dict:
    assigned = labels[labels != UNASSIGNED]
    counts = np.bincount(assigned, minlength=k)
    share = counts / max(counts.sum(), 1)
    nz = share[share > 0]
    return {"coverage": float((labels != UNASSIGNED).mean()), "counts": counts.tolist(),
            "min_count": int(counts.min()), "max_count": int(counts.max()),
            "size_entropy_normalised": float(-(nz * np.log(nz)).sum() / np.log(k))}


def compare_rules(u, v, model) -> dict:
    """Label-free comparison of pair-assignment rules."""
    rng = np.random.default_rng(SEED)
    out = {}
    q1, q2 = model.predict(u), model.predict(v)
    labels = {r: assign_pairs(u, v, model, r) for r in ("q1", "pair_avg", "same_only")}
    swapped = {r: assign_pairs(v, u, model, r) for r in labels}
    noise_u = u + rng.normal(0, NOISE_SIGMA, u.shape).astype(np.float32)
    noise_v = v + rng.normal(0, NOISE_SIGMA, v.shape).astype(np.float32)
    noisy = {r: assign_pairs(noise_u / np.linalg.norm(noise_u, axis=1, keepdims=True),
                             noise_v / np.linalg.norm(noise_v, axis=1, keepdims=True), model, r) for r in labels}
    mean_cos_noise = float(np.mean(np.sum(u * (noise_u / np.linalg.norm(noise_u, axis=1, keepdims=True)), axis=1)))
    geo_pair = centroid_geometry(pair_embedding(u, v), model)
    geo_q1 = centroid_geometry(u, model)
    for r, lab in labels.items():
        both = (lab != UNASSIGNED) & (noisy[r] != UNASSIGNED)
        out[r] = {**size_profile(lab, model.k),
                  "symmetric_share_unchanged_when_swapped": float((lab == swapped[r]).mean()),
                  "noise_robustness_share_unchanged": float((lab[both] == noisy[r][both]).mean()),
                  "noise_drops_to_unassigned": float(((lab != UNASSIGNED) & (noisy[r] == UNASSIGNED)).mean())}
    pa = labels["pair_avg"]
    out["pair_avg"].update({
        "equals_q1_cluster": float((pa == q1).mean()), "equals_q2_cluster": float((pa == q2).mean()),
        "equals_either_question_cluster": float(((pa == q1) | (pa == q2)).mean()),
        "neither_question_cluster": float(((pa != q1) & (pa != q2)).mean()),
        "margin_median": float(np.median(geo_pair["margin"])),
        "margin_p25": float(np.percentile(geo_pair["margin"], 25))})
    out["q1"].update({"margin_median": float(np.median(geo_q1["margin"])),
                      "margin_p25": float(np.percentile(geo_q1["margin"], 25))})
    out["ari_q1_vs_pair_avg"] = float(adjusted_rand_score(labels["q1"], pa))
    out["questions_in_same_cluster"] = float((q1 == q2).mean())
    out["noise_sigma"] = NOISE_SIGMA
    out["noise_mean_cos_to_original"] = mean_cos_noise
    return out, labels, geo_pair


def date_diff(q1: str, q2: str) -> bool:
    d1 = {m.lower() for m in _DATE.findall(q1)}
    d2 = {m.lower() for m in _DATE.findall(q2)}
    return bool(d1 ^ d2) and bool(d1 | d2)


def boundary_analysis(y, s, pred, margin, cos_assigned, straddle) -> dict:
    def stats(mask):
        yt, st, pt = y[mask], s[mask], pred[mask]
        m = classification_metrics(yt, st, 0.5)  # threshold-free parts only from here
        tp = int(((pt == 1) & (yt == 1)).sum()); fp = int(((pt == 1) & (yt == 0)).sum())
        fn = int(((pt == 0) & (yt == 1)).sum()); tn = int(((pt == 0) & (yt == 0)).sum())
        return {"n": int(mask.sum()), "prevalence": float(yt.mean()), "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                "pr_auc_lift": m["pr_auc"] / yt.mean(),
                "f1_at_global": 2 * tp / max(2 * tp + fp + fn, 1),
                "fpr_at_global": fp / max(fp + tn, 1), "fnr_at_global": fn / max(fn + tp, 1)}
    out = {}
    qs = np.quantile(margin, [0.25, 0.5, 0.75])
    bins = np.digitize(margin, qs)
    out["by_margin_quartile"] = {f"Q{i + 1}" + (" (lowest margin)" if i == 0 else " (highest margin)" if i == 3 else ""):
                                 stats(bins == i) for i in range(4)}
    out["margin_quartile_edges"] = qs.tolist()
    cq = np.quantile(cos_assigned, [0.25, 0.5, 0.75])
    cb = np.digitize(cos_assigned, cq)
    out["by_cos_to_centroid_quartile"] = {f"Q{i + 1}": stats(cb == i) for i in range(4)}
    out["questions_in_same_cluster"] = stats(~straddle)
    out["questions_in_different_clusters"] = stats(straddle)
    return out


def plots(table, boot, null, observed, names, global_thr, overall_f1, boundary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax, title, xl="", yl=""):
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlabel(xl, color=INK2); ax.set_ylabel(yl, color=INK2)
        ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2); ax.set_facecolor(SURFACE)

    order = table.sort_values("f1").index
    labels = [f"C{c} {names.get(int(c), '')[:30]}" for c in order]
    yy = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=SURFACE, sharey=True)
    for ax, (col, lo, hi, ref, title, xl) in zip(axes, [
            ("f1", "f1_ci_low", "f1_ci_high", overall_f1, "F1 at the global threshold (95% bootstrap CI)", "F1"),
            ("opt_threshold", "opt_threshold_ci_low", "opt_threshold_ci_high", global_thr,
             "Cluster-optimal threshold (95% bootstrap CI)", "threshold")]):
        vals, l, h = table.loc[order, col], boot.loc[order, lo], boot.loc[order, hi]
        ax.hlines(yy, l, h, color=BLUE, linewidth=2)
        ax.scatter(vals, yy, color=BLUE, s=40, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(ref, color=INK2, linestyle="--", linewidth=1)
        ax.text(ref, -0.9, f" overall {ref:.2f}" if col == "f1" else f" global τ {ref:.2f}",
                color=INK2, fontsize=8, va="center")
        ax.set_ylim(-1.4, len(order) - 0.5)
        style(ax, title, xl)
    axes[0].set_yticks(yy, labels, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "phase5_cluster_f1_thresholds.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    for ax, key, title in zip(axes, ["f1_std", "opt_threshold_std"],
                              ["Spread (std) of cluster F1", "Spread (std) of cluster-optimal thresholds"]):
        raw = null["_raw"][key]
        ax.hist(raw, bins=30, color="#c9c8c1", edgecolor=SURFACE)
        ax.axvline(observed[key], color=ORANGE, linewidth=2)
        ax.text(observed[key], ax.get_ylim()[1] * 0.9, f" observed {observed[key]:.3f}", color=INK, fontsize=8.5)
        style(ax, f"{title}\nreal clusters (line) vs {null['n_perm']} random partitions of equal sizes",
              "std across 12 groups", "random partitions")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout(); fig.savefig(FIG / "phase5_null_comparison.png", dpi=130); plt.close(fig)

    q = boundary["by_margin_quartile"]
    keys = list(q)
    panels = [("roc_auc", "ROC-AUC (threshold- and prevalence-free)"), ("pr_auc_lift", "PR-AUC / prevalence"),
              ("prevalence", "Duplicate prevalence"), ("f1_at_global", "F1 at global τ")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), facecolor=SURFACE)
    for ax, (metric, title) in zip(axes, panels):
        ax.plot(range(4), [q[k][metric] for k in keys], color=BLUE, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xticks(range(4), ["Q1\nlow", "Q2", "Q3", "Q4\nhigh"])
        style(ax, title, "assignment-margin quartile")
        if metric == "f1_at_global":
            ax.set_ylim(0.70, 0.85)  # unzoomed: F1 varies by only ~0.013 across quartiles
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.suptitle("Model quality by pair-assignment margin (low margin = pair midpoint near a cluster boundary)",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "phase5_boundary.png", dpi=130); plt.close(fig)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    val = load_validation()
    scores = pd.read_csv(MODEL_SCORES / "val_scores.csv")[["id", PRIMARY_MODEL]]
    df = val.merge(scores, on="id")
    assert len(df) == len(val)
    global_thr = json.loads((MODEL_SCORES / "metrics.json").read_text())[PRIMARY_MODEL]["tuned_threshold"]
    model = CentroidModel.load(CLUSTER_DIR)
    names = {int(c): v["name"] for c, v in json.loads((PROJECT_ROOT / "artifacts" / "phase4" / "cluster_names.json")
                                                      .read_text()).items() if not c.startswith("_")}
    matrix, index = EmbeddingCache(ENCODER, "val").load()
    u = matrix[[index[clean(t)] for t in df.question1]]
    v = matrix[[index[clean(t)] for t in df.question2]]
    y, s = df["is_duplicate"].to_numpy(), df[PRIMARY_MODEL].to_numpy()

    # ---------------- 1. rule comparison (label-free)
    rules, labels, geo = compare_rules(u, v, model)
    rules["primary_rule"] = PRIMARY_RULE
    (ART / "rule_comparison.json").write_text(json.dumps(rules, indent=2))
    print(json.dumps({r: {k: (round(x, 4) if isinstance(x, float) else x) for k, x in rules[r].items() if k != "counts"}
                      for r in ("q1", "pair_avg", "same_only")}, indent=1), flush=True)

    # ---------------- 2. per-cluster metrics (primary rule)
    groups = labels[PRIMARY_RULE]
    pred = (s >= global_thr).astype(int)
    overall = classification_metrics(y, s, global_thr)
    table = cluster_table(y, s, groups, global_thr)
    table.insert(0, "name", [names[c] for c in table.index])
    boot = bootstrap_clusters(y, s, groups, global_thr)
    table = table.join(boot)
    summary = robustness_summary(table, overall["f1"], global_thr)
    null = random_partition_null(y, s, groups, global_thr)
    observed = {"f1_std": summary["f1_std"], "opt_threshold_std": summary["opt_threshold_std"],
                "f1_range": summary["f1_range"], "opt_threshold_range": summary["opt_threshold_range"],
                "roc_auc_range": float(np.ptp(table["roc_auc"]))}
    summary["null_comparison"] = {k: {"observed": observed[k], "null_mean": null[k]["mean"], "null_p95": null[k]["p95"],
                                      "null_max": null[k]["max"],
                                      "p_value_one_sided": float((np.array(null["_raw"][k]) >= observed[k]).mean())}
                                  for k in observed}
    gi = int(np.argmin(np.abs(THRESHOLDS - global_thr)))
    table["f1_gain_in_sample_if_cluster_threshold"] = table["f1_at_opt_in_sample"] - table["f1"]
    table["threshold_ci_excludes_global"] = (table["opt_threshold_ci_low"] > global_thr) | (table["opt_threshold_ci_high"] < global_thr)
    table["opt_threshold_ci_width"] = table["opt_threshold_ci_high"] - table["opt_threshold_ci_low"]
    summary["corr_n_vs_threshold_ci_width"] = float(np.corrcoef(table["n"], table["opt_threshold_ci_width"])[0, 1])
    summary["in_sample_gain_overall_if_all_cluster_thresholds"] = float(
        2 * sum(((s[groups == c] >= table.loc[c, "opt_threshold"]) & (y[groups == c] == 1)).sum() for c in table.index)
        / sum(((s[groups == c] >= table.loc[c, "opt_threshold"]).sum() + y[groups == c].sum()) for c in table.index)
        - overall["f1"])
    table.to_csv(ART / "cluster_metrics.csv")

    # ---------------- 3. sensitivity to the assignment rule (reported only)
    sens = {}
    for r in ("q1", "same_only"):
        g = labels[r]
        m = g != UNASSIGNED
        t = cluster_table(y[m], s[m], g[m], global_thr)
        sens[r] = {**robustness_summary(t, classification_metrics(y[m], s[m], global_thr)["f1"], global_thr),
                   "per_cluster_f1": t["f1"].round(4).to_dict(), "per_cluster_opt_threshold": t["opt_threshold"].to_dict(),
                   "coverage": float(m.mean())}
    sens["pair_avg"] = {"per_cluster_f1": table["f1"].round(4).to_dict(),
                        "per_cluster_opt_threshold": table["opt_threshold"].to_dict()}
    (ART / "rule_sensitivity.json").write_text(json.dumps(sens, indent=2, default=float))

    # ---------------- 6. boundary sensitivity (diagnostic)
    straddle = model.predict(u) != model.predict(v)
    boundary = boundary_analysis(y, s, pred, geo["margin"], geo["cos_assigned"], straddle)
    (ART / "boundary.json").write_text(json.dumps(boundary, indent=2))

    # ---------------- per-pair record
    lex = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "lexical_val.csv")[["id", "word_jaccard"]]
    tags = pd.DataFrame([{**error_tags(a, b), "date_diff": date_diff(a, b)} for a, b in zip(df.question1, df.question2)])
    pairs = pd.DataFrame({"id": df["id"], "cluster": groups, "cluster_q1": labels["q1"], "cluster_same_only": labels["same_only"],
                          "margin": geo["margin"], "cos_assigned": geo["cos_assigned"], "second_cluster": geo["second_cluster"],
                          "straddle": straddle, "score": s, "pred_global": pred, "is_duplicate": y})
    pairs = pairs.merge(lex, on="id")
    pairs = pd.concat([pairs, tags[["entity_diff", "number_diff", "negation_diff", "date_diff"]]], axis=1)
    pairs["outcome"] = np.select([(pairs.pred_global == 1) & (pairs.is_duplicate == 1), (pairs.pred_global == 1) & (pairs.is_duplicate == 0),
                                  (pairs.pred_global == 0) & (pairs.is_duplicate == 0)], ["TP", "FP", "TN"], "FN")
    pairs.to_csv(ART / "pair_assignments.csv", index=False)

    # ---------------- 4. weakest clusters + error tags within clusters + samples for manual inspection
    ranks = pd.DataFrame({"f1": table["f1"].rank(), "precision": table["precision"].rank(),
                          "recall": table["recall"].rank(), "fpr": table["fpr"].rank(ascending=False),
                          "fnr": table["fnr"].rank(ascending=False)})
    ranks["mean_rank"] = ranks.mean(axis=1)
    weakest = table["f1"].nsmallest(3).index.tolist()
    tag_rates = {}
    for c in table.index:
        sub = pairs[pairs.cluster == c]
        neg, pos = sub[sub.is_duplicate == 0], sub[sub.is_duplicate == 1]
        tag_rates[int(c)] = {
            **{f"share_of_FP_with_{t}": float(sub[sub.outcome == "FP"][t].mean()) for t in ("entity_diff", "number_diff", "negation_diff", "date_diff")},
            "share_of_FP_high_overlap(j>0.6)": float((sub[sub.outcome == "FP"].word_jaccard > 0.6).mean()),
            "share_of_FN_low_overlap(j<=0.2)": float((sub[sub.outcome == "FN"].word_jaccard <= 0.2).mean()),
            "fpr_high_overlap_negatives": float((neg[neg.word_jaccard > 0.6].pred_global == 1).mean()),
            "fnr_low_overlap_positives": float((pos[pos.word_jaccard <= 0.2].pred_global == 0).mean()),
        }
    rng = np.random.default_rng(SEED)
    samples = []
    text = df.set_index("id")[["question1", "question2"]]
    for c in weakest + ["rest"]:
        pool_mask = pairs.cluster.isin(weakest) == False if c == "rest" else pairs.cluster == c  # noqa: E712
        for kind in ("FP", "FN"):
            pool = pairs[pool_mask & (pairs.outcome == kind)]
            pick = pool.iloc[rng.choice(len(pool), min(15, len(pool)), replace=False)]
            for _, r in pick.iterrows():
                samples.append({"id": r.id, "group": f"C{c}" if c != "rest" else "other clusters", "cluster": r.cluster,
                                "outcome": kind, "score": round(r.score, 3), "jaccard": round(r.word_jaccard, 2),
                                "question1": text.loc[r.id, "question1"], "question2": text.loc[r.id, "question2"]})
    pd.DataFrame(samples).to_csv(ART / "error_samples.csv", index=False)
    summary.update({"weakest_by_f1": weakest, "rank_table": ranks.round(2).to_dict("index"),
                    "tag_rates_by_cluster": tag_rates, "primary_rule": PRIMARY_RULE, "primary_model": PRIMARY_MODEL})
    null_out = {k: v for k, v in null.items() if k != "_raw"}
    (ART / "null_partition.json").write_text(json.dumps(null_out, indent=2))
    (ART / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    plots(table, boot, null, observed, names, global_thr, overall["f1"], boundary)
    cols = ["n", "prevalence", "precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc", "opt_threshold",
            "opt_threshold_ci_low", "opt_threshold_ci_high", "f1_ci_low", "f1_ci_high"]
    print(table[cols].round(3).to_string(), flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rank_table", "tag_rates_by_cluster")}, indent=1, default=float))


if __name__ == "__main__":
    main()
