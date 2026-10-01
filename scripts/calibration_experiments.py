"""Phase 7: global vs cluster-calibrated thresholds, evaluated out-of-sample WITHIN validation.

Thresholds may only be learned from validation, and the test split is reserved for Phase 8. Tuning per-cluster
thresholds on validation and scoring them on the same pairs would favour them by construction, so every
method is CROSS-FITTED: validation is split into 5 question-disjoint folds, thresholds are fitted on 4 folds and
applied to the 5th. This is repeated with 20 different fold shufflings. All systems use the same folds.

Systems (score source x threshold policy):
  B_global  - Phase 3 semantic model, one global threshold               (baseline)
  B_cluster - Phase 3 semantic model, per-cluster thresholds              (adaptive)
  C_global  - Phase 6 entity/constraint-aware meta-model, global threshold
  C_cluster - Phase 6 meta-model + per-cluster thresholds                 (combined; valid: the meta-model never saw
                                                                            validation, and thresholds are cross-fitted)
Per-cluster thresholds require >= MIN_PAIRS fitting pairs and >= MIN_POSITIVES duplicates, otherwise global fallback.

Decision rule (written before running): adopt per-cluster thresholds for a score source only if they improve
macro-cluster F1 over that source's global threshold with a 95% CI excluding 0, AND the CI of the overall F1
change does not lie entirely below 0.

Usage:  python scripts/calibration_experiments.py
Writes: artifacts/phase7/*.json, docs/figures/phase7_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibration.thresholds import (MIN_PAIRS, MIN_POSITIVES, ThresholdPolicy, cross_fitted_predictions,  # noqa: E402
                                        decision_metrics, group_folds)
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402
from src.utils.splits import build_graph  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "phase7"
FIG = PROJECT_ROOT / "docs" / "figures"
N_FOLDS, N_REPS, N_BOOT, SEED = 5, 20, 1000, 42
SOURCES = {"B": "B_base", "C": "C_no_spacy_hgb"}
SYSTEMS = [("B", "global"), ("B", "cluster"), ("C", "global"), ("C", "cluster")]
COMPARISONS = [("B_cluster", "B_global"), ("C_global", "B_global"), ("C_cluster", "B_global"), ("C_cluster", "C_global")]
MIN_PAIRS_GRID = [250, 500, 1000, 2000, 2500, 3000, 3500]
METRICS = ["f1", "precision", "recall", "fpr", "fnr", "macro_cluster_f1", "worst_cluster_f1"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
COLORS = {"B_global": "#2a78d6", "B_cluster": "#eb6834", "C_global": "#1baf7a", "C_cluster": "#eda100"}


def load_validation() -> pd.DataFrame:
    path = resolve(load_config()["split_dir"]) / "val.csv"
    assert path.name == "val.csv", "Phase 7 must only read the validation split"
    val = pd.read_csv(path, keep_default_na=False, na_values=[""])
    scores = pd.read_csv(PROJECT_ROOT / "artifacts" / "phase6" / "val_scores.csv")[["id", *SOURCES.values()]]
    clusters = pd.read_csv(PROJECT_ROOT / "artifacts" / "phase5" / "pair_assignments.csv")[["id", "cluster"]]
    df = val.merge(scores, on="id").merge(clusters, on="id")
    assert len(df) == len(val), "score / cluster files must cover the validation split exactly"
    df["component"] = build_graph(df).pair_component
    return df


def run_reps(df: pd.DataFrame, min_pairs: int = MIN_PAIRS, systems=SYSTEMS, reps: int = N_REPS) -> dict:
    """OOF decisions for every system and repetition: {system: [pred_rep0, pred_rep1, ...]}, plus fitted policies."""
    y, cl = df["is_duplicate"].to_numpy(), df["cluster"].to_numpy()
    preds = {f"{s}_{m}": [] for s, m in systems}
    policies = {f"{s}_{m}": [] for s, m in systems}
    for r in range(reps):
        folds = group_folds(df["component"].to_numpy(), N_FOLDS, SEED + r)
        for s, m in systems:
            p, pols = cross_fitted_predictions(y, df[SOURCES[s]].to_numpy(), cl, folds, m, min_pairs, MIN_POSITIVES)
            preds[f"{s}_{m}"].append(p)
            policies[f"{s}_{m}"].extend(pols)
    return {"preds": preds, "policies": policies}


def summarise(y, cl, preds: dict) -> dict:
    out = {}
    for name, plist in preds.items():
        ms = [decision_metrics(y, p, cl) for p in plist]
        out[name] = {k: {"mean": float(np.mean([m[k] for m in ms])), "min": float(np.min([m[k] for m in ms])),
                         "max": float(np.max([m[k] for m in ms]))} for k in METRICS}
        out[name]["per_cluster_f1_mean"] = np.mean([m["per_cluster_f1"] for m in ms], axis=0).round(4).tolist()
    return out


def paired_bootstrap(y, cl, preds: dict, comparisons, n_boot: int = N_BOOT) -> dict:
    """CI of the repetition-AVERAGED difference: resample validation pairs, recompute every repetition's metrics."""
    rng = np.random.default_rng(SEED)
    keys = ["f1", "precision", "recall", "fpr", "fnr", "macro_cluster_f1", "worst_cluster_f1"]
    deltas = {c: {k: [] for k in keys} for c in comparisons}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        cache = {}
        for name in {n for c in comparisons for n in c}:
            ms = [decision_metrics(y[idx], p[idx], cl[idx]) for p in preds[name]]
            cache[name] = {k: np.mean([m[k] for m in ms]) for k in keys}
        for a, b in comparisons:
            for k in keys:
                deltas[(a, b)][k].append(cache[a][k] - cache[b][k])
    out = {}
    for (a, b), d in deltas.items():
        out[f"{a}_vs_{b}"] = {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
                              for k, v in d.items()}
    return out


def decide(boot: dict, source: str) -> dict:
    d = boot[f"{source}_cluster_vs_{source}_global"]
    macro_ok = d["macro_cluster_f1"]["ci95"][0] > 0
    f1_not_hurt = d["f1"]["ci95"][1] >= 0
    return {"adopt_cluster_thresholds": bool(macro_ok and f1_not_hurt),
            "macro_cluster_f1_delta": d["macro_cluster_f1"], "f1_delta": d["f1"], "worst_cluster_f1_delta": d["worst_cluster_f1"],
            "rule": "adopt iff macro-cluster F1 CI > 0 and overall-F1 CI not entirely < 0"}


def plots(summary, boot, sens, names) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax, title, xl=""):
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_xlabel(xl, color=INK2)
        ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2); ax.set_facecolor(SURFACE)

    comps = [f"{a}_vs_{b}" for a, b in COMPARISONS]
    labels = ["B: cluster vs global τ", "C (meta) vs B, both global τ", "C + cluster τ vs B global",
              "C: cluster vs global τ"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4), facecolor=SURFACE, sharey=True)
    yy = np.arange(len(comps))
    for ax, k, title in zip(axes, ["f1", "macro_cluster_f1", "worst_cluster_f1"],
                            ["Δ overall F1", "Δ macro-cluster F1", "Δ worst-cluster F1"]):
        m = [boot[c][k]["mean"] for c in comps]
        lo = [boot[c][k]["ci95"][0] for c in comps]
        hi = [boot[c][k]["ci95"][1] for c in comps]
        ax.hlines(yy, lo, hi, color="#2a78d6", linewidth=2)
        ax.scatter(m, yy, color="#2a78d6", s=36, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        style(ax, f"{title} (95% CI)", "difference (cross-fitted, 20 repetitions)")
    axes[0].set_yticks(yy, labels, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "phase7_deltas.png", dpi=130); plt.close(fig)

    order = np.argsort(summary["B_global"]["per_cluster_f1_mean"])
    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=SURFACE)
    for off, name in zip((-0.24, -0.08, 0.08, 0.24), ["B_global", "B_cluster", "C_global", "C_cluster"]):
        v = np.array(summary[name]["per_cluster_f1_mean"])[order]
        ax.scatter(v, np.arange(12) + off, color=COLORS[name], s=30, label=name, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(np.arange(12), [f"C{c} {names[c][:28]}" for c in order], fontsize=8.5)
    style(ax, "Per-cluster F1 (cross-fitted, mean of 20 repetitions)", "F1")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "phase7_per_cluster.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), facecolor=SURFACE)
    for src, color in (("B", "#2a78d6"), ("C", "#eda100")):
        xs = [r["min_pairs"] for r in sens[src]]
        axes[0].plot(xs, [r["macro_cluster_f1"] for r in sens[src]], color=color, marker="o", linewidth=2, label=f"{src}: cluster τ")
        axes[0].axhline(summary[f"{src}_global"]["macro_cluster_f1"]["mean"], color=color, linestyle="--", linewidth=1)
    xs = [r["min_pairs"] for r in sens["B"]]
    axes[1].plot(xs, [r["clusters_with_own_threshold"] for r in sens["B"]], color=INK2, marker="o", linewidth=2)
    style(axes[0], "Macro-cluster F1 vs minimum cluster size (dashed = global τ)", "min fitting pairs per cluster")
    style(axes[1], "Clusters given their own threshold (same for B and C)", "min fitting pairs per cluster")
    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "phase7_min_size.png", dpi=130); plt.close(fig)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    df = load_validation()
    y, cl = df["is_duplicate"].to_numpy(), df["cluster"].to_numpy()
    names = {int(c): v["name"] for c, v in json.loads((PROJECT_ROOT / "artifacts" / "phase4" / "cluster_names.json")
                                                      .read_text()).items() if not c.startswith("_")}
    print(f"val={len(df)} components={df.component.nunique()} folds={N_FOLDS} reps={N_REPS}", flush=True)

    # ---- in-sample reference (optimistic: thresholds fitted and scored on the same pairs)
    in_sample = {}
    for s, m in SYSTEMS:
        pol = ThresholdPolicy(m).fit(y, df[SOURCES[s]], cl)
        met = decision_metrics(y, pol.predict(df[SOURCES[s]], cl), cl)
        in_sample[f"{s}_{m}"] = {k: float(met[k]) for k in METRICS}
    print("in-sample:", {k: round(v["f1"], 4) for k, v in in_sample.items()}, flush=True)

    # ---- cross-fitted main comparison
    runs = run_reps(df)
    summary = summarise(y, cl, runs["preds"])
    for name, s in summary.items():
        print(f"{name:10s} " + " ".join(f"{k}={s[k]['mean']:.4f}" for k in METRICS), flush=True)
    boot = paired_bootstrap(y, cl, runs["preds"], COMPARISONS)
    for c, d in boot.items():
        print(f"{c:26s} " + " ".join(f"d{k}={d[k]['mean']:+.4f}[{d[k]['ci95'][0]:+.4f},{d[k]['ci95'][1]:+.4f}]"
                                     for k in ("f1", "macro_cluster_f1", "worst_cluster_f1")), flush=True)

    # ---- threshold stability across the 100 fold fits
    stability = {}
    for name in ("B_cluster", "C_cluster"):
        pols = runs["policies"][name]
        g = [p.global_threshold for p in pols]
        stability[name] = {"global": {"mean": float(np.mean(g)), "std": float(np.std(g)), "min": float(np.min(g)), "max": float(np.max(g))}}
        for c in range(12):
            t = [p.cluster_thresholds.get(c, np.nan) for p in pols]
            stability[name][c] = {"mean": float(np.nanmean(t)), "std": float(np.nanstd(t)), "min": float(np.nanmin(t)),
                                  "max": float(np.nanmax(t)), "own_threshold_share": float(np.mean(~np.isnan(t)))}

    # ---- minimum-size sensitivity (fallback behaviour)
    sens = {}
    for src in ("B", "C"):
        sens[src] = []
        for mp in MIN_PAIRS_GRID:
            r = run_reps(df, mp, systems=[(src, "cluster")], reps=5)
            ms = [decision_metrics(y, p, cl) for p in r["preds"][f"{src}_cluster"]]
            own = np.mean([len(p.cluster_thresholds) for p in r["policies"][f"{src}_cluster"]])
            sens[src].append({"min_pairs": mp, "clusters_with_own_threshold": float(own),
                              **{k: float(np.mean([m[k] for m in ms])) for k in ("f1", "macro_cluster_f1", "worst_cluster_f1")}})
        print(src, "min_pairs sensitivity:", [(r["min_pairs"], round(r["clusters_with_own_threshold"], 1), round(r["macro_cluster_f1"], 4)) for r in sens[src]], flush=True)

    # ---- pre-registered decision and frozen configuration for Phase 8
    decision = {"B": decide(boot, "B"), "C": decide(boot, "C")}
    final_mode = "cluster" if decision["C"]["adopt_cluster_thresholds"] else "global"
    baseline = ThresholdPolicy("global").fit(y, df[SOURCES["B"]], cl)
    final = ThresholdPolicy(final_mode).fit(y, df[SOURCES["C"]], cl)
    baseline.save(ART / "frozen_policy_baseline.json", {"score_source": "phase3 sbert_mlp (B)", "fitted_on": "full validation"})
    final.save(ART / "frozen_policy_final.json", {"score_source": "phase6 meta C_no_spacy_hgb (C)", "fitted_on": "full validation"})
    reference_cluster = ThresholdPolicy("cluster").fit(y, df[SOURCES["C"]], cl)  # for reporting only
    print("DECISION:", {k: v["adopt_cluster_thresholds"] for k, v in decision.items()}, "final mode:", final_mode, flush=True)

    (ART / "results.json").write_text(json.dumps({
        "setup": {"n_folds": N_FOLDS, "n_reps": N_REPS, "n_boot": N_BOOT, "min_pairs": MIN_PAIRS, "min_positives": MIN_POSITIVES},
        "in_sample": in_sample, "cross_fitted": summary, "bootstrap": boot, "threshold_stability": stability,
        "min_pairs_sensitivity": sens, "decision": decision, "final_threshold_mode": final_mode,
        "full_val_cluster_thresholds_C": {str(k): v for k, v in reference_cluster.cluster_thresholds.items()},
        "full_val_global_threshold_C": reference_cluster.global_threshold}, indent=2, default=float))
    plots(summary, boot, sens, names)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        saved = json.loads((ART / "results.json").read_text())
        nm = {int(c): v["name"] for c, v in json.loads((PROJECT_ROOT / "artifacts" / "phase4" / "cluster_names.json")
                                                       .read_text()).items() if not c.startswith("_")}
        plots(saved["cross_fitted"], saved["bootstrap"], saved["min_pairs_sensitivity"], nm)
    else:
        main()
