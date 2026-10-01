"""Phase 6c: train and compare the pre-specified meta-classifier variants; evaluate on VALIDATION.

Training data: all TRAIN pairs, with the base probability taken from cross-fitted (out-of-fold) models
(crossfit_base.py), so the meta-model sees honest base scores. Validation base scores come from the deployed,
unchanged Phase 3 model. Validation is used for decision thresholds and for comparing the fixed variants.
The test split is never read.

Comparisons required by the specification:
  A  semantic similarity only     - MiniLM cosine, threshold tuned
  B  semantic model probability   - deployed sbert_mlp, threshold tuned (the bar: F1 0.771)
  C  semantic model + constraints - meta-classifier (several pre-specified variants and ablations)

The variant carried forward is chosen by a rule written down BEFORE the run (choose_variant below).

Usage:  python scripts/train_meta.py
Writes: artifacts/phase6/{variant_results.json, chosen.json, val_scores.csv, meta_model.joblib,
        slice_rates.json, manual_sample_flips.json, probes.json}, docs/figures/phase6_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_phase3 import paired_bootstrap  # noqa: E402
from src.evaluation.cluster_analysis import cluster_table, robustness_summary  # noqa: E402
from src.evaluation.metrics import validation_report  # noqa: E402
from src.models.meta import GROUP_ABLATIONS, HGB_ABLATIONS, LEXICAL, VARIANTS, MetaModel, logit, variant_by_name  # noqa: E402
from src.utils.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "phase6"
FEAT = PROJECT_ROOT / "data" / "processed" / "features"
P3 = PROJECT_ROOT / "artifacts" / "phase3" / "sbert_all-MiniLM-L6-v2"
P5 = PROJECT_ROOT / "artifacts" / "phase5"
FIG = PROJECT_ROOT / "docs" / "figures"
INK, INK2, GRID, SURFACE, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#2a78d6", "#eb6834"
MIN_GAIN = 0.005  # a more complex variant must beat the simpler one by >= this PR-AUC AND with a CI excluding 0

EXTRA_PROBES = [
    ("paraphrase", "How can I lose weight quickly?", "What is the fastest way to lose weight?"),
    ("paraphrase", "Is it possible to hack fb?", "How do we hack a Facebook account?"),
    ("entity_mismatch", "Who founded Microsoft?", "Who founded Apple?"),
    ("entity_mismatch", "What is the temperament of a Doberman/Lab mix?", "What is the temperament of a Lab/Pitbull mix?"),
    ("number_mismatch", "How can I lose 5 kg in a month?", "How can I lose 20 kg in a month?"),
    ("number_mismatch", "What is the best phone under 10000 rupees?", "What is the best phone under 20000 rupees?"),
    ("date_mismatch", "Who won the 2012 US presidential election?", "Who won the 2016 US presidential election?"),
    ("location_mismatch", "What are the best hotels in Delhi?", "What are the best hotels in Mumbai?"),
    ("negation_mismatch", "Why do people believe in God?", "Why do people not believe in God?"),
    ("negation_mismatch", "What are some examples of movable joints?", "What are some examples of non-movable joints?"),
    ("broader_narrower", "How do I learn programming?", "How do I learn Python programming for data science?"),
    ("identical_case", "What are the best places to visit in india?", "What are the best places to visit in India?"),
    ("acronym_variant", "How do I apply to the U.S. army?", "How do I apply to the US Army?"),
]


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(FEAT / "constraints_train.csv")
    tr = tr.merge(pd.read_csv(ART / "train_oof_base.csv")[["id", "base_prob_oof"]], on="id")
    tr = tr.merge(pd.read_csv(FEAT / "lexical_train.csv")[["id"] + LEXICAL], on="id")
    tr["base_prob"] = tr["base_prob_oof"]
    va = pd.read_csv(FEAT / "constraints_val.csv")
    va = va.merge(pd.read_csv(P3 / "val_scores.csv")[["id", "sbert_mlp", "sbert_cosine"]], on="id")
    va = va.merge(pd.read_csv(FEAT / "lexical_val.csv")[["id"] + LEXICAL], on="id")
    va["base_prob"] = va["sbert_mlp"]
    for d in (tr, va):
        d["base_logit"] = logit(d["base_prob"])
    assert not tr.isna().any().any() and not va.isna().any().any()
    return tr, va


def evaluate(name: str, y: np.ndarray, s: np.ndarray, clusters: np.ndarray, is_prob: bool = True) -> dict:
    rep = validation_report(y, s, is_probability=is_prob)
    thr = rep["tuned_threshold"]
    table = cluster_table(y, s, clusters, thr)
    summ = robustness_summary(table, rep["val_at_tuned_threshold"]["f1"], thr)
    rep["cluster"] = {k: summ[k] for k in ("macro_cluster_f1", "worst_cluster_f1", "worst_cluster", "best_cluster_f1")}
    rep["cluster"]["per_cluster_f1"] = table["f1"].round(4).to_dict()
    rep["cluster"]["per_cluster_precision"] = table["precision"].round(4).to_dict()
    rep["cluster"]["per_cluster_fpr"] = table["fpr"].round(4).to_dict()
    m = rep["val_at_tuned_threshold"]
    print(f"{name:24s} thr={thr:.2f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
          f"PR={m['pr_auc']:.4f} ROC={m['roc_auc']:.4f} macroF1={summ['macro_cluster_f1']:.4f} "
          f"worst={summ['worst_cluster_f1']:.4f}(C{summ['worst_cluster']})", flush=True)
    return rep


def slice_rates(va: pd.DataFrame, preds: dict) -> dict:
    """FPR / FNR on the Phase 5 heuristic-tag slices (independent of the spaCy-based features)."""
    tags = pd.read_csv(P5 / "pair_assignments.csv")[["id", "entity_diff", "number_diff", "negation_diff", "date_diff",
                                                      "word_jaccard"]]
    d = va[["id", "is_duplicate"]].merge(tags, on="id")
    hi = d["word_jaccard"] > 0.6
    slices = {"all": np.ones(len(d), bool), "high_overlap(j>0.6)": hi.to_numpy()}
    for t in ("entity_diff", "number_diff", "negation_diff", "date_diff"):
        slices[f"{t}"] = d[t].to_numpy(bool)
        slices[f"{t}&high_overlap"] = (d[t] & hi).to_numpy(bool)
    y = d["is_duplicate"].to_numpy()
    out = {}
    for sname, m in slices.items():
        neg, pos = m & (y == 0), m & (y == 1)
        out[sname] = {"negatives": int(neg.sum()), "positives": int(pos.sum())}
        for name, p in preds.items():
            out[sname][f"fpr_{name}"] = float(p[neg].mean()) if neg.any() else None
            out[sname][f"fnr_{name}"] = float(1 - p[pos].mean()) if pos.any() else None
    return out


def manual_flips(va: pd.DataFrame, preds: dict) -> dict:
    """How the Phase 5 hand-labelled errors (base model at tau=0.32) change under each prediction rule."""
    s = pd.read_csv(P5 / "error_samples.csv").merge(pd.read_csv(P5 / "manual_error_labels.csv"), on="id")
    idx = va.reset_index(drop=True).reset_index().set_index("id")["index"]
    out = {}
    for name, p in preds.items():
        s[name] = [p[idx[i]] for i in s["id"]]
        res = {}
        for kind, fixed_when in (("FP", 0), ("FN", 1)):
            part = s[s.outcome == kind]
            fixed = part[name] == fixed_when
            res[kind] = {"n": int(len(part)), "fixed": int(fixed.sum()),
                         "fixed_by_category": part[fixed].category.value_counts().to_dict(),
                         "total_by_category": part.category.value_counts().to_dict()}
        out[name] = res
    return out


def choose_variant(results: dict, boot: dict) -> dict:
    """Pre-registered selection rule:
    1. Start from C_constraints (all constraint groups, logistic regression).
    2. Drop spaCy if C_no_spacy is not clearly worse (C's PR-AUC gain over C_no_spacy < MIN_GAIN or CI includes 0).
    3. Switch to a more complex variant built on the SAME feature set as step 2 (+lexical, then gradient boosting)
       only if it beats the current choice by >= MIN_GAIN PR-AUC with a bootstrap CI excluding 0.
       C_plus_cluster is never chosen here (per-cluster offsets belong to Phase 7).
    Correction made after the first run: that run compared C_hgb (built WITH spaCy features) against C_no_spacy
    after step 2 had dropped spaCy. Step 3 now uses the feature set chosen in step 2 (reported in docs).
    """
    log, choice = [], "C_constraints"
    b = boot["C_constraints_vs_C_no_spacy"]
    if b["delta_pr_auc_mean"] < MIN_GAIN or b["delta_pr_auc_ci95"][0] <= 0:
        choice = "C_no_spacy"
        log.append(f"spaCy not clearly helpful (dPR-AUC {b['delta_pr_auc_mean']:+.4f}, CI {np.round(b['delta_pr_auc_ci95'], 4).tolist()}): drop it")
    else:
        log.append(f"spaCy helps (dPR-AUC {b['delta_pr_auc_mean']:+.4f}): keep it")
    complex_for = {"C_constraints": ("C_plus_lexical", "C_hgb"),
                   "C_no_spacy": ("C_no_spacy_plus_lexical", "C_no_spacy_hgb")}[choice]
    for complex_name in complex_for:
        b = boot[f"{complex_name}_vs_{choice}"]
        if b["delta_pr_auc_mean"] >= MIN_GAIN and b["delta_pr_auc_ci95"][0] > 0:
            log.append(f"{complex_name} beats {choice} (dPR-AUC {b['delta_pr_auc_mean']:+.4f}, "
                       f"CI {np.round(b['delta_pr_auc_ci95'], 4).tolist()}): switch")
            choice = complex_name
        else:
            log.append(f"{complex_name} does not clearly beat {choice} (dPR-AUC {b['delta_pr_auc_mean']:+.4f}, "
                       f"CI {np.round(b['delta_pr_auc_ci95'], 4).tolist()}): keep {choice}")
    return {"chosen": choice, "log": log, "min_gain": MIN_GAIN}


def plots(results: dict, boot_vs_b: dict, slices: dict, chosen: str) -> None:
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

    names = [n for n in boot_vs_b]
    yy = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(names) + 1.6), facecolor=SURFACE, sharey=True)
    for ax, key, title in zip(axes, ["f1", "pr_auc"], ["ΔF1 vs B (deployed base model)", "ΔPR-AUC vs B"]):
        mean = [boot_vs_b[n][f"delta_{key}_mean"] for n in names]
        lo = [boot_vs_b[n][f"delta_{key}_ci95"][0] for n in names]
        hi = [boot_vs_b[n][f"delta_{key}_ci95"][1] for n in names]
        colors = [ORANGE if n == chosen else BLUE for n in names]
        ax.hlines(yy, lo, hi, color=colors, linewidth=2)
        ax.scatter(mean, yy, color=colors, s=36, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(0, color=INK2, linewidth=1)
        style(ax, f"{title} (95% paired bootstrap CI)", "difference")
    axes[0].set_yticks(yy, names, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "phase6_variant_deltas.png", dpi=130); plt.close(fig)

    keys = ["high_overlap(j>0.6)", "entity_diff&high_overlap", "number_diff&high_overlap", "negation_diff&high_overlap",
            "date_diff&high_overlap", "all"]
    labels = ["high overlap\n(all)", "entity diff", "number diff", "negation diff", "date diff", "all pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), facecolor=SURFACE, sharey=True)
    for ax, kind, count, title in ((axes[0], "fpr", "negatives", "False-positive rate (negatives)"),
                                   (axes[1], "fnr", "positives", "False-negative rate (positives): the cost")):
        x = np.arange(len(keys)); w = 0.38
        for i, (name, color, lab) in enumerate((("B_base", BLUE, "B: base model"), (chosen, ORANGE, f"C: {chosen}"))):
            vals = [slices[k][f"{kind}_{name}"] for k in keys]
            bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, color=color, label=lab)
            for r, val in zip(bars, vals):
                ax.text(r.get_x() + r.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(x, [f"{l}\n(n={slices[k][count]})" for l, k in zip(labels, keys)], fontsize=7.5)
        style(ax, title)
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 0.85)
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=8)
    fig.suptitle("Validation slices: high overlap = word Jaccard > 0.6; tags are the Phase 5 heuristics",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "phase6_slice_fpr.png", dpi=130); plt.close(fig)


def plot_only() -> None:
    saved = json.loads((ART / "variant_results.json").read_text())
    chosen = json.loads((ART / "chosen.json").read_text())["chosen"]
    order = ["A_cosine", "B_recal", "B_hgb"] + [v.name for v in VARIANTS if v.name not in ("B_recal", "B_hgb")]
    plots(saved["results"], {n: saved["bootstrap"][f"{n}_vs_B_base"] for n in order},
          json.loads((ART / "slice_rates.json").read_text()), chosen)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    tr, va = load_frames()
    y_tr, y_va, cl = tr["is_duplicate"].to_numpy(), va["is_duplicate"].to_numpy(), va["cluster"].to_numpy()
    print(f"train={len(tr)} val={len(va)}", flush=True)

    scores = {"A_cosine": va["sbert_cosine"].to_numpy(), "B_base": va["sbert_mlp"].to_numpy()}
    results = {"A_cosine": evaluate("A_cosine", y_va, scores["A_cosine"], cl, is_prob=False),
               "B_base": evaluate("B_base", y_va, scores["B_base"], cl)}
    models = {}
    for v in VARIANTS + [variant_by_name(n) for n in GROUP_ABLATIONS + HGB_ABLATIONS]:
        m = MetaModel(v).fit(tr, y_tr)
        models[v.name] = m
        scores[v.name] = m.predict_proba(va)
        results[v.name] = evaluate(v.name, y_va, scores[v.name], cl)
        results[v.name]["description"] = v.description
        results[v.name]["coefficients"] = m.coefficients()

    thr = {n: results[n]["tuned_threshold"] for n in results}
    boot = {}
    for n in results:
        if n != "B_base":
            boot[f"{n}_vs_B_base"] = paired_bootstrap(y_va, scores["B_base"], thr["B_base"], scores[n], thr[n])
    pairs = [("C_constraints", "C_no_spacy"), ("C_plus_lexical", "C_constraints"), ("C_hgb", "C_constraints"),
             ("C_no_spacy_plus_lexical", "C_no_spacy"), ("C_no_spacy_hgb", "C_no_spacy"), ("C_constraints", "B_recal"),
             ("C_no_spacy", "B_recal"), ("C_no_spacy_hgb", "B_hgb"), ("B_hgb", "B_recal"), ("C_hgb", "C_no_spacy_hgb")]
    pairs += [(n, "C_no_spacy_hgb") for n in HGB_ABLATIONS]
    for a, b in pairs:
        boot[f"{a}_vs_{b}"] = paired_bootstrap(y_va, scores[b], thr[b], scores[a], thr[a])
    for k, b in boot.items():
        print(f"bootstrap {k:34s} dF1={b['delta_f1_mean']:+.4f} {np.round(b['delta_f1_ci95'], 4)} "
              f"dPR={b['delta_pr_auc_mean']:+.4f} {np.round(b['delta_pr_auc_ci95'], 4)}", flush=True)

    decision = choose_variant(results, boot)
    chosen = decision["chosen"]
    print("\n".join(decision["log"]), "\nCHOSEN:", chosen, flush=True)

    preds = {n: (scores[n] >= thr[n]).astype(int) for n in ("B_base", "B_recal", chosen)}
    slices = slice_rates(va, preds)
    flips = manual_flips(va, preds)

    bundle = {"model": models[chosen], "threshold": thr[chosen], "uses_spacy": models[chosen].variant.uses_spacy,
              "variant": chosen,
              "trained_on": "all train pairs; base probability from 5-fold question-disjoint cross-fitting"}
    joblib.dump(bundle, ART / "meta_model.joblib")
    (ART / "variant_results.json").write_text(json.dumps({"results": results, "bootstrap": boot}, indent=2, default=float))
    (ART / "chosen.json").write_text(json.dumps({**decision, "threshold": thr[chosen],
                                                 "metrics": results[chosen]["val_at_tuned_threshold"],
                                                 "cluster": results[chosen]["cluster"]}, indent=2, default=float))
    (ART / "slice_rates.json").write_text(json.dumps(slices, indent=2))
    (ART / "manual_sample_flips.json").write_text(json.dumps(flips, indent=2))
    pd.DataFrame({"id": va["id"], **{n: np.round(s, 6) for n, s in scores.items()}}).to_csv(ART / "val_scores.csv", index=False)

    order = ["A_cosine", "B_recal", "B_hgb"] + [v.name for v in VARIANTS if v.name not in ("B_recal", "B_hgb")]
    plots(results, {n: boot[f"{n}_vs_B_base"] for n in order}, slices, chosen)

    # probes through the full raw-text pipeline
    from src.models.predictors import MetaPredictor
    mp = MetaPredictor(ART / "meta_model.joblib")
    q1 = [p[1] for p in EXTRA_PROBES]; q2 = [p[2] for p in EXTRA_PROBES]
    base = mp.base.predict(q1, q2)
    meta = mp.predict(q1, q2)
    probes = [{"type": t, "question1": a, "question2": b, "base_prob": round(float(pb), 3),
               "base_dup": bool(pb >= thr["B_base"]), "meta_prob": round(float(pm), 3), "meta_dup": bool(pm >= thr[chosen])}
              for (t, a, b), pb, pm in zip(EXTRA_PROBES, base, meta)]
    (ART / "probes.json").write_text(json.dumps(probes, indent=2))
    print(pd.DataFrame(probes)[["type", "base_prob", "base_dup", "meta_prob", "meta_dup"]].to_string(), flush=True)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot_only()
    else:
        main()
