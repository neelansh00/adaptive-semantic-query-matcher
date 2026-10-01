"""Phase 3 comparison on VALIDATION: Phase 2 best baseline vs Siamese BiLSTM vs sentence-transformer heads.

Reads saved validation scores (no retraining), then adds error analysis, sanity probes, latency and size.
Usage:  python scripts/compare_phase3.py [--skip-latency]
Writes: artifacts/phase3/comparison.json, docs/figures/phase3_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_baselines import PROBES  # noqa: E402
from src.evaluation.error_analysis import error_tags  # noqa: E402
from src.evaluation.metrics import threshold_sweep, validation_report  # noqa: E402
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402
from src.utils.torch_utils import configure_threads, count_parameters, dir_size_mb  # noqa: E402

ART = PROJECT_ROOT / "artifacts"
FIG = PROJECT_ROOT / "docs" / "figures"
BASELINE = "lr_tfidf_pair_lexical"
MODELS = [BASELINE, "siamese_bilstm", "sbert_cosine", "sbert_lr", "sbert_mlp"]
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
TAGS = ["entity_diff", "number_diff", "negation_diff"]
SBERT_DIR = ART / "phase3" / "sbert_all-MiniLM-L6-v2"


def load_scores() -> pd.DataFrame:
    val = pd.read_csv(resolve(load_config()["split_dir"]) / "val.csv", keep_default_na=False, na_values=[""])
    df = val.merge(pd.read_csv(ART / "phase2" / "val_predictions.csv")[["id", BASELINE]], on="id")
    bilstm = ART / "phase3" / "bilstm" / "val_scores.csv"
    if bilstm.exists():
        df = df.merge(pd.read_csv(bilstm), on="id")
    else:
        print("WARNING: BiLSTM scores missing - comparing without it (dry run)")
        MODELS.remove("siamese_bilstm")
    df = df.merge(pd.read_csv(SBERT_DIR / "val_scores.csv"), on="id")
    lex = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "lexical_val.csv")[["id", "word_jaccard"]]
    df = df.merge(lex, on="id")
    assert len(df) == len(val), "score files do not align with the frozen validation split"
    return df


def error_breakdown(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df["is_duplicate"].to_numpy()
    jac = df["word_jaccard"].to_numpy()
    out = {}
    for name, lo, hi in (("jaccard<=0.2", -1, 0.2), ("0.2-0.4", 0.2, 0.4), ("0.4-0.6", 0.4, 0.6), (">0.6", 0.6, 1.1)):
        m = (jac > lo) & (jac <= hi)
        neg, pos = m & (y == 0), m & (y == 1)
        out[name] = {"fpr": float(pred[neg].mean()), "fnr": float(1 - pred[pos].mean()),
                     "negatives": int(neg.sum()), "positives": int(pos.sum())}
    hi_neg = (jac > 0.6) & (y == 0)
    mid_neg = (jac > 0.4) & (jac <= 0.6) & (y == 0)
    for tag in TAGS + ["any_constraint_diff"]:
        t = df[tag].to_numpy()
        out[f"fpr_{tag}_overlap>0.6"] = float(pred[hi_neg & t].mean())
        out[f"fpr_{tag}_overlap0.4-0.6"] = float(pred[mid_neg & t].mean())
    return out


def transitions(df, pred_base, pred_new, name, n_examples=8, seed=42) -> dict:
    """Pairs the new model fixes / breaks relative to the Phase 2 baseline, split by overlap."""
    y = df["is_duplicate"].to_numpy()
    base_ok, new_ok = pred_base == y, pred_new == y
    rng = np.random.default_rng(seed)
    res = {}
    for label, mask in (("fixed", ~base_ok & new_ok), ("broken", base_ok & ~new_ok)):
        sub = df[mask]
        res[label] = {"n": int(mask.sum()),
                      "false_negatives_among": int(((y == 1) & mask).sum()),
                      "false_positives_among": int(((y == 0) & mask).sum()),
                      "median_jaccard": float(sub["word_jaccard"].median()) if len(sub) else None}
        cols = ["question1", "question2", "is_duplicate", BASELINE, name]
        for cls, cls_name in ((1, "positives"), (0, "negatives")):
            part = sub[sub.is_duplicate == cls]
            k = min(n_examples, len(part))
            res[label][f"examples_{cls_name}"] = part.iloc[rng.choice(len(part), k, replace=False)][cols] \
                .round(3).to_dict("records") if k else []
    return res


def paired_bootstrap(y, s_base, t_base, s_new, t_new, n_boot=1000, seed=42) -> dict:
    """95% CI of (new - baseline) for F1 (each at its own frozen val threshold) and PR-AUC,
    resampling validation pairs with replacement. Thresholds are NOT re-tuned per resample."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    pb, pn = s_base >= t_base, s_new >= t_new

    def f1(y_, p_):
        tp = (p_ & (y_ == 1)).sum()
        return 2 * tp / (p_.sum() + (y_ == 1).sum())

    d_f1, d_pr = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d_f1.append(f1(y[i], pn[i]) - f1(y[i], pb[i]))
        d_pr.append(average_precision_score(y[i], s_new[i]) - average_precision_score(y[i], s_base[i]))
    ci = lambda d: [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]  # noqa: E731
    return {"delta_f1_mean": float(np.mean(d_f1)), "delta_f1_ci95": ci(d_f1),
            "delta_pr_auc_mean": float(np.mean(d_pr)), "delta_pr_auc_ci95": ci(d_pr),
            "p_new_f1_not_better": float(np.mean(np.array(d_f1) <= 0)), "n_boot": n_boot}


def confident_errors(df, name, thr, k=10) -> dict:
    y, s = df["is_duplicate"], df[name]
    cols = ["question1", "question2", name, "word_jaccard"]
    return {"false_positives": df[(y == 0) & (s >= thr)].nlargest(k, name)[cols].round(3).to_dict("records"),
            "false_negatives": df[(y == 1) & (s < thr)].nsmallest(k, name)[cols].round(3).to_dict("records")}


def latency_and_size(df: pd.DataFrame) -> dict:
    from src.models.predictors import BiLSTMPredictor, SBERTPredictor, TfidfLexicalPredictor

    configure_threads()
    rng = np.random.default_rng(0)
    sample = df.iloc[rng.choice(len(df), 2000, replace=False)]
    q1, q2 = sample["question1"].tolist(), sample["question2"].tolist()

    preds = {BASELINE: TfidfLexicalPredictor(), "siamese_bilstm": BiLSTMPredictor(),
             "sbert_cosine": SBERTPredictor("cosine"), "sbert_lr": SBERTPredictor("lr"),
             "sbert_mlp": SBERTPredictor("mlp")}
    # Encoder size = fp32 weights actually loaded (the HF repo also ships ONNX/OpenVINO copies).
    enc_mb = count_parameters(preds["sbert_cosine"].encoder) * 4 / 2**20
    out = {}
    for name, p in preds.items():
        p.predict(q1[:50], q2[:50])  # warm-up
        single = []
        for i in range(200):
            t = time.perf_counter()
            p.predict([q1[i]], [q2[i]])
            single.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        p.predict(q1, q2)
        batch_ms = (time.perf_counter() - t) * 1000 / len(q1)
        if name == BASELINE:
            size, params = dir_size_mb(p.path), int(p.model.coef_.size + 1)
        elif name == "siamese_bilstm":
            size, params = dir_size_mb(p.path), count_parameters(p.model)
        else:
            head = {"sbert_cosine": [], "sbert_lr": [SBERT_DIR / "lr.joblib"],
                    "sbert_mlp": [SBERT_DIR / "mlp.pt"]}[name]
            size = enc_mb + sum(dir_size_mb(h) for h in head)
            params = count_parameters(p.encoder) + (0 if name == "sbert_cosine" else
                                                   p.clf.coef_.size + 1 if name == "sbert_lr" else count_parameters(p.clf))
        out[name] = {"single_pair_ms_median": float(np.median(single)), "single_pair_ms_p95": float(np.percentile(single, 95)),
                     "batch_ms_per_pair": batch_ms, "size_mb": round(size, 1), "parameters": int(params)}
        print(f"{name:22s} single={out[name]['single_pair_ms_median']:.2f}ms batch={batch_ms:.3f}ms/pair "
              f"size={size:.1f}MB params={params:,}", flush=True)
    return out, preds


def probes(preds: dict, thresholds: dict) -> list[dict]:
    pr = pd.DataFrame(PROBES, columns=["type", "question1", "question2"])
    for name, p in preds.items():
        pr[name] = p.predict(pr.question1.tolist(), pr.question2.tolist()).round(3)
        pr[name + "_dup"] = pr[name] >= thresholds[name]
    return pr.to_dict("records")


def plots(df: pd.DataFrame, reports: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax, title, xl, yl):
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel(xl, color=INK2); ax.set_ylabel(yl, color=INK2)
        ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2); ax.set_facecolor(SURFACE)

    y = df["is_duplicate"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor=SURFACE)
    for name in MODELS:
        p, r, _ = precision_recall_curve(y, df[name])
        axes[0].plot(r, p, color=COLORS[name], linewidth=2, label=f"{name} (PR-AUC {reports[name]['val_at_tuned_threshold']['pr_auc']:.3f})")
        sw = threshold_sweep(y, df[name])
        axes[1].plot(sw.threshold, sw.f1, color=COLORS[name], linewidth=2, label=name)
        t = reports[name]["tuned_threshold"]
        axes[1].scatter([t], [sw.loc[(sw.threshold - t).abs().idxmin(), "f1"]], color=COLORS[name], s=40, zorder=3,
                        edgecolor=SURFACE, linewidth=1.5)
    axes[0].axhline(y.mean(), color=INK2, linestyle="--", linewidth=1)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.01)
    axes[1].axhline(0.730, color=INK2, linestyle=":", linewidth=1)
    axes[1].text(0.01, 0.705, "Phase 2 bar F1 = 0.730", color=INK2, fontsize=8, ha="left", va="top")
    axes[1].set_ylim(0, 0.9)
    style(axes[0], "Precision-recall (validation)", "recall", "precision")
    style(axes[1], "F1 vs threshold (dot = tuned threshold)", "threshold", "F1")
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=7.5, loc="lower left")
    axes[1].legend(frameon=False, labelcolor=INK2, fontsize=7.5, loc="lower center")
    fig.tight_layout(); fig.savefig(FIG / "phase3_pr_and_f1.png", dpi=130); plt.close(fig)

    # Error rates by overlap band, small multiples: FPR (left) and FNR (right)
    bands = ["jaccard<=0.2", "0.2-0.4", "0.4-0.6", ">0.6"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=SURFACE, sharey=True)
    x = np.arange(len(bands)); w = 0.8 / len(MODELS)
    for k, name in enumerate(MODELS):
        eb = reports[name]["errors_by_overlap"]
        for ax, key in zip(axes, ("fpr", "fnr")):
            ax.bar(x + (k - (len(MODELS) - 1) / 2) * w, [eb[b][key] for b in bands], w * 0.92,
                   color=COLORS[name], label=name)
    for ax, title in zip(axes, ("False-positive rate among negatives", "False-negative rate among positives")):
        ax.set_xticks(x, ["≤0.2", "0.2-0.4", "0.4-0.6", ">0.6"])
        style(ax, title, "word Jaccard of the pair", "")
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "phase3_errors_by_overlap.png", dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()

    df = load_scores()
    tags = pd.DataFrame([error_tags(a, b) for a, b in zip(df.question1, df.question2)], index=df.index)
    df = pd.concat([df, tags[TAGS]], axis=1)
    df["any_constraint_diff"] = df[TAGS].any(axis=1)
    y = df["is_duplicate"].to_numpy()

    reports, preds_bin = {}, {}
    for name in MODELS:
        rep = validation_report(y, df[name].to_numpy(), is_probability=name != "sbert_cosine")
        pred = (df[name].to_numpy() >= rep["tuned_threshold"]).astype(int)
        rep["errors_by_overlap"] = error_breakdown(df, pred)
        rep["confident_errors"] = confident_errors(df, name, rep["tuned_threshold"])
        reports[name], preds_bin[name] = rep, pred
        m = rep["val_at_tuned_threshold"]
        print(f"{name:22s} thr={rep['tuned_threshold']:.2f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f} beats_0.730={m['f1'] > 0.730}", flush=True)

    trans = {n: transitions(df, preds_bin[BASELINE], preds_bin[n], n) for n in MODELS if n != BASELINE}
    boot = {}
    for n in MODELS:
        if n == BASELINE:
            continue
        boot[n] = paired_bootstrap(y, df[BASELINE].to_numpy(), reports[BASELINE]["tuned_threshold"],
                                   df[n].to_numpy(), reports[n]["tuned_threshold"])
        b = boot[n]
        print(f"bootstrap {n:16s} dF1={b['delta_f1_mean']:+.4f} CI{[round(x, 4) for x in b['delta_f1_ci95']]} "
              f"dPR={b['delta_pr_auc_mean']:+.4f} CI{[round(x, 4) for x in b['delta_pr_auc_ci95']]}", flush=True)
    thresholds = {n: reports[n]["tuned_threshold"] for n in MODELS}
    # Justifies the MLP head over the simpler LR head (same encoder, same features).
    boot_mlp_lr = paired_bootstrap(y, df["sbert_lr"].to_numpy(), reports["sbert_lr"]["tuned_threshold"],
                                   df["sbert_mlp"].to_numpy(), reports["sbert_mlp"]["tuned_threshold"])
    out = {"models": reports, "vs_phase2_baseline": trans, "bootstrap_vs_phase2_baseline": boot,
           "bootstrap_sbert_mlp_vs_sbert_lr": boot_mlp_lr, "phase2_bar": {"f1": 0.730, "pr_auc": 0.771}}
    if not args.skip_latency:
        out["latency_size"], predictors = latency_and_size(df)
        out["sanity_probes"] = probes(predictors, thresholds)
    (ART / "phase3" / "comparison.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    plots(df, reports)
    for n in MODELS:
        eb = reports[n]["errors_by_overlap"]
        print(f"{n:22s} FNR(j<=0.2)={eb['jaccard<=0.2']['fnr']:.3f} FPR(j>0.6)={eb['>0.6']['fpr']:.3f} "
              f"FPR ent>0.6={eb['fpr_entity_diff_overlap>0.6']:.3f} num>0.6={eb['fpr_number_diff_overlap>0.6']:.3f} "
              f"neg>0.6={eb['fpr_negation_diff_overlap>0.6']:.3f}")


if __name__ == "__main__":
    main()
