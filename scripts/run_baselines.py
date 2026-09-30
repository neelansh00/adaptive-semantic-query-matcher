"""Phase 2: lexical and shallow-ML baselines, evaluated on VALIDATION only.

Usage:  python scripts/run_baselines.py
Reads:  data/processed/splits/{train,val}.csv      (test.csv is never opened)
Writes: artifacts/phase2/{baseline_metrics.json, threshold_sweeps.csv, val_predictions.csv,
                          error_analysis.json, *.joblib}
        data/processed/features/lexical_{train,val}.csv
        docs/figures/phase2_*.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.error_analysis import tag_error_rates, tag_table  # noqa: E402
from src.evaluation.metrics import (best_f1_threshold, calibration_metrics,  # noqa: E402
                                    classification_metrics, threshold_sweep)
from src.features.lexical import FEATURE_NAMES, lexical_features  # noqa: E402
from src.preprocessing.text import normalize, tokenize  # noqa: E402
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402

SEED = 42
ART = PROJECT_ROOT / "artifacts" / "phase2"
FIG = PROJECT_ROOT / "docs" / "figures"
FEAT = PROJECT_ROOT / "data" / "processed" / "features"
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
TFIDF_PARAMS = dict(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                    min_df=2, sublinear_tf=True, dtype=np.float32)

# Reference palette (categorical slots 1-4, fixed order) + neutral inks.
COLORS = {"cosine_tfidf": "#2a78d6", "lr_lexical": "#eb6834", "lr_tfidf_pair": "#1baf7a",
          "lr_tfidf_pair_lexical": "#eda100"}
# Hand-written probes (not from any split) to expose behaviour on the project's mismatch types.
PROBES = [
    ("identical", "What is formwork?", "What is formwork?"),
    ("identical", "Who founded Microsoft?", "Who founded Microsoft?"),
    ("paraphrase", "How can I lose weight quickly?", "What is the fastest way to lose weight?"),
    ("entity_mismatch", "Who founded Microsoft?", "Who founded Apple?"),
    ("number_mismatch", "How can I lose 5 kg in a month?", "How can I lose 20 kg in a month?"),
    ("negation_mismatch", "Why do people believe in God?", "Why do people not believe in God?"),
    ("broader_narrower", "How do I learn programming?", "How do I learn Python programming for data science?"),
]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def load_split(name: str) -> pd.DataFrame:
    assert name in ("train", "val"), "Phase 2 must not read the test split"
    path = resolve(load_config()["split_dir"]) / f"{name}.csv"
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def cached_lexical(df: pd.DataFrame, name: str) -> pd.DataFrame:
    FEAT.mkdir(parents=True, exist_ok=True)
    path = FEAT / f"lexical_{name}.csv"
    if path.exists():
        cached = pd.read_csv(path)
        if len(cached) == len(df) and (cached["id"].to_numpy() == df["id"].to_numpy()).all():
            return cached[FEATURE_NAMES]
    feats = lexical_features(df)
    feats.insert(0, "id", df["id"].to_numpy())
    feats.to_csv(path, index=False)
    return feats[FEATURE_NAMES]


def pair_matrix(v1: sp.csr_matrix, v2: sp.csr_matrix) -> sp.csr_matrix:
    """Order-invariant pair representation: [|v1 - v2|, v1 * v2]."""
    return sp.hstack([abs(v1 - v2), v1.multiply(v2)], format="csr")


def rowwise_cosine(v1, v2) -> np.ndarray:
    # TF-IDF rows are already L2-normalised, so the dot product is the cosine.
    return np.asarray(v1.multiply(v2).sum(axis=1)).ravel()


def fit_lr(X_tr, y_tr, X_va, y_va, name):
    """Pick C on validation PR-AUC (threshold-free), return best model and the grid."""
    grid, best = [], None
    for C in C_GRID:
        t = time.time()
        m = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        res = classification_metrics(y_va, p, 0.5)
        grid.append({"C": C, "val_pr_auc": res["pr_auc"], "val_roc_auc": res["roc_auc"],
                     "fit_seconds": round(time.time() - t, 1)})
        print(f"  [{name}] C={C}: PR-AUC={res['pr_auc']:.4f} ({grid[-1]['fit_seconds']}s)")
        if best is None or res["pr_auc"] > best[1]:
            best = (m, res["pr_auc"], p, C)
    return best[0], best[2], best[3], grid


def threshold_transfer(y, s, seed=SEED) -> dict:
    """Optimism check: tune the threshold on one random half of val, evaluate on the other."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    out = []
    for tune, ev in ((a, b), (b, a)):
        t = best_f1_threshold(y[tune], s[tune])
        out.append(classification_metrics(y[ev], s[ev], t)["f1"])
    return {"cross_half_f1_mean": float(np.mean(out)), "cross_half_f1": [float(x) for x in out]}


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.grid(color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2); ax.set_facecolor(SURFACE)


def make_plots(y, scores: dict, best: str, thresholds: dict, prevalence: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PR curves
    fig, ax = plt.subplots(figsize=(6.5, 4.6), facecolor=SURFACE)
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=COLORS[name], linewidth=2, label=name)
    ax.axhline(prevalence, color=INK2, linestyle="--", linewidth=1)
    ax.text(0.01, prevalence + 0.015, f"majority / chance = {prevalence:.2f}", color=INK2, fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    style(ax, "Precision-recall curves (validation)", "recall", "precision")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "phase2_pr_curves.png", dpi=130); plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 4.6), facecolor=SURFACE)
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2, label=name)
    ax.plot([0, 1], [0, 1], color=INK2, linestyle="--", linewidth=1)
    style(ax, "ROC curves (validation)", "false-positive rate", "true-positive rate")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "phase2_roc_curves.png", dpi=130); plt.close(fig)

    # Threshold sweeps: small multiples (cosine baseline vs best model), one y-scale each
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE, sharey=True)
    for ax, name in zip(axes, ["cosine_tfidf", best]):
        sw = threshold_sweep(y, scores[name])
        ax.plot(sw.threshold, sw.precision, color="#2a78d6", linewidth=2, label="precision")
        ax.plot(sw.threshold, sw.recall, color="#eb6834", linewidth=2, label="recall")
        ax.plot(sw.threshold, sw.f1, color=INK, linewidth=2, label="F1")
        t = thresholds[name]
        ax.axvline(t, color=INK2, linestyle=":", linewidth=1)
        ax.text(t + 0.01, 0.05, f"best F1 @ {t:.2f}", color=INK2, fontsize=8)
        style(ax, name, "threshold", "score" if name == "cosine_tfidf" else "")
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper right")
    fig.suptitle("Precision / recall / F1 vs threshold (validation)", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "phase2_threshold_sweep.png", dpi=130); plt.close(fig)

    # Confusion matrix of best model at its tuned threshold
    cm = confusion_matrix(y, (scores[best] >= thresholds[best]).astype(int))
    fig, ax = plt.subplots(figsize=(4.4, 3.8), facecolor=SURFACE)
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n({v / cm.sum():.1%})", ha="center", va="center",
                color="white" if v > cm.max() * 0.5 else INK, fontsize=9)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(f"{best} @ {thresholds[best]:.2f} (val)", loc="left", color=INK, fontsize=10)
    ax.tick_params(colors=INK2)
    fig.tight_layout(); fig.savefig(FIG / "phase2_confusion_matrix.png", dpi=130); plt.close(fig)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    train, val = load_split("train"), load_split("val")
    y_tr, y_va = train["is_duplicate"].to_numpy(), val["is_duplicate"].to_numpy()
    print(f"train={len(train)} val={len(val)}")

    # ---- preprocessing
    t = time.time()
    norm = {s: (d["question1"].map(normalize), d["question2"].map(normalize)) for s, d in
            (("train", train), ("val", val))}
    print(f"normalised in {time.time() - t:.1f}s")

    # ---- lexical features
    t = time.time()
    lex_tr, lex_va = cached_lexical(train, "train"), cached_lexical(val, "val")
    print(f"lexical features in {time.time() - t:.1f}s")

    # ---- TF-IDF fit on TRAIN questions only (each distinct question once)
    t = time.time()
    train_questions = pd.unique(pd.concat(norm["train"]))
    tfidf = TfidfVectorizer(**TFIDF_PARAMS).fit(train_questions)
    V = {s: (tfidf.transform(a), tfidf.transform(b)) for s, (a, b) in norm.items()}
    print(f"tfidf vocab={len(tfidf.vocabulary_)} in {time.time() - t:.1f}s")

    results, scores, extra = {}, {}, {}

    # ---- 0. majority baseline
    prevalence = float(y_tr.mean())
    maj = classification_metrics(y_va, np.full(len(y_va), prevalence), 0.5)
    results["majority"] = {"val": maj, "notes": "always predicts the train majority class (0)"}

    # ---- 1. cosine similarity (unsupervised score, threshold tuned on val)
    cos_va = rowwise_cosine(*V["val"])
    scores["cosine_tfidf"] = cos_va

    # ---- 2. LR on lexical features (interpretable)
    scaler = StandardScaler().fit(lex_tr)
    Xl_tr, Xl_va = scaler.transform(lex_tr), scaler.transform(lex_va)
    print("lr_lexical")
    lr_lex, scores["lr_lexical"], c_lex, grid_lex = fit_lr(Xl_tr, y_tr, Xl_va, y_va, "lr_lexical")
    extra["lr_lexical"] = {"C": c_lex, "c_grid": grid_lex,
                           "coefficients_standardised": dict(zip(FEATURE_NAMES, lr_lex.coef_[0].round(4).tolist()))}

    # ---- 3. LR on TF-IDF pair representation
    Xp_tr, Xp_va = pair_matrix(*V["train"]), pair_matrix(*V["val"])
    print("lr_tfidf_pair")
    lr_tf, scores["lr_tfidf_pair"], c_tf, grid_tf = fit_lr(Xp_tr, y_tr, Xp_va, y_va, "lr_tfidf_pair")
    extra["lr_tfidf_pair"] = {"C": c_tf, "c_grid": grid_tf, "n_features": Xp_tr.shape[1]}

    # ---- 4. LR on TF-IDF pair + lexical features + cosine
    cos_tr = rowwise_cosine(*V["train"])
    cscale = StandardScaler().fit(cos_tr[:, None])
    Xc_tr = sp.hstack([Xp_tr, sp.csr_matrix(Xl_tr), sp.csr_matrix(cscale.transform(cos_tr[:, None]))], format="csr")
    Xc_va = sp.hstack([Xp_va, sp.csr_matrix(Xl_va), sp.csr_matrix(cscale.transform(cos_va[:, None]))], format="csr")
    print("lr_tfidf_pair_lexical")
    lr_all, scores["lr_tfidf_pair_lexical"], c_all, grid_all = fit_lr(Xc_tr, y_tr, Xc_va, y_va, "lr_tfidf_pair_lexical")
    extra["lr_tfidf_pair_lexical"] = {"C": c_all, "c_grid": grid_all, "n_features": Xc_tr.shape[1]}

    # ---- evaluation (validation)
    thresholds, sweeps = {}, []
    for name, s in scores.items():
        thr = best_f1_threshold(y_va, s)
        thresholds[name] = thr
        entry = {"val_at_0.5": classification_metrics(y_va, s, 0.5),
                 "val_at_tuned_threshold": classification_metrics(y_va, s, thr),
                 "tuned_threshold": thr,
                 "threshold_transfer": threshold_transfer(y_va, s)}
        if name != "cosine_tfidf":
            entry["calibration"] = calibration_metrics(y_va, s)
        entry.update(extra.get(name, {}))
        results[name] = entry
        sweeps.append(threshold_sweep(y_va, s).assign(model=name))
        m = entry["val_at_tuned_threshold"]
        print(f"{name:24s} thr={thr:.2f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f}")

    best = max((n for n in scores), key=lambda n: results[n]["val_at_tuned_threshold"]["pr_auc"])
    results["best_model_by_val_pr_auc"] = best
    results["config"] = {"tfidf": {k: str(v) for k, v in TFIDF_PARAMS.items()},
                         "tfidf_vocab_size": len(tfidf.vocabulary_), "c_grid": C_GRID, "seed": SEED}

    # ---- latency (per pair, batch of full val set, single core python)
    t = time.time()
    a, b = tfidf.transform(norm["val"][0]), tfidf.transform(norm["val"][1])
    lr_tf.predict_proba(pair_matrix(a, b))
    results["lr_tfidf_pair"]["latency_ms_per_pair_batch"] = 1000 * (time.time() - t) / len(val)

    # ---- sanity probes
    pr = pd.DataFrame(PROBES, columns=["type", "question1", "question2"])
    pa, pb = tfidf.transform(pr.question1.map(normalize)), tfidf.transform(pr.question2.map(normalize))
    pl = scaler.transform(lexical_features(pr))
    pc = rowwise_cosine(pa, pb)
    pr["cosine_tfidf"] = pc
    pr["lr_lexical"] = lr_lex.predict_proba(pl)[:, 1]
    pr["lr_tfidf_pair"] = lr_tf.predict_proba(pair_matrix(pa, pb))[:, 1]
    pr["lr_tfidf_pair_lexical"] = lr_all.predict_proba(sp.hstack(
        [pair_matrix(pa, pb), sp.csr_matrix(pl), sp.csr_matrix(cscale.transform(pc[:, None]))], format="csr"))[:, 1]
    results["sanity_probes"] = pr.round(4).to_dict("records")
    print(pr.drop(columns=["question2"]).round(3).to_string())

    # ---- error analysis of best model
    y_pred = (scores[best] >= thresholds[best]).astype(int)
    tags, summary = tag_table(val, y_pred)
    val_out = val[["id", "question1", "question2", "is_duplicate"]].copy()
    for name, s in scores.items():
        val_out[name] = s
    lexical_view = val_out.join(lex_va[["word_jaccard"]])
    jac_bins = pd.cut(lexical_view["word_jaccard"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
    by_jac = pd.DataFrame({"y": y_va, "pred": y_pred}).groupby(jac_bins, observed=True).apply(
        lambda g: pd.Series({"n": len(g), "prevalence": g.y.mean(),
                             "fpr": ((g.pred == 1) & (g.y == 0)).sum() / max((g.y == 0).sum(), 1),
                             "fnr": ((g.pred == 0) & (g.y == 1)).sum() / max((g.y == 1).sum(), 1)}),
        include_groups=False)
    s_best = scores[best]
    fp = val_out[(y_pred == 1) & (y_va == 0)].assign(score=s_best[(y_pred == 1) & (y_va == 0)])
    fn = val_out[(y_pred == 0) & (y_va == 1)].assign(score=s_best[(y_pred == 0) & (y_va == 1)])
    cols = ["id", "question1", "question2", "score"]
    rng = np.random.default_rng(SEED)
    err = {
        "model": best, "threshold": thresholds[best],
        "tag_share_by_outcome": summary.round(4).to_dict(),
        "tag_conditional_error_rates": tag_error_rates(tags).round(4).to_dict("index"),
        "error_rates_by_word_jaccard": by_jac.round(4).reset_index().astype({"word_jaccard": str}).to_dict("records"),
        "most_confident_false_positives": fp.nlargest(15, "score")[cols].to_dict("records"),
        "most_confident_false_negatives": fn.nsmallest(15, "score")[cols].to_dict("records"),
        "random_false_positives": fp.iloc[rng.choice(len(fp), 20, replace=False)][cols].to_dict("records"),
        "random_false_negatives": fn.iloc[rng.choice(len(fn), 20, replace=False)][cols].to_dict("records"),
    }

    # ---- save
    (ART / "baseline_metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    (ART / "error_analysis.json").write_text(json.dumps(err, indent=2, default=str), encoding="utf-8")
    pd.concat(sweeps).to_csv(ART / "threshold_sweeps.csv", index=False)
    val_out.drop(columns=["question1", "question2"]).to_csv(ART / "val_predictions.csv", index=False)
    joblib.dump({"tfidf": tfidf, "model": lr_tf, "threshold": thresholds["lr_tfidf_pair"]},
                ART / "lr_tfidf_pair.joblib", compress=3)
    joblib.dump({"scaler": scaler, "model": lr_lex, "features": FEATURE_NAMES,
                 "threshold": thresholds["lr_lexical"]}, ART / "lr_lexical.joblib")
    joblib.dump({"tfidf": tfidf, "lex_scaler": scaler, "cos_scaler": cscale, "model": lr_all,
                 "threshold": thresholds["lr_tfidf_pair_lexical"]}, ART / "lr_tfidf_pair_lexical.joblib", compress=3)

    make_plots(y_va, scores, best, thresholds, float(y_va.mean()))
    print("best:", best)
    print(tag_error_rates(tags).round(3))
    print(by_jac.round(3))


if __name__ == "__main__":
    main()
