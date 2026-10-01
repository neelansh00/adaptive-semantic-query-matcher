"""Phase 8b: the single, frozen evaluation on the TEST split.

Before reading test it (1) verifies every file hash in artifacts/phase8/freeze_manifest.json, (2) verifies test.csv
is byte-identical to the Phase 1 frozen split, and (3) refuses to run twice (artifacts/phase8/TEST_EVALUATED.json).

Scoring path (identical for every split; nothing is fitted here):
  questions -> MiniLM embeddings (float16 cache, as for train/val) -> Phase 3 MLP head -> base probability (B)
  questions -> deterministic spaCy-free constraint annotations -> pair features -> + logit(B) -> Phase 6 meta-model (C)
  pair embeddings -> frozen Phase 4 centroids (pair-average rule) -> cluster
  B, C -> frozen Phase 7 global thresholds (0.32 / 0.38) -> decisions
Reference systems (Phase 2 TF-IDF+lexical LR, Phase 3 BiLSTM, MiniLM cosine) are scored with their own frozen
validation thresholds, for context only.

Usage:
  python scripts/final_evaluation.py --dry-run-on-val   # same pipeline on VALIDATION; must reproduce known numbers
  python scripts/final_evaluation.py                    # the one-time test evaluation
Writes: artifacts/phase8/{test_results.json, test_scores.csv, error_samples.csv, TEST_EVALUATED.json},
        artifacts/phase8/dry_run_val/*, docs/figures/phase8_*.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_sbert import PairMLP, mlp_proba  # noqa: E402
from src.calibration.thresholds import ThresholdPolicy, decision_metrics  # noqa: E402
from src.clustering.core import CentroidModel  # noqa: E402
from src.clustering.pairs import assign_pairs  # noqa: E402
from src.evaluation.cluster_analysis import cluster_table  # noqa: E402
from src.evaluation.error_analysis import error_tags  # noqa: E402
from src.evaluation.metrics import calibration_metrics, classification_metrics  # noqa: E402
from src.features.constraints import annotate, pair_frame  # noqa: E402
from src.features.lexical import lexical_features  # noqa: E402
from src.models.meta import base_logit  # noqa: E402
from src.models.sentence_encoder import EmbeddingCache, clean, embedding_pair_features  # noqa: E402
from src.utils.data import PROJECT_ROOT, content_sha256, file_sha256, identity_key, load_config, resolve  # noqa: E402

ART = PROJECT_ROOT / "artifacts"
OUT = ART / "phase8"
FIG = PROJECT_ROOT / "docs" / "figures"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SEED, N_BOOT = 42, 1000
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
REFERENCE_THRESHOLDS = {  # frozen validation thresholds of the reference systems (Phases 2-3)
    "ref_tfidf_lexical": ("phase2/baseline_metrics.json", ["lr_tfidf_pair_lexical", "tuned_threshold"]),
    "ref_bilstm": ("phase3/bilstm/metrics.json", ["tuned_threshold"]),
    "ref_cosine": ("phase3/sbert_all-MiniLM-L6-v2/metrics.json", ["sbert_cosine", "tuned_threshold"]),
}
# Known validation results that the dry run must reproduce (Phases 2, 3, 6, 7).
EXPECTED_VAL_F1 = {"baseline": 0.7711, "final": 0.7829, "ref_tfidf_lexical": 0.7299, "ref_bilstm": 0.7323, "ref_cosine": 0.7347}


def verify_manifest() -> dict:
    manifest = json.loads((OUT / "freeze_manifest.json").read_text())
    bad = [k for k, v in manifest["files"].items() if content_sha256(PROJECT_ROOT / v["path"]) != v["sha256"]]
    if bad:
        raise SystemExit(f"FROZEN ARTIFACTS CHANGED since the manifest: {bad}")
    from huggingface_hub import hf_hub_download
    enc = file_sha256(hf_hub_download(ENCODER, "model.safetensors"))
    if enc != manifest["encoder"]["model.safetensors_sha256"]:
        raise SystemExit("Encoder weights differ from the frozen manifest")
    print(f"[freeze] {len(manifest['files'])} files + encoder verified against manifest from {manifest['frozen_at_utc']}", flush=True)
    return manifest


def load_eval_split(name: str) -> pd.DataFrame:
    split_dir = resolve(load_config()["split_dir"])
    df = pd.read_csv(split_dir / f"{name}.csv", keep_default_na=False, na_values=[""])
    meta = json.loads((split_dir / "split_metadata.json").read_text())
    assert file_sha256(split_dir / f"{name}.csv") == meta["split_file_sha256"][name], f"{name}.csv was modified"
    return df


def leakage_check(df: pd.DataFrame, name: str) -> dict:
    split_dir = resolve(load_config()["split_dir"])
    qs = lambda d: set(d.question1.map(identity_key)) | set(d.question2.map(identity_key))  # noqa: E731
    target = qs(df)
    out = {}
    # the validation dry run must never open test.csv, so it is checked against train only
    for other in ("train",) if name == "val" else ("train", "val"):
        o = pd.read_csv(split_dir / f"{other}.csv", usecols=["question1", "question2"], keep_default_na=False, na_values=[""])
        out[f"questions_shared_with_{other}"] = len(target & qs(o))
    assert all(v == 0 for v in out.values()), f"question overlap detected: {out}"
    return out


def score_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Every system's score for every pair. Nothing is fitted."""
    cache = EmbeddingCache(ENCODER, name)
    if not cache.exists():
        print(f"[encode] {name}: {cache.build(pd.concat([df.question1, df.question2]))}", flush=True)
    u, v = cache.pair_embeddings(df)
    X = embedding_pair_features(u, v)
    head_dir = ART / "phase3" / "sbert_all-MiniLM-L6-v2"
    cfg = json.loads((head_dir / "mlp_config.json").read_text())
    head = PairMLP(cfg["in_dim"], cfg["hidden"], cfg["dropout"])
    head.load_state_dict(torch.load(head_dir / "mlp.pt", map_location="cpu", weights_only=True))
    base = mlp_proba(head, X, torch.device("cpu"))

    t = time.time()
    texts = pd.unique(pd.concat([df.question1, df.question2]).map(clean))
    annotations = {q: annotate(q, None) for q in texts}  # spaCy-free, as the chosen meta-model requires
    feats = pair_frame(df, annotations, use_spacy=False)
    feats["base_logit"] = base_logit(base)
    bundle = __import__("joblib").load(ART / "phase6" / "meta_model.joblib")
    assert not bundle["uses_spacy"]
    meta = bundle["model"].predict_proba(feats)
    print(f"[score] constraint features + meta-model in {time.time() - t:.0f}s", flush=True)

    clusters = assign_pairs(u, v, CentroidModel.load(ART / "phase4" / "cluster_model"), "pair_avg")
    from src.models.predictors import BiLSTMPredictor, TfidfLexicalPredictor
    ref_tfidf = TfidfLexicalPredictor().predict(df.question1.tolist(), df.question2.tolist())
    ref_bilstm = BiLSTMPredictor().predict(df.question1.tolist(), df.question2.tolist())
    return pd.DataFrame({"id": df["id"].to_numpy(), "is_duplicate": df["is_duplicate"].to_numpy(), "cluster": clusters,
                         "baseline": base, "final": meta, "ref_tfidf_lexical": ref_tfidf, "ref_bilstm": ref_bilstm,
                         "ref_cosine": X[:, -1]})


def thresholds() -> dict:
    thr = {"baseline": ThresholdPolicy.load(ART / "phase7" / "frozen_policy_baseline.json"),
           "final": ThresholdPolicy.load(ART / "phase7" / "frozen_policy_final.json")}
    out = {k: p.global_threshold for k, p in thr.items()}
    assert all(p.mode == "global" for p in thr.values())
    for name, (path, keys) in REFERENCE_THRESHOLDS.items():
        d = json.loads((ART / path).read_text())
        for k in keys:
            d = d[k]
        out[name] = float(d)
    return out


def evaluate(scores: pd.DataFrame, thr: dict) -> dict:
    y, cl = scores["is_duplicate"].to_numpy(), scores["cluster"].to_numpy()
    res = {}
    for name, t in thr.items():
        s = scores[name].to_numpy()
        m = classification_metrics(y, s, t)
        tab = cluster_table(y, s, cl, t)
        res[name] = {"threshold": t, **{k: m[k] for k in ("f1", "precision", "recall", "accuracy", "roc_auc", "pr_auc",
                                                         "fpr", "fnr", "tp", "fp", "tn", "fn")},
                     "macro_cluster_f1": float(tab["f1"].mean()), "worst_cluster_f1": float(tab["f1"].min()),
                     "worst_cluster": int(tab["f1"].idxmin()), "best_cluster_f1": float(tab["f1"].max()),
                     "per_cluster": tab[["n", "prevalence", "precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc"]]
                     .round(4).to_dict("index")}
        if name in ("baseline", "final", "ref_bilstm", "ref_tfidf_lexical"):
            res[name]["calibration"] = calibration_metrics(y, s)
    return res


def paired_bootstrap(scores: pd.DataFrame, thr: dict, a: str = "final", b: str = "baseline") -> dict:
    y, cl = scores["is_duplicate"].to_numpy(), scores["cluster"].to_numpy()
    pa, pb = (scores[a] >= thr[a]).to_numpy().astype(int), (scores[b] >= thr[b]).to_numpy().astype(int)
    sa, sb = scores[a].to_numpy(), scores[b].to_numpy()
    rng = np.random.default_rng(SEED)
    keys = ["f1", "precision", "recall", "macro_cluster_f1", "worst_cluster_f1", "fpr", "fnr"]
    d = {k: [] for k in keys + ["pr_auc"]}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        ma, mb = decision_metrics(y[i], pa[i], cl[i]), decision_metrics(y[i], pb[i], cl[i])
        for k in keys:
            d[k].append(ma[k] - mb[k])
        d["pr_auc"].append(average_precision_score(y[i], sa[i]) - average_precision_score(y[i], sb[i]))
    return {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in d.items()}


def slices_and_transitions(df: pd.DataFrame, scores: pd.DataFrame, thr: dict) -> tuple[dict, dict]:
    tags = pd.DataFrame([error_tags(a, b) for a, b in zip(df.question1, df.question2)])
    jac = lexical_features(df)["word_jaccard"].to_numpy()
    y = scores["is_duplicate"].to_numpy()
    pred = {k: (scores[k] >= thr[k]).to_numpy() for k in ("baseline", "final")}
    hi = jac > 0.6
    slices = {"all": np.ones(len(y), bool), "high_overlap(j>0.6)": hi, "low_overlap(j<=0.2)": jac <= 0.2}
    for t in ("entity_diff", "number_diff", "negation_diff"):
        slices[f"{t}&high_overlap"] = tags[t].to_numpy() & hi
    out = {}
    for sn, m in slices.items():
        neg, pos = m & (y == 0), m & (y == 1)
        out[sn] = {"negatives": int(neg.sum()), "positives": int(pos.sum()),
                   **{f"fpr_{k}": float(p[neg].mean()) for k, p in pred.items()},
                   **{f"fnr_{k}": float(1 - p[pos].mean()) for k, p in pred.items()}}
    b, f = pred["baseline"], pred["final"]
    trans = {"fp_fixed": int(((y == 0) & b & ~f).sum()), "fn_fixed": int(((y == 1) & ~b & f).sum()),
             "new_fp": int(((y == 0) & ~b & f).sum()), "new_fn": int(((y == 1) & b & ~f).sum())}
    return out, trans


def plots(scores: pd.DataFrame, res: dict, names: dict, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = scores["is_duplicate"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.4, 1, 1]})
    for name, color in (("ref_tfidf_lexical", "#c9c8c1"), ("ref_cosine", "#1baf7a"), ("baseline", "#2a78d6"),
                        ("final", "#eb6834")):
        p, r, _ = precision_recall_curve(y, scores[name])
        axes[0].plot(r, p, color=color, linewidth=2, label=f"{name} (PR-AUC {res[name]['pr_auc']:.3f})")
        t = res[name]["threshold"]
        axes[0].scatter([res[name]["recall"]], [res[name]["precision"]], color=color, s=40, zorder=3, edgecolor=SURFACE)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.01)
    axes[0].set_title(f"Precision-recall on {tag} (dot = frozen threshold)", loc="left", color=INK, fontsize=10)
    axes[0].set_xlabel("recall", color=INK2); axes[0].set_ylabel("precision", color=INK2)
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left")
    axes[0].grid(color=GRID, linewidth=0.8)
    for ax, name in zip(axes[1:], ("baseline", "final")):
        r = res[name]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}\n({v / cm.sum():.1%})", ha="center", va="center",
                    color="white" if v > cm.max() * 0.5 else INK, fontsize=9)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"]); ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{name} @ τ={r['threshold']:.2f}", loc="left", color=INK, fontsize=10)
    for ax in axes:
        ax.tick_params(colors=INK2); ax.set_facecolor(SURFACE)
    fig.tight_layout(); fig.savefig(FIG / f"phase8_{tag}_pr_confusion.png", dpi=130); plt.close(fig)

    b = pd.DataFrame(res["baseline"]["per_cluster"]).T["f1"]
    f = pd.DataFrame(res["final"]["per_cluster"]).T["f1"]
    order = b.sort_values().index
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    yy = np.arange(len(order))
    ax.hlines(yy, b[order], f[order], color=GRID, linewidth=2)
    ax.scatter(b[order], yy, color="#2a78d6", s=36, zorder=3, label="baseline", edgecolor=SURFACE)
    ax.scatter(f[order], yy, color="#eb6834", s=36, zorder=3, label="final", edgecolor=SURFACE)
    ax.set_yticks(yy, [f"C{c} {names[int(c)][:28]}" for c in order], fontsize=8.5)
    ax.set_title(f"Per-cluster F1 on {tag}: baseline -> final", loc="left", color=INK, fontsize=10)
    ax.set_xlabel("F1", color=INK2); ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2); ax.set_facecolor(SURFACE)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / f"phase8_{tag}_per_cluster.png", dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-on-val", action="store_true", help="run the identical pipeline on validation")
    args = ap.parse_args()
    split = "val" if args.dry_run_on_val else "test"
    out_dir = OUT / "dry_run_val" if args.dry_run_on_val else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = OUT / "TEST_EVALUATED.json"
    if split == "test" and marker.exists():
        raise SystemExit(f"The test split was already evaluated ({json.loads(marker.read_text())['evaluated_at_utc']}). "
                         "Phase 8 evaluates exactly once.")

    manifest = verify_manifest()
    df = load_eval_split(split)
    leakage = leakage_check(df, split)
    print(f"[{split}] {len(df)} pairs; leakage check {leakage}", flush=True)
    if split == "test":
        marker.write_text(json.dumps({"evaluated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                      "manifest_frozen_at_utc": manifest["frozen_at_utc"],
                                      "manifest_git_commit": manifest["git_commit"]}, indent=2))

    thr = thresholds()
    scores = score_split(df, split)
    res = evaluate(scores, thr)
    boot = paired_bootstrap(scores, thr)
    slices, trans = slices_and_transitions(df, scores, thr)
    names = {int(c): v["name"] for c, v in json.loads((ART / "phase4" / "cluster_names.json").read_text()).items()
             if not c.startswith("_")}

    for name, r in res.items():
        print(f"{name:18s} tau={r['threshold']:.2f} F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
              f"ROC={r['roc_auc']:.4f} PR={r['pr_auc']:.4f} macro={r['macro_cluster_f1']:.4f} "
              f"worst={r['worst_cluster_f1']:.4f}(C{r['worst_cluster']})", flush=True)
    print("final - baseline:", {k: f"{v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in boot.items()}, flush=True)
    print("transitions:", trans, flush=True)

    if split == "val":
        check = {k: (round(res[k]["f1"], 4), v) for k, v in EXPECTED_VAL_F1.items()}
        ok = all(abs(a - b) <= 0.0002 for a, b in check.values())
        print("DRY RUN reproduces known validation F1:", ok, check, flush=True)
        stored = pd.read_csv(ART / "phase6" / "val_scores.csv")[["id", "B_base", "C_no_spacy_hgb"]].merge(scores, on="id")
        diff = {"baseline_max_abs": float((stored.B_base - stored.baseline).abs().max()),
                "final_max_abs": float((stored.C_no_spacy_hgb - stored.final).abs().max())}
        print("score parity vs stored validation scores:", diff, flush=True)
        (out_dir / "dry_run_check.json").write_text(json.dumps({"f1_check": check, "reproduced": ok, "score_parity": diff}, indent=2))
        if not ok:
            raise SystemExit("Dry run did not reproduce validation results: do NOT evaluate on test")

    scores.round(6).to_csv(out_dir / f"{split}_scores.csv", index=False)
    rng = np.random.default_rng(SEED)
    fin = scores["final"] >= thr["final"]
    y = scores["is_duplicate"] == 1
    sample = []
    for kind, mask in (("FP", fin & ~y), ("FN", ~fin & y)):
        idx = rng.choice(np.flatnonzero(mask.to_numpy()), 30, replace=False)
        for i in idx:
            sample.append({"id": int(df.id.iloc[i]), "outcome": kind, "cluster": int(scores.cluster.iloc[i]),
                           "final_score": round(float(scores.final.iloc[i]), 3), "baseline_score": round(float(scores.baseline.iloc[i]), 3),
                           "question1": df.question1.iloc[i], "question2": df.question2.iloc[i]})
    pd.DataFrame(sample).to_csv(out_dir / "error_samples.csv", index=False)
    (out_dir / f"{split}_results.json").write_text(json.dumps(
        {"split": split, "n_pairs": len(df), "prevalence": float(y.mean()), "leakage_check": leakage, "thresholds": thr,
         "results": res, "final_vs_baseline_bootstrap": boot, "slices": slices, "transitions": trans,
         "manifest_git_commit": manifest["git_commit"]}, indent=2, default=float))
    plots(scores, res, names, split)
    print(f"[done] results -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
