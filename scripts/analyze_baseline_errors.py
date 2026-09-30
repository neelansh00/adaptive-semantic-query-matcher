"""Phase 2 error analysis controlled for lexical overlap (reads saved val predictions; no retraining).

Raw tag rates are confounded: pairs that differ in an entity usually differ in many other words,
so they have low overlap and are easy to reject. Here the comparison is made WITHIN overlap bands.

Usage:  python scripts/analyze_baseline_errors.py
Writes: artifacts/phase2/error_analysis_by_overlap.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.error_analysis import error_tags  # noqa: E402
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "phase2"
TAGS = ["entity_diff", "number_diff", "negation_diff"]


def main() -> None:
    metrics = json.loads((ART / "baseline_metrics.json").read_text(encoding="utf-8"))
    model = metrics["best_model_by_val_pr_auc"]
    thr = metrics[model]["tuned_threshold"]

    val = pd.read_csv(resolve(load_config()["split_dir"]) / "val.csv", keep_default_na=False, na_values=[""])
    preds = pd.read_csv(ART / "val_predictions.csv")
    df = val.merge(preds[["id", model]], on="id")
    lex = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "lexical_val.csv")
    df = df.merge(lex[["id", "word_jaccard"]], on="id")
    tags = pd.DataFrame([error_tags(a, b) for a, b in zip(df.question1, df.question2)], index=df.index)
    df = pd.concat([df, tags[TAGS]], axis=1)
    df["pred"] = (df[model] >= thr).astype(int)
    df["band"] = pd.cut(df["word_jaccard"], [-0.01, 0.4, 0.6, 1.0], labels=["<=0.4", "0.4-0.6", ">0.6"])
    df["any_constraint_diff"] = df[TAGS].any(axis=1)

    neg = df[df.is_duplicate == 0]
    rows = []
    for band, g in neg.groupby("band", observed=True):
        for tag in TAGS + ["any_constraint_diff"]:
            w, wo = g[g[tag]], g[~g[tag]]
            rows.append({"overlap_band": str(band), "tag": tag, "negatives_with_tag": len(w),
                         "fpr_with_tag": round(w.pred.mean(), 4) if len(w) else None,
                         "negatives_without_tag": len(wo),
                         "fpr_without_tag": round(wo.pred.mean(), 4) if len(wo) else None})
    table = pd.DataFrame(rows)

    hi_fp = df[(df.band == ">0.6") & (df.is_duplicate == 0) & (df.pred == 1)]
    share_hi_fp = {t: round(float(hi_fp[t].mean()), 4) for t in TAGS + ["any_constraint_diff"]}
    all_fp = df[(df.is_duplicate == 0) & (df.pred == 1)]
    out = {
        "model": model, "threshold": thr,
        "fpr_by_band_and_tag": table.to_dict("records"),
        "high_overlap_false_positives": int(len(hi_fp)),
        "share_of_all_false_positives_that_are_high_overlap": round(len(hi_fp) / len(all_fp), 4),
        "tag_share_among_high_overlap_false_positives": share_hi_fp,
    }
    (ART / "error_analysis_by_overlap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k != "fpr_by_band_and_tag"}, indent=1))


if __name__ == "__main__":
    main()
