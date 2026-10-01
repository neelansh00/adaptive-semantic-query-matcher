"""Phase 3 contamination control: does MiniLM's Quora pretraining inflate its scores?

all-MiniLM-L6-v2's model card lists 103,663 "Quora Question Triplets" from the same Quora release as
this dataset, so some validation duplicates may have been seen in pretraining. As a control, both
encoders get identical heads trained on the identical 60k-pair train subset and are compared on val:
  - all-MiniLM-L6-v2           (Quora triplets in pretraining data)
  - nli-distilroberta-base-v2  (NLI model; its card lists no Quora data)
A gap does NOT isolate contamination (the encoders also differ in size and objective); it bounds it.

Usage:  python scripts/run_encoder_control.py
Needs:  encode_questions.py --control-subset for each encoder (MiniLM's full train cache also works)
Writes: artifacts/phase3/encoder_control.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.encode_questions import SUBSET_PATH, load_split  # noqa: E402
from src.evaluation.metrics import validation_report  # noqa: E402
from src.models.sentence_encoder import EmbeddingCache, embedding_pair_features  # noqa: E402
from src.utils.data import PROJECT_ROOT  # noqa: E402

ENCODERS = {"all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "Quora triplets in pretraining"),
            "nli-distilroberta-base-v2": ("sentence-transformers/nli-distilroberta-base-v2", "no Quora data listed")}


def main() -> None:
    train, val = load_split("train"), load_split("val")
    ids = set(json.loads(SUBSET_PATH.read_text()))
    sub = train[train["id"].isin(ids)]
    y_sub, y_va = sub["is_duplicate"].to_numpy(), val["is_duplicate"].to_numpy()
    out = {"subset_pairs": len(sub), "head": "LogisticRegression(C=1.0), fixed; no tuning on val", "results": {}}
    for short, (name, note) in ENCODERS.items():
        tr_cache = EmbeddingCache(name, "train_subset")
        if not tr_cache.exists():
            tr_cache = EmbeddingCache(name, "train")
        X_sub = embedding_pair_features(*tr_cache.pair_embeddings(sub))
        X_va = embedding_pair_features(*EmbeddingCache(name, "val").pair_embeddings(val))
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X_sub, y_sub)
        res = {"pretraining_note": note}
        for head, s in (("cosine", X_va[:, -1]), ("lr_60k", lr.predict_proba(X_va)[:, 1])):
            rep = validation_report(y_va, s, is_probability=head != "cosine")
            m = rep["val_at_tuned_threshold"]
            res[head] = {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                         "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "threshold": rep["tuned_threshold"]}
            print(f"{short:26s} {head:7s} F1={m['f1']:.4f} ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f}", flush=True)
        out["results"][short] = res
    path = PROJECT_ROOT / "artifacts" / "phase3" / "encoder_control.json"
    path.write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
