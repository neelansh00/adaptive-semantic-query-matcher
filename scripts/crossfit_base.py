"""Phase 6b: out-of-fold (OOF) base-model probabilities for every TRAIN pair (stacking without leakage).

The meta-classifier must learn how far to trust the base model's probability. In-sample train probabilities
are over-confident (the model saw those pairs), so each train pair is scored by a copy of the Phase 3 MLP head
trained on the OTHER folds. Folds are question-disjoint (whole question-graph components). Each fold model uses
the Phase 3 recipe with a FIXED 25 epochs (the Phase 3 best epoch), so no early-stopping data is needed.

The deployed Phase 3 model is unchanged and still produces the validation/test base scores. To quantify the
train/val mismatch this creates, every fold model also scores validation and is compared with the deployed model.

Usage:  python scripts/crossfit_base.py
Writes: artifacts/phase6/train_oof_base.csv, artifacts/phase6/crossfit_report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_sbert import PairMLP, mlp_proba  # noqa: E402
from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.models.sentence_encoder import EmbeddingCache, embedding_pair_features  # noqa: E402
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402
from src.utils.splits import build_graph  # noqa: E402
from src.utils.torch_utils import configure_threads, get_device, set_seed  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "phase6"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
N_FOLDS, EPOCHS, SEED = 5, 25, 42


def load_split(name: str) -> pd.DataFrame:
    assert name in ("train", "val"), "Phase 6 must not read the test split"
    return pd.read_csv(resolve(load_config()["split_dir"]) / f"{name}.csv", keep_default_na=False, na_values=[""])


def train_fixed(X, y, device, epochs=EPOCHS, lr=1e-3, batch=512) -> PairMLP:
    """Phase 3 MLP recipe (same architecture/optimiser/seed), fixed number of epochs."""
    set_seed(SEED)
    model = PairMLP(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=g).split(batch):
            opt.zero_grad()
            loss_fn(model(Xt[idx].to(device)), yt[idx].to(device)).backward()
            opt.step()
    return model


def main() -> None:
    configure_threads()
    device = get_device()
    ART.mkdir(parents=True, exist_ok=True)
    train, val = load_split("train"), load_split("val")
    X = embedding_pair_features(*EmbeddingCache(ENCODER, "train").pair_embeddings(train))
    Xv = embedding_pair_features(*EmbeddingCache(ENCODER, "val").pair_embeddings(val))
    y, yv = train["is_duplicate"].to_numpy(), val["is_duplicate"].to_numpy()
    groups = build_graph(train).pair_component

    oof = np.full(len(train), np.nan, dtype=np.float32)
    fold_of = np.full(len(train), -1)
    val_preds, report = [], {"n_folds": N_FOLDS, "epochs": EPOCHS, "folds": []}
    for k, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, groups)):
        t = time.time()
        model = train_fixed(X[tr], y[tr], device)
        oof[te] = mlp_proba(model, X[te], device)
        fold_of[te] = k
        vp = mlp_proba(model, Xv, device)
        val_preds.append(vp)
        m_oof = classification_metrics(y[te], oof[te], 0.32)
        m_val = classification_metrics(yv, vp, 0.32)
        rec = {"fold": k, "train_pairs": int(len(tr)), "oof_pairs": int(len(te)), "oof_pr_auc": m_oof["pr_auc"],
               "oof_roc_auc": m_oof["roc_auc"], "val_pr_auc_of_fold_model": m_val["pr_auc"],
               "seconds": round(time.time() - t, 1)}
        report["folds"].append(rec)
        print(json.dumps(rec), flush=True)
    assert not np.isnan(oof).any()

    deployed = pd.read_csv(PROJECT_ROOT / "artifacts" / "phase3" / "sbert_all-MiniLM-L6-v2" / "val_scores.csv")
    deployed = val[["id"]].merge(deployed, on="id")["sbert_mlp"].to_numpy()
    fold_mean = np.mean(val_preds, axis=0)
    report["oof_overall"] = {k: classification_metrics(y, oof, 0.32)[k] for k in ("pr_auc", "roc_auc", "f1")}
    report["val_deployed"] = {k: classification_metrics(yv, deployed, 0.32)[k] for k in ("pr_auc", "roc_auc", "f1")}
    report["val_fold_mean"] = {k: classification_metrics(yv, fold_mean, 0.32)[k] for k in ("pr_auc", "roc_auc", "f1")}
    report["deployed_vs_fold_mean_pearson"] = float(np.corrcoef(deployed, fold_mean)[0, 1])
    report["deployed_minus_fold_mean_mean_abs"] = float(np.mean(np.abs(deployed - fold_mean)))
    pd.DataFrame({"id": train["id"], "base_prob_oof": oof, "fold": fold_of}).to_csv(ART / "train_oof_base.csv", index=False)
    (ART / "crossfit_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}, indent=1))


if __name__ == "__main__":
    main()
