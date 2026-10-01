"""Phase 3, Model B: frozen sentence-transformer embeddings + light pair heads.

Heads (all on order-invariant features [|u-v|, u*v, cos]):
  sbert_cosine  - cosine similarity only, threshold tuned on val (no training)
  sbert_lr      - logistic regression; C chosen on the question-disjoint train-dev slice
  sbert_mlp     - one-hidden-layer MLP; early stopping on train-dev. Kept only as a measured comparison.
Validation is used only for the decision threshold and the final report. The test split is never read.

Usage:  python scripts/run_sbert.py [--model sentence-transformers/all-MiniLM-L6-v2]
Needs:  python scripts/encode_questions.py --model <same model>
Writes: artifacts/phase3/sbert_<model>/{metrics.json, val_scores.csv, lr.joblib, mlp.pt, mlp_config.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import classification_metrics, validation_report  # noqa: E402
from src.models.sentence_encoder import EmbeddingCache, embedding_pair_features, model_slug  # noqa: E402
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402
from src.utils.splits import train_dev_split  # noqa: E402
from src.utils.torch_utils import configure_threads, get_device, set_seed  # noqa: E402

SEED = 42
C_GRID = [0.1, 1.0, 10.0]


def load_split(name: str) -> pd.DataFrame:
    assert name in ("train", "val"), "Phase 3 must not read the test split"
    return pd.read_csv(resolve(load_config()["split_dir"]) / f"{name}.csv", keep_default_na=False, na_values=[""])


class PairMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.no_grad()
def mlp_proba(model: PairMLP, X: np.ndarray, device, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(X[i:i + batch]).to(device))).cpu().numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out)


def train_mlp(X_fit, y_fit, X_dev, y_dev, device, epochs=30, patience=3, lr=1e-3, batch=512):
    set_seed(SEED)
    model = PairMLP(X_fit.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(X_fit), torch.from_numpy(y_fit.astype(np.float32))
    g = torch.Generator().manual_seed(SEED)
    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for idx in torch.randperm(len(Xt), generator=g).split(batch):
            opt.zero_grad()
            loss_fn(model(Xt[idx].to(device)), yt[idx].to(device)).backward()
            opt.step()
        pr = classification_metrics(y_dev, mlp_proba(model, X_dev, device), 0.5)["pr_auc"]
        history.append({"epoch": epoch, "dev_pr_auc": pr, "seconds": round(time.time() - t0, 1)})
        print(f"  [mlp] epoch {epoch}: dev PR-AUC={pr:.4f}", flush=True)
        if pr > best:
            best, bad = pr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    configure_threads()
    device = get_device()
    out = PROJECT_ROOT / "artifacts" / "phase3" / f"sbert_{model_slug(args.model)}"
    out.mkdir(parents=True, exist_ok=True)

    train, val = load_split("train"), load_split("val")
    is_fit, is_dev = train_dev_split(train, seed=SEED)
    u_tr, v_tr = EmbeddingCache(args.model, "train").pair_embeddings(train)
    u_va, v_va = EmbeddingCache(args.model, "val").pair_embeddings(val)
    X_tr, X_va = embedding_pair_features(u_tr, v_tr), embedding_pair_features(u_va, v_va)
    y_tr, y_va = train["is_duplicate"].to_numpy(), val["is_duplicate"].to_numpy()
    X_fit, y_fit, X_dev, y_dev = X_tr[is_fit], y_tr[is_fit], X_tr[is_dev], y_tr[is_dev]
    print(f"fit={len(X_fit)} dev={len(X_dev)} val={len(X_va)} dim={X_tr.shape[1]} device={device}", flush=True)

    scores, extra = {}, {}
    scores["sbert_cosine"] = X_va[:, -1]

    grid, best = [], None
    for C in C_GRID:
        t0 = time.time()
        lr = LogisticRegression(C=C, max_iter=2000).fit(X_fit, y_fit)
        pr = classification_metrics(y_dev, lr.predict_proba(X_dev)[:, 1], 0.5)["pr_auc"]
        grid.append({"C": C, "dev_pr_auc": pr, "fit_seconds": round(time.time() - t0, 1)})
        print(f"  [lr] C={C}: dev PR-AUC={pr:.4f}", flush=True)
        if best is None or pr > best[1]:
            best = (lr, pr, C)
    lr = best[0]
    scores["sbert_lr"] = lr.predict_proba(X_va)[:, 1]
    extra["sbert_lr"] = {"C": best[2], "c_grid_dev": grid}
    joblib.dump(lr, out / "lr.joblib")

    mlp, hist = train_mlp(X_fit, y_fit, X_dev, y_dev, device)
    scores["sbert_mlp"] = mlp_proba(mlp, X_va, device)
    extra["sbert_mlp"] = {"history_dev": hist, "best_dev_pr_auc": max(h["dev_pr_auc"] for h in hist),
                          "parameters": sum(p.numel() for p in mlp.parameters())}
    torch.save(mlp.state_dict(), out / "mlp.pt")
    (out / "mlp_config.json").write_text(json.dumps({"in_dim": X_tr.shape[1], "hidden": 256, "dropout": 0.2}))

    results = {"encoder": args.model, "device": str(device)}
    for name, s in scores.items():
        results[name] = {**validation_report(y_va, s, is_probability=name != "sbert_cosine"), **extra.get(name, {})}
        m = results[name]["val_at_tuned_threshold"]
        print(f"{name:14s} thr={results[name]['tuned_threshold']:.2f} F1={m['f1']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f}", flush=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    pd.DataFrame({"id": val["id"], **scores}).to_csv(out / "val_scores.csv", index=False)


if __name__ == "__main__":
    main()
