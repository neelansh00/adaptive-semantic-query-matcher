"""Phase 3, Model A: train the Siamese BiLSTM (PyTorch; CPU by default, CUDA if available).

Training data: train split minus a question-disjoint 5% dev slice (early stopping on dev PR-AUC).
Validation split: used once at the end for threshold tuning + reporting. Test split: never read.

Usage:  python scripts/train_bilstm.py [--epochs 8] [--patience 1]
Writes: artifacts/phase3/bilstm/{model.pt, vocab.json, config.json, history.json, metrics.json, val_scores.csv}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import classification_metrics, validation_report  # noqa: E402
from src.models.siamese_bilstm import (BiLSTMConfig, SiameseBiLSTM, Vocab, bucketed_batches,  # noqa: E402
                                       pad_batch, predict_proba, save_model)
from src.preprocessing.text import preprocess  # noqa: E402
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402
from src.utils.splits import train_dev_split  # noqa: E402
from src.utils.torch_utils import configure_threads, count_parameters, get_device, set_seed  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "phase3" / "bilstm"
SEED = 42


def load_split(name: str) -> pd.DataFrame:
    assert name in ("train", "val"), "Phase 3 must not read the test split"
    return pd.read_csv(resolve(load_config()["split_dir"]) / f"{name}.csv", keep_default_na=False, na_values=[""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-freq", type=int, default=2)
    args = ap.parse_args()

    set_seed(SEED)
    threads = configure_threads()
    device = get_device()
    print(f"device={device} threads={threads} cuda_available={torch.cuda.is_available()}", flush=True)

    train, val = load_split("train"), load_split("val")
    is_fit, is_dev = train_dev_split(train, seed=SEED)
    fit, dev = train[is_fit].reset_index(drop=True), train[is_dev].reset_index(drop=True)
    print(f"fit={len(fit)} dev={len(dev)} val={len(val)}", flush=True)

    tok = {name: (df["question1"].map(preprocess).tolist(), df["question2"].map(preprocess).tolist())
           for name, df in (("fit", fit), ("dev", dev), ("val", val))}
    # Vocabulary from the fit questions only (each distinct question once).
    uniq = pd.unique(pd.concat([fit["question1"], fit["question2"]]))
    vocab = Vocab.build((preprocess(q) for q in uniq), min_freq=args.min_freq)
    cfg = BiLSTMConfig(vocab_size=len(vocab))
    ids = {k: ([vocab.encode(t, cfg.max_len) for t in a], [vocab.encode(t, cfg.max_len) for t in b])
           for k, (a, b) in tok.items()}
    unk_rate = {k: float(np.mean([i == 1 for s in a + b for i in s])) for k, (a, b) in ids.items()}
    print(f"vocab={len(vocab)} unk_rate={unk_rate}", flush=True)

    model = SiameseBiLSTM(cfg).to(device)
    n_params = count_parameters(model)
    print(f"parameters={n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    y_fit = torch.tensor(fit["is_duplicate"].to_numpy(), dtype=torch.float32)
    fit_a, fit_b = ids["fit"]
    lengths = np.array([max(len(x), len(y)) for x, y in zip(fit_a, fit_b)])
    rng = np.random.default_rng(SEED)

    history, best_pr, best_state, bad_epochs, start = [], -1.0, None, 0, 1
    ckpt_path = OUT / "checkpoint.pt"
    if ckpt_path.exists():
        # Exact resume: model, optimiser, both RNG streams and early-stopping state.
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"])
        history, best_pr, best_state, bad_epochs = ck["history"], ck["best_pr"], ck["best_state"], ck["bad_epochs"]
        start = len(history) + 1
        print(f"resumed from checkpoint after epoch {len(history)}", flush=True)
    stopped = bad_epochs > args.patience
    for epoch in range(start, args.epochs + 1):
        if stopped:
            break
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for batch in bucketed_batches(lengths, args.batch_size, rng):
            ai, al = pad_batch([fit_a[i] for i in batch])
            bi, bl = pad_batch([fit_b[i] for i in batch])
            opt.zero_grad()
            loss = loss_fn(model(ai.to(device), al, bi.to(device), bl), y_fit[batch].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch)
            n += len(batch)
        dev_p = predict_proba(model, *ids["dev"], device=device)
        dev_m = classification_metrics(dev["is_duplicate"], dev_p, 0.5)
        rec = {"epoch": epoch, "train_loss": total / n, "dev_pr_auc": dev_m["pr_auc"],
               "dev_roc_auc": dev_m["roc_auc"], "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if dev_m["pr_auc"] > best_pr:
            best_pr, bad_epochs = dev_m["pr_auc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            stopped = bad_epochs > args.patience
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "np_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "history": history, "best_pr": best_pr,
                    "best_state": best_state, "bad_epochs": bad_epochs}, ckpt_path)

    model.load_state_dict(best_state)
    best_epoch = max(history, key=lambda r: r["dev_pr_auc"])["epoch"]
    save_model(model, vocab, OUT, extra={"best_epoch": best_epoch, "seed": SEED, "lr": args.lr,
                                         "batch_size": args.batch_size, "min_freq": args.min_freq})

    # ---- validation: threshold tuning + reporting (once, after training is finished)
    val_p = predict_proba(model, *ids["val"], device=device)
    report = validation_report(val["is_duplicate"].to_numpy(), val_p)
    report.update({"best_epoch": best_epoch, "dev_pr_auc": best_pr, "parameters": n_params,
                   "vocab_size": len(vocab), "unk_rate": unk_rate, "device": str(device),
                   "train_seconds_total": round(sum(r["seconds"] for r in history), 1)})
    (OUT / "history.json").write_text(json.dumps(history, indent=2))
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    pd.DataFrame({"id": val["id"], "siamese_bilstm": val_p}).to_csv(OUT / "val_scores.csv", index=False)
    m = report["val_at_tuned_threshold"]
    print(f"VAL thr={report['tuned_threshold']:.2f} F1={m['f1']:.4f} P={m['precision']:.4f} "
          f"R={m['recall']:.4f} ROC={m['roc_auc']:.4f} PR={m['pr_auc']:.4f}", flush=True)
    ckpt_path.unlink()  # training finished; the saved model.pt is the artifact


if __name__ == "__main__":
    main()
