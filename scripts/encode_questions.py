"""Encode unique train/val questions with frozen sentence encoders (Phase 3). The test split is never read.

Usage:
  python scripts/encode_questions.py --model sentence-transformers/all-MiniLM-L6-v2
  python scripts/encode_questions.py --model sentence-transformers/nli-distilroberta-base-v2 --control-subset
      (contamination control: encodes val + the fixed 60k-pair train subset only)
Writes: data/processed/embeddings/<model>/{train|train_subset,val}.npy (+ texts json, info json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.sentence_encoder import EmbeddingCache  # noqa: E402
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402
from src.utils.splits import train_dev_split  # noqa: E402
from src.utils.torch_utils import configure_threads, get_device  # noqa: E402

SUBSET_PATH = PROJECT_ROOT / "artifacts" / "phase3" / "control_subset_ids.json"
SUBSET_SIZE = 60_000
SEED = 42


def load_split(name: str) -> pd.DataFrame:
    assert name in ("train", "val"), "Phase 3 must not read the test split"
    return pd.read_csv(resolve(load_config()["split_dir"]) / f"{name}.csv", keep_default_na=False, na_values=[""])


def control_subset(train: pd.DataFrame) -> pd.DataFrame:
    """Fixed random subset of the train-fit rows (dev slice excluded), shared by both encoders."""
    if SUBSET_PATH.exists():
        ids = set(json.loads(SUBSET_PATH.read_text()))
        return train[train["id"].isin(ids)]
    is_fit, _ = train_dev_split(train, seed=SEED)
    fit_ids = train.loc[is_fit, "id"].to_numpy()
    ids = np.sort(np.random.default_rng(SEED).choice(fit_ids, SUBSET_SIZE, replace=False))
    SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBSET_PATH.write_text(json.dumps(ids.tolist()))
    return train[train["id"].isin(set(ids.tolist()))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--control-subset", action="store_true")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    configure_threads()
    device = str(get_device())

    train, val = load_split("train"), load_split("val")
    jobs = {"val": val}
    jobs["train_subset" if args.control_subset else "train"] = control_subset(train) if args.control_subset else train
    for name, df in jobs.items():
        cache = EmbeddingCache(args.model, name)
        if cache.exists():
            print(f"[skip] {name}: cache exists")
            continue
        info = cache.build(pd.concat([df["question1"], df["question2"]]), args.batch_size, device)
        print(f"[done] {name}: {info}", flush=True)


if __name__ == "__main__":
    main()
