"""Frozen sentence-transformer encoding with an on-disk cache of unique questions.

Encoders receive the raw question text (whitespace-trimmed): they were pretrained on raw text
with their own tokenisers, so the Phase 2 normalisation is not applied here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.data import PROJECT_ROOT

EMB_ROOT = PROJECT_ROOT / "data" / "processed" / "embeddings"
MAX_SEQ_LENGTH = 128  # word p99 is 31; 128 word-pieces only truncates the rare long narratives


def clean(text) -> str:
    return "" if text is None or text != text else " ".join(str(text).split())


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def load_encoder(model_name: str, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = min(model.max_seq_length, MAX_SEQ_LENGTH)
    return model


class EmbeddingCache:
    """Maps cleaned question text -> row of a float16 matrix stored under EMB_ROOT/<model>/<name>."""

    def __init__(self, model_name: str, name: str):
        self.model_name = model_name
        self.dir = EMB_ROOT / model_slug(model_name)
        self.name = name
        self.matrix_path = self.dir / f"{name}.npy"
        self.keys_path = self.dir / f"{name}_texts.json"

    def exists(self) -> bool:
        return self.matrix_path.exists() and self.keys_path.exists()

    def build(self, texts, batch_size: int = 128, device: str = "cpu") -> dict:
        uniq = pd.unique(pd.Series([clean(t) for t in texts]))
        model = load_encoder(self.model_name, device)
        t = time.time()
        emb = model.encode(list(uniq), batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
        seconds = time.time() - t
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.matrix_path, emb.astype(np.float16))
        self.keys_path.write_text(json.dumps(list(uniq)), encoding="utf-8")
        info = {"model": self.model_name, "n_texts": len(uniq), "dim": int(emb.shape[1]),
                "seconds": round(seconds, 1), "texts_per_second": round(len(uniq) / seconds, 1)}
        (self.dir / f"{self.name}_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        return info

    def load(self) -> tuple[np.ndarray, dict]:
        matrix = np.load(self.matrix_path).astype(np.float32)
        keys = json.loads(self.keys_path.read_text(encoding="utf-8"))
        return matrix, {k: i for i, k in enumerate(keys)}

    def pair_embeddings(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix, index = self.load()
        ia = np.array([index[clean(t)] for t in df["question1"]])
        ib = np.array([index[clean(t)] for t in df["question2"]])
        return matrix[ia], matrix[ib]


def embedding_pair_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Order-invariant pair representation [|u - v|, u * v, cos(u, v)] for L2-normalised u, v."""
    cos = np.sum(u * v, axis=1, keepdims=True)
    return np.hstack([np.abs(u - v), u * v, cos]).astype(np.float32)
