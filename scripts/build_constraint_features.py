"""Phase 6a: annotate every unique train/val question once and build pair-level constraint features.

Per pair it also attaches the semantic cosine (frozen MiniLM embeddings) and the frozen Phase 4 cluster
(pair-average rule). The test split is never read.

Usage:  python scripts/build_constraint_features.py
Writes: data/processed/features/annotations_{train,val}.joblib
        data/processed/features/constraints_{train,val}.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clustering.core import CentroidModel  # noqa: E402
from src.clustering.pairs import assign_pairs  # noqa: E402
from src.features.constraints import annotate, pair_frame  # noqa: E402
from src.models.sentence_encoder import EmbeddingCache, clean  # noqa: E402
from src.utils.data import PROJECT_ROOT, load_config, resolve  # noqa: E402

FEAT = PROJECT_ROOT / "data" / "processed" / "features"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"


def load_split(name: str) -> pd.DataFrame:
    assert name in ("train", "val"), "Phase 6 must not read the test split"
    return pd.read_csv(resolve(load_config()["split_dir"]) / f"{name}.csv", keep_default_na=False, na_values=[""])


def annotate_questions(texts: list[str], nlp) -> dict:
    out = {}
    t = time.time()
    for i, (text, doc) in enumerate(zip(texts, nlp.pipe(texts, batch_size=1000))):
        out[text] = annotate(text, doc)
        if (i + 1) % 100_000 == 0:
            print(f"  annotated {i + 1:,} ({(i + 1) / (time.time() - t):.0f}/s)", flush=True)
    return out


def main() -> None:
    import spacy
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
    model = CentroidModel.load(PROJECT_ROOT / "artifacts" / "phase4" / "cluster_model")
    FEAT.mkdir(parents=True, exist_ok=True)
    for name in ("val", "train"):
        df = load_split(name)
        ann_path = FEAT / f"annotations_{name}.joblib"
        if ann_path.exists():
            annotations = joblib.load(ann_path)
        else:
            texts = list(pd.unique(pd.concat([df.question1, df.question2]).map(clean)))
            print(f"[{name}] annotating {len(texts):,} unique questions", flush=True)
            annotations = annotate_questions(texts, nlp)
            joblib.dump(annotations, ann_path, compress=3)
        t = time.time()
        feats = pair_frame(df, annotations, use_spacy=True)
        u, v = EmbeddingCache(ENCODER, name).pair_embeddings(df)
        feats.insert(0, "cosine", np.sum(u * v, axis=1))
        feats.insert(0, "cluster", assign_pairs(u, v, model, "pair_avg"))
        feats.insert(0, "is_duplicate", df["is_duplicate"].to_numpy())
        feats.insert(0, "id", df["id"].to_numpy())
        feats.to_csv(FEAT / f"constraints_{name}.csv", index=False)
        print(f"[{name}] {feats.shape} pair features in {time.time() - t:.0f}s", flush=True)
        rates = feats[[c for c in feats.columns if c.endswith(("_mismatch", "_one_sided", "_xor", "_subset"))]].mean()
        print(rates.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
