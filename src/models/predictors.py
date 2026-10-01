"""Uniform raw-text -> probability predictors for every trained model (used for probes, latency, demo).

Each predictor exposes `predict(q1_list, q2_list) -> np.ndarray` of duplicate scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.features.lexical import lexical_features
from src.preprocessing.text import normalize, preprocess
from src.utils.data import PROJECT_ROOT

ART = PROJECT_ROOT / "artifacts"


class TfidfLexicalPredictor:
    """Phase 2 best baseline (lr_tfidf_pair_lexical)."""
    name = "lr_tfidf_pair_lexical"

    def __init__(self, path: Path = ART / "phase2" / "lr_tfidf_pair_lexical.joblib"):
        self.path = Path(path)
        b = joblib.load(self.path)
        self.tfidf, self.lex_scaler, self.cos_scaler, self.model = b["tfidf"], b["lex_scaler"], b["cos_scaler"], b["model"]
        self.threshold = b["threshold"]

    def predict(self, q1, q2) -> np.ndarray:
        a = self.tfidf.transform([normalize(q) for q in q1])
        b = self.tfidf.transform([normalize(q) for q in q2])
        cos = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        lex = self.lex_scaler.transform(lexical_features(pd.DataFrame({"question1": q1, "question2": q2})))
        X = sp.hstack([abs(a - b), a.multiply(b), sp.csr_matrix(lex),
                       sp.csr_matrix(self.cos_scaler.transform(cos[:, None]))], format="csr")
        return self.model.predict_proba(X)[:, 1]


class BiLSTMPredictor:
    name = "siamese_bilstm"

    def __init__(self, directory: Path = ART / "phase3" / "bilstm", device: str = "cpu"):
        from src.models.siamese_bilstm import load_model
        self.path = Path(directory)
        self.model, self.vocab, self.meta = load_model(self.path, device)
        self.device = device

    def predict(self, q1, q2) -> np.ndarray:
        from src.models.siamese_bilstm import predict_proba
        L = self.model.cfg.max_len
        a = [self.vocab.encode(preprocess(q), L) for q in q1]
        b = [self.vocab.encode(preprocess(q), L) for q in q2]
        return predict_proba(self.model, a, b, device=self.device)


class SBERTPredictor:
    """Frozen sentence encoder + head in {'cosine', 'lr', 'mlp'}."""

    def __init__(self, head: str = "lr", model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        from src.models.sentence_encoder import load_encoder, model_slug
        self.head, self.device, self.model_name = head, device, model_name
        self.name = f"sbert_{head}" if head != "cosine" else "sbert_cosine"
        self.encoder = load_encoder(model_name, device)
        self.dir = ART / "phase3" / f"sbert_{model_slug(model_name)}"
        if head == "lr":
            self.clf = joblib.load(self.dir / "lr.joblib")
        elif head == "mlp":
            from scripts.run_sbert import PairMLP
            cfg = json.loads((self.dir / "mlp_config.json").read_text())
            self.clf = PairMLP(cfg["in_dim"], cfg["hidden"], cfg["dropout"])
            self.clf.load_state_dict(torch.load(self.dir / "mlp.pt", map_location=device, weights_only=True))
            self.clf.eval()

    def embed(self, texts) -> np.ndarray:
        from src.models.sentence_encoder import clean
        return self.encoder.encode([clean(t) for t in texts], batch_size=128, normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_numpy=True)

    def predict(self, q1, q2) -> np.ndarray:
        from src.models.sentence_encoder import embedding_pair_features
        X = embedding_pair_features(self.embed(q1), self.embed(q2))
        if self.head == "cosine":
            return X[:, -1]
        if self.head == "lr":
            return self.clf.predict_proba(X)[:, 1]
        with torch.no_grad():
            return torch.sigmoid(self.clf(torch.from_numpy(X))).numpy()
