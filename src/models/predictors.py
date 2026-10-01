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
        """Embeddings rounded through float16, exactly like the on-disk cache that every training and
        evaluation score was computed from (avoids train/serve skew; the boosted Phase 6 trees amplify
        even 1e-4 differences in the base score at split points)."""
        from src.models.sentence_encoder import clean
        emb = self.encoder.encode([clean(t) for t in texts], batch_size=128, normalize_embeddings=True,
                                  show_progress_bar=False, convert_to_numpy=True)
        return emb.astype(np.float16).astype(np.float32)

    def predict(self, q1, q2) -> np.ndarray:
        return self.predict_from_embeddings(self.embed(q1), self.embed(q2))

    def predict_from_embeddings(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        from src.models.sentence_encoder import embedding_pair_features
        X = embedding_pair_features(u, v)
        if self.head == "cosine":
            return X[:, -1]
        if self.head == "lr":
            return self.clf.predict_proba(X)[:, 1]
        with torch.no_grad():
            return torch.sigmoid(self.clf(torch.from_numpy(X))).numpy()


class MetaPredictor:
    """Phase 6: frozen base model (MiniLM + MLP) + deterministic constraint features -> meta-classifier.
    `bundle` is the joblib written by scripts/train_meta.py (model, variant, threshold)."""

    def __init__(self, bundle_path: Path = ART / "phase6" / "meta_model.joblib", device: str = "cpu"):
        from src.clustering.core import CentroidModel
        self.path = Path(bundle_path)
        b = joblib.load(self.path)
        self.meta, self.threshold, self.uses_spacy = b["model"], b["threshold"], b["uses_spacy"]
        if self.uses_spacy:
            import spacy
        self.name = f"meta_{self.meta.variant.name}"
        self.base = SBERTPredictor("mlp", device=device)
        # spaCy is loaded only if the chosen variant actually uses spaCy-derived features
        self.nlp = (spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
                    if self.uses_spacy else None)
        self.clusters = CentroidModel.load(ART / "phase4" / "cluster_model")

    def features(self, q1, q2) -> pd.DataFrame:
        from src.clustering.pairs import assign_pairs
        from src.features.constraints import annotate, pair_constraint_features
        from src.features.lexical import lexical_features
        from src.models.meta import base_logit
        from src.models.sentence_encoder import clean, embedding_pair_features
        q1, q2 = [clean(q) for q in q1], [clean(q) for q in q2]  # same text the training annotations used
        u, v = self.base.embed(q1), self.base.embed(q2)
        base_prob = self.base.predict_from_embeddings(u, v)  # reuse embeddings: encode each question once
        if self.nlp is not None:
            docs_a, docs_b = list(self.nlp.pipe(list(q1))), list(self.nlp.pipe(list(q2)))
        else:
            docs_a, docs_b = [None] * len(q1), [None] * len(q2)
        rows = [pair_constraint_features(annotate(a, da), annotate(b, db), use_spacy=self.uses_spacy)
                for a, b, da, db in zip(q1, q2, docs_a, docs_b)]
        f = pd.DataFrame(rows)
        f["base_prob"], f["base_logit"] = base_prob, base_logit(base_prob)
        f["cosine"] = embedding_pair_features(u, v)[:, -1]
        f["cluster"] = assign_pairs(u, v, self.clusters, "pair_avg")
        from src.models.meta import LEXICAL
        lex = lexical_features(pd.DataFrame({"question1": list(q1), "question2": list(q2)}))[LEXICAL]
        return pd.concat([f, lex.reset_index(drop=True)], axis=1)  # LEXICAL only: avoids duplicate columns

    def predict(self, q1, q2) -> np.ndarray:
        return self.meta.predict_proba(self.features(q1, q2))
