"""Phase 6 meta-classifier: base semantic probability + deterministic constraint features.

All variants are pre-specified here (features + model) BEFORE looking at validation results.
Hyper-parameters are fixed (LogisticRegression C=1.0 on standardised features; HistGradientBoosting
defaults with internal early stopping), so validation is used only for decision thresholds and for
comparing these fixed variants.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.constraints import FEATURE_GROUPS

LEXICAL = ["word_jaccard", "content_jaccard", "char3_jaccard", "common_ratio_min", "common_ratio_max"]
CONSTRAINT_GROUPS = ["number", "date", "entity_heuristic", "entity_spacy", "location", "negation", "intent_word",
                     "specificity"]


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


@dataclass(frozen=True)
class Variant:
    name: str
    groups: tuple = ()             # constraint feature groups
    lexical: bool = False          # add Phase 2 lexical overlap features
    cluster_onehot: bool = False   # add frozen-cluster one-hot (ablation; overlaps Phase 7)
    model: str = "lr"              # "lr" or "hgb"
    description: str = ""

    @property
    def uses_spacy(self) -> bool:
        return bool(SPACY_GROUPS & set(self.groups))

    def columns(self) -> list[str]:
        cols = ["base_logit"]
        for g in self.groups:
            cols += FEATURE_GROUPS[g]
        if self.lexical:
            cols += LEXICAL
        return cols


VARIANTS = [
    Variant("B_recal", description="LR on base logit only: controls for the effect of re-fitting a meta-model"),
    Variant("C_constraints", tuple(CONSTRAINT_GROUPS), description="base + all constraint/specificity features (primary)"),
    Variant("C_no_spacy", tuple("date_heuristic" if g == "date" else g for g in CONSTRAINT_GROUPS
                                if g not in ("entity_spacy", "location")),
            description="spaCy-free: heuristic entities, regex-only dates"),
    Variant("C_no_specificity", tuple(g for g in CONSTRAINT_GROUPS if g != "specificity"),
            description="explicit constraints only, no specificity / length proxies"),
    Variant("C_plus_lexical", tuple(CONSTRAINT_GROUPS), lexical=True, description="C + Phase 2 lexical overlap"),
    Variant("C_plus_cluster", tuple(CONSTRAINT_GROUPS), cluster_onehot=True,
            description="C + frozen-cluster one-hot (ablation only; per-cluster offsets belong to Phase 7)"),
    Variant("C_hgb", tuple(CONSTRAINT_GROUPS), model="hgb", description="same features as C, gradient-boosted trees"),
    # Complex variants built on the no-spaCy feature set (used if the spaCy step drops spaCy):
    Variant("C_no_spacy_plus_lexical", tuple("date_heuristic" if g == "date" else g for g in CONSTRAINT_GROUPS
                                             if g not in ("entity_spacy", "location")),
            lexical=True, description="C_no_spacy + Phase 2 lexical overlap"),
    Variant("C_no_spacy_hgb", tuple("date_heuristic" if g == "date" else g for g in CONSTRAINT_GROUPS
                                    if g not in ("entity_spacy", "location")),
            model="hgb", description="C_no_spacy features, gradient-boosted trees"),
    # Control: does boosting help WITHOUT constraint features (i.e. is the gain just a non-linear recalibration)?
    Variant("B_hgb", model="hgb", description="gradient-boosted trees on the base logit only (control)"),
]
GROUP_ABLATIONS = [f"C_minus_{g}" for g in CONSTRAINT_GROUPS]
# spaCy-free set: drop spaCy entities/locations AND use regex-only dates (the "date" group includes spaCy DATE spans)
NO_SPACY_GROUPS = tuple("date_heuristic" if g == "date" else g for g in CONSTRAINT_GROUPS
                        if g not in ("entity_spacy", "location"))
SPACY_GROUPS = {"entity_spacy", "location", "date"}
HGB_ABLATIONS = [f"HGB_minus_{g}" for g in NO_SPACY_GROUPS]


def variant_by_name(name: str) -> Variant:
    for v in VARIANTS:
        if v.name == name:
            return v
    if name.startswith("C_minus_"):
        g = name.removeprefix("C_minus_")
        return Variant(name, tuple(x for x in CONSTRAINT_GROUPS if x != g), description=f"C without {g}")
    if name.startswith("HGB_minus_"):
        g = name.removeprefix("HGB_minus_")
        return Variant(name, tuple(x for x in NO_SPACY_GROUPS if x != g), model="hgb",
                       description=f"C_no_spacy_hgb without {g}")
    raise KeyError(name)


@dataclass
class MetaModel:
    variant: Variant
    n_clusters: int = 12
    scaler: StandardScaler = field(default=None)
    clf: object = field(default=None)

    def design(self, feats: pd.DataFrame) -> np.ndarray:
        X = feats[self.variant.columns()].to_numpy(dtype=float)
        if self.variant.cluster_onehot:
            X = np.hstack([X, np.eye(self.n_clusters)[feats["cluster"].to_numpy().astype(int)]])
        return X

    def fit(self, feats: pd.DataFrame, y: np.ndarray, seed: int = 42) -> "MetaModel":
        X = self.design(feats)
        if self.variant.model == "lr":
            self.scaler = StandardScaler().fit(X)
            self.clf = LogisticRegression(C=1.0, max_iter=5000).fit(self.scaler.transform(X), y)
        else:
            self.clf = HistGradientBoostingClassifier(random_state=seed, early_stopping=True,
                                                      validation_fraction=0.1).fit(X, y)
        return self

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        X = self.design(feats)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.clf.predict_proba(X)[:, 1]

    def coefficients(self) -> dict:
        """Standardised LR coefficients (effect of +1 SD on the log-odds); cluster one-hots excluded."""
        if self.variant.model != "lr":
            return {}
        names = self.variant.columns()
        return dict(zip(names, np.round(self.clf.coef_[0][: len(names)], 4).tolist()))
