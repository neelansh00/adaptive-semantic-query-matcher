"""Interpretable, symmetric lexical pair features.

Every feature is symmetric in (A, B), so swapping the two questions never changes a prediction.
Entity / number / negation *consistency* features are intentionally left for Phase 6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.preprocessing.text import normalize, tokenize

# Negations and question words are kept as content: they change intent.
KEEP = {"not", "no", "nor", "never", "without", "what", "how", "why", "when", "where", "which", "who",
        "whom", "whose", "can", "should", "will", "would", "could", "is", "are", "do", "does"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - KEEP

FEATURE_NAMES = [
    "word_jaccard", "content_jaccard", "common_word_count", "common_ratio_min", "common_ratio_max",
    "char3_jaccard", "word_count_absdiff", "char_len_absdiff", "length_ratio",
    "first_word_equal", "last_word_equal",
]


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _char_ngrams(text: str, n: int = 3) -> set:
    text = f" {text} "
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def pair_features(q1: str, q2: str) -> list[float]:
    n1, n2 = normalize(q1), normalize(q2)
    t1, t2 = tokenize(n1), tokenize(n2)
    s1, s2 = set(t1), set(t2)
    c1, c2 = s1 - STOP_WORDS, s2 - STOP_WORDS
    common = len(s1 & s2)
    shorter, longer = sorted((len(s1), len(s2)))
    lw = sorted((len(t1), len(t2)))
    return [
        _jaccard(s1, s2),
        _jaccard(c1, c2),
        float(common),
        common / shorter if shorter else 0.0,
        common / longer if longer else 0.0,
        _jaccard(_char_ngrams(n1), _char_ngrams(n2)),
        float(abs(len(t1) - len(t2))),
        float(abs(len(n1) - len(n2))),
        lw[0] / lw[1] if lw[1] else 0.0,
        float(bool(t1) and bool(t2) and t1[0] == t2[0]),
        float(bool(t1) and bool(t2) and t1[-1] == t2[-1]),
    ]


def lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = [pair_features(a, b) for a, b in zip(df["question1"], df["question2"])]
    return pd.DataFrame(np.asarray(rows, dtype=float), columns=FEATURE_NAMES, index=df.index)
