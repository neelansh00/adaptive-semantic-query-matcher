"""Heuristic tags for *describing* errors (analysis only, never used as model features here).

Tags look at the tokens that differ between the two questions:
  number_diff      a differing token contains a digit
  entity_diff      a differing token is capitalised mid-sentence in the raw text (crude entity proxy)
  negation_diff    a negation word appears in only one question
  high_overlap     word Jaccard >= 0.6
  low_overlap      word Jaccard < 0.2
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.preprocessing.text import normalize, tokenize

NEGATIONS = {"not", "no", "never", "without", "nor", "non"}
_PRONOUN_I = {"i", "i'm", "i've", "i'll", "i'd"}
_CAP = re.compile(r"(?<![.?!]\s)(?<!^)\b([A-Z][\w'-]*)")


def _capitalised(raw: str) -> set[str]:
    return {m.lower() for m in _CAP.findall(str(raw).strip())} - _PRONOUN_I


def error_tags(q1: str, q2: str) -> dict:
    t1, t2 = set(tokenize(normalize(q1))), set(tokenize(normalize(q2)))
    diff = t1 ^ t2
    union = t1 | t2
    jac = len(t1 & t2) / len(union) if union else 0.0
    caps = (_capitalised(q1) | _capitalised(q2)) & diff
    return {
        "number_diff": any(any(c.isdigit() for c in t) for t in diff),
        "entity_diff": bool(caps),
        "negation_diff": bool(diff & NEGATIONS) or any(t.startswith("non") and t[3:] in union for t in diff),
        "high_overlap": jac >= 0.6,
        "low_overlap": jac < 0.2,
    }


def outcome(y_true, y_pred) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.select([(y_true == 1) & (y_pred == 1), (y_true == 0) & (y_pred == 1),
                      (y_true == 0) & (y_pred == 0)], ["TP", "FP", "TN"], "FN")


def tag_table(df: pd.DataFrame, y_pred) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-pair tags + outcome, share of each tag within each outcome)."""
    tags = pd.DataFrame([error_tags(a, b) for a, b in zip(df["question1"], df["question2"])], index=df.index)
    tags["outcome"] = outcome(df["is_duplicate"], y_pred)
    summary = tags.groupby("outcome").mean().T
    summary.loc["n"] = tags["outcome"].value_counts()
    return tags, summary[["TP", "FP", "TN", "FN"]]


def tag_error_rates(tags: pd.DataFrame) -> pd.DataFrame:
    """For each tag: FPR among negatives with vs without the tag, FNR among positives with vs without."""
    neg = tags["outcome"].isin(["FP", "TN"])
    pos = ~neg
    is_fp = tags["outcome"] == "FP"
    is_fn = tags["outcome"] == "FN"
    rows = {}
    for tag in [c for c in tags.columns if c != "outcome"]:
        t = tags[tag].astype(bool)
        rows[tag] = {
            "negatives_with_tag": int((neg & t).sum()),
            "fpr_with_tag": is_fp[neg & t].mean(), "fpr_without_tag": is_fp[neg & ~t].mean(),
            "positives_with_tag": int((pos & t).sum()),
            "fnr_with_tag": is_fn[pos & t].mean(), "fnr_without_tag": is_fn[pos & ~t].mean(),
        }
    return pd.DataFrame(rows).T
