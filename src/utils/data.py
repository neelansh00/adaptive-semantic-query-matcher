"""Dataset loading and question-identity helpers.

The raw CSV is only ever read, never written.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COLUMNS = ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"]
TARGET = "is_duplicate"


def load_config(path: str | Path = "configs/data.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(resolve(path), "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_raw_pairs(path: str | Path | None = None) -> pd.DataFrame:
    """Load the labelled pair file exactly as supplied (no rows dropped)."""
    path = resolve(path or load_config()["raw_train_path"])
    # keep_default_na=False + explicit na_values: only truly empty cells become NaN,
    # so questions like "NA" or "null" stay as text.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], encoding="utf-8")
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {sorted(missing)}")
    return df


# Placeholder strings found in the raw file that stand in for a missing question.
PLACEHOLDER_QUESTIONS = {"", "n/a"}


def invalid_pair_mask(df: pd.DataFrame) -> pd.Series:
    """True where either question is empty or a placeholder (3 rows in the raw file)."""
    q1 = df["question1"].fillna("").str.strip().str.lower()
    q2 = df["question2"].fillna("").str.strip().str.lower()
    return q1.isin(PLACEHOLDER_QUESTIONS) | q2.isin(PLACEHOLDER_QUESTIONS)


def clean_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal, documented cleaning: drop pairs with an empty/placeholder question.
    Text itself is left untouched."""
    return df[~invalid_pair_mask(df)].reset_index(drop=True)


_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def identity_key(text: str) -> str:
    """Key used to decide whether two question strings are 'the same question'.

    Lowercase + whitespace collapse. Deliberately conservative: it merges exact
    re-posts with different qids but does not merge paraphrases.
    """
    return _WS.sub(" ", str(text).lower()).strip()


def loose_key(text: str) -> str:
    """Aggressive key (alphanumerics only), used only to *measure* residual leakage."""
    return _NON_ALNUM.sub(" ", str(text).lower()).strip()


TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".csv", ".toml", ".ini", ".cfg"}


def content_sha256(path: str | Path) -> str:
    """Platform-stable content hash: text files are hashed with CRLF normalised to LF (git stores LF and a Windows
    working copy may hold CRLF); binary files (weights, joblib, npy) are hashed byte-for-byte."""
    path = resolve(path)
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()
