"""Minimal, information-preserving text preprocessing.

Deliberately NOT done: stop-word removal, stemming, number removal, punctuation stripping
of numbers ("3.5", "1,000"), or dropping negations. Those carry exactly the constraints
(entities, quantities, polarity) that separate near-identical non-duplicates.
"""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "′": "'"})
# Negation contractions -> explicit "not" so the negation survives tokenisation.
_NEGATIONS = [
    (re.compile(r"\bcan't\b|\bcannot\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"\b(\w+)n't\b"), r"\1 not"),
]
_MATH = re.compile(r"\[/?math\]")
# numbers (with internal . or ,) | words (letters/digits/underscore, internal apostrophe) | single symbol
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|\w+(?:'\w+)?|[^\w\s]")


def normalize(text: str | None) -> str:
    """Unicode NFKC, straight quotes, lowercase, explicit negation, whitespace collapse.
    `[math]` tags are removed but the formula text is kept."""
    if text is None or text != text:  # None or NaN
        return ""
    t = unicodedata.normalize("NFKC", str(text)).translate(_QUOTES).lower()
    t = _MATH.sub(" ", t)
    for pattern, repl in _NEGATIONS:
        t = pattern.sub(repl, t)
    return _WS.sub(" ", t).strip()


def tokenize(text: str, keep_punct: bool = False) -> list[str]:
    """Tokenise already-normalised text. Punctuation tokens are dropped unless requested."""
    tokens = _TOKEN.findall(text)
    if keep_punct:
        return tokens
    return [t for t in tokens if t[0].isalnum() or t[0] == "_"]


def preprocess(text: str | None) -> list[str]:
    return tokenize(normalize(text))
