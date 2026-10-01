"""Deterministic entity / constraint consistency features (Phase 6). No LLMs, no learned extractors
except spaCy's small NER model (optional, ablated against a capitalisation heuristic).

Each question is annotated ONCE (`annotate`); pair features are then computed from two annotations
(`pair_constraint_features`). Every pair feature is symmetric in (A, B).

Matching rule: an item (entity, number, date) of one question counts as "matched" if it also appears in the
other question's annotation OR its text occurs as whole words in the other question's normalised text.
This tolerates Quora's inconsistent casing ("india" vs "India"), which defeats NER on one side.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.features.lexical import STOP_WORDS
from src.preprocessing.text import normalize, tokenize

NEGATIONS = {"not", "no", "never", "without", "nor", "none", "nobody", "nothing", "neither", "cannot"}
WH_WORDS = {"what", "how", "why", "when", "where", "which", "who", "whom", "whose", "is", "are", "can",
            "should", "do", "does", "will", "would", "could"}
# "one"/"zero" omitted: "how does one ..." is a pronoun far more often than a number.
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
                "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)s?\b")
# A whole number token (thousands separators, decimals), optionally followed directly by a unit ("5kg", "2.5k").
# Single alternative so the regex cannot backtrack "2.5k" into "2" and "5".
_NUM = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)(?=$|[^\w.]|\.(?!\d)|[a-z%$])", re.I)
_DATE_WORDS = re.compile(r"\b(" + "|".join(MONTHS + DAYS + ["today", "tomorrow", "yesterday", "tonight"]) + r")\b")
_CAP = re.compile(r"(?<![.?!]\s)(?<!^)\b([A-Z][\w'&.-]*[\w])")
_ACRONYM = re.compile(r"\b([A-Z]{2,}[0-9]*|[A-Za-z]+[0-9]+[A-Za-z0-9]*)\b")
_DOTTED_ACRONYM = re.compile(r"\b(?:[A-Za-z]\.){2,}")  # U.S. / u.s.a. -> US / usa
_PRONOUN_I = {"i", "i'm", "i've", "i'll", "i'd"}
LOCATION_LABELS = {"GPE", "LOC", "FAC"}
SPACY_SKIP = {"CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY", "DATE", "TIME"}  # handled by number/date rules


def _undot(text: str) -> str:
    return _DOTTED_ACRONYM.sub(lambda m: m.group(0).replace(".", ""), str(text))


def _canon(text: str) -> str:
    """Canonical item text: dotted acronyms collapsed, normalised, punctuation-free tokens joined by spaces."""
    return " ".join(tokenize(normalize(_undot(text))))


def _numbers(norm: str) -> set[str]:
    nums = set()
    for m in _NUM.finditer(norm):
        raw = m.group(1).replace(",", "")
        try:
            v = float(raw)
            nums.add(str(int(v)) if v == int(v) else str(v))
        except ValueError:
            continue
    for tok in tokenize(norm):
        if tok in NUMBER_WORDS:
            nums.add(str(NUMBER_WORDS[tok]))
    return nums


def _dates(norm: str) -> set[str]:
    return {m.group(1) for m in _YEAR.finditer(norm)} | set(_DATE_WORDS.findall(norm))


def _heuristic_entities(raw: str) -> set[str]:
    raw = _undot(str(raw).strip())
    if raw.isupper():  # ALL-CAPS question: capitalisation carries no information
        return set()
    caps = {_canon(m) for m in _CAP.findall(raw)}
    caps = {e for e in caps if e not in STOP_WORDS}
    acr = {_canon(m) for m in _ACRONYM.findall(raw)}  # acronyms kept even if they spell a stop word (US, IT)
    return {e for e in caps | acr if e and e not in WH_WORDS and e not in _PRONOUN_I}


def annotate(raw: str, doc=None) -> dict:
    """Per-question annotation. `doc` is an optional spaCy Doc of the raw text."""
    norm = normalize(_undot(raw))
    toks = tokenize(norm)
    ann = {
        "norm": f" {' '.join(toks)} ",  # punctuation-free, for whole-word matching
        "n_words": len(toks),
        "content": {t for t in toks if t not in STOP_WORDS and t not in NEGATIONS},
        "numbers": _numbers(norm),
        "dates": _dates(norm),
        "negation": sum(t in NEGATIONS for t in toks) + len(re.findall(r"\bnon-?\w", norm)),
        "wh": toks[0] if toks and toks[0] in WH_WORDS else "",
        "ents_heur": _heuristic_entities(raw),
        "ents_spacy": set(), "locs_spacy": set(), "dates_spacy": set(),
    }
    if doc is not None:
        for e in doc.ents:
            text = _canon(e.text).removeprefix("the ").strip()
            if not text:
                continue
            if e.label_ in LOCATION_LABELS:
                ann["locs_spacy"].add(text)
                ann["ents_spacy"].add(text)
            elif e.label_ in ("DATE", "TIME"):
                ann["dates_spacy"].add(text)
            elif e.label_ not in SPACY_SKIP:
                ann["ents_spacy"].add(text)
    return ann


def _unmatched(a: set, b: set, b_norm: str) -> int:
    """Items of a absent from b and not found as whole words in b's text."""
    return sum(1 for x in a if x not in b and f" {x} " not in b_norm)


def _consistency(a: set, b: set, a_norm: str, b_norm: str, prefix: str) -> dict:
    ua, ub = _unmatched(a, b, b_norm), _unmatched(b, a, a_norm)
    both = bool(a) and bool(b)
    return {
        f"{prefix}_any": float(bool(a) or bool(b)),
        f"{prefix}_both": float(both),
        f"{prefix}_unmatched_total": float(ua + ub),
        f"{prefix}_unmatched_max_side": float(max(ua, ub)),
        f"{prefix}_mismatch": float(both and (ua + ub) > 0),       # both mention items, and they disagree
        f"{prefix}_one_sided": float(bool(a) != bool(b) and (ua + ub) > 0),
    }


def pair_constraint_features(a: dict, b: dict, use_spacy: bool = True) -> dict:
    f = {}
    f.update(_consistency(a["numbers"], b["numbers"], a["norm"], b["norm"], "num"))
    dates_a = a["dates"] | (a["dates_spacy"] if use_spacy else set())
    dates_b = b["dates"] | (b["dates_spacy"] if use_spacy else set())
    f.update(_consistency(dates_a, dates_b, a["norm"], b["norm"], "date"))
    # regex-only dates (years, months, weekdays): what a spaCy-free model can compute at inference time
    f.update(_consistency(a["dates"], b["dates"], a["norm"], b["norm"], "dateh"))
    f.update(_consistency(a["ents_heur"], b["ents_heur"], a["norm"], b["norm"], "enth"))
    if use_spacy:
        f.update(_consistency(a["ents_spacy"], b["ents_spacy"], a["norm"], b["norm"], "ents"))
        f.update(_consistency(a["locs_spacy"], b["locs_spacy"], a["norm"], b["norm"], "loc"))
    # negation: polarity disagreement
    f["neg_xor"] = float((a["negation"] > 0) != (b["negation"] > 0))
    f["neg_count_diff"] = float(abs(a["negation"] - b["negation"]))
    # question-word (intent) disagreement, only when both start with a wh/aux word
    f["wh_mismatch"] = float(bool(a["wh"]) and bool(b["wh"]) and a["wh"] != b["wh"])
    # specificity / scope proxies (broader vs narrower)
    only_a, only_b = len(a["content"] - b["content"]), len(b["content"] - a["content"])
    f["extra_content_max"] = float(max(only_a, only_b))
    f["extra_content_min"] = float(min(only_a, only_b))
    f["content_subset"] = float(min(only_a, only_b) == 0 and max(only_a, only_b) > 0)  # one contains the other
    f["log_len_ratio"] = float(abs(np.log(a["n_words"] + 1) - np.log(b["n_words"] + 1)))  # exactly symmetric
    f["word_count_absdiff"] = float(abs(a["n_words"] - b["n_words"]))
    return f


FEATURE_GROUPS = {
    "number": ["num_any", "num_both", "num_unmatched_total", "num_unmatched_max_side", "num_mismatch", "num_one_sided"],
    "date": ["date_any", "date_both", "date_unmatched_total", "date_unmatched_max_side", "date_mismatch", "date_one_sided"],
    "date_heuristic": ["dateh_any", "dateh_both", "dateh_unmatched_total", "dateh_unmatched_max_side", "dateh_mismatch",
                       "dateh_one_sided"],
    "entity_heuristic": ["enth_any", "enth_both", "enth_unmatched_total", "enth_unmatched_max_side", "enth_mismatch",
                         "enth_one_sided"],
    "entity_spacy": ["ents_any", "ents_both", "ents_unmatched_total", "ents_unmatched_max_side", "ents_mismatch",
                     "ents_one_sided"],
    "location": ["loc_any", "loc_both", "loc_unmatched_total", "loc_unmatched_max_side", "loc_mismatch", "loc_one_sided"],
    "negation": ["neg_xor", "neg_count_diff"],
    "intent_word": ["wh_mismatch"],
    "specificity": ["extra_content_max", "extra_content_min", "content_subset", "log_len_ratio", "word_count_absdiff"],
}


def pair_frame(df: pd.DataFrame, annotations: dict, use_spacy: bool = True) -> pd.DataFrame:
    """Pair features for every row of df, looking up per-question annotations by cleaned raw text."""
    from src.models.sentence_encoder import clean
    rows = [pair_constraint_features(annotations[clean(q1)], annotations[clean(q2)], use_spacy)
            for q1, q2 in zip(df["question1"], df["question2"])]
    return pd.DataFrame(rows, index=df.index)
