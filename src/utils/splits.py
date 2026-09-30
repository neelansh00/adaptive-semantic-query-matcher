"""Split strategies and leakage measurement for question-pair data.

Questions are nodes, pairs are edges. Two questions are the same node when their
`identity_key` matches (so re-posts with a new qid are not treated as new questions).

Strategies (all deterministic given `seed`):
  random_row        - stratified random row split (the naive baseline)
  component_whole   - every connected component goes to exactly one split
  node_level        - every question is assigned a split; cross-split pairs dropped
  hybrid_component  - components <= cap assigned whole, larger ones split per node
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split

from src.utils.data import identity_key, loose_key

SPLITS = ("train", "val", "test")
DROPPED = "dropped_cross_split"


@dataclass
class QuestionGraph:
    node_a: np.ndarray        # node id of question1 per pair
    node_b: np.ndarray        # node id of question2 per pair
    n_nodes: int
    node_component: np.ndarray  # component id per node
    pair_component: np.ndarray  # component id per pair
    component_pairs: np.ndarray  # number of pairs in each component (indexed by component id)

    @property
    def pair_component_size(self) -> np.ndarray:
        return self.component_pairs[self.pair_component]


def build_graph(df: pd.DataFrame, key=identity_key) -> QuestionGraph:
    keys = pd.concat([df["question1"], df["question2"]], ignore_index=True).map(key)
    codes, uniques = pd.factorize(keys)
    n = len(df)
    a, b = codes[:n], codes[n:]
    adj = sp.coo_matrix((np.ones(n), (a, b)), shape=(len(uniques), len(uniques)))
    _, node_comp = connected_components(adj, directed=False)
    pair_comp = node_comp[a]
    comp_pairs = np.bincount(pair_comp, minlength=node_comp.max() + 1)
    return QuestionGraph(a, b, len(uniques), node_comp, pair_comp, comp_pairs)


# --------------------------------------------------------------------------- strategies

def random_row_split(df: pd.DataFrame, ratios: dict, seed: int) -> pd.Series:
    idx = np.arange(len(df))
    y = df["is_duplicate"].to_numpy()
    rest, test = train_test_split(idx, test_size=ratios["test"], stratify=y, random_state=seed)
    val_frac = ratios["val"] / (ratios["train"] + ratios["val"])
    train, val = train_test_split(rest, test_size=val_frac, stratify=y[rest], random_state=seed)
    out = np.empty(len(df), dtype=object)
    out[train], out[val], out[test] = "train", "val", "test"
    return pd.Series(out, index=df.index, name="split")


def _assign_components_stratified(comp_ids: np.ndarray, comp_pairs: np.ndarray, ratios: dict,
                                  buckets: list[int], rng: np.random.Generator) -> dict:
    """Assign whole components to splits, separately inside each component-size bucket,
    so each split receives ~ratio of the pairs of every bucket."""
    sizes = comp_pairs[comp_ids]
    edges = [0, *buckets, np.inf]
    bucket_of = np.digitize(sizes, edges[1:-1], right=True)
    cut_test, cut_val = ratios["test"], ratios["test"] + ratios["val"]
    assignment = {}
    for bucket in np.unique(bucket_of):
        ids = comp_ids[bucket_of == bucket]
        ids = ids[rng.permutation(len(ids))]
        w = comp_pairs[ids].astype(float)
        start = (np.cumsum(w) - w) / w.sum()  # cumulative share *before* this component
        labels = np.where(start < cut_test, "test", np.where(start < cut_val, "val", "train"))
        assignment.update(zip(ids.tolist(), labels.tolist()))
    return assignment


def node_probabilities(ratios: dict) -> np.ndarray:
    """Per-question split probabilities such that *retained pairs* follow `ratios`.

    A pair survives in split s only if both questions land in s (prob p_s**2), so
    drawing questions with p_s = ratio_s would shrink val/test to ~1% each.
    Using p_s proportional to sqrt(ratio_s) makes p_s**2 proportional to ratio_s."""
    p = np.sqrt([ratios[s] for s in SPLITS])
    return p / p.sum()


def _node_level_labels(g: QuestionGraph, nodes_mask: np.ndarray, ratios: dict,
                       rng: np.random.Generator) -> np.ndarray:
    node_split = np.full(g.n_nodes, None, dtype=object)
    draws = rng.choice(np.array(SPLITS, dtype=object), size=int(nodes_mask.sum()),
                       p=node_probabilities(ratios))
    node_split[np.flatnonzero(nodes_mask)] = draws
    return node_split


def component_whole_split(df, g: QuestionGraph, ratios, seed, buckets) -> pd.Series:
    rng = np.random.default_rng(seed)
    comp_ids = np.unique(g.pair_component)
    assign = _assign_components_stratified(comp_ids, g.component_pairs, ratios, buckets, rng)
    return pd.Series([assign[c] for c in g.pair_component], index=df.index, name="split")


def node_level_split(df, g: QuestionGraph, ratios, seed) -> pd.Series:
    rng = np.random.default_rng(seed)
    node_split = _node_level_labels(g, np.ones(g.n_nodes, bool), ratios, rng)
    sa, sb = node_split[g.node_a], node_split[g.node_b]
    return pd.Series(np.where(sa == sb, sa, DROPPED), index=df.index, name="split")


def hybrid_component_split(df, g: QuestionGraph, ratios, seed, buckets, cap) -> pd.Series:
    rng = np.random.default_rng(seed)
    comp_ids = np.unique(g.pair_component)
    small = comp_ids[g.component_pairs[comp_ids] <= cap]
    assign = _assign_components_stratified(small, g.component_pairs, ratios, buckets, rng)

    large_nodes = g.component_pairs[g.node_component] > cap
    node_split = _node_level_labels(g, large_nodes, ratios, rng)

    out = np.empty(len(df), dtype=object)
    is_large = g.pair_component_size > cap
    out[~is_large] = [assign[c] for c in g.pair_component[~is_large]]
    sa, sb = node_split[g.node_a[is_large]], node_split[g.node_b[is_large]]
    out[is_large] = np.where(sa == sb, sa, DROPPED)
    return pd.Series(out, index=df.index, name="split")


def make_split(df: pd.DataFrame, cfg: dict, strategy: str | None = None,
               g: QuestionGraph | None = None) -> pd.Series:
    strategy = strategy or cfg["final_strategy"]
    ratios, seed = cfg["ratios"], cfg["seed"]
    if strategy == "random_row":
        return random_row_split(df, ratios, seed)
    g = g or build_graph(df)
    if strategy == "component_whole":
        return component_whole_split(df, g, ratios, seed, cfg["size_buckets"])
    if strategy == "node_level":
        return node_level_split(df, g, ratios, seed)
    if strategy == "hybrid_component":
        return hybrid_component_split(df, g, ratios, seed, cfg["size_buckets"],
                                      cfg["large_component_cap"])
    raise ValueError(f"unknown strategy {strategy!r}")


# --------------------------------------------------------------------------- leakage

def question_overlap(df: pd.DataFrame, split: pd.Series, eval_split: str,
                     ref_split: str = "train", key=identity_key) -> dict:
    """How much of `eval_split` has already been seen in `ref_split`."""
    ref = df[split == ref_split]
    ev = df[split == eval_split]
    ref_q = set(ref["question1"].map(key)) | set(ref["question2"].map(key))
    a_seen = ev["question1"].map(key).isin(ref_q)
    b_seen = ev["question2"].map(key).isin(ref_q)
    ev_q = set(ev["question1"].map(key)) | set(ev["question2"].map(key))
    return {
        "unique_questions": len(ev_q),
        "questions_seen_in_ref": len(ev_q & ref_q),
        "question_overlap_rate": len(ev_q & ref_q) / max(len(ev_q), 1),
        "pairs_any_question_seen": float((a_seen | b_seen).mean()) if len(ev) else 0.0,
        "pairs_both_questions_seen": float((a_seen & b_seen).mean()) if len(ev) else 0.0,
    }


def transitive_label_leakage(df: pd.DataFrame, split: pd.Series, eval_split: str) -> dict:
    """Memorisation probe: predict 'duplicate' for an eval pair iff both questions
    are already linked by *training* duplicate pairs (transitive closure).
    No model involved - any signal here is pure leakage."""
    train = df[(split == "train") & (df["is_duplicate"] == 1)]
    keys = pd.concat([train["question1"], train["question2"]]).map(identity_key)
    codes, uniq = pd.factorize(keys)
    n = len(train)
    adj = sp.coo_matrix((np.ones(n), (codes[:n], codes[n:])), shape=(len(uniq), len(uniq)))
    _, comp = connected_components(adj, directed=False)
    comp_of = dict(zip(uniq, comp))

    ev = df[split == eval_split]
    ca = ev["question1"].map(identity_key).map(comp_of)
    cb = ev["question2"].map(identity_key).map(comp_of)
    fires = (ca.notna() & cb.notna() & (ca == cb)).to_numpy()
    y = ev["is_duplicate"].to_numpy()
    tp = int((fires & (y == 1)).sum())
    return {
        "rule_fires_on_pairs": float(fires.mean()) if len(ev) else 0.0,
        "rule_precision": tp / max(int(fires.sum()), 1),
        "share_of_positives_recovered": tp / max(int((y == 1).sum()), 1),
    }


def split_report(df: pd.DataFrame, split: pd.Series, g: QuestionGraph) -> dict:
    n = len(df)
    size = pd.Series(g.pair_component_size, index=df.index)
    profile_bins = [0, 1, 10, 500, np.inf]
    profile_names = ["isolated_pair(1)", "small(2-10)", "medium(11-500)", "large(>500)"]
    report = {"retention": float((split != DROPPED).mean()), "splits": {}}
    for s in SPLITS:
        m = split == s
        prof = pd.cut(size[m], profile_bins, labels=profile_names).value_counts(normalize=True)
        entry = {
            "pairs": int(m.sum()),
            "share_of_retained": float(m.sum() / max((split != DROPPED).sum(), 1)),
            "positive_rate": float(df.loc[m, "is_duplicate"].mean()),
            "component_size_profile": {k: round(float(prof.get(k, 0.0)), 4) for k in profile_names},
        }
        if s != "train":
            entry["overlap_with_train"] = question_overlap(df, split, s, "train")
            entry["overlap_with_train_loose_key"] = question_overlap(df, split, s, "train", loose_key)
            entry["transitive_label_leakage"] = transitive_label_leakage(df, split, s)
        report["splits"][s] = entry
    report["splits"]["test"]["overlap_with_val"] = question_overlap(df, split, "test", "val")
    report["dropped_pairs"] = int((split == DROPPED).sum())
    report["dropped_positive_rate"] = (float(df.loc[split == DROPPED, "is_duplicate"].mean())
                                       if (split == DROPPED).any() else None)
    assert report["splits"]["train"]["pairs"] + report["splits"]["val"]["pairs"] + \
        report["splits"]["test"]["pairs"] + report["dropped_pairs"] == n
    return report
