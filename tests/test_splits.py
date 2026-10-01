import hashlib

import numpy as np
import pytest

from src.utils.data import file_sha256, identity_key, invalid_pair_mask, resolve
from src.utils.splits import DROPPED, SPLITS, build_graph, make_split, node_probabilities, question_overlap

TOY_CFG = {"seed": 7, "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
           "size_buckets": [1, 2, 5, 10], "large_component_cap": 10}


def questions(df):
    return set(df["question1"].map(identity_key)) | set(df["question2"].map(identity_key))


# ---------------------------------------------------------------- logic on a toy graph

@pytest.mark.parametrize("strategy", ["component_whole", "node_level", "hybrid_component"])
def test_group_strategies_are_question_disjoint(toy_pairs, strategy):
    sp = make_split(toy_pairs, TOY_CFG, strategy)
    for s in ("val", "test"):
        assert question_overlap(toy_pairs, sp, s, "train")["questions_seen_in_ref"] == 0


def test_random_row_split_leaks_on_hub_questions(toy_pairs):
    sp = make_split(toy_pairs, TOY_CFG, "random_row")
    leaked = sum(question_overlap(toy_pairs, sp, s)["questions_seen_in_ref"] for s in ("val", "test"))
    assert leaked > 0


def test_split_is_reproducible_and_seed_sensitive(toy_pairs):
    a = make_split(toy_pairs, TOY_CFG, "hybrid_component")
    b = make_split(toy_pairs, TOY_CFG, "hybrid_component")
    c = make_split(toy_pairs, {**TOY_CFG, "seed": 8}, "hybrid_component")
    assert a.equals(b)
    assert not a.equals(c)


def test_hybrid_only_drops_pairs_from_large_components(toy_pairs):
    g = build_graph(toy_pairs)
    sp = make_split(toy_pairs, TOY_CFG, "hybrid_component", g)
    dropped = sp == DROPPED
    assert (g.pair_component_size[dropped.to_numpy()] > TOY_CFG["large_component_cap"]).all()


def test_node_probabilities_give_target_pair_ratios():
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    p2 = node_probabilities(ratios) ** 2
    np.testing.assert_allclose(p2 / p2.sum(), [0.8, 0.1, 0.1])


# ---------------------------------------------------------------- frozen split on real data

def test_frozen_split_reproduces_from_raw(raw_df, cfg, split_meta):
    df = raw_df[~invalid_pair_mask(raw_df)].reset_index(drop=True)
    sp = make_split(df, cfg)
    for s in SPLITS:
        ids = ",".join(map(str, sorted(df.loc[sp == s, "id"])))
        assert hashlib.sha256(ids.encode()).hexdigest() == split_meta["split_id_sha256"][s]


def test_split_files_not_mutated(cfg, split_meta):
    if not (resolve(cfg["split_dir"]) / "train.csv").exists():
        pytest.skip("split CSVs not regenerated (python scripts/reproduce.py --run extract splits)")
    for s in SPLITS:
        assert file_sha256(resolve(cfg["split_dir"]) / f"{s}.csv") == split_meta["split_file_sha256"][s]


def test_split_sizes_and_balance(split_frames, split_meta):
    sizes = {s: len(f) for s, f in split_frames.items()}
    assert sizes == {s: split_meta["counts"][s] for s in SPLITS}
    total = sum(sizes.values())
    for s, target in {"train": 0.8, "val": 0.1, "test": 0.1}.items():
        assert abs(sizes[s] / total - target) < 0.01
    rates = [f["is_duplicate"].mean() for f in split_frames.values()]
    assert max(rates) - min(rates) < 0.02


def test_splits_are_id_and_question_disjoint(split_frames):
    ids = [set(f["id"]) for f in split_frames.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    q = {s: questions(f) for s, f in split_frames.items()}
    assert not (q["train"] & q["val"])
    assert not (q["train"] & q["test"])
    assert not (q["val"] & q["test"])
