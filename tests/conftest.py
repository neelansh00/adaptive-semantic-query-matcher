import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data import load_config, load_raw_pairs, resolve  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_df(cfg):
    path = resolve(cfg["raw_train_path"])
    if not path.exists():
        pytest.skip("raw dataset not present (extract quora-question-pairs.zip into data/raw/)")
    return load_raw_pairs(path)


@pytest.fixture(scope="session")
def split_meta(cfg):
    path = resolve(cfg["split_dir"]) / "split_metadata.json"
    if not path.exists():
        pytest.skip("frozen split not created yet (run scripts/make_splits.py)")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def split_frames(cfg, split_meta):
    d = resolve(cfg["split_dir"])
    if not all((d / f"{s}.csv").exists() for s in ("train", "val", "test")):
        pytest.skip("split CSVs not regenerated (python scripts/reproduce.py --run extract splits)")
    return {s: pd.read_csv(d / f"{s}.csv", keep_default_na=False, na_values=[""])
            for s in ("train", "val", "test")}


@pytest.fixture
def toy_pairs():
    """Small graph: a hub question (q0) with many partners + isolated pairs."""
    rows = []
    for i in range(1, 31):  # hub component: 30 pairs
        rows.append(("Hub question?", f"Partner {i}?", i % 2))
    for i in range(200):  # 200 isolated pairs
        rows.append((f"Left {i}?", f"Right {i}?", int(i % 3 == 0)))
    df = pd.DataFrame(rows, columns=["question1", "question2", "is_duplicate"])
    df.insert(0, "id", range(len(df)))
    return df
