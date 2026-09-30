"""Compare split strategies and create the frozen train/val/test split (Phase 1).

Usage:
  python scripts/make_splits.py            # compare strategies, create split if none exists,
                                           # otherwise verify the existing split is reproduced
  python scripts/make_splits.py --force    # overwrite an existing frozen split (don't, after Phase 1)

Writes:
  artifacts/phase1/split_comparison.json
  data/processed/splits/split_assignments.csv   (id -> train/val/test/dropped_cross_split/excluded_invalid)
  data/processed/splits/{train,val,test}.csv    (raw rows, text untouched)
  data/processed/splits/split_metadata.json     (config, hashes, sizes, leakage stats)
  docs/figures/split_leakage_comparison.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.data import (PROJECT_ROOT, file_sha256, invalid_pair_mask, load_config,  # noqa: E402
                            load_raw_pairs, resolve)
from src.utils.splits import DROPPED, SPLITS, build_graph, make_split, split_report  # noqa: E402

STRATEGIES = ["random_row", "component_whole", "node_level", "hybrid_component"]
EXCLUDED = "excluded_invalid"


def ids_hash(ids) -> str:
    return hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()


def stability(df, g, cfg, strategy, seeds=range(10)) -> dict:
    """Spread of split size / positive rate across seeds (sensitivity to where big components land)."""
    rows = []
    for s in seeds:
        sp = make_split(df, {**cfg, "seed": s}, strategy, g)
        for name in SPLITS:
            m = sp == name
            rows.append({"seed": s, "split": name, "share": m.mean(), "pos": df.loc[m, "is_duplicate"].mean()})
    t = pd.DataFrame(rows).groupby("split").agg(share_min=("share", "min"), share_max=("share", "max"),
                                                pos_min=("pos", "min"), pos_max=("pos", "max"))
    return t.round(4).to_dict("index")


def plot_comparison(comparison: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
    names = list(comparison)
    anyq = [comparison[n]["report"]["splits"]["test"]["overlap_with_train"]["pairs_any_question_seen"] for n in names]
    both = [comparison[n]["report"]["splits"]["test"]["overlap_with_train"]["pairs_both_questions_seen"] for n in names]
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=SURFACE)
    b1 = ax.bar(x - w / 2 - 0.01, anyq, w, color=BLUE, label=">= 1 question also in train")
    b2 = ax.bar(x + w / 2 + 0.01, both, w, color=ORANGE, label="both questions also in train")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.1%}",
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [n.replace("_", "\n") for n in names])
    ax.set_ylim(0, max(anyq) * 1.25 + 0.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Test pairs whose questions were already seen in train", loc="left", color=INK, fontsize=11)
    ax.set_ylabel("share of test pairs", color=INK2)
    ax.legend(frameon=False, labelcolor=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2); ax.set_facecolor(SURFACE)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing frozen split")
    args = parser.parse_args()

    cfg = load_config()
    raw_path = resolve(cfg["raw_train_path"])
    split_dir = resolve(cfg["split_dir"])
    meta_path = split_dir / "split_metadata.json"
    art = PROJECT_ROOT / "artifacts" / "phase1"
    art.mkdir(parents=True, exist_ok=True)

    raw = load_raw_pairs(raw_path)
    invalid = invalid_pair_mask(raw)
    df = raw[~invalid].reset_index(drop=True)
    g = build_graph(df)

    # ---------------- strategy comparison
    comparison = {}
    for strat in STRATEGIES:
        sp = make_split(df, cfg, strat, g)
        comparison[strat] = {"report": split_report(df, sp, g)}
        if strat != "random_row":
            comparison[strat]["stability_over_10_seeds"] = stability(df, g, cfg, strat)
        print(f"[compare] {strat}: retention={comparison[strat]['report']['retention']:.4f}")
    (art / "split_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    plot_comparison(comparison, PROJECT_ROOT / "docs" / "figures" / "split_leakage_comparison.png")

    # ---------------- final split
    strategy = cfg["final_strategy"]
    split = make_split(df, cfg, strategy, g)
    assignments = pd.concat([
        pd.DataFrame({"id": df["id"], "split": split.to_numpy()}),
        pd.DataFrame({"id": raw.loc[invalid, "id"], "split": EXCLUDED}),
    ]).sort_values("id").reset_index(drop=True)
    split_ids = {s: assignments.loc[assignments.split == s, "id"].tolist() for s in SPLITS}
    new_hashes = {s: ids_hash(ids) for s, ids in split_ids.items()}

    if meta_path.exists() and not args.force:
        frozen = json.loads(meta_path.read_text(encoding="utf-8"))
        ok = frozen["split_id_sha256"] == new_hashes and frozen["raw_sha256"] == file_sha256(raw_path)
        print("[freeze] existing split found -> reproduced exactly" if ok else
              "[freeze] WARNING: regenerated split differs from frozen split; NOT overwriting")
        sys.exit(0 if ok else 1)

    split_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(split_dir / "split_assignments.csv", index=False)
    file_hashes = {}
    for s in SPLITS:
        out = split_dir / f"{s}.csv"
        df[split == s].to_csv(out, index=False)
        file_hashes[s] = file_sha256(out)

    report = comparison[strategy]["report"]
    meta = {
        "strategy": strategy,
        "config": cfg,
        "raw_sha256": file_sha256(raw_path),
        "raw_rows": int(len(raw)),
        "excluded_invalid_ids": raw.loc[invalid, "id"].tolist(),
        "counts": assignments["split"].value_counts().to_dict(),
        "positive_rate": {s: report["splits"][s]["positive_rate"] for s in SPLITS},
        "split_id_sha256": new_hashes,
        "split_file_sha256": file_hashes,
        "leakage": {s: {k: v for k, v in report["splits"][s].items() if k.startswith(("overlap", "transitive"))}
                    for s in ("val", "test")},
        "note": "Test split is frozen. Use only in Phase 8 final evaluation.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("strategy", "counts", "positive_rate", "leakage")}, indent=1))


if __name__ == "__main__":
    main()
