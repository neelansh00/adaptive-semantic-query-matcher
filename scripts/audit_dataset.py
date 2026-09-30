"""Reproducible dataset audit (Phase 1).

Usage:  python scripts/audit_dataset.py
Writes: artifacts/phase1/audit_stats.json, artifacts/phase1/audit_examples.json,
        docs/figures/{question_length,question_reuse,positive_rate_by_component}.png
Reads the raw CSV only; never modifies it.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.data import (PROJECT_ROOT, TARGET, file_sha256, identity_key, invalid_pair_mask,  # noqa: E402
                            load_config, load_raw_pairs, resolve)
from src.utils.splits import build_graph  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "phase1"
FIG = PROJECT_ROOT / "docs" / "figures"
SEED = 42

# Plot styling (reference palette: blue slot 1, neutral inks).
BLUE, ORANGE, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.set_facecolor(SURFACE)


def archive_inventory() -> list[dict]:
    zpath = PROJECT_ROOT / "quora-question-pairs.zip"
    if not zpath.exists():
        return []
    with zipfile.ZipFile(zpath) as z:
        return [{"name": i.filename, "bytes": i.file_size} for i in z.infolist()]


def word_jaccard(a: pd.Series, b: pd.Series) -> np.ndarray:
    tok = re.compile(r"\w+")
    out = np.empty(len(a))
    for i, (x, y) in enumerate(zip(a.str.lower(), b.str.lower())):
        sx, sy = set(tok.findall(x)), set(tok.findall(y))
        out[i] = len(sx & sy) / len(sx | sy) if sx | sy else 0.0
    return out


def text_issue_counts(q: pd.Series) -> dict:
    patterns = {
        "non_ascii": r"[^\x00-\x7f]",
        "mojibake_like": r"Ã.|â€|Â",
        "html_entity": r"&(?:amp|lt|gt|quot|#\d+);",
        "latex_math_tag": r"\[math\]",
        "url": r"https?://|www\.",
        "control_char": r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        "newline": r"[\r\n]",
        "no_letters": r"^[^A-Za-z]*$",
    }
    counts = {k: int(q.str.contains(p, regex=True).sum()) for k, p in patterns.items()}
    counts["no_question_mark"] = int((~q.str.contains(r"\?", regex=True)).sum())
    counts["multiple_question_marks"] = int((q.str.count(r"\?") > 1).sum())
    return counts


def ex(df, rows, n=5):
    cols = ["id", "question1", "question2", TARGET]
    return df.loc[rows[:n], cols].to_dict("records")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    raw_path = resolve(cfg["raw_train_path"])
    df = load_raw_pairs(raw_path)
    stats: dict = {"archive_inventory": archive_inventory(), "raw_train_path": cfg["raw_train_path"],
                   "raw_sha256": file_sha256(raw_path)}

    # ---- structure
    stats["shape"] = list(df.shape)
    stats["columns"] = df.columns.tolist()
    stats["dtypes"] = df.dtypes.astype(str).to_dict()
    stats["missing_values"] = df.isna().sum().astype(int).to_dict()
    stats["rows_with_missing_question"] = df[df[["question1", "question2"]].isna().any(axis=1)] \
        .fillna("<NaN>").to_dict("records")
    literal_null_like = {"NA", "N/A", "n/a", "nan", "NaN", "null", "None", "none"}
    stats["invalid_pairs_excluded_from_splits"] = df[invalid_pair_mask(df)].fillna("<empty>").to_dict("records")
    stats["literal_null_like_questions"] = int(pd.concat([df.question1, df.question2])
                                               .isin(literal_null_like).sum())

    q1, q2 = df["question1"].fillna(""), df["question2"].fillna("")
    k1, k2 = q1.map(identity_key), q2.map(identity_key)

    # ---- duplicates / integrity
    unordered_qid = pd.Series(list(map(tuple, np.sort(df[["qid1", "qid2"]].to_numpy(), axis=1))))
    unordered_txt = pd.Series([tuple(sorted(t)) for t in zip(k1, k2)])
    same_text = k1 == k2
    stats["integrity"] = {
        "duplicate_rows_all_columns": int(df.duplicated().sum()),
        "duplicate_rows_excluding_id": int(df.drop(columns="id").duplicated().sum()),
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "duplicate_pairs_unordered_qid": int(unordered_qid.duplicated().sum()),
        "duplicate_pairs_unordered_normalized_text": int(unordered_txt.duplicated().sum()),
        "self_pairs_same_qid": int((df.qid1 == df.qid2).sum()),
        "pairs_identical_after_normalization": int(same_text.sum()),
        "label_of_identical_pairs": df.loc[same_text, TARGET].value_counts().to_dict(),
        "labels_values": sorted(df[TARGET].unique().tolist()),
    }
    dup_txt_mask = unordered_txt.duplicated(keep=False)
    if dup_txt_mask.any():
        conflicting = df[dup_txt_mask].groupby(unordered_txt[dup_txt_mask])[TARGET].nunique()
        stats["integrity"]["repeated_text_pairs_with_conflicting_labels"] = int((conflicting > 1).sum())

    qid_text = pd.concat([pd.DataFrame({"qid": df.qid1, "q": q1}), pd.DataFrame({"qid": df.qid2, "q": q2})])
    stats["integrity"]["qids_with_multiple_texts"] = int((qid_text.groupby("qid").q.nunique() > 1).sum())
    stats["integrity"]["normalized_texts_with_multiple_qids"] = int(
        (qid_text.assign(k=qid_text.q.map(identity_key)).groupby("k").qid.nunique() > 1).sum())

    # ---- labels
    vc = df[TARGET].value_counts().sort_index()
    stats["label_distribution"] = {"counts": vc.to_dict(), "positive_rate": float(df[TARGET].mean())}

    # ---- question reuse
    qid_counts = qid_text["qid"].value_counts()
    key_counts = pd.concat([k1, k2]).value_counts()
    stats["questions"] = {
        "unique_qids": int(qid_counts.size),
        "unique_raw_texts": int(qid_text["q"].nunique()),
        "unique_normalized_texts": int(key_counts.size),
        "question_slots_total": int(2 * len(df)),
        "qids_appearing_more_than_once": int((qid_counts > 1).sum()),
        "share_of_qids_repeated": float((qid_counts > 1).mean()),
        "max_appearances": int(qid_counts.max()),
        "appearance_quantiles": qid_counts.quantile([.5, .9, .99, .999]).to_dict(),
        "pairs_where_a_question_is_repeated": float(
            ((df.qid1.map(qid_counts) > 1) | (df.qid2.map(qid_counts) > 1)).mean()),
        "top_repeated": [{"question": qid_text.drop_duplicates("qid").set_index("qid").q[q], "count": int(c)}
                         for q, c in qid_counts.head(10).items()],
    }
    freq_hist = qid_counts.value_counts().sort_index()

    # ---- question graph / frequency shortcut
    g = build_graph(df.assign(question1=q1, question2=q2))
    comp_size = pd.Series(g.pair_component_size)
    bins = [0, 1, 2, 5, 10, 50, 100, 500, 1000, 5000, np.inf]
    by_size = df.groupby(pd.cut(comp_size, bins), observed=True)[TARGET].agg(["size", "mean"])
    stats["question_graph"] = {
        "nodes": int(g.n_nodes),
        "components": int(len(g.component_pairs)),
        "largest_components_pairs": np.sort(g.component_pairs)[::-1][:10].tolist(),
        "largest_component_share": float(g.component_pairs.max() / len(df)),
        "positive_rate_by_component_size": {str(k): {"pairs": int(r["size"]), "positive_rate": round(r["mean"], 4)}
                                            for k, r in by_size.iterrows()},
    }
    max_freq = np.maximum(df.qid1.map(qid_counts), df.qid2.map(qid_counts))
    from sklearn.metrics import roc_auc_score
    stats["question_graph"]["auc_of_max_question_frequency_alone"] = float(roc_auc_score(df[TARGET], max_freq))

    # ---- lengths
    lens = {}
    for name, s in {"chars": pd.concat([q1, q2]).str.len(),
                    "words": pd.concat([q1, q2]).str.split().str.len()}.items():
        lens[name] = {"mean": float(s.mean()), "median": float(s.median()), "min": int(s.min()),
                      "p1": float(s.quantile(.01)), "p99": float(s.quantile(.99)), "max": int(s.max())}
    stats["lengths"] = lens
    all_q = pd.concat([q1, q2], ignore_index=True)
    all_len = all_q.str.len()
    stats["length_extremes"] = {
        "shortest": all_q[all_len.sort_values().index].drop_duplicates().head(10).tolist(),
        "longest_chars": all_len.sort_values(ascending=False).head(3).tolist(),
        "longest_preview": [t[:200] + " ..." for t in
                            all_q[all_len.sort_values(ascending=False).index].drop_duplicates().head(3)],
        "questions_under_10_chars": int((all_len < 10).sum()),
        "questions_over_500_chars": int((all_len > 500).sum()),
    }
    stats["length_by_label"] = df.assign(
        abs_word_diff=(q1.str.split().str.len() - q2.str.split().str.len()).abs()
    ).groupby(TARGET)["abs_word_diff"].median().to_dict()

    # ---- text issues
    stats["text_issues_question_slots"] = text_issue_counts(all_q)

    # ---- examples
    rng = np.random.default_rng(SEED)
    jac = word_jaccard(q1, q2)
    stats["word_jaccard_by_label"] = pd.Series(jac).groupby(df[TARGET]).describe().round(3).to_dict("index")
    pos = np.flatnonzero(df[TARGET] == 1)
    neg = np.flatnonzero(df[TARGET] == 0)
    hard_pos = np.flatnonzero((df[TARGET] == 1) & (jac < 0.2))
    hard_neg = np.flatnonzero((df[TARGET] == 0) & (jac > 0.8))
    stats["difficulty_counts"] = {"positives_word_jaccard_lt_0.2": int(len(hard_pos)),
                                  "negatives_word_jaccard_gt_0.8": int(len(hard_neg))}
    examples = {
        "duplicate_pairs": ex(df, rng.choice(pos, 5, replace=False)),
        "non_duplicate_pairs": ex(df, rng.choice(neg, 5, replace=False)),
        "difficult_positives_low_overlap": ex(df, rng.choice(hard_pos, 6, replace=False), 6),
        "difficult_negatives_high_overlap": ex(df, rng.choice(hard_neg, 8, replace=False), 8),
        "identical_text_labelled_non_duplicate": ex(df, np.flatnonzero(same_text & (df[TARGET] == 0))),
    }

    (OUT / "audit_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    (OUT / "audit_examples.json").write_text(json.dumps(examples, indent=2, default=str), encoding="utf-8")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    words = all_q.str.split().str.len()
    ax.hist(words.clip(upper=60), bins=range(0, 62), color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(words.median(), color=INK2, linestyle="--", linewidth=1)
    ax.text(words.median() + 1, ax.get_ylim()[1] * 0.9, f"median = {words.median():.0f} words", color=INK2)
    style(ax, "Question length (words, clipped at 60)", "words per question", "questions")
    fig.tight_layout(); fig.savefig(FIG / "question_length.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    ax.scatter(freq_hist.index, freq_hist.values, s=14, color=BLUE)
    ax.set_xscale("log"); ax.set_yscale("log")
    style(ax, "Question reuse: how many pairs each qid appears in", "appearances of a qid (log)",
          "number of qids (log)")
    fig.tight_layout(); fig.savefig(FIG / "question_reuse.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    labels = ["1", "2", "3-5", "6-10", "11-50", "51-100", "101-500", "501-1k", "1k-5k", ">5k"]
    ax.bar(labels, by_size["mean"].values, color=BLUE, width=0.7)
    for i, (r, n) in enumerate(zip(by_size["mean"].values, by_size["size"].values)):
        ax.text(i, r + 0.015, f"{r:.2f}", ha="center", color=INK, fontsize=8)
    ax.axhline(df[TARGET].mean(), color=INK2, linestyle="--", linewidth=1)
    ax.text(-0.45, df[TARGET].mean() + 0.10, f"overall rate {df[TARGET].mean():.2f}", color=INK2, ha="left", fontsize=8)
    ax.set_ylim(0, 1)
    style(ax, "Duplicate rate rises with question-graph component size",
          "pairs in the connected component of the pair", "positive rate")
    fig.tight_layout(); fig.savefig(FIG / "positive_rate_by_component.png", dpi=130); plt.close(fig)

    print(json.dumps({k: stats[k] for k in ["shape", "missing_values", "integrity", "label_distribution",
                                            "questions", "question_graph", "lengths", "length_extremes",
                                            "length_by_label", "text_issues_question_slots",
                                            "difficulty_counts", "word_jaccard_by_label",
                                            "literal_null_like_questions", "archive_inventory"]},
                     indent=1, default=str))
    print(json.dumps(examples, indent=1, default=str))


if __name__ == "__main__":
    main()
