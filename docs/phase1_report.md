# Phase 1 Report: Dataset Audit and Problem Definition

## Dataset summary

- The repository contained only `quora-question-pairs.zip`. It holds `train.csv` (labelled), two different
  unlabelled `test.csv` variants (2.35M and 3.56M rows), and a constant sample submission.
  **Only `train.csv` has labels**, so all train/val/test data comes from it.
- `train.csv`: **404,290 rows × 6 columns** (`id, qid1, qid2, question1, question2, is_duplicate`).
  There are no duplicate rows, ids or reversed pairs, and no self-pairs.
- **Missing values:** 2 empty `question2` cells and 1 literal `n/a` placeholder in `question1`.
  Pandas' default parsing hides the `n/a` as NaN. These 3 rows (all labelled 0) are excluded.
- **Questions:** 537,933 unique qids and 537,151 unique normalised texts. 715 texts were posted under several qids.
- **Lengths:** 11.1 words on average, median 10, p99 31, max 237. There are 110 questions under 10 characters and 33 over 500.
- **Text quality:** decodes cleanly as UTF-8 (no mojibake or HTML entities). 949 questions contain `[math]` LaTeX tags,
  and 41,863 are multi-question posts.
- **Label noise:** 5 identical-text pairs are labelled 0, and 10 repeated pairs carry conflicting labels.

Details: [data_audit.md](data_audit.md).

## Label distribution

0 → 255,027 (63.1%) and 1 → 149,263 (**36.9%**). The imbalance is moderate, so accuracy is not a primary metric.

## Repeated-question analysis

- 20.8% of qids appear in more than one pair, and 61.3% of pairs contain a reused question. The maximum is 157 appearances
  ("What are the best ways to lose weight?").
- In the question graph (questions as nodes, pairs as edges) there are 200,338 components. The largest holds 17,624 pairs (4.4%).
- **The duplicate rate rises with component size**, from 22% for isolated pairs to 87% in 501–1,000-pair components.
  Question frequency alone gives ROC-AUC 0.698. This is a sampling shortcut, so frequency and graph
  features are excluded from all models.

## Leakage investigation

Under a stratified random row split:
- 39% of unique test questions also appear in train;
- 58.4% of test pairs have at least one question in train, and 29.9% have both;
- a transitive-closure rule over training duplicates, which uses no model and no text, fires on 21.1% of test pairs
  with **99.9% precision** and recovers **57.1% of all test positives**.

Random-split scores would therefore be inflated by memorisation to an unknown degree.

## Split strategies considered

| Strategy | Retained | Question overlap | Problem |
|---|---|---|---|
| Random row | 100% | 58% of test pairs | leakage (above) |
| Whole components | 100% | 0 | one 4.4% component dominates whichever split it lands in; test positive rate varies 37–43% across seeds |
| Node-level | 43% | 0 | discards 57% of data |
| **Hybrid** | **94.2%** | **0** | hub questions under-represented (see limitations) |

## Final split strategy

**Hybrid question-disjoint split** (seed 42):
- components with ≤ 500 pairs are assigned whole, stratified by component-size bucket;
- the 19 components with > 500 pairs are split per question with probabilities ∝ √ratio. Pairs whose questions land
  in different splits are dropped.

The √ correction was added after the first run showed hub pairs at 8.1% of train but only ~1.2% of val/test.
Rationale and full comparison: [split_strategy.md](split_strategy.md).

## Final train/validation/test sizes

| Split | Pairs | Share | Positive rate |
|---|---|---|---|
| train | 304,384 | 79.9% | 34.78% |
| val | 38,138 | 10.0% | 35.61% |
| test | 38,420 | 10.1% | 35.93% |
| dropped (cross-split hub pairs) | 23,345 | – | 68.6% |
| excluded (invalid) | 3 | – | 0% |

The question overlap between every pair of splits is 0, measured on the normalised-text identity. Under an aggressive
punctuation-stripping key, residual overlap touches 0.22% of val pairs and 0.10% of test pairs.
The component-size profile is matched across splits (large-component share 4.4 / 4.4 / 4.8%).

Saved in `data/processed/splits/`: `split_assignments.csv` (every id → split), `{train,val,test}.csv`,
and `split_metadata.json` (config, raw SHA-256, per-split id and file hashes, leakage statistics).

## Evaluation metrics

- **Headline:** F1 at a validation-tuned threshold.
- **Also reported:** precision, recall, PR-AUC and ROC-AUC. Accuracy is reported only.
- **Threshold analysis:** threshold sweep, PR curve, F1 vs threshold, and the validation-optimal threshold frozen for test.
- **Calibration:** Brier score and ECE, from Phase 6 onward.
- **Cluster level:** per-cluster P/R/F1/FPR/FNR and optimal threshold, plus macro and worst-cluster F1.

All of these are implemented and unit-tested in `src/evaluation/metrics.py`. See [evaluation_plan.md](evaluation_plan.md).

## Known limitations

1. The hybrid split drops 23,345 hub pairs (5.8%, 68.6% positive). Large-component pairs fall from 10% of the raw data
   to ~4.5% of each split, and the positive rate from 36.9% to ~35%. Results describe this filtered distribution.
2. Scores will not be comparable to published Quora results, which mostly use leaky random or Kaggle splits.
   Expect lower numbers.
3. Question identity is exact-text-based. Trivially edited near-duplicates can cross splits (< 0.25% of pairs).
4. Labels are noisy (identical-text negatives, conflicting repeats, debatable positives), which caps achievable accuracy.
5. The positive rate is a sampling artifact of the dataset construction, so precision does not transfer to deployment.
6. The cap of 500 was chosen by inspecting the component-size distribution, not by tuning. Alternatives are documented.
7. Git was not initialised: the folder was not a repository, so nothing was committed (see verification).

## Phase 1 verification

| Check | Result |
|---|---|
| `python scripts/audit_dataset.py` | runs, exit 0; writes `artifacts/phase1/audit_stats.json`, `audit_examples.json` and 3 figures |
| `python scripts/make_splits.py` (first run) | creates the frozen split and `split_comparison.json` |
| `python scripts/make_splits.py` (re-run) | `[freeze] existing split found -> reproduced exactly`, no overwrite |
| `python -m pytest` | **19 passed** |
| Raw data unmodified | `data/raw/train.csv` SHA-256 recorded in `split_metadata.json` and re-checked on every re-run |
| Split reproducibility | test regenerates the split from raw data and matches the stored per-split id hashes |
| No test-set mutation | test compares split-file SHA-256 against the metadata |
| Disjointness | tests assert 0 shared ids and 0 shared normalised questions across all three splits |
| Sizes and balance | tests assert 80/10/10 ± 1 pp and positive-rate spread < 2 pp |
| Test set usage | no model, threshold or statistic was computed on the test split beyond the split-balance checks above |

Tests (`tests/`): `test_data.py` (schema, target, invalid rows, loader behaviour), `test_splits.py`
(disjointness on toy and real data, reproducibility, seed sensitivity, hybrid drop rule, √ balancing,
frozen-split hashes, sizes), and `test_metrics.py` (confusion counts, threshold selection, ECE, per-cluster fallback).

PHASE 1 VERIFIED — READY FOR PHASE 2
