# Adaptive Semantic Query Matcher

**Research question:** can query type (semantic groups discovered from embeddings), entity/constraint
consistency, and calibrated thresholds improve semantic duplicate detection beyond a single
embedding-similarity threshold?

Dataset: Quora Question Pairs (the labelled `train.csv` only; see [docs/data_audit.md](docs/data_audit.md)).

> **Status: Phase 3 complete** (Phase 1: audit, leakage, frozen splits; Phase 2: lexical baselines;
> Phase 3: Siamese BiLSTM + sentence-transformer models). Clustering starts in Phase 4. The full README (architecture, results, demo) is written in Phase 11.

## Phase 1 key findings

- **Size and balance:** 404,290 labelled pairs, 36.9% duplicates. 3 rows are invalid (an empty or `n/a` question) and are excluded.
- **Heavy question reuse:** 61% of pairs contain a question that appears in other pairs.
- **Random splits leak badly:** under a random row split, 58% of test pairs share a question with train.
  A text-free transitive-label rule then labels 21% of test pairs at 99.9% precision.
- **Question frequency is a shortcut:** it predicts the label with ROC-AUC 0.70 without reading any text.
  Frequency and graph features are therefore banned as model inputs.
- **Chosen split:** a hybrid question-disjoint split. Question overlap between splits is 0, 94.2% of pairs are retained,
  and the split is 80/10/10 with matched positive rates (34.8 / 35.6 / 35.9%) and matched component-size profiles.

## Phase 2 key findings (validation)

| Model | F1 (tuned τ) | PR-AUC |
|---|---|---|
| cosine TF-IDF | 0.618 | 0.479 |
| LR on 11 lexical features | 0.659 | 0.614 |
| LR on TF-IDF pair vector | 0.698 | 0.736 |
| **LR on TF-IDF pair + lexical** | **0.730** | **0.771** |

- **Threshold:** the F1-optimal threshold is 0.20–0.32, not 0.5. Tuning it adds 5–16 F1 points.
- **Overlap is non-monotonic:** near-identical pairs are *less* often duplicates (44%) than moderately similar ones (55%).
  They are mostly templated questions that differ in one entity, number or attribute.
- **Where the errors fall:** false positives concentrate at high overlap and false negatives at low overlap (synonyms, aliases).
- **Template learning:** the best lexical model gives an exact duplicate ("What is formwork?" ×2) probability 0.00.

## Phase 3 key findings (validation; Phase 2 bar F1 0.730 / PR-AUC 0.771)

| Model | F1 | PR-AUC | ΔF1 vs Phase 2 (95% CI) |
|---|---|---|---|
| Siamese BiLSTM (PyTorch, from scratch) | 0.732 | 0.773 | +0.002 [−0.002, +0.007]: not significant |
| MiniLM cosine (single similarity threshold) | 0.735 | 0.765 | +0.005 [−0.001, +0.010]: not significant |
| MiniLM + logistic-regression head | 0.750 | 0.792 | +0.020 [+0.015, +0.025] |
| **MiniLM + MLP head (primary model)** | **0.771** | **0.824** | **+0.041 [+0.036, +0.047]** |

- **Paraphrases:** embeddings cut low-overlap false negatives from 46% to 28%.
- **Constraints:** a raw embedding-similarity threshold is the *worst* model on high-overlap negatives. It accepts 81% of number
  mismatches and 84% of negation mismatches. The best model still accepts 47% and 72%, which motivates Phase 6.
- **Contamination:** `all-MiniLM-L6-v2` was pretrained on Quora duplicate triplets. In a control with identical heads, an encoder with no
  Quora data listed scores 3–4.5 F1 points lower. Absolute scores are therefore optimistic (see [docs/deep_models.md](docs/deep_models.md)).
- **Hardware:** CPU only (no CUDA GPU available). The frozen encoder costs ~28 ms per pair end to end; no fine-tuning was needed.

## Documentation

| Doc | Content |
|---|---|
| [docs/problem_definition.md](docs/problem_definition.md) | target, what "duplicate" means, mismatch types, excluded signals |
| [docs/data_audit.md](docs/data_audit.md) | full dataset audit |
| [docs/split_strategy.md](docs/split_strategy.md) | leakage analysis, 4 split strategies compared, final choice |
| [docs/evaluation_plan.md](docs/evaluation_plan.md) | metrics, threshold analysis, cluster-level metrics |
| [docs/phase1_report.md](docs/phase1_report.md) | Phase 1 summary and verification |
| [docs/baseline_results.md](docs/baseline_results.md) | Phase 2 preprocessing, lexical features, baselines, failure analysis |
| [docs/deep_models.md](docs/deep_models.md) | Phase 3 Siamese BiLSTM, sentence-transformer heads, contamination control, error analysis, latency |

## Reproduce

```bash
pip install -r requirements.txt

# 1. Extract the Kaggle archive into data/raw/ (the archive itself is left untouched)
python -c "import zipfile; z=zipfile.ZipFile('quora-question-pairs.zip'); z.extract('train.csv.zip','data/raw'); zipfile.ZipFile('data/raw/train.csv.zip').extractall('data/raw')"

# 2. Audit (writes artifacts/phase1/audit_*.json and docs/figures/*.png)
python scripts/audit_dataset.py

# 3. Compare split strategies and create / verify the frozen split
python scripts/make_splits.py

# 4. Phase 2 baselines (~10 min) and overlap-controlled error analysis
python scripts/run_baselines.py
python scripts/analyze_baseline_errors.py

# 5. Phase 3 deep models (CPU; see docs/deep_models.md for timings)
python scripts/encode_questions.py --model sentence-transformers/all-MiniLM-L6-v2
python scripts/run_sbert.py
python scripts/train_bilstm.py
python scripts/encode_questions.py --model sentence-transformers/nli-distilroberta-base-v2 --control-subset
python scripts/run_encoder_control.py
python scripts/compare_phase3.py

# 6. Tests
python -m pytest
```

`make_splits.py` never overwrites an existing frozen split. It regenerates the split and checks the result
against the hashes in `data/processed/splits/split_metadata.json`.

## Project structure

```
configs/data.yaml          split configuration (seed, ratios, cap)
data/raw/                  extracted Kaggle files (not versioned, never modified)
data/processed/splits/     frozen train/val/test + split_metadata.json
scripts/                   Phase 1-2: audit_dataset, make_splits, run_baselines, analyze_baseline_errors
                           Phase 3: encode_questions, run_sbert, train_bilstm, run_encoder_control, compare_phase3
src/utils/                 data loading (data.py), split strategies + leakage metrics (splits.py)
src/evaluation/            metrics.py (F1/PR/ROC, threshold sweep, ECE, per-cluster metrics)
src/preprocessing/         minimal text normalisation + tokenisation
src/features/              symmetric lexical pair features
src/evaluation/error_analysis.py   heuristic error tags (analysis only)
src/models/                 siamese_bilstm.py, sentence_encoder.py (cached embeddings), predictors.py (raw text -> score)
src/utils/torch_utils.py   device (CUDA if present, else CPU), seeding, threads
src/{clustering,calibration}/   placeholders for later phases
tests/                     Phase 1-3 tests
docs/                      documentation + figures
artifacts/phase1/          audit statistics, split comparison
artifacts/phase2/          baseline metrics, val predictions, error analysis, fitted models (joblib)
artifacts/phase3/          BiLSTM + SBERT heads, val scores, encoder control, comparison.json
data/processed/embeddings/ cached MiniLM question embeddings (train, val)
```
