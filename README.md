# Adaptive Semantic Query Matcher

**Research question:** can query type (semantic groups discovered from embeddings), entity/constraint
consistency, and calibrated thresholds improve semantic duplicate detection beyond a single
embedding-similarity threshold?

Dataset: Quora Question Pairs (the labelled `train.csv` only; see [docs/data_audit.md](docs/data_audit.md)).

> **Status: Phase 6 complete** (1: audit and frozen splits; 2: lexical baselines; 3: deep models; 4: query clustering;
> 5: cluster-level error analysis; 6: entity/constraint features). Cluster-calibrated thresholds start in Phase 7. The full README (architecture, results, demo) is written in Phase 11.

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

## Phase 4 key findings (train questions only, no labels used)

- **Method:** K-Means on 424,012 unique train-question embeddings, compared for K ∈ {5, 8, 10, 12, 15, 20}, plus HDBSCAN.
- **No separated structure:** silhouette ≈ 0.02 at every K. HDBSCAN returns 32–100% noise or one dominant blob, so it is rejected for a measured reason.
- **K = 12 chosen on stability.** Robust stability (6 seed pairs, 3 split-halves) peaks at K = 12 (ARI 0.79 / 0.80). It also has the most balanced
  sizes (6.4–13.4%) and 10 of 12 clearly interpretable clusters. Single-run stability estimates were misleading; K = 5 looked perfectly stable but is the least stable.
- **Clusters (named after inspection):** a mix of topic and intent, e.g. definitions, product "which is best", how-to accounts/apps, learning
  and exam preparation, relationships, health, India-specific, and personal-experience questions.
- **Soft boundaries:** a third of validation pairs straddle two clusters. The centroids are frozen because K-Means refits are not bit-reproducible.

## Phase 5 key findings (validation; frozen clusters and model; nothing tuned)

- **Pairs are assigned by the pair-average embedding**, chosen on label-free grounds: it is symmetric and has full coverage. The question-1 rule changes a third
  of assignments if the questions are swapped.
- **Performance varies beyond chance.** Cluster F1 ranges 0.708–0.808 (std 0.033, against 0.009 for random groups), but F1 largely tracks
  duplicate prevalence (ρ = 0.69). The lowest-F1 regions (Education, Definitions, Accounts/apps) rank pairs normally; the advice regions
  (Learning, Relationships, Health) rank worst.
- **Optimal thresholds vary from 0.20 to 0.46**, beyond chance, but F1 plateaus are wide. Only 3 of 12 clusters have CIs excluding the global
  τ = 0.32, and the total in-sample headroom is just +0.006 F1. Thresholds tuned on fewer than about 500 pairs are mostly noise.
- **Manual error analysis (120 pairs):** among false positives, 27% likely label noise, 25% scope, 20% explicit constraint mismatch, 12% attribute
  swap. Among false negatives, 57% are low-overlap paraphrases.

## Phase 6 key findings (validation)

| Model | F1 | PR-AUC | Worst-cluster F1 |
|---|---|---|---|
| A. cosine similarity only | 0.735 | 0.765 | 0.651 |
| B. semantic model probability (Phase 3) | 0.771 | 0.824 | 0.708 |
| **C. semantic model + constraint features (boosted trees, spaCy-free)** | **0.783** | **0.839** | **0.719** |

- **The gain is significant:** C vs B is +0.012 F1 [+0.009, +0.015]. Controls show re-fitting or boosting the base score alone adds nothing.
- **High-overlap false positives roughly halve** for entity (0.22 → 0.12), number (0.47 → 0.18) and negation (0.72 → 0.33) differences.
  The cost is more false negatives on true duplicates that contain a superficial constraint difference.
- **Specificity/length signals are the largest contributor**; numbers, negation and entities add smaller, significant gains.
  spaCy NER added only +0.001 PR-AUC and was dropped.
- **Stacking without leakage:** base scores for training the meta-model come from 5-fold, question-disjoint cross-fitting.

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
| [docs/clustering.md](docs/clustering.md) | Phase 4 K-Means vs HDBSCAN, K selection, stability, cluster interpretation |
| [docs/cluster_error_analysis.md](docs/cluster_error_analysis.md) | Phase 5 pair assignment, per-cluster metrics, threshold variation, error categories |
| [docs/entity_constraint_features.md](docs/entity_constraint_features.md) | Phase 6 constraint features, cross-fitted meta-classifier, ablations, trade-offs |

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

# 6. Phase 4 clustering (train questions only)
python scripts/compare_clustering.py
python scripts/stability_recheck.py --k 5 8 10 12 15 20
python scripts/build_clusters.py        # uses frozen centroids; --refit to re-fit deliberately

# 7. Phase 5 cluster-level error analysis (validation only)
python scripts/analyze_clusters.py
python scripts/threshold_reliability.py
python scripts/summarize_manual_errors.py

# 8. Phase 6 entity/constraint features (spaCy model needed only for the NER ablation)
python -m spacy download en_core_web_sm
python scripts/build_constraint_features.py
python scripts/crossfit_base.py
python scripts/train_meta.py

# 9. Tests
python -m pytest
```

`make_splits.py` never overwrites an existing frozen split. It regenerates the split and checks the result
against the hashes in `data/processed/splits/split_metadata.json`.

## Project structure

```
configs/data.yaml          split configuration (seed, ratios, cap)
configs/clustering.yaml    clustering configuration (K grid, chosen K = 12, HDBSCAN settings)
data/raw/                  extracted Kaggle files (not versioned, never modified)
data/processed/splits/     frozen train/val/test + split_metadata.json
scripts/                   Phase 1-2: audit_dataset, make_splits, run_baselines, analyze_baseline_errors
                           Phase 3: encode_questions, run_sbert, train_bilstm, run_encoder_control, compare_phase3
                           Phase 4: compare_clustering, stability_recheck, build_clusters
                           Phase 5: analyze_clusters, threshold_reliability, summarize_manual_errors
                           Phase 6: build_constraint_features, crossfit_base, train_meta
src/utils/                 data loading (data.py), split strategies + leakage metrics (splits.py)
src/evaluation/            metrics.py (F1/PR/ROC, threshold sweep, ECE, per-cluster metrics)
src/preprocessing/         minimal text normalisation + tokenisation
src/features/              symmetric lexical pair features (lexical.py), entity/number/date/negation/specificity features (constraints.py)
src/evaluation/error_analysis.py   heuristic error tags (analysis only)
src/models/                 siamese_bilstm.py, sentence_encoder.py (cached embeddings), meta.py (Phase 6 variants),
                           predictors.py (raw text -> score, incl. MetaPredictor)
src/utils/torch_utils.py   device (CUDA if present, else CPU), seeding, threads
src/clustering/core.py     K-Means fitting, frozen centroid model, metrics, stability, c-TF-IDF descriptions
src/clustering/pairs.py    pair-to-cluster rules (q1 / pair-average / same-only), assignment margin
src/evaluation/cluster_analysis.py   per-cluster metrics, bootstrap CIs, random-partition null
src/calibration/           placeholder for later phases
tests/                     Phase 1-6 tests
docs/                      documentation + figures
artifacts/phase1/          audit statistics, split comparison
artifacts/phase2/          baseline metrics, val predictions, error analysis, fitted models (joblib)
artifacts/phase3/          BiLSTM + SBERT heads, val scores, encoder control, comparison.json
data/processed/embeddings/ cached MiniLM question embeddings (train, val)
artifacts/phase4/          K sweep, stability, HDBSCAN, frozen cluster centroids, cluster descriptions/names
artifacts/phase5/          per-cluster metrics, null, threshold reliability, pair assignments, manual error labels
artifacts/phase6/          meta-model bundle, variant results + bootstraps, slices, probes, cross-fit report, latency
```
