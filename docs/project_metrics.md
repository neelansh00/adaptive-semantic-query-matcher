# Project Metrics

Every number below is measured and traceable to the artifact named in its section. "val" = frozen validation split; "test" = frozen test split, evaluated once.

## Dataset / scale

Sources: `artifacts/phase1/audit_stats.json`, `split_comparison.json`, `data/processed/splits/split_metadata.json`.

| Metric | Value |
|---|---|
| Labelled pairs (Kaggle `train.csv`) | 404,290 (6 columns) |
| Duplicate rate | 36.92% (149,263 / 404,290) |
| Unique question ids | 537,933 |
| Pairs containing a reused question | 61.3% |
| Max appearances of a single question | 157 |
| ROC-AUC of question frequency alone (no text): a leakage shortcut | 0.698 |
| Random row split: test pairs sharing a question with train | 58.4% |
| Random row split: text-free transitive rule | fires on 21.1% of test pairs, 99.9% precision, recovers 57.1% of positives |
| Question-disjoint split, train / val / test pairs | 304,384 / 38,138 / 38,420 |
| Duplicate rate, train / val / test | 34.78% / 35.61% / 35.93% |
| Questions shared between splits | 0 |
| Pairs retained | 94.2% (23,345 cross-split hub pairs dropped, 3 invalid rows excluded) |
| Unique train questions embedded / clustered | 424,012 |

## Baseline results (val)

Source: `artifacts/phase2/baseline_metrics.json`. τ = validation-F1-optimal threshold.

| Model | τ | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Majority class | – | 0.000 | – | – | 0.500 | 0.356 |
| TF-IDF cosine | 0.20 | 0.6181 | 0.4738 | 0.8886 | 0.7032 | 0.4791 |
| LR, 11 lexical features | 0.23 | 0.6594 | 0.5143 | 0.9186 | 0.7864 | 0.6137 |
| LR, TF-IDF pair vector | 0.25 | 0.6976 | 0.6135 | 0.8084 | 0.8457 | 0.7356 |
| **LR, TF-IDF pair + lexical** | 0.32 | **0.7299** | 0.6591 | 0.8177 | 0.8739 | **0.7705** |

- Tuning the threshold instead of using 0.5 raised F1 by 5–16 points across these models.
- Near-identical pairs (word Jaccard 0.8–1.0) are 44.0% duplicates, fewer than moderately similar pairs (0.4–0.6: 54.9%).

## Deep-model results (val)

Sources: `artifacts/phase3/comparison.json`, `encoder_control.json`.

| Model | τ | F1 | Precision | Recall | ROC-AUC | PR-AUC | ΔF1 vs best lexical [95% CI] |
|---|---|---|---|---|---|---|---|
| Siamese BiLSTM (9.09M params) | 0.57 | 0.7323 | 0.6615 | 0.8200 | 0.8742 | 0.7730 | +0.0024 [−0.0023, +0.0069] |
| MiniLM cosine | 0.76 | 0.7347 | 0.6395 | 0.8632 | 0.8760 | 0.7645 | +0.0049 [−0.0006, +0.0101] |
| MiniLM + LR head | 0.33 | 0.7500 | 0.6585 | 0.8711 | 0.8910 | 0.7924 | +0.0203 [+0.0150, +0.0252] |
| **MiniLM + MLP head** | 0.32 | **0.7711** | 0.6935 | 0.8683 | 0.9066 | **0.8237** | **+0.0413 [+0.0358, +0.0470]** |

- **Low-overlap duplicates missed** (FNR at word Jaccard ≤ 0.2): 45.5% for the lexical model → 27.9% for MiniLM + MLP.
- **High-overlap negatives wrongly accepted by MiniLM cosine:** 81.3% of those with a number difference, 83.6% of those with a negation difference.
- **Contamination control** (identical LR head, same 60k train pairs): MiniLM cosine / LR F1 0.7347 / 0.7434, against 0.6896 / 0.7131 for an
  NLI-only encoder with no Quora data listed.
- **Latency, CPU, single pair (Phase 3 measurement):** TF-IDF 2.7 ms, BiLSTM 3.8 ms, MiniLM + MLP 27.9 ms.
- **Size:** MiniLM encoder 86.6 MB + MLP head 0.76 MB.

## Final frozen evaluation (test, n = 38,420; evaluated once)

Sources: `artifacts/phase8/test_results.json`, `latency.json`, `manual_error_summary.json`.

| System | τ | F1 | Precision | Recall | ROC-AUC | PR-AUC | Macro-cluster F1 | Worst-cluster F1 |
|---|---|---|---|---|---|---|---|---|
| TF-IDF pair + lexical LR | 0.32 | 0.7322 | 0.6623 | 0.8185 | 0.8759 | 0.7756 | 0.7241 | 0.5994 |
| Siamese BiLSTM | 0.57 | 0.7362 | 0.6690 | 0.8185 | 0.8783 | 0.7810 | 0.7295 | 0.6011 |
| MiniLM cosine | 0.76 | 0.7330 | 0.6338 | 0.8691 | 0.8735 | 0.7666 | 0.7298 | 0.6444 |
| **Baseline:** MiniLM + MLP | 0.32 | 0.7750 | 0.6988 | 0.8700 | 0.9064 | 0.8243 | 0.7709 | 0.6684 |
| **Final:** + constraint meta-classifier | 0.35 | **0.78559** | 0.7154 | 0.8710 | 0.9144 | **0.83355** | 0.7824 | 0.6954 |

| Final − baseline (paired bootstrap, 1,000 resamples) | Δ | 95% CI |
|---|---|---|
| F1 | +0.0106 | [+0.0076, +0.0136] |
| Precision | +0.0167 | [+0.0132, +0.0200] |
| Recall | +0.0010 | [−0.0032, +0.0050] (n.s.) |
| PR-AUC | +0.0092 | [+0.0060, +0.0121] |
| Macro-cluster F1 | +0.0115 | [+0.0085, +0.0147] |
| Worst-cluster F1 | +0.0262 | [+0.0097, +0.0417] |
| FPR | −0.0161 | [−0.0191, −0.0127] |

- **Confusion counts (baseline → final):** FP 5,177 → 4,782 · FN 1,795 → 1,781 · TP 12,009 → 12,023 · TN 19,439 → 19,834.
- **Calibration:** ECE 0.0267 → 0.0189; Brier 0.1202 → 0.1142.
- **High-overlap FPR, baseline → final:**
  - entity difference 0.174 → 0.075 (n = 1,778 negatives);
  - number difference 0.502 → 0.218 (n = 239);
  - negation difference 0.641 → 0.315 (n = 92).
- **Per cluster:** 10 of 12 clusters improve (+0.006 to +0.027), 2 are unchanged (Δ ≤ 0.001), 0 worsen. The worst cluster, C6, goes from 0.668 to 0.695.
- **Latency, CPU, measured side by side:** batch 10.65 → 11.05 ms per pair; single pair median 50.6 → 69.2 ms.
- **Added model size:** meta-model 0.35 MB.
- **Remaining errors** (30 false positives + 30 false negatives of the final system, manually labelled):
  - false positives: 27% likely label noise, 27% broader/narrower scope, 20% different aspect, 17% attribute/role swap, 7% explicit constraint mismatch;
  - false negatives: 53% low-overlap paraphrase.

## Cluster analysis

Sources: `artifacts/phase4/*`, `artifacts/phase5/summary.json`, `artifacts/phase7/results.json`.

| Metric | Value |
|---|---|
| K-Means K tested | 5, 8, 10, 12, 15, 20 |
| Chosen K | 12 |
| Silhouette (20k sample) at K = 12 | 0.020 (0.015–0.021 across all K) |
| Davies-Bouldin / Calinski-Harabasz at K = 12 | 5.20 / 3,781 |
| Robust stability at K = 12: seed ARI / split-half ARI | 0.792 / 0.803 (best of K tested) |
| HDBSCAN (5 settings, 30k sample) | noise 32.4–100%, or one cluster holding 59–68% of the sample |
| Cluster sizes | 6.4%–13.4% of train questions |
| Questions within 0.02 cosine of a second centroid | 13.0% |
| Val pairs whose two questions fall in the same cluster | 67.3% |
| Per-cluster F1 at the global τ (val, base model) | 0.708–0.808 (std 0.033; random-partition null std 0.009) |
| Correlation of cluster F1 with duplicate prevalence | Spearman ρ = 0.69 |
| Cluster-optimal thresholds (val, base model) | 0.20–0.46 (std 0.066) |
| Cross-fitted Δ macro-cluster F1, cluster vs global τ | base model −0.0032 [−0.0051, −0.0014]; final model −0.0033 [−0.0050, −0.0016] |
| Cross-fitted Δ macro-cluster F1, constraint model vs base (both global τ) | +0.0119 [+0.0087, +0.0149] |

## Reproducibility

Sources: `artifacts/MODEL_REGISTRY.json`, `artifacts/phase8/freeze_manifest.json`, `docs/final_verification.md`.

| Metric | Value |
|---|---|
| Artifacts in the model registry | 22 (hash, size, git status, regenerate command each) |
| Files hashed in the Phase 8 freeze manifest | 25 + encoder weights |
| Encoder | `all-MiniLM-L6-v2` pinned at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Validation dry run before the test run | all 5 systems reproduced exactly; 0 decision flips |
| Train/serve parity | inference design matrix bit-identical on all 38,138 val pairs |
| Demo vs evaluated scores | max difference 5e-7, 0 of 200 decision flips |
| Golden probe pairs | 14, checked to 1e-5 |
| Recomputed test metrics vs reported | identical (final F1 0.78559, PR-AUC 0.83355) |
| Fresh clone + Kaggle archive | split CSVs byte-identical; 129 passed / 4 skipped; `--verify` passes |

## Tests

| Metric | Value |
|---|---|
| Test count | 133 across 13 files |
| Result on the working repository | 133 passed |
| Bare fresh clone (no data) | 121 passed, 12 skipped with reasons, 0 failed |
