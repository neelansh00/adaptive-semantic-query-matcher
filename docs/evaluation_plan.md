# Evaluation Plan

Implemented in [src/evaluation/metrics.py](../src/evaluation/metrics.py), with unit tests in `tests/test_metrics.py`.
Every later phase reports these metrics with the same functions.

## Data usage

| Split | Used for |
|---|---|
| train | fitting models, vocabularies, embeddings used to fit clustering |
| val | model selection, threshold tuning, feature selection, clustering decisions (K), calibration |
| test | **one** final evaluation in Phase 8, after all choices are frozen |

No metric is ever computed on `test` before Phase 8, not even "just to look".

## Primary classification metrics

| Metric | Role |
|---|---|
| **F1 (positive class)** | headline metric, at the validation-chosen threshold |
| **Precision** | of the pairs flagged as duplicates, how many really are |
| **Recall** | of the true duplicates, how many are found |
| **PR-AUC** (average precision) | threshold-free ranking quality under class imbalance (~35% positives) |
| **ROC-AUC** | threshold-free ranking quality; less sensitive to imbalance, reported for comparability |
| Accuracy | reported only. A majority-class predictor already scores ~64%, so accuracy is not informative. |
| FPR / FNR | error rates by type, used heavily in Phases 5–7 |

**Why precision and recall are reported separately.** The two errors cost different things.
- A **false positive** merges two different questions. A user asking about *Apple* sees answers about *Microsoft*.
  This is the entity/constraint failure mode the project targets.
- A **false negative** leaves a duplicate unmerged. The cost is fragmentation and redundant answers, which is annoying but recoverable.

F1 is the headline because it balances both. Every phase also reports precision and recall so that a
change which trades one for the other is visible and not hidden inside F1.

## Threshold analysis

A score only becomes a decision through a threshold τ. **0.5 is not automatically right**:
- A calibrated probability at 0.5 maximises accuracy, not F1. The F1-optimal cut-off for a
  minority positive class is usually lower.
- Cosine similarities and uncalibrated network outputs have no natural 0.5 at all.

So for every model we report:
1. a **threshold sweep** of precision, recall and F1 over τ ∈ [0, 1] in steps of 0.01 (`threshold_sweep`);
2. the **precision–recall curve**;
3. **F1 vs threshold**;
4. the **validation-optimal threshold** τ* = argmax F1 on val (`best_f1_threshold`; ties go to the lowest τ).
   τ* is then frozen and applied unchanged to test.

For similarity scores outside [0, 1], or scores concentrated in a narrow range, the sweep grid
is built from score quantiles instead.

## Calibration (when probabilities are interpreted)

- **Brier score**: mean squared error of the probability.
- **Expected calibration error (ECE)**: 10 equal-width bins, weighted mean |confidence − accuracy|.

These matter from Phase 6 onward, where a probability is shown to the user and thresholds are
compared across clusters. Thresholds are only comparable if the underlying scores mean the same thing.

## Cluster-level metrics (Phases 5–8)

Each val/test pair is assigned to one discovered cluster. The assignment rule is chosen in Phase 5. Per cluster we report:
`n`, duplicate prevalence, precision, recall, F1, FPR, FNR, mean score, and the validation-optimal threshold.

Summaries:
- **Macro cluster F1**: unweighted mean of per-cluster F1.
- **Worst-cluster F1**: minimum per-cluster F1, plus which cluster it is.

**Why overall F1 is not enough.** Overall F1 is dominated by the largest and easiest clusters.
A model can gain 1 point of global F1 while getting *worse* on a small entity-heavy cluster,
and that segment is exactly where the users who suffer false merges are. Macro and worst-cluster F1 make that visible.

Rules:
- Clusters with fewer than a minimum number of validation pairs are reported but flagged. They fall back
  to the global threshold in Phase 7. The minimum is fixed before running Phase 7.
- Per-cluster metrics on small clusters have wide uncertainty. Phase 7 reports bootstrap intervals
  for per-cluster F1 before any claim of improvement.

## Efficiency (Phase 3 and later)

Inference latency per pair (batch and single), and model size on disk.

## Reporting rules

- Report only measured numbers, and include negative results.
- Every comparison states which split it was computed on.
- Improvements are claimed only if they hold on val **and** are confirmed once on test.
