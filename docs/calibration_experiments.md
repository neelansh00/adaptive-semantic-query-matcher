# Phase 7: Global vs Cluster-Calibrated Thresholds

**The central hypothesis:** semantic query regions behave differently, so thresholds calibrated *per cluster* should make duplicate detection more robust
than a single global threshold.

**Answer: no.** On this dataset, with these clusters, cluster-specific thresholds **significantly reduce** overall F1 and macro-cluster F1 once they are
evaluated out of sample, for both the semantic model and the entity/constraint-aware model. They do not improve the worst cluster. What *does* improve
robustness is the Phase 6 entity/constraint-aware model with a **single global threshold**. That is the configuration frozen for Phase 8.

All work is on the frozen **validation** split. Thresholds were learned only from validation, never set by hand. The test split was never read (enforced by tests).

Reproduce: `python scripts/calibration_experiments.py` (about 10 minutes on CPU). Use `--plot-only` to redraw the figures from `artifacts/phase7/results.json`.

## 1. Why cross-fitting

Thresholds must be learned on validation, and the test set is reserved for Phase 8. If per-cluster thresholds were tuned on validation and then scored on
those same pairs, they would win by construction: 12 extra free parameters fitted to the evaluation data. In-sample, they do look better:

| In-sample (optimistic: thresholds fitted and scored on the same pairs) | F1 | Macro-cluster F1 | Worst-cluster F1 |
|---|---|---|---|
| B: semantic model, global τ | 0.771 | 0.770 | 0.708 |
| B: semantic model, cluster τ | 0.777 | 0.775 | 0.710 |
| C: constraint-aware model, global τ | 0.783 | 0.781 | 0.719 |
| C: constraint-aware model, cluster τ | **0.788** | **0.785** | 0.722 |

This is the illusion Phase 5 warned about: its estimated in-sample headroom was +0.006 F1.

So every system is **cross-fitted within validation**:
1. **Folds:** validation is split into 5 question-disjoint folds, assigning whole question-graph components so no question sits on both sides.
2. **Fit and apply:** thresholds (global and per cluster) are fitted on 4 folds and applied to the held-out fold, giving out-of-fold decisions for every pair.
3. **Repetition:** this is repeated with **20 different fold shufflings**; all systems share the same folds in each repetition.
4. **Uncertainty:** a **paired bootstrap** (1,000 resamples of validation pairs) of the *repetition-averaged* difference between systems.

`tests/test_calibration.py` checks that changing a fold's labels never changes that fold's own decisions.

## 2. Rules (fixed before running)

| Rule | Value |
|---|---|
| Threshold search | F1-optimal on a 0.01 grid (ties → lowest), the same rule as every earlier phase; no manual thresholds |
| Minimum size for a cluster's own threshold | **≥ 1,000 fitting pairs and ≥ 100 duplicates.** From Phase 5's reliability experiment: below about 1,000 pairs a tuned threshold has std > 0.05 and its noise rivals the whole between-cluster spread |
| Fallback | clusters below the minimum (or unseen) use the global threshold fitted on the same data |
| Pair → cluster | frozen Phase 4 centroids, Phase 5 pair-average rule |
| Decision rule for Phase 8 | adopt per-cluster thresholds for a model only if they improve **macro-cluster F1 with a 95% CI excluding 0** *and* the CI of the overall F1 change is not entirely below 0 |

With the 1,000-pair minimum, all 12 clusters qualify in every fold (each has 1,950–3,650 fitting pairs), so the fallback was exercised separately (§5).

## 3. Systems compared

| # | System | Scores | Threshold |
|---|---|---|---|
| 1 | **Global threshold** (baseline) | B: Phase 3 semantic model | one global τ |
| 2 | **Cluster-specific thresholds** | B | per-cluster τ, global fallback |
| 3 | **Entity/constraint-aware classifier** | C: Phase 6 meta-model (`C_no_spacy_hgb`) | one global τ |
| 4 | **Combined** | C | per-cluster τ, global fallback |

**Why system 4 is methodologically valid:** the Phase 6 meta-model was trained on *train* only (with cross-fitted base scores) and never saw validation.
Its cluster thresholds are cross-fitted within validation exactly like everyone else's. No score or threshold is evaluated on data it was fitted to.

## 4. Results (cross-fitted, mean of 20 repetitions)

| System | F1 | Precision | Recall | FPR | FNR | Macro-cluster F1 | Worst-cluster F1 |
|---|---|---|---|---|---|---|---|
| 1. B + global τ | 0.7706 | 0.693 | 0.868 | 0.213 | 0.132 | 0.7695 | 0.7071 |
| 2. B + cluster τ | 0.7674 | 0.697 | 0.853 | 0.205 | 0.147 | 0.7664 | 0.7075 |
| **3. C + global τ** | **0.7809** | **0.716** | 0.859 | 0.189 | 0.141 | **0.7793** | **0.7170** |
| 4. C + cluster τ | 0.7793 | 0.716 | 0.856 | 0.188 | 0.145 | 0.7771 | 0.7111 |

Range across the 20 repetitions: system 1 F1 0.769–0.771 and worst-cluster 0.706–0.708; system 3 F1 0.779–0.782 and worst-cluster 0.714–0.721.

![deltas](figures/phase7_deltas.png)

| Comparison (paired bootstrap of the 20-repetition mean) | ΔF1 [95% CI] | Δ macro-cluster F1 | Δ worst-cluster F1 |
|---|---|---|---|
| **Cluster vs global τ, semantic model (2 vs 1)** | **−0.0033 [−0.0053, −0.0014]** | **−0.0032 [−0.0051, −0.0014]** | −0.0000 [−0.0095, +0.0072] |
| **Cluster vs global τ, constraint model (4 vs 3)** | **−0.0017 [−0.0034, −0.0001]** | **−0.0022 [−0.0038, −0.0006]** | −0.0037 [−0.0102, +0.0062] |
| Constraint model vs semantic model, both global (3 vs 1) | **+0.0103 [+0.0074, +0.0132]** | **+0.0098 [+0.0067, +0.0127]** | +0.0097 [−0.0078, +0.0249] |
| Combined vs baseline (4 vs 1) | +0.0086 [+0.0055, +0.0116] | +0.0075 [+0.0045, +0.0106] | +0.0060 [−0.0074, +0.0200] |

Reading the table:
- **Cluster thresholds hurt, significantly, for both models,** on overall F1 and on macro-cluster F1, the very robustness metric they were meant to improve.
- They **do not help the worst cluster** (C6 Education): the change is about 0 with a wide CI.
- **The entity/constraint-aware model is the only intervention that improves robustness:** +0.010 overall and +0.010 macro-cluster F1, both significant.
  Its worst-cluster gain (+0.010) is consistent in direction (all 20 repetitions: 0.714–0.721 vs 0.706–0.708) but its bootstrap CI includes 0. The worst-cluster
  minimum over 12 noisy cluster estimates is itself a noisy statistic.
- Adding cluster thresholds on top of the constraint model (system 4) only gives back part of its gain.

**Pre-registered decision:** adopt cluster thresholds for B? **No** (macro Δ CI entirely < 0). For C? **No** (macro Δ CI entirely < 0).

### Per cluster

![per-cluster F1](figures/phase7_per_cluster.png)

| Cluster | B global | B cluster τ (Δ) | C global (Δ vs B global) | C cluster τ (Δ vs C global) |
|---|---|---|---|---|
| C0 Definitions | 0.711 | −0.001 | **+0.023** | −0.005 |
| C1 Opinions & beliefs | 0.767 | +0.003 | +0.013 | +0.000 |
| C2 Accounts & apps | 0.733 | −0.001 | **−0.010** | +0.005 |
| C3 Health & body | 0.772 | **−0.012** | +0.012 | **−0.012** |
| C4 Relationships | 0.791 | −0.008 | +0.006 | +0.004 |
| C5 Countries | 0.798 | +0.008 | +0.009 | −0.004 |
| C6 Education (worst) | 0.707 | +0.001 | +0.010 | −0.006 |
| C7 Money & business | 0.775 | **−0.019** | **+0.018** | −0.003 |
| C8 Product comparison | 0.781 | −0.009 | +0.010 | −0.001 |
| C9 India-specific | 0.805 | −0.008 | +0.004 | −0.002 |
| C10 Learning & exams | 0.787 | −0.004 | +0.012 | −0.006 |
| C11 Personal experiences | 0.808 | **+0.010** | +0.011 | +0.003 |

- **Cluster thresholds:** 3 to 4 clusters gain (C11 +0.010, C5 +0.008), and more lose (C7 −0.019, C3 −0.012). It is a mixed picture with a negative net, not a systematic improvement.
- **Phase 5's one "clear" case also fails.** C4 had an in-sample optimal threshold of 0.20 (bootstrap CI [0.18, 0.21]), the only cluster whose F1 plateau
  excluded the global τ. Cross-fitted, its own threshold *lowers* its F1 by 0.008 (model B). Phase 5's within-cluster bootstrap of the arg-max threshold
  understated how unstable a tuned threshold is on new data.
- **The constraint-aware model improves 11 of 12 clusters** at a single threshold. The exception is C2 (accounts/apps), whose errors are mostly paraphrases (Phase 5).

### Why the cluster thresholds fail: instability

Thresholds fitted in the 100 fold fits (20 repetitions × 5 folds):

| | Global τ: mean (std) | Per-cluster τ: std range | Per-cluster τ: widest min–max |
|---|---|---|---|
| B (semantic) | 0.319 (0.005) | 0.005 – **0.075** | C7: 0.19 – 0.42 |
| C (constraint-aware) | 0.368 (0.022) | 0.011 – **0.059** | C3: 0.30 – 0.52 |

A cluster threshold is fitted on about 2–4k pairs and moves by up to ±0.06 between folds. The global threshold, fitted on about 30k pairs, barely moves.
Phase 5 found the *true* between-cluster differences are modest and the F1 curves flat near their optima. So the variance added by 12 noisy thresholds
outweighs the bias removed.

## 5. Minimum cluster size and the fallback

The minimum was varied (5 repetitions each) to exercise the global fallback. Fitting folds hold about 1,950–3,650 pairs per cluster.

![minimum size](figures/phase7_min_size.png)

| Minimum fitting pairs | Clusters with own τ (mean) | Macro-cluster F1, B | Macro-cluster F1, C |
|---|---|---|---|
| 250 / 500 / 1,000 | 12.0 | 0.7662 | 0.7770 |
| 2,000 | 10.8 | 0.7667 | 0.7774 |
| 2,500 | 5.8 | 0.7672 | 0.7786 |
| 3,000 | 1.4 | 0.7694 | 0.7789 |
| 3,500 | 0.9 | 0.7694 | 0.7790 |
| *global τ only* | 0 | *0.7695* | *0.7793* |

Performance improves **monotonically as fewer clusters get their own threshold**, converging on the global-threshold result. The fallback mechanism works
(small clusters revert to the global τ, unit-tested), but on this data the best "minimum size" is effectively infinite.

## 6. Answer

> **"Does cluster-aware calibration improve robustness compared with a single threshold?"**
>
> **No.**
> - Evaluated out of sample (cross-fitted within validation), per-cluster thresholds **lower** macro-cluster F1 by 0.003 (semantic model)
>   and 0.002 (constraint-aware model), and overall F1 by similar amounts. Both effects are statistically significant.
> - They leave worst-cluster F1 unchanged within noise.
> - The in-sample improvement (+0.005–0.006 F1) is **entirely an artefact of fitting and scoring on the same data**.
> - Robustness across semantic regions *was* improved, but by a different mechanism: the Phase 6 entity/constraint-aware model at a single global threshold
>   (+0.010 macro-cluster F1, 11 of 12 clusters better).

This negative result is consistent with Phases 4 and 5:
- the semantic regions are soft (silhouette ≈ 0.02);
- much of the between-cluster F1 variation is prevalence, not a different optimal operating point;
- optimal thresholds differ beyond chance, but by too little relative to their estimation noise at 2–4k pairs per cluster.

## 7. Frozen configuration for Phase 8

| System | Scores | Threshold | File |
|---|---|---|---|
| **Baseline** (best non-adaptive semantic model, global τ) | Phase 3 `sbert_mlp` | global **τ = 0.32** | `artifacts/phase7/frozen_policy_baseline.json` |
| **Final** | Phase 6 meta-model `C_no_spacy_hgb` | global **τ = 0.38** (cluster thresholds rejected) | `artifacts/phase7/frozen_policy_final.json` |

Both thresholds were refitted on the full validation set and reproduce the Phase 3 and Phase 6 values exactly; a test checks this.
For transparency, the per-cluster thresholds the rejected policy *would* have used (C, full validation) are 0.27–0.45, recorded in `results.json`.

## 8. Limitations

1. **The evaluation is cross-fitted within validation, not on independent data.** Phase 8's single test run is the final check. Cross-fitting removes the
   in-sample bias, but each fold's thresholds are fitted on 80% of validation, slightly less data than a deployed threshold would use.
2. **The result is specific to these 12 MiniLM K-Means clusters and these two models.** A different clustering (larger K, a different encoder) or far more
   validation data per cluster could change it. In Phase 5's reliability experiment the threshold std was still about 0.03 at 4,000 pairs, so substantially larger clusters would be needed.
3. **Only hard per-cluster thresholds were tested.** Smoother alternatives (shrinking cluster thresholds toward the global one, or per-cluster calibration
   curves) were deliberately left out to keep the comparison simple. The monotone trend in §5 suggests shrinkage would at best approach the global result.
4. **Worst-cluster F1 is a noisy statistic** (the minimum over 12 estimates), so its CIs are wide for every comparison.
