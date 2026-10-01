# Resume Bullets

All figures are measured on the frozen Quora Question Pairs split ([project_metrics.md](project_metrics.md)). "Held-out test" means a test split
evaluated once, after freezing.

## Four bullets (distinct contributions)

1. **Leakage-controlled evaluation design.** Found that a random split of Quora Question Pairs leaks 58% of test pairs (a text-free transitive rule labels
   21% of test pairs at 99.9% precision). Built a question-disjoint split from question-graph components (0 shared questions, 94% of 404k pairs retained)
   and ran a hash-verified, one-time held-out test evaluation.

2. **Semantic matching models.** Benchmarked TF-IDF + logistic regression, a PyTorch Siamese BiLSTM and frozen MiniLM sentence embeddings with an MLP
   head; the MiniLM model improved validation F1 from 0.730 to 0.771 with a significant paired-bootstrap gain. Ran a pretraining-contamination
   control (the encoder saw Quora data): an encoder without Quora data scored 3.0–4.5 F1 lower under identical heads.

3. **Entity/constraint meta-classifier.** Added deterministic number, date, entity, negation and specificity consistency features, stacked on cross-fitted
   base scores with gradient boosting. Raised held-out test F1 from 0.775 to 0.786 and worst-cluster F1 from 0.668 to 0.695, and roughly halved
   high-overlap entity/number/negation false positives at unchanged recall.

4. **Clustering and calibration experiments.** Discovered 12 query regions with K-Means on 424k question embeddings, choosing K by stability
   (ARI 0.8) because silhouette was about 0.02. Showed with cross-fitted, pre-registered tests that per-region thresholds reduce macro-cluster F1
   out of sample, and kept a single calibrated threshold.

## Three one-line variants

- Built a leakage-free duplicate-question detector (question-disjoint split, one-time test evaluation): F1 0.786 / PR-AUC 0.834 on held-out Quora pairs.
- Raised held-out F1 from 0.775 to 0.786 and worst-cluster F1 from 0.668 to 0.695 by stacking entity/number/negation consistency features on a MiniLM semantic model.
- Showed with cross-fitted tests that per-cluster threshold calibration does not beat a single threshold; documented the negative result with reproducible, hash-verified artifacts.

## Accuracy notes (for interviews)

- The clusters are **embedding regions named after inspection**, not ground-truth question categories.
- This is a **research system evaluated on one dataset**, not a production deployment.
- Absolute scores are optimistic because the encoder saw Quora data in pretraining. The baseline-vs-final comparison is like-for-like.
- Scores are not comparable to random-split or Kaggle leaderboard results.
