# Project Summary

**Problem.** Duplicate-question detection usually applies one embedding-similarity threshold. Does it help to add (a) automatically discovered semantic
query regions with their own thresholds and (b) explicit entity/constraint consistency? Dataset: Quora Question Pairs (404,290 labelled pairs).

**Leakage first.** 61% of pairs reuse a question. With a random split, 58% of test pairs share a question with train, and a text-free transitive-label
rule labels 21% of test pairs at 99.9% precision. I built a **question-disjoint** 80/10/10 split from question-graph components (0 shared questions,
94% of pairs retained). I froze the test split, and banned question-frequency features (ROC-AUC 0.70 on their own).

**Lexical baseline.** TF-IDF pair vectors `[|v1−v2|, v1⊙v2]` + overlap features + logistic regression: val F1 **0.730**. Lessons:
- the F1-optimal threshold is far from 0.5;
- near-identical pairs are often *not* duplicates (templated questions differing in one slot).

**Deep semantic model.**
- A Siamese BiLSTM matched the baseline (0.732, not significant).
- Frozen **MiniLM** sentence embeddings + MLP head reached val F1 **0.771** (+0.041). It mostly fixed low-overlap paraphrases, but raw similarity accepted
  81–84% of near-identical number and negation mismatches.
- A control showed MiniLM's pretraining included Quora data, so absolute scores are optimistic.

**Clustering.** K-Means on 424k train-question embeddings. K = 12 chosen on robust stability (ARI about 0.8), since internal metrics can't choose.
Silhouette ≈ 0.02 and HDBSCAN found no density structure, so the clusters are **soft regions, named after inspection**, not true question types.

**Cluster-level error analysis.**
- Per-region F1 varied 0.708–0.808 (beyond a random-partition null), but tracked duplicate prevalence.
- Region-optimal thresholds ranged 0.20–0.46, but F1 curves were flat.
- Manual review of false positives: about 20% entity/number/negation mismatches, 25% scope differences, 27% likely label noise.

**Constraint features.** Deterministic number/date/entity/negation/question-word/specificity features, stacked with the base model's logit in a
gradient-boosted meta-classifier. To avoid leakage, base scores were cross-fitted. Controls showed the gain comes from the features: val F1 **0.784**.
spaCy NER didn't earn its dependency and was dropped.

**Adaptive calibration.** Global vs per-cluster thresholds, **cross-fitted within validation**. Per-cluster thresholds looked better in-sample but
*lowered* macro-cluster F1 out of sample (−0.003, significant), so they were rejected under a rule written in advance. Negative result kept.

**Frozen evaluation.** I hashed every artifact, verified the pipeline on validation (a dry run caught and fixed a train/serve skew), then read the test
split once.
- Final system: **F1 0.786, PR-AUC 0.834**, against 0.775 / 0.824 for the baseline (F1 +0.011 [+0.008, +0.014]).
- Macro-cluster F1 0.771 → 0.782; worst-cluster F1 0.668 → 0.695. No region worsened.
- Precision went up at unchanged recall.
- High-overlap entity/number/negation false positives roughly halved.

**Reproducibility.** A model registry (22 artifacts with hashes), a pinned encoder revision, feature configuration, golden outputs, and
`scripts/reproduce.py --verify`. Tested from a fresh clone. Retraining is close but not bit-identical (floating-point nondeterminism), so the frozen
artifacts are the reference. A Streamlit demo shows each decision with its signals and a deterministic rationale.

**Bottom line.** Entity/constraint consistency improved robustness on held-out data. Per-region threshold calibration did not, because the regions are
soft and threshold estimates at 2–4k pairs per region are too noisy.
