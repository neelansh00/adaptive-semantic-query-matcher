# Interview Notes

How to explain each part of the project in depth. Numbers are from [project_metrics.md](project_metrics.md). "val" = validation split, "test" = the frozen test split (evaluated once).

**30-second version.** I built a duplicate-question detector on Quora Question Pairs to test whether semantic query regions, entity/constraint checks and
per-region thresholds beat a single embedding-similarity threshold.
- **Setup:** a question-disjoint split, after finding that a random split leaks badly.
- **Models:** lexical baselines, then a Siamese BiLSTM, then a frozen MiniLM sentence-embedding model.
- **Analysis:** clustered questions with K-Means, then added deterministic entity/number/negation/specificity features through a stacked meta-classifier.
- **Calibration:** tested global vs per-cluster thresholds with cross-fitting.
- **Result:** the constraint features improved held-out F1 from 0.775 to 0.786 and worst-cluster F1 from 0.668 to 0.695. Per-cluster thresholds did
  *not* help out of sample, and I kept that negative result.

---

## 1. Dataset and leakage

**What I did.** Audited the data before modelling. 61.3% of pairs contain a question that also appears in other pairs (one appears in 157), so I built a
question graph (questions = nodes, pairs = edges) and split by graph components:
- small components are assigned whole;
- the 19 largest hub components are split per question, with node probabilities proportional to √ratio so the retained pairs still come out 80/10/10.

The result is zero shared questions between splits, 94.2% of pairs kept, and matched duplicate rates (34.8/35.6/35.9%).

**Why.** With a random row split, 58.4% of test pairs share a question with train. A rule that never reads the text (A≡B and B≡C in train ⇒ A≡C)
labels 21.1% of test pairs at 99.9% precision. Any model trained on a random split can harvest that shortcut, so its scores measure memorisation as
well as language understanding. Question frequency alone also predicts the label (ROC-AUC 0.70), so I banned frequency and graph features.

**Alternatives.**
- *Random split:* rejected for leakage.
- *Whole-component split:* zero loss, but one component holds 4.4% of the data, so the test duplicate rate swung 37–43% across seeds.
- *Pure per-question split:* only 43% of pairs retained.

**Trade-offs.** The hybrid drops 23,345 hub pairs, which are 69% duplicates. Hub questions are therefore under-represented, and scores aren't comparable to
leaderboard numbers.

**Why the test set stayed frozen.**
- Every decision (model choice, features, K, thresholds) used validation.
- Before reading test I hashed every artifact into a manifest and committed it.
- A dry run on validation had to reproduce every known number exactly.
- The evaluation script refuses to run twice.

If I looked at test and then changed anything, the test score would stop being an unbiased estimate.

**Likely questions.**
- *"Why not GroupKFold on question ids?"* A pair has two questions, so grouping by one id still leaks through the other; graph components are the
  correct unit. Their sizes are very skewed, which is why the largest ones needed special handling.
- *"How do you know the split has no leakage?"* There are zero shared questions at the normalised-text level (enforced by a test), and the transitive-label
  rule fires on 0 test pairs.
- *"What did the dry run catch?"* A train/serve skew: features read from CSV differed by one unit in the last place from freshly computed ones, and
  gradient-boosted tree split points sat exactly on those values. 90 of 38,138 validation decisions flipped. I fixed it by canonicalising inputs,
  retrained, and re-froze, all before touching test.

## 2. TF-IDF baseline

**What I did.** I built a strong lexical baseline: TF-IDF (word 1–2-grams, fitted on train questions only) turned into a pair vector `[|v1−v2|, v1⊙v2]`,
plus 11 symmetric overlap features, fed to logistic regression. Val F1 0.730, PR-AUC 0.771.

**Why a strong baseline.** A deep model only means something relative to the best cheap model. A weak baseline would have inflated every later gain.

**Why `|v1−v2|` and `v1⊙v2`.**
- The product captures *shared* terms and the absolute difference captures terms *in only one* question, so a linear model can weigh both.
- Both are symmetric, so swapping A and B can't change the score.
- Concatenating `[v1, v2]` would be order-dependent and would make the model learn interactions it can't express linearly.

**Why threshold tuning mattered.** The F1-optimal thresholds were 0.20–0.32, not 0.5. Using 0.5 cost 5–16 F1 points. 0.5 is only optimal for *accuracy* on a
calibrated model; F1 with a 36% positive class wants a lower cut-off. I tuned on half of validation and scored the other half: within 0.002 F1, so no optimism.

**What it showed.** Lexical overlap is *non-monotonic*: near-identical pairs are fewer duplicates (44%) than moderately similar ones (55%), because many are
templated questions differing in one slot. The model gave an exact duplicate, "What is formwork?" ×2, probability 0.00, because the shared bigram
"what is" carries a large negative weight learned from templated negatives.

**Likely questions.**
- *"Why logistic regression and not boosting here?"* It's interpretable, fast on about 700k sparse features, and C was tuned. It was a baseline, not the final model.
- *"Why is cosine TF-IDF so weak (0.618)?"* Its precision tops out around 0.5 at every threshold. Similarity alone doesn't separate "same question" from "same template".

## 3. Siamese BiLSTM

**What I did.**
- *Architecture:* a PyTorch Siamese network, where **one** BiLSTM encoder (shared weights) reads both questions. Embedding 200 → BiLSTM 128 per direction → masked max-pool → u, v.
- *Head:* an MLP on `[|u−v|, u⊙v, cos]`.
- *Training:* early stopping on a question-disjoint dev slice carved from train, so validation stayed clean.
- *Engineering:* length-bucketed batches with `pack_padded_sequence`, and checkpointing with exact resume (an interrupted run reproduced epoch 1 bit-for-bit).

**Why shared weights.** Both inputs are questions from the same distribution. Sharing weights halves the parameters, gives identical questions identical
vectors, and makes the model symmetric by construction (unit-tested). It also lets each question be encoded once and compared against many, which suits retrieval.

**Why a BiLSTM.** It reads order in both directions, which bag-of-words can't ("India→Pakistan" vs "Pakistan→India"). Max-pooling keeps the strongest
feature from any position.

**What it showed.** Val F1 0.732, not significantly better than TF-IDF (CI [−0.002, +0.007]), poorly calibrated (ECE 0.106), and 9.1M parameters, mostly
embeddings trained from scratch. Interestingly it handled number and negation mismatches *better* than the transformer, because "not" and "20" are explicit
tokens for it. That observation motivated the explicit constraint features later.

**Comparison with transformer embeddings.** MiniLM (pretrained on over a billion sentence pairs) beat it by about +0.04 F1, mainly on low-overlap paraphrases,
where a from-scratch model can't know that "fb" means "Facebook".

**Likely questions.**
- *"Why not pretrained GloVe?"* I kept the model self-contained; it's the main reason it can't learn rare synonyms, and the doc says so.
- *"Why include it if it didn't win?"* It's the classic Siamese architecture the task calls for, and its failure modes were informative.

## 4. Sentence transformers

**What I did.** I used **frozen** `all-MiniLM-L6-v2` (22.7M parameters, 384-dimensional), encoded each unique question once and cached it, and compared three heads on
`[|u−v|, u⊙v, cos]`:
- cosine threshold: F1 0.735;
- logistic regression: 0.750;
- one-hidden-layer MLP: **0.771**, PR-AUC 0.824.

The MLP beats LR by +0.021 [+0.018, +0.024], which justified it.

**Why MiniLM.** It's the smallest mainstream SBERT-style encoder, which matters on a CPU-only laptop (about 300 questions/s). It wasn't fine-tuned: the
spec allowed fine-tuning only if the frozen model underperformed, and it beat the bar by +0.041 F1.

**Cosine similarity / embeddings.** Embeddings are L2-normalised, so cosine is a dot product. Questions with the same meaning land close together even
when they share no words.

**Why dense models help low-overlap paraphrases.** Missed duplicates with word Jaccard ≤ 0.2 fell from 46% (lexical) to 28%.

**The catch.** Raw cosine is *worst* on near-identical non-duplicates: on high-overlap negatives it accepted 81% of number mismatches and 84% of negation
mismatches. Embeddings blur "5 kg" vs "20 kg".

**Contamination.** The model card lists 103,663 Quora triplets in pretraining. With identical heads on the same 60k pairs, an NLI-only encoder scored 3.0–4.5
F1 lower. That's an upper bound on contamination, not a measurement, because the encoders also differ in training data. I report absolute scores as optimistic.

**Likely questions.**
- *"Why not a cross-encoder?"* It's more accurate but doesn't give per-question embeddings for clustering or retrieval, and it's slower. The project needed embeddings.
- *"Why an MLP and not just cosine?"* A learned head uses *where* the vectors differ, not just how far apart they are; +0.036 F1 over cosine on val.

## 5. Clustering

**What I did.** I ran K-Means on 424,012 unique train-question embeddings for K ∈ {5, 8, 10, 12, 15, 20}, plus HDBSCAN, then chose **K = 12**. No labels
were used anywhere in clustering.

**Why K = 12.**
- *Internal metrics couldn't decide:* Davies-Bouldin and coherence always improve with K, and Calinski-Harabasz always worsens.
- *Stability decided it:* single-run stability estimates were misleading (K = 5 looked perfect), so I re-measured with 6 seed pairs and 3 split-halves.
  K = 12 scored seed ARI 0.79 and split-half ARI 0.80, the best of all K. It also had the most balanced sizes and enough validation pairs per cluster.

**ARI.** The adjusted Rand index compares two partitions up to relabelling: 1 means identical, 0 means chance.

**Silhouette ≈ 0.02.** Points are about as close to neighbouring clusters as to their own, at every K. The questions form a continuum; K-Means *imposes*
regions on it.

**Why HDBSCAN was rejected.** Across 5 settings it labelled 32–100% of points as noise, or found one blob holding 59–68% of the sample. Every pair needs a
region to get a threshold, and there was no density structure to find. I tried small cluster sizes and low-dimensional PCA so it wasn't a strawman.

**Why names came afterwards.** Defining bins like "factual / opinion" up front would bake in my assumptions. I read each cluster's distinctive terms and
nearest-centroid questions and *then* named them. Some are topical (health, India), some intent-like (definitions, "which is best", how-to).

**Why they're regions, not types.**
- the silhouette is about 0.02;
- 13% of questions are almost equidistant between two centroids;
- a third of pairs straddle two clusters;
- refits move a few dozen questions.

So I froze the centroids as the source of truth.

**Likely questions.**
- *"Why cluster questions, not pairs?"* The research question was about query types. Pairs were assigned via the normalised average of their two
  embeddings, which is symmetric, unlike using question 1's cluster (which changed a third of assignments when the questions were swapped).
- *"Is a silhouette of 0.02 a failure?"* It's a finding. It predicted that per-cluster behaviour would be weak, and Phase 7 confirmed it.

## 6. Error analysis

**What I did.**
- *Per-cluster metrics* with bootstrap CIs, compared against a **random-partition null** (500 random groupings of the same sizes) to tell real spread from noise.
- *Overlap-controlled error rates,* so that error types weren't confounded with how many words the questions share.
- *Hand-labelled error samples:* 120 errors in Phase 5 and 60 in Phase 8.

**What it showed.**
- **High-overlap false positives:** near-identical templates differing in one slot.
- **Low-overlap false negatives:** synonyms, aliases, misspellings ("security counsel").
- **Entity / number / negation mismatches:** 20% of the base model's sampled false positives.
- **Scope / specificity (broader vs narrower):** about 25%, and it cuts both ways. "Free iPhone" vs "free iPhone 7" is labelled a duplicate.
- **Label noise:** about 27% of false positives look mislabelled ("most followed on Instagram" vs "most followers" labelled 0), which caps achievable scores.
- **Cluster F1** varied 0.708–0.808, more than the random null's spread, but it tracked duplicate prevalence (ρ = 0.69). The worst-F1 clusters ranked pairs normally.

**Likely questions.**
- *"How do you know the cluster differences are real?"* The random-partition null: real F1 std 0.033 against 0.009 for random groups. But prevalence explains much of it, which is why I also reported ROC-AUC per cluster.
- *"How reliable are your manual labels?"* Single annotator with scores visible, so I treat them as indicative proportions and say so.

## 7. Entity / constraint features

**How they were extracted.** Deterministically, with no LLM and, in the end, no spaCy. Each question is annotated once:
- numbers (handles "1,000", "2.5k", "5kg"; not the pronoun "one");
- regex dates;
- capitalised entities and acronyms;
- negations including "n't" and "non-";
- the question word;
- content words.

Pair features count unmatched items per group, plus specificity: extra content words, subset relation, length ratio. An item counts as matched if it
appears in the other question's text, which handles "india"/"India" and "U.S."/"US". Every feature is exactly symmetric (tested).

**How they're used.** A stacked meta-classifier on the base model's logit plus the features. To avoid leakage, the base scores for training came from
**5-fold question-disjoint cross-fitting** (copies of the base head scoring pairs they never trained on). Variants and the selection rule were fixed in advance.
- Controls showed that re-fitting or boosting the base score alone added nothing.
- spaCy NER added only +0.001 PR-AUC, so it was dropped.
- Boosted trees beat logistic regression by +0.008 PR-AUC, so they were chosen.

**What they improved.**
- Test F1 +0.011 and worst-cluster F1 +0.026.
- High-overlap FPR on entity/number/negation differences roughly halved (0.174 → 0.075, 0.502 → 0.218, 0.641 → 0.315).
- Explicit constraint errors fell from 20% to 7% of sampled false positives.

**What they didn't fix.**
- *Role swaps* ("introvert parents → extrovert child" vs the reverse).
- *Different aspects of one topic,* scope, and label noise.
- *Paraphrases* needing world knowledge.
- *They introduced false alarms* on duplicates with a superficial constraint difference ("GRE 299 vs 319", labelled duplicate).

The largest contributor was actually the **specificity/length** group; numbers were the only explicit group significant on its own.

**Likely questions.**
- *"Why stacking and not fine-tuning?"* Interpretable signals, a cheap model (0.35 MB, +4% batch latency), and direct control over constraint types.
- *"Why cross-fitting?"* In-sample base probabilities are over-confident, so a meta-model trained on them would over-trust the base model.

## 8. Calibration (thresholds)

**What I did.** Compared one global threshold against per-cluster thresholds, for both the base and the final model, **cross-fitted within validation**
(5 question-disjoint folds × 20 repetitions; thresholds fitted on 4 folds, applied to the 5th) with paired bootstrap CIs. The decision rule was written
before running.

**Why 0.5 is arbitrary.** A threshold turns a score into a decision. 0.5 maximises accuracy only for a calibrated probability, and F1 under a 36%
positive class wants something lower (0.32–0.35 here). Cosine similarities have no natural 0.5 at all (the cosine optimum was 0.76).

**Fallback.** A cluster gets its own threshold only with ≥ 1,000 fitting pairs and ≥ 100 duplicates; otherwise, and for any unseen cluster, the global
threshold applies. The 1,000 minimum came from a reliability experiment: below about 500 pairs, a tuned threshold's std (≥ 0.08) is as large as the whole
between-cluster spread.

**Avoiding overfitting small clusters.**
- *In-sample, cluster thresholds look better* (+0.005 F1), because that's 12 extra parameters fitted to the scored data.
- *Cross-fitted, they're worse:* macro-cluster F1 −0.003 for both models, CIs excluding 0.
- *They were unstable,* moving up to ±0.06 between folds.
- *Fewer is better:* raising the minimum size so more clusters fall back improved results monotonically towards the global result.

**Likely questions.**
- *"So the main hypothesis failed?"* Per-cluster calibration did. Constraint consistency improved robustness instead. I pre-registered the rule and reported the negative result.
- *"Would shrinkage help?"* Possibly a little, but the monotone fallback trend suggests at best matching global. I deliberately didn't add it.

## 9. Metrics

| Metric | Why it is in the project |
|---|---|
| **F1** | headline: balances false merges against missed duplicates; threshold-dependent |
| **Precision / recall** | different costs. A false positive merges different questions (wrong answer shown); a false negative leaves a duplicate unmerged (fragmentation). The final system gained precision at unchanged recall |
| **PR-AUC** | threshold-free ranking under class imbalance; chance level equals prevalence (0.36) |
| **ROC-AUC** | threshold-free ranking, insensitive to prevalence; used to compare clusters with different duplicate rates |
| **Calibration (ECE, Brier)** | whether probabilities mean what they say; needed before comparing thresholds across clusters. Final ECE 0.019 |
| **Macro-cluster F1** | average over regions, so large easy clusters don't hide weak ones |
| **Worst-cluster F1** | the robustness floor; it's a noisy statistic (the minimum of 12 estimates), hence the wide CIs |

**Likely questions.**
- *"Why not accuracy?"* Predicting "not duplicate" for everything already gives 64%.
- *"Why paired bootstrap?"* Both systems are scored on the same pairs, so resampling pairs jointly gives the CI of the *difference*.

## 10. Reproducibility

**What I did.**
- **Model registry:** 22 artifacts, each with content hash, size, git status and regenerate command.
- **Phase 8 freeze manifest:** 25 files plus encoder weights, committed before the test run.
- **Pinned encoder** revision.
- **Generated feature configuration.**
- **Golden outputs:** 14 probe pairs checked to 1e-5.
- **`reproduce.py --verify`,** which recomputes the reported test metrics from the saved per-pair scores.
- **Fresh-clone test:** clone plus archive gives byte-identical split files, 129 passed / 4 skipped.

**Hashes.** Text files are hashed with line endings normalised, because git stores LF and a Windows checkout may hold CRLF; binaries are hashed byte-for-byte.

**Why retraining isn't bit-identical.** Multithreaded floating-point sums happen in different orders between runs. Seeds fix initialisation, not summation
order. Two K-Means refits with the same seed agreed at ARI 0.9997, not 1.0.

**Why the frozen artifacts are the reference.** Retraining would produce a slightly different model, which is a *different system* from the one evaluated
on test. The evaluated artifacts are versioned and hash-checked, and every reported number is recomputable from them.

**Likely questions.**
- *"What was the hardest bug?"* The train/serve skew found by the dry run, where 1-ulp CSV differences flipped tree splits. It was invisible to a tolerance-based
  parity test (`pytest.approx`); an exact, all-pairs parity test catches it now.
- *"What would you do next?"* Handle role and argument-structure swaps (for example with a cross-encoder on the residual errors), evaluate on a second
  dataset, and fine-tune an encoder not pretrained on Quora to remove the contamination caveat.
