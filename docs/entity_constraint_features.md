# Phase 6: Entity and Constraint Consistency Features

**Question:** do explicit, deterministic entity/number/date/negation/specificity signals improve duplicate detection beyond the Phase 3
semantic model? In particular, do they reduce the entity-sensitive false positives found in Phases 3 and 5?

**Answer (validation): yes, modestly and significantly.**
- The chosen spaCy-free model raises F1 from **0.771 to 0.784** (+0.013, 95% CI [+0.010, +0.016]) and PR-AUC from **0.824 to 0.838**.
- It **lowers both error rates** (FPR 0.212 → 0.196, FNR 0.132 → 0.127) and raises the worst-cluster F1 from 0.708 to 0.721.
- On high-overlap negatives it roughly **halves** false positives with an entity, number or negation difference. The price is more false negatives
  in those same slices.
- **The largest single contributor is the specificity/length signal**; the number features are the only explicit-constraint group that is
  individually significant.

All numbers are on the frozen **validation** split. The test split was never read (enforced by tests). No LLM is used anywhere.

> **Revision note (made during Phase 8, before the test split was read).** The Phase 8 dry run on validation found a train/serve skew. Training
> features came through CSV files, and freshly computed features differ from them by about one unit in the last place. Some tree bin edges sit
> exactly on training values, so 90 of 38,138 validation decisions flipped between the two paths. Two fixes were made, and the model was retrained
> and re-evaluated:
> - meta-model inputs are rounded to 6 decimals inside `MetaModel.design()`;
> - the base probability is cast to float32 before its logit (`base_logit()`), which restores the exact value on every path.
>
> A strict test now requires the inference-path design matrix to equal the training one **bit-for-bit on all validation pairs**. The chosen variant
> did not change; its operating point moved within a flat F1 plateau (τ 0.38 → 0.35, F1 0.783 → 0.784). Earlier logs are kept
> (`artifacts/logs/phase6_train_meta_run*.log`). The Phase 6 claim that small decision differences came from *embedding numerics* was wrong:
> it was this CSV skew.

Reproduce:
```bash
python scripts/build_constraint_features.py   # annotate unique train/val questions once; pair features (~12 min CPU)
python scripts/crossfit_base.py               # 5-fold question-disjoint out-of-fold base scores for train (~9 min)
python scripts/train_meta.py                  # pre-specified variants, ablations, bootstrap, slices, probes (~15 min)
```
Artifacts: `artifacts/phase6/`. These are `variant_results.json` (every variant, every bootstrap), `chosen.json`, `meta_model.joblib`
(the frozen model, 0.4 MB), `val_scores.csv`, `slice_rates.json`, `manual_sample_flips.json`, `probes.json`, `crossfit_report.json` and `latency.json`.

## 1. Features ([src/features/constraints.py](../src/features/constraints.py))

Each unique question is annotated once; pair features are then computed from two annotations. **Every pair feature is exactly symmetric** in (A, B)
(unit-tested). An item of question A counts as *matched* if it appears in B's annotation **or as whole words in B's normalised text**. That makes
"india" vs "India" and "U.S." vs "US" consistent despite Quora's casing.

| Group | What is extracted | Pair features (6 per extracted-item group) |
|---|---|---|
| **number** | digits incl. "1,000", "2.5k", "5kg"; number words *two…billion* ("one" omitted: "how does one…") | `any`, `both`, `unmatched_total`, `unmatched_max_side`, `mismatch` (both have numbers that disagree), `one_sided` |
| **date** (regex) | years 1800–2099 incl. "1990s", months, weekdays, today/tomorrow/yesterday | same six |
| **entity (heuristic)** | mid-sentence capitalised tokens and acronyms ("IT", "US", "i5"), dotted acronyms collapsed; skipped for ALL-CAPS questions | same six |
| entity / location / date (spaCy) | `en_core_web_sm` NER: ORG, PERSON, PRODUCT, … / GPE, LOC, FAC / DATE, TIME | same six (ablated, see §3) |
| **negation** | not, no, never, without, nor, none, nobody, nothing, neither, cannot, "n't" (via Phase 2 preprocessing), "non-" | `neg_xor` (polarity disagrees), `neg_count_diff` |
| **intent word** | first token if it is a wh-/auxiliary word | `wh_mismatch` |
| **specificity** | content words (stop words and negations removed), word counts | `extra_content_max/min` (content words only one side has), `content_subset` (one question's content is contained in the other's), `log_len_ratio`, `word_count_absdiff` |

**Bugs caught by unit tests before any model was trained:**
- **"2.5k" parsed as {2, 5}:** regex backtracking.
- **"india" vs "India" flagged as a mismatch:** punctuation was attached to the matched text.
- **"U.S." vs "US" mismatch, and "US"/"IT" dropped as stop words.**

The features were rebuilt after the regex fix (`artifacts/logs/phase6_build_features_stale_regex.log`).

**How often signals fire (validation, negatives vs duplicates):** number mismatch 5.0% vs 1.4%; heuristic entity mismatch 26.9% vs 10.6%;
negation disagreement 11.5% vs 4.1%. The exception is `content_subset`, which is **twice as common among duplicates** (13.6% vs 6.0%): scope
differences cut both ways, as Phase 5 found.

## 2. Meta-classifier design

**Stacking without leakage (cross-fitting).** The meta-classifier learns how far to trust the base model's probability. In-sample base
probabilities are over-confident, so every train pair is scored by a copy of the Phase 3 MLP head trained on the **other 4 of 5 question-disjoint
folds** (`crossfit_base.py`). Each copy uses the same recipe and a fixed 25 epochs.

| Check | Value |
|---|---|
| Out-of-fold base PR-AUC on train (per fold) | 0.801 – 0.808 |
| Fold models scoring validation, PR-AUC | 0.818 – 0.822 (deployed model: 0.824) |
| Deployed vs mean fold model on validation | Pearson 0.994, mean absolute difference 0.023 |

The deployed Phase 3 model is **unchanged** and supplies validation and test base probabilities.

**Pre-specified variants** ([src/models/meta.py](../src/models/meta.py)) used fixed hyperparameters: LogisticRegression with C = 1 on standardised
features, and HistGradientBoosting with defaults and internal early stopping. Validation was used only for thresholds and for comparing the fixed variants.

| Variant | Inputs | Role |
|---|---|---|
| **A** `A_cosine` | MiniLM cosine only | specification baseline A |
| **B** `B_base` | deployed Phase 3 probability | specification baseline B, **the bar: F1 0.771** |
| `B_recal` / `B_hgb` | LR / boosted trees on the base logit only | controls: re-fitting or non-linearity alone |
| **C** `C_constraints` | base + all constraint groups incl. spaCy (LR) | specification variant C |
| `C_no_spacy` | base + spaCy-free groups (LR) | spaCy ablation |
| `C_*_plus_lexical`, `C_*hgb` | + Phase 2 overlap features; gradient-boosted trees | more features / non-linear model |
| `C_plus_cluster` | + frozen-cluster one-hot | ablation only (Phase 7's question) |
| `C_minus_*`, `HGB_minus_*` | leave one feature group out | attribution |

**Selection rule** (`choose_variant`, written before the run):
1. Start from `C_constraints`.
2. Drop spaCy unless it adds ≥ 0.005 PR-AUC with a CI excluding 0.
3. Switch to a more complex variant built on the step-2 feature set only if it beats the current choice by ≥ 0.005 PR-AUC with a CI excluding 0.

Corrections made after seeing results, all logged and none changing the conclusion:
- *Run 1:* step 3 compared a spaCy-feature model with the no-spaCy set; fixed to use the step-2 feature set.
- *Run 2:* "spaCy-free" dates still contained spaCy spans; a regex-only date group was added.
- *Run 4/5:* input canonicalisation (revision note above).

## 3. Results (validation)

![variant deltas](figures/phase6_variant_deltas.png)

| Model | τ\* | **F1** | Precision | Recall | ROC-AUC | **PR-AUC** | Macro-cluster F1 | **Worst-cluster F1** | ΔF1 vs B [95% CI] | ΔPR-AUC vs B [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| A: cosine similarity only | 0.76 | 0.735 | 0.640 | 0.863 | 0.876 | 0.765 | 0.733 | 0.651 (C6) | −0.036 [−0.040, −0.033] | −0.059 [−0.064, −0.054] |
| **B: semantic model probability** | 0.32 | **0.771** | 0.694 | 0.868 | 0.907 | **0.824** | 0.770 | **0.708** (C6) | – | – |
| B_recal (control) | 0.36 | 0.771 | 0.695 | 0.866 | 0.907 | 0.824 | 0.770 | 0.709 | −0.000 [−0.001, +0.000] | 0.000 |
| B_hgb (control) | 0.34 | 0.771 | 0.692 | 0.871 | 0.907 | 0.822 | 0.770 | 0.707 | −0.000 [−0.001, +0.001] | −0.002 [−0.002, −0.001] |
| C: constraints incl. spaCy (LR) | 0.38 | 0.775 | 0.710 | 0.854 | 0.910 | 0.831 | 0.773 | 0.704 | +0.004 [+0.001, +0.006] | +0.008 [+0.006, +0.009] |
| C_no_spacy (LR) | 0.38 | 0.774 | 0.709 | 0.853 | 0.910 | 0.830 | 0.773 | 0.706 | +0.003 [+0.001, +0.005] | +0.006 [+0.005, +0.008] |
| C_no_spacy_plus_lexical (LR) | 0.36 | 0.777 | 0.701 | 0.873 | 0.912 | 0.832 | 0.776 | 0.710 | – | – |
| C_hgb (incl. spaCy) | 0.38 | 0.784 | 0.724 | 0.855 | 0.916 | 0.839 | 0.782 | 0.721 | – | – |
| **C_no_spacy_hgb (chosen)** | **0.35** | **0.784** | **0.711** | **0.873** | **0.916** | **0.838** | **0.782** | **0.721** (C6) | **+0.013 [+0.010, +0.016]** | **+0.014 [+0.012, +0.017]** |
| C_plus_cluster (ablation) | 0.36 | 0.774 | 0.701 | 0.864 | 0.910 | 0.831 | 0.772 | 0.698 | +0.003 [+0.001, +0.005] | +0.007 [+0.005, +0.009] |

(The CIs come from paired bootstraps over 1,000 resamples of validation pairs, with each model at its own frozen τ\*.)

**Selection log:**
1. *spaCy:* +0.0014 PR-AUC over `C_no_spacy`, below the 0.005 bar, so it is **dropped**. With boosted trees it adds +0.0014 [−0.000, +0.003].
2. *Lexical overlap:* +0.0018 [−0.0001, +0.0036], not adopted.
3. *Boosted trees:* +0.0080 [+0.0056, +0.0100] over `C_no_spacy`, so the rule **switches** to sklearn's HistGradientBoosting. No new dependency is added.

**Controls:** re-fitting a model on the base score, linearly or with boosting, changes nothing. **The gain comes from the added features.**
**Calibration** improves: ECE 0.024 → 0.015 and Brier 0.120 → 0.113. The cross-half threshold check gives F1 0.783, against 0.784 in-sample.

### Which features matter (leave-one-group-out from the chosen model)

| Removed group | ΔF1 [95% CI] | ΔPR-AUC [95% CI] | Verdict |
|---|---|---|---|
| specificity | **−0.011 [−0.014, −0.008]** | **−0.008 [−0.010, −0.006]** | largest contributor |
| number | **−0.004 [−0.005, −0.002]** | **−0.003 [−0.005, −0.001]** | significant |
| entity (heuristic) | −0.001 [−0.003, +0.000] | −0.000 [−0.002, +0.002] | not significant alone |
| negation | −0.001 [−0.003, +0.001] | −0.002 [−0.004, +0.001] | not significant alone |
| intent word | −0.000 [−0.002, +0.001] | −0.001 [−0.004, +0.001] | not significant alone |
| date (regex) | −0.001 [−0.003, +0.001] | −0.001 [−0.003, +0.002] | not significant (regex date mismatch in 0.3% of pairs) |

Specificity pays off only **non-linearly**: removing it costs −0.002 F1 in the logistic model but −0.011 in the boosted one. The entity, negation,
intent and date groups are each individually not significant on validation. They may overlap with one another and with specificity, so removing
one is compensated by the others. Their *slice-level* effect is still large (§4).

## 4. Did entity-sensitive false positives decrease?

![slice error rates](figures/phase6_slice_fpr.png)

| Validation slice (Phase 5 heuristic tags; high overlap = word Jaccard > 0.6) | Negatives | FPR B → C | Positives | FNR B → C |
|---|---|---|---|---|
| All pairs | 24,557 | 0.212 → **0.196** | 13,581 | 0.132 → **0.127** |
| High overlap | 3,272 | 0.425 → **0.301** | 3,348 | 0.043 → 0.072 |
| Entity difference, high overlap | 1,425 | 0.216 → **0.126** (−42%) | 273 | 0.150 → 0.267 |
| Number difference, high overlap | 300 | 0.470 → **0.190** (−60%) | 66 | 0.136 → 0.333 |
| Negation difference, high overlap | 61 | 0.721 → **0.344** (−52%) | 30 | 0.100 → 0.233 |
| Date difference, high overlap | 67 | 0.463 → **0.269** | 19 | 0.053 → 0.158 |

**Yes. Entity, number and negation false positives at high overlap fall by roughly half.** In those slices, true duplicates with a superficial
constraint difference are rejected more often. Overall, though, the chosen operating point lowers **both** error rates:
- 955 false positives fixed and 428 false negatives fixed;
- 557 new false positives and 362 new false negatives;
- **net 398 fewer false positives and 66 fewer false negatives** on validation.

New false negatives include number-format false alarms ("2016-2017" vs "2016-17") and rhetorical negation ("Is X going to…" vs "Why isn't X going to…").

**Phase 5 hand-labelled errors** (60 false positives and 60 false negatives of the base model, rescored by C at its own τ):
- **13 of 60 false positives fixed:** 4 of 6 entity mismatches, 1 of 2 number, 1 of 2 negation, 3 of 9 different-aspect, 2 of 15 scope, 1 of 7 attribute swaps,
  and 1 of 16 likely-noise pairs.
- **16 of 60 false negatives fixed:** 9 low-overlap paraphrases, 6 broader/narrower, 1 debatable merge.

### Sanity probes (hand-written; `probes.json`)

| Type | Pair | B prob → dup? | C prob → dup? (τ 0.35) |
|---|---|---|---|
| paraphrase | "How can I lose weight quickly?" / "What is the fastest way to lose weight?" | 0.82 ✓ | 0.80 ✓ |
| paraphrase | "Is it possible to hack fb?" / "How do we hack a Facebook account?" | 0.53 ✓ | 0.56 ✓ |
| entity | "temperament of a Doberman/Lab mix" / "…a Lab/Pitbull mix" | 0.40 **✓** | 0.20 ✗ |
| number | "lose **5** kg in a month" / "lose **20** kg in a month" | 0.34 **✓** | 0.11 ✗ |
| number | "best phone under **10000** rupees" / "…under **20000** rupees" | 0.62 **✓** | 0.36 **✓** (still accepted, just above τ) |
| negation | "Why do people believe in God?" / "…**not** believe…" | 0.32 **✓** | 0.12 ✗ |
| entity / date / location | Microsoft vs Apple; 2012 vs 2016 election; Delhi vs Mumbai hotels | ✗ | ✗ |
| case only / acronym | "india"/"India"; "U.S."/"US" | ✓ | ✓ |

### Per cluster

| Cluster | F1 B → C | Precision B → C |
|---|---|---|
| C7 Money & business | 0.775 → **0.798** (+0.023) | 0.734 → 0.761 |
| C0 Definitions & technical concepts | 0.711 → **0.733** (+0.022) | 0.600 → 0.638 |
| C1, C3, C8, C4, C6, C10 | +0.013 to +0.016 | up |
| C5, C11, C9 | +0.003 to +0.010 | up |
| **C2 Accounts, apps & platforms** | 0.733 → **0.725** (−0.008) | 0.648 → 0.638 |

11 of 12 clusters improve, and the worst cluster (C6) rises from 0.708 to 0.721. The one decline is C2, whose errors are mostly paraphrases (Phase 5).

## 5. Cost

| | Base (Phase 3) | Meta (chosen) |
|---|---|---|
| Extra dependencies | – | none (spaCy dropped; sklearn trees) |
| Model artifact | 87 MB encoder + 0.8 MB head | + 0.4 MB `meta_model.joblib` |
| Latency, single pair (CPU, median) | 46 ms | 72 ms |
| Latency, batch of 2,000 (per pair) | 10.7 ms | 11.2 ms |

## 6. Limitations

1. **The gain is modest** (+0.013 F1). Most of it comes from generic specificity/length signals; numbers are the only explicit constraint group that
   is significant on its own.
2. **There is an FN trade-off within the constraint slices** (format variants like "2016-17", rhetorical negation, scope).
3. **The entity heuristic is crude,** and spaCy NER did not help enough (+0.001 PR-AUC). Neither handles aliases ("fb" vs "Facebook").
4. **The tree model's operating point is sensitive to tiny input changes.** Canonicalising inputs (changes below 1e-6) moved τ from 0.38 to 0.35
   within a flat plateau. Validation F1 barely changed (0.783 → 0.784), but precision and recall shifted (0.723/0.853 → 0.711/0.873).
5. Validation is used for thresholds and for choosing among about 20 fixed variants, with three logged rule corrections. Phase 8 is the real check.
6. Around 27% of the base model's false positives look mislabelled (Phase 5), a ceiling no feature can lift.

## 7. Hand-off to Phase 7

The global system to compare against is `C_no_spacy_hgb`, F1 0.784 at its own global τ = 0.35 (`meta_model.joblib`, `val_scores.csv`).
