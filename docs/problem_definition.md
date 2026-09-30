# Problem Definition

## Task

Given two user questions **A** and **B**, estimate the probability that they express
**sufficiently equivalent intent** that they should be treated as duplicates. For example,
one could be merged into the other, or an answer to one would fully answer the other.

- **Target:** `is_duplicate` ∈ {0, 1}, taken from `data/raw/train.csv`, the only labelled file in the archive
  (see [data_audit.md](data_audit.md)).
- **Input:** the raw text of `question1` and `question2`. The IDs `qid1`/`qid2` and anything derived
  from the question graph (how often a question appears, how many neighbours it has) are
  **not model inputs** (see "Excluded signals" below).
- **Output:** a probability-like score *s*(A, B) and a binary decision `s ≥ τ`, where the threshold τ
  is chosen on **validation** data. The global threshold is one τ; later phases test per-cluster τ.

## What "duplicate" means, and what it does not

Duplicate means **same information need**. It does **not** mean lexically similar or topically related.
Two questions can share most of their words, or sit close together in embedding space, and still
ask for different things. The dataset contains many such pairs. All of the following are real
`is_duplicate = 0` examples found during the audit:

| Mismatch type | Example (Question A / Question B) |
|---|---|
| Different **entity** | "…if **Austria** had won the War of Spanish Succession?" / "…if **France** had won…" |
| Different **location** | "…public transit in **Susano**, Brazil?" / "…public transit in **Araçatuba**, Brazil?" |
| Different **entity** (movie) | "…end credits to the 2002 movie **Chance**?" / "…the 2002 movie **The Rookie**?" |
| **Negation / polarity** | "examples of **non-movable** joints?" / "examples of **movable** joints?" |
| **Role reversal** (same entities, different direction) | "advantages **India** received from its split with **Pakistan**" / "…**Pakistan** received…with **India**" |
| Different **attribute / field** | "careers in **biology** changing…" / "careers in **English** changing…" |

The following are also expected to break the equivalence (documented now, measured in Phases 5–6):

- different **numbers** or quantities ("lose 10 kg" vs "lose 30 kg")
- different **dates** or time periods ("in 2016" vs "in 2017")
- different **constraints** or scope ("…in India" vs "…" with no location; "for beginners" vs "for experts")
- **broader vs. narrower** questions (a question vs. a strictly more specific sub-question)
- **context-dependent questions** such as "Who is this?" or "What does this sentence mean?". Their literal
  text can be identical across posts while the referent differs. The audit found 5 exact-text pairs labelled 0 of this kind.

The reverse also happens. Duplicates can share very little vocabulary
("Is it possible to hack fb?" / "How do we hack a Facebook account?").
So neither lexical overlap nor raw embedding similarity is a sufficient definition.

## Why this matters for the project

The research question is whether **query type** (clusters discovered from embeddings, Phase 4),
**entity/constraint consistency** (Phase 6), and **calibrated thresholds** (Phase 7) improve on a
single embedding-similarity threshold. The mismatch types above are exactly the cases where a
single global similarity threshold is expected to fail:

- entity-heavy factual questions reach high similarity while differing in the one token that matters;
- open-ended advice questions ("best way to lose weight") are duplicates even when phrased very differently.

## Label semantics and noise (known caveats)

- Labels come from Quora's human process and are **not guaranteed correct**. The audit found
  5 pairs with identical text (after lowercasing and whitespace normalisation) labelled 0, and
  10 repeated pairs whose copies carry conflicting labels.
- The label is **asymmetric in coverage**. Only the pairs Quora chose to sample are labelled. A 0 means
  "this sampled pair was judged non-duplicate", not that the two questions are unrelated.
- The class balance (36.9% positive) reflects the sampling procedure, not real-world prevalence.
  Precision measured here is therefore specific to this sampling.

## Excluded signals (leakage shortcuts)

The audit shows that **how often a question appears** predicts the label on its own:
maximum question frequency alone reaches ROC-AUC ≈ 0.70, and pairs in large question-graph
components are 53–87% positive versus 22% for isolated pairs. This is an artifact of how the
dataset was sampled, not a property of intent. So we do **not** use:

- qid-derived features or question frequency / reuse counts;
- question-graph features (shared neighbours, component size, transitive labels);
- any statistic computed over validation or test questions.

A model that exploits these would not transfer to a system scoring genuinely new questions.
