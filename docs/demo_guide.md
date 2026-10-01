# Demo Guide (3–5 minutes)

```bash
streamlit run app/main.py          # first load ~15 s (models); then ~100 ms per pair
```

All five pairs are built-in examples (sidebar), and every value below was produced by the frozen final system. The numbers are reproducible: the same
pairs are in `tests/golden/expected_scores.json`.

**Read the screen in this order:**
1. the green/red **decision** banner;
2. the four metric tiles: **final probability**, **semantic model only**, **semantic similarity**, **global threshold** (0.35);
3. **Why** (the rationale);
4. the **mismatch signals** and **what-if** tables;
5. the **discovered query group** and the threshold note.

## Opening (30 s)

> "This decides whether two questions ask the same thing. The baseline is a sentence-embedding model with one threshold. That handles paraphrases, but it
> accepts near-identical questions that differ in one crucial detail. The final system adds deterministic checks for numbers, dates, entities, negation
> and scope. On a frozen held-out test set it improves F1 from 0.775 to 0.786 and the worst query region from 0.668 to 0.695. Every explanation on screen
> is computed from the model and its features; there's no language model."

## 1. Clear paraphrase (40 s)

**"How can I lose weight quickly?"** / **"What is the fastest way to lose weight?"**

| Expected | Value |
|---|---|
| Semantic similarity | 0.868 |
| Semantic model only | 0.818 (duplicate) |
| Signals | question word differs ("how" vs "what"); only in A: "quickly", only in B: "fastest", "way" |
| Cluster | C3 Health, body, food & remedies |
| Final | **0.804 → duplicate** |

> "Different words, same need. The embedding model recognises the paraphrase, and the constraint checks find nothing that changes the meaning.
> The only differences are wording, so the duplicate decision stands."

## 2. High-overlap entity mismatch (50 s)

**"What is the temperament of a Doberman/Lab mix?"** / **"What is the temperament of a Lab/Pitbull mix?"**

| Expected | Value |
|---|---|
| Semantic similarity | 0.866 |
| Semantic model only | 0.403 (**duplicate**: the baseline is wrong) |
| Signals | named entities differ: "doberman" vs "pitbull" |
| What-if | if the entities matched, the score would be 0.321 |
| Cluster | C0 Definitions & technical/science concepts (near a group boundary) |
| Final | **0.202 → not a duplicate** |

> "Same template, different breed. The embedding model alone calls it a duplicate. The entity check flags Doberman vs Pitbull, and the score drops from
> 0.40 to 0.20. The what-if shows the entity difference is responsible for part of that drop. On the test set, false positives of this kind fell from
> 17.4% to 7.5%."

## 3. Number mismatch (40 s)

**"How can I lose 5 kg in a month?"** / **"How can I lose 20 kg in a month?"**

| Expected | Value |
|---|---|
| Semantic similarity | 0.880 |
| Semantic model only | 0.336 (**duplicate**: the baseline is wrong) |
| Signals | numbers differ: 5 vs 20 |
| What-if | if the numbers matched, the score would be 0.267 |
| Cluster | C3 Health, body, food & remedies |
| Final | **0.114 → not a duplicate** |

> "Embeddings barely distinguish 5 from 20. On validation, a plain similarity threshold accepted 81% of near-identical pairs that differ only in a number.
> The number check catches it. This is also the most consistent constraint signal in the ablations."

## 4. Negation mismatch (40 s)

**"Why do people believe in God?"** / **"Why do people not believe in God?"**

| Expected | Value |
|---|---|
| Semantic similarity | 0.898 (highest of the five) |
| Semantic model only | 0.320 (**duplicate**: the baseline is wrong, right at its threshold) |
| Signals | only question B is negated |
| What-if | if the negation matched, the score would be 0.269 |
| Cluster | C1 Opinions on public figures, beliefs & speculation (near a group boundary) |
| Final | **0.123 → not a duplicate** |

> "The most similar pair of the five, yet the opposite question. Similarity alone can't see 'not'. The negation signal does."

## 5. Ambiguous broader vs narrower (40 s)

**"How do I learn programming?"** / **"How do I learn Python programming for data science?"**

| Expected | Value |
|---|---|
| Semantic similarity | 0.660 |
| Semantic model only | 0.265 (not duplicate) |
| Signals | entity only in B: "python"; broader/narrower: A is broader, B adds "data", "python", "science" |
| What-if | if the entities matched, the score would be 0.394 |
| Cluster | C10 Learning, study & exam preparation |
| Final | **0.313 → not a duplicate (borderline)** |

> "This one is genuinely ambiguous: one question is a narrower version of the other. Quora's own labels are inconsistent on scope; sometimes they merge,
> sometimes not. The app flags it as borderline instead of pretending to be certain. Scope differences are about a quarter of the remaining false positives."

## Close (30 s): query groups and limits

Point at the **discovered query group** and the threshold note:

> "Each pair lands in one of 12 query regions that K-Means found from embeddings; I named them only afterwards. I tested giving each region its own
> threshold. In-sample it looked better, but evaluated out of sample it was worse, so the frozen system uses one global threshold. The app says so.
> The regions are soft: silhouette is about 0.02, so they're a coordinate, not true question types."

**If there's time,** show the built-in **"Known failure: role swap"** example. It's accepted at 0.64, which is wrong:

> "Role reversals aren't captured by token-level checks. That's one of the remaining error types I documented."

## Fallbacks

- **The first load is slow (~15 s):** start the app before the meeting and run one example to warm the cache.
- **Asked about a custom pair:** type it in. Decisions within 0.05 of 0.35 are shown as borderline.
- **Asked "is this production-ready?":** "No. It's an evaluated research system on one dataset; the README lists the limitations."
