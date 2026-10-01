# Phase 9: Streamlit Demo

```bash
streamlit run app/main.py        # http://localhost:8501 ; first load takes ~15 s (models), then ~100 ms per pair
```

The demo runs the **frozen final system** evaluated in Phase 8 and nothing else:
1. MiniLM embeddings → Phase 3 MLP head (base probability);
2. deterministic spaCy-free constraint features → Phase 6 meta-model (final probability);
3. one **global threshold, τ = 0.35**.

It reproduces the evaluated validation scores to within 5e-7, with 0 decision flips on 200 checked pairs (`tests/test_service.py`).

**No language model is used.** Every number and sentence on screen is computed deterministically from the frozen models and features.

## What it shows

| Output | Source |
|---|---|
| Duplicate probability and decision | meta-model probability vs global τ 0.35; flagged *borderline* if within 0.05 of τ |
| Semantic model only | Phase 3 probability and its own decision at τ 0.32, so the effect of the constraint layer is visible |
| Semantic similarity | cosine of the two MiniLM embeddings |
| Discovered query group | frozen Phase 4 centroid nearest the pair's average embedding, its post-hoc name, and whether the pair sits near a group boundary |
| Global / cluster threshold | the global τ is applied. **No cluster threshold is applied**, and the app says why: Phase 7 found per-cluster thresholds lowered macro-cluster F1 out of sample |
| Extracted entities/constraints | per question: numbers, dates, capitalised entities and acronyms, negation, question word, content words |
| Mismatch signals | derived from the same features the model uses: numbers/dates/entities that differ or appear on one side only, negation disagreement, question-word difference, broader/narrower scope |
| What-if | the meta-model re-scored **as if one constraint matched** across the two questions (items kept but treated as identical). This is a counterfactual on the model, not a causal claim about the text. Scope/specificity has no what-if, because "identical content" is not a coherent state for two different questions |
| Rationale | a fixed template filled from the values above |

## Built-in examples

| Example | Semantic model only | Final system | Point |
|---|---|---|---|
| Paraphrase ("lose weight quickly" / "fastest way to lose weight") | 0.82 dup | **0.80 dup** | different wording, same need |
| Entity: Doberman/Lab vs Lab/Pitbull mix | 0.40 **dup** | **0.20 not dup** | the constraint layer fixes a base-model false positive |
| Entity: "Who founded Microsoft?" / "…Apple?" | 0.03 not dup | 0.05 not dup | spec example; the base model already handles it |
| Number: lose 5 kg vs 20 kg | 0.34 **dup** | **0.11 not dup** | numbers differ |
| Negation: believe vs not believe in God | 0.32 **dup** | **0.12 not dup** | polarity differs |
| Date: 2012 vs 2016 election | 0.04 not dup | 0.02 not dup | |
| Location: hotels in Delhi vs Mumbai | 0.00 not dup | 0.01 not dup | |
| Broader vs narrower: "learn programming" / "learn Python programming for data science" | 0.26 not dup | 0.31 not dup, **borderline** | scope differences are genuinely ambiguous |
| **Known failure:** role swap (introvert parents → extrovert child, and the reverse) | 0.56 dup | **0.64 dup (wrong)** | token features do not capture role reversal |
| **Debatable:** GRE score 299 vs 319 | 0.22 not dup | 0.06 not dup | Quora labels it a duplicate; the number check disagrees (the cost side of Phase 6) |

The last two are included on purpose so the demo shows the system's limits as well as its strengths.

## Code

- `src/service/matcher.py`: `DuplicateMatcher.match(a, b) -> MatchResult`. All logic, no UI.
- `app/main.py`: a thin Streamlit renderer (`st.cache_resource` loads the models once).
- `tests/test_service.py`: fields and frozen policy, symmetry and determinism, documented example behaviour, coherent what-if,
  extraction, empty input, **parity with the evaluated scores**, and a headless `AppTest` of the UI.
