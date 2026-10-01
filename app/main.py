"""Adaptive Semantic Query Matcher: Streamlit demo of the frozen final system.

Run:  streamlit run app/main.py
All logic lives in src/service/matcher.py; this file only renders it. No language model is used:
every number and sentence shown is computed deterministically from the frozen models and features.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.service.matcher import DuplicateMatcher  # noqa: E402

EXAMPLES = {
    "Paraphrase": ("How can I lose weight quickly?", "What is the fastest way to lose weight?",
                   "Different wording, same need: should be a duplicate."),
    "Entity mismatch (the base model accepts it)": ("What is the temperament of a Doberman/Lab mix?",
                                                    "What is the temperament of a Lab/Pitbull mix?",
                                                    "Same template, different breed. The semantic model alone says duplicate."),
    "Entity mismatch": ("Who founded Microsoft?", "Who founded Apple?", "Different company: not a duplicate."),
    "Number mismatch": ("How can I lose 5 kg in a month?", "How can I lose 20 kg in a month?",
                        "Different target: not a duplicate. The semantic model alone says duplicate."),
    "Negation mismatch": ("Why do people believe in God?", "Why do people not believe in God?",
                          "Opposite question: not a duplicate. The semantic model alone says duplicate."),
    "Date mismatch": ("Who won the 2012 US presidential election?", "Who won the 2016 US presidential election?",
                      "Different election: not a duplicate."),
    "Location mismatch": ("What are the best hotels in Delhi?", "What are the best hotels in Mumbai?",
                          "Different city: not a duplicate."),
    "Broader vs narrower": ("How do I learn programming?", "How do I learn Python programming for data science?",
                            "One question is a narrower version of the other: borderline by design."),
    "Known failure: role swap": ("Can two introverted parents produce an extroverted child?",
                                 "If parents are extroverts, can their child be an introvert?",
                                 "The roles are reversed, so these are different questions, but the system says duplicate. "
                                 "Token-level features do not capture role swaps."),
    "Debatable: Quora label says duplicate": ("Is 299 a good enough GRE score?", "Is a score of 319 in GRE a good score?",
                                              "Quora labelled this pair a duplicate; the number check rejects it. "
                                              "This is the cost side of constraint features."),
}


@st.cache_resource(show_spinner="Loading the frozen models (first run only)…")
def get_matcher() -> DuplicateMatcher:
    return DuplicateMatcher()


def set_example() -> None:
    qa, qb, _ = EXAMPLES[st.session_state["example"]]
    st.session_state["qa"], st.session_state["qb"] = qa, qb


def main() -> None:
    st.set_page_config(page_title="Adaptive Semantic Query Matcher", layout="centered")
    st.title("Adaptive Semantic Query Matcher")
    st.caption("Are two questions asking the same thing? Frozen final system: MiniLM semantic model + deterministic "
               "entity/number/date/negation/scope checks, one global threshold. No language model is used for explanations.")

    if "qa" not in st.session_state:
        st.session_state["example"] = "Paraphrase"
        set_example()
    with st.sidebar:
        st.header("Examples")
        st.radio("Load an example pair", list(EXAMPLES), key="example", on_change=set_example)
        st.info(EXAMPLES[st.session_state["example"]][2])
        st.header("About")
        st.markdown(
            "- **Data:** Quora Question Pairs, question-disjoint split\n"
            "- **Held-out test:** F1 0.786, PR-AUC 0.834 (baseline 0.775 / 0.824)\n"
            "- **Decision:** one global threshold. Per-cluster thresholds were tested and rejected (Phase 7)\n"
            "- **Details:** `docs/final_evaluation.md`")

    qa = st.text_area("Question A", key="qa", height=80)
    qb = st.text_area("Question B", key="qb", height=80)
    if not st.button("Compare", type="primary") and "result" not in st.session_state:
        return
    if not qa.strip() or not qb.strip():
        st.warning("Please enter both questions.")
        return

    matcher = get_matcher()
    t = time.perf_counter()
    r = matcher.match(qa, qb)
    elapsed = (time.perf_counter() - t) * 1000
    st.session_state["result"] = True

    verdict = "Duplicate" if r.decision else "Not a duplicate"
    msg = f"**{verdict}**: duplicate probability {r.final_probability:.2f} (global threshold {r.global_threshold:.2f})"
    if r.borderline:
        msg += ", borderline"
    (st.success if r.decision else st.error)(msg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final probability", f"{r.final_probability:.2f}")
    c2.metric("Semantic model only", f"{r.base_probability:.2f}",
              help=f"Phase 3 model; its own threshold is {r.base_threshold:.2f} "
                   f"({'duplicate' if r.base_decision else 'not duplicate'})")
    c3.metric("Semantic similarity", f"{r.semantic_similarity:.2f}", help="Cosine of the MiniLM sentence embeddings")
    c4.metric("Global threshold", f"{r.global_threshold:.2f}")

    st.subheader("Why")
    for line in r.rationale:
        st.markdown(f"- {line}")

    st.subheader("Mismatch signals")
    if r.signals:
        st.dataframe(pd.DataFrame(r.signals).rename(columns={"type": "signal", "detail": "detail"}),
                     hide_index=True, width="stretch")
    else:
        st.markdown("None detected.")
    if r.what_if:
        st.markdown("**What-if** (model re-scored as if that constraint matched across the two questions)")
        st.dataframe(pd.DataFrame([{"constraint": w["group"], "probability if it matched": round(w["probability_if_no_difference"], 3),
                                    "effect on score": round(w["effect"], 3)} for w in r.what_if]),
                     hide_index=True, width="stretch")

    st.subheader("Discovered query group")
    st.markdown(f"**C{r.cluster_id}: {r.cluster_name}** (assignment margin {r.cluster_margin:.3f}"
                f"{', near a group boundary' if r.near_cluster_boundary else ''})")
    st.markdown(f"**Cluster-specific threshold:** {r.cluster_threshold_note}")
    st.caption("Groups were discovered by K-Means on question embeddings and named after inspection. They are soft regions, "
               "not fixed question types.")

    with st.expander("Extracted entities and constraints"):
        rows = []
        for key, label in (("numbers", "numbers"), ("dates", "dates"), ("entities", "entities (capitalised / acronyms)"),
                           ("has_negation", "negation"), ("question_word", "question word"), ("content_words", "content words")):
            a, b = r.extracted["question_a"][key], r.extracted["question_b"][key]
            fmt = lambda v: ", ".join(v) if isinstance(v, list) else str(v)  # noqa: E731
            rows.append({"item": label, "question A": fmt(a) or "-", "question B": fmt(b) or "-"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Computed in {elapsed:.0f} ms on CPU.")


main()
