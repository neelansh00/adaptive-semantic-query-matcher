"""Phase 10: record golden outputs of the frozen systems on fixed probe pairs (deterministic-inference test).

Probe pairs are hand-written (no dataset needed), so the golden test also runs on a fresh clone that has only the
git-tracked artifacts and the pinned encoder. Re-run only if a frozen model is deliberately changed.

Usage:  python scripts/make_golden.py
Writes: tests/golden/expected_scores.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.data import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "tests" / "golden" / "expected_scores.json"
PROBES = [
    ("How can I lose weight quickly?", "What is the fastest way to lose weight?"),
    ("Is it possible to hack fb?", "How do we hack a Facebook account?"),
    ("Who founded Microsoft?", "Who founded Apple?"),
    ("What is the temperament of a Doberman/Lab mix?", "What is the temperament of a Lab/Pitbull mix?"),
    ("How can I lose 5 kg in a month?", "How can I lose 20 kg in a month?"),
    ("What is the best phone under 10000 rupees?", "What is the best phone under 20000 rupees?"),
    ("Who won the 2012 US presidential election?", "Who won the 2016 US presidential election?"),
    ("What are the best hotels in Delhi?", "What are the best hotels in Mumbai?"),
    ("Why do people believe in God?", "Why do people not believe in God?"),
    ("How do I learn programming?", "How do I learn Python programming for data science?"),
    ("What are the best places to visit in india?", "What are the best places to visit in India?"),
    ("How do I apply to the U.S. army?", "How do I apply to the US Army?"),
    ("Can two introverted parents produce an extroverted child?", "If parents are extroverts, can their child be an introvert?"),
    ("What is formwork?", "What is formwork?"),
]


def main() -> None:
    from src.service.matcher import DuplicateMatcher
    m = DuplicateMatcher()
    rows = []
    for a, b in PROBES:
        r = m.match(a, b)
        rows.append({"question_a": a, "question_b": b, "base_probability": r.base_probability,
                     "final_probability": r.final_probability, "decision": r.decision,
                     "semantic_similarity": r.semantic_similarity, "cluster_id": r.cluster_id,
                     "signal_types": [s["type"] for s in r.signals]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"system": "frozen Phase 8 final system via src/service/matcher.py",
                               "tolerance": 1e-5, "probes": rows}, indent=2))
    for row in rows:
        print(f"{row['final_probability']:.4f} {str(row['decision']):5s} base {row['base_probability']:.4f}  {row['question_a'][:50]}")


if __name__ == "__main__":
    main()
