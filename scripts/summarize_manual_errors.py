"""Phase 5: tabulate the manual error categories (artifacts/phase5/manual_error_labels.csv).

The labels are one primary category per sampled error, assigned by reading each pair (single annotator,
model score visible). Sample: 15 FP + 15 FN from each of the 3 weakest clusters by F1 and from the
remaining clusters pooled (seed 42; see analyze_clusters.py).

Usage:  python scripts/summarize_manual_errors.py [--dir artifacts/phase8]
Writes: artifacts/phase5/manual_error_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.data import PROJECT_ROOT  # noqa: E402

ART = PROJECT_ROOT / "artifacts" / "phase5"
CONSTRAINT = {"entity_mismatch", "number_mismatch", "date_time_mismatch", "location_mismatch", "negation_mismatch"}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ART), help="folder with error_samples.csv + manual_error_labels.csv")
    art = Path(ap.parse_args().dir)
    samples = pd.read_csv(art / "error_samples.csv")
    labels = pd.read_csv(art / "manual_error_labels.csv")
    assert set(samples.id) == set(labels.id) and labels.id.is_unique, "labels must cover the sample exactly"
    m = samples.merge(labels, on="id")
    out = {"n_labelled": len(m), "annotation": "single annotator, model score visible, one primary category per pair"}
    for kind in ("FP", "FN"):
        part = m[m.outcome == kind]
        out[kind] = {"counts": part.category.value_counts().to_dict(),
                     "share": part.category.value_counts(normalize=True).round(3).to_dict()}
        if "group" in part:
            out[kind]["by_group"] = pd.crosstab(part.category, part.group).to_dict()
    fp = m[m.outcome == "FP"]
    out["FP"]["share_explicit_constraint_mismatch"] = float(fp.category.isin(CONSTRAINT).mean())
    out["FP"]["share_constraint_or_attribute_swap"] = float(fp.category.isin(CONSTRAINT | {"attribute_swap"}).mean())
    out["FP"]["share_likely_label_noise"] = float((fp.category == "likely_label_noise").mean())
    (art / "manual_error_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("FP", "FN")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
