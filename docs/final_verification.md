# Final Verification

Verification of the finished project on 2026-10-02: Windows 11, Python 3.12.2, CPU only. No model, feature, clustering, threshold or calibration
logic was changed in Phase 11, and the frozen Phase 8 evaluation artifacts are untouched.

## 1. Test results

| Check | Result |
|---|---|
| `python -m pytest` (working repository) | **133 passed**, 0 failed, 0 skipped (48.5 s) |
| Bare fresh `git clone`, no data | **121 passed, 12 skipped** (each skip names the step that regenerates its data), 0 failed |
| Fresh clone + Kaggle archive, after `reproduce.py --run extract splits` | **129 passed, 4 skipped** (Phase 6 feature caches; the two large untracked reference models), 0 failed |

The tests cover preprocessing, splits, lexical and constraint features, entity overlap, numeric mismatch, cluster assignment, threshold selection,
global fallback, model loading, deterministic inference (golden outputs), the Streamlit service layer and app, and the no-test-leakage and freeze
guarantees. The mapping is in [reproducibility.md](reproducibility.md).

## 2. Reproducibility verification

`python scripts/reproduce.py --verify`:

| Check | Result |
|---|---|
| Model registry: 22 artifacts vs recorded hashes | unchanged; no final-system artifact missing |
| Phase 8 freeze manifest: 25 files + encoder weights | **OK** |
| Test metrics recomputed from saved per-pair scores vs reported | identical. Final: F1 0.78559, PR-AUC 0.83355, ROC-AUC 0.91439, macro-cluster F1 0.78238, worst-cluster F1 0.69537. Baseline: 0.77502 / 0.82434 / 0.90643 / 0.77088 / 0.6684 |
| Split regenerated from the raw archive | reproduced exactly (id hashes) |
| Full test suite | passed |

Further checks:
- **Phase 8 artifacts unchanged since the test run:** `git diff 2277777 -- artifacts/phase8/` is empty, committed and in the working tree.
- **Fresh clone + archive:** regenerated split CSVs are **byte-identical** to the frozen files, and `--verify` passes.
- **Encoder:** pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; its weight hash equals the one in the freeze manifest.

## 3. Documentation, import and startup checks

`python scripts/check_docs.py`:

| Check | Result |
|---|---|
| Relative Markdown links in `README.md` and `docs/*.md` (21 files) | all resolve |
| Repository file paths named in the README | all exist |
| Modules in `src/` and `scripts/` imported (52) | no import errors |

**Service and app:**
- `DuplicateMatcher` loads in 18.8 s on a cold start. The first pair takes 218 ms; warm pairs take about 100 ms.
- On the number-mismatch probe the output is final 0.1141, not a duplicate, τ 0.35, no cluster threshold applied. That matches the golden value.
- The Streamlit app renders the decision, metrics, rationale, signals, what-if, query group and threshold note under `AppTest` (headless) with no
  exceptions. The real server answered `/_stcore/health` with `ok` in Phase 9.

## 4. Issues found and fixed during Phase 11

Both were found by actually cloning the repository and running it, not by inspection. Neither touched models or evaluation artifacts.

1. **Data-dependent tests failed on a fresh clone** instead of skipping: split CSVs, `val.csv` and the large untracked reference models were assumed to
   exist. They now skip with an instruction. The freeze test allows only the deliberately untracked reference models to be absent.
2. **`make_splits.py` could not materialise the split on a fresh clone.** With the versioned metadata present, it only verified. It now writes the split
   CSVs when the regenerated split matches the frozen hashes; the metadata is never rewritten. The result is byte-identical files.

## 5. Known limitations

- **Cluster boundaries are soft** (silhouette ≈ 0.02; HDBSCAN found no density structure). The 12 regions are a coarse coordinate, and their names are
  post-hoc interpretations, not universal question types.
- **Per-cluster thresholds did not help** out of sample and are not part of the final system.
- **Labels are noisy and ambiguous** (about 27% of the remaining sampled false positives look mislabelled; scope is judged inconsistently).
- **Remaining errors:** role/attribute swaps, different aspects of the same topic, broader/narrower scope, and paraphrases needing world knowledge.
  Constraint features also cause some false alarms on true duplicates.
- **The encoder was pretrained partly on Quora data,** so absolute scores are optimistic. The baseline-vs-final comparison is like-for-like.
- **Results apply to this frozen Quora Question Pairs split only.** No production robustness or cross-dataset generalisation is claimed.
- **Retraining the neural and K-Means steps is numerically close but not bit-identical.** The committed frozen artifacts are the reference for every
  reported number. The test evaluation ran once and cannot be re-run for optimisation.

## 6. Final project state

| Item | State |
|---|---|
| Final system | MiniLM + MLP head → spaCy-free constraint meta-classifier (HistGradientBoosting) → global τ = 0.35 |
| Held-out test (evaluated once) | F1 **0.78559**, PR-AUC **0.83355**, ROC-AUC 0.91439, macro-cluster F1 0.78238, worst-cluster F1 0.69537 |
| vs baseline (MiniLM + MLP, τ 0.32) | F1 +0.0106 [+0.0076, +0.0136]; worst-cluster F1 +0.0262 [+0.0097, +0.0417]; no cluster worse |
| Research question | constraint consistency improved robustness; per-cluster threshold calibration did not (negative result kept) |
| Versioned artifacts | 22 in the registry; all final-system artifacts tracked in git; encoder pinned |
| Tests | 133 passing |
| Documentation | README + 20 documents in `docs/`: phase reports, summary, metrics, interview notes, demo guide, resume bullets, this report |

PROJECT COMPLETE — DOCUMENTATION AND REPRODUCIBILITY VERIFIED
