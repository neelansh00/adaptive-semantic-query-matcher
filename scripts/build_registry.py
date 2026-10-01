"""Phase 10: model / artifact registry and feature configuration.

Records, for every artifact the project produces or depends on: role, phase, content hash, size, whether git
tracks it, and the exact command that regenerates it. Also writes the versioned feature configuration of the
final system and the software environment. Reads nothing from the test split.

Usage:  python scripts/build_registry.py            # (re)write the registry
        python scripts/build_registry.py --check    # verify every present artifact against the registry
Writes: artifacts/MODEL_REGISTRY.json, artifacts/feature_config.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.data import PROJECT_ROOT, content_sha256  # noqa: E402

REGISTRY = PROJECT_ROOT / "artifacts" / "MODEL_REGISTRY.json"
FEATURE_CONFIG = PROJECT_ROOT / "artifacts" / "feature_config.json"

# (path, phase, role, used_by_final_system, producer command)
ARTIFACTS = [
    ("data/processed/splits/split_metadata.json", 1, "frozen split definition + hashes", True, "python scripts/make_splits.py"),
    ("artifacts/phase2/lr_tfidf_pair_lexical.joblib", 2, "reference model: TF-IDF pair + lexical LR", False, "python scripts/run_baselines.py"),
    ("artifacts/phase2/lr_tfidf_pair.joblib", 2, "reference model: TF-IDF pair LR", False, "python scripts/run_baselines.py"),
    ("artifacts/phase2/lr_lexical.joblib", 2, "reference model: lexical-feature LR", False, "python scripts/run_baselines.py"),
    ("artifacts/phase3/bilstm/model.pt", 3, "reference model: Siamese BiLSTM weights", False, "python scripts/train_bilstm.py"),
    ("artifacts/phase3/bilstm/vocab.json", 3, "reference model: BiLSTM vocabulary (tokenizer)", False, "python scripts/train_bilstm.py"),
    ("artifacts/phase3/bilstm/config.json", 3, "reference model: BiLSTM hyper-parameters", False, "python scripts/train_bilstm.py"),
    ("artifacts/phase3/sbert_all-MiniLM-L6-v2/mlp.pt", 3, "BASE MODEL head weights (MiniLM + MLP)", True, "python scripts/run_sbert.py"),
    ("artifacts/phase3/sbert_all-MiniLM-L6-v2/mlp_config.json", 3, "base model head architecture", True, "python scripts/run_sbert.py"),
    ("artifacts/phase3/sbert_all-MiniLM-L6-v2/lr.joblib", 3, "reference: MiniLM + LR head", False, "python scripts/run_sbert.py"),
    ("artifacts/phase4/cluster_model/centroids.npy", 4, "frozen K-Means centroids (12 query groups)", True, "python scripts/build_clusters.py --refit"),
    ("artifacts/phase4/cluster_model/cluster_model.json", 4, "cluster model metadata", True, "python scripts/build_clusters.py --refit"),
    ("artifacts/phase4/cluster_names.json", 4, "post-hoc cluster names (manual)", True, "written by hand after inspection (Phase 4)"),
    ("artifacts/phase6/meta_model.joblib", 6, "FINAL MODEL: constraint-aware meta-classifier", True, "python scripts/crossfit_base.py && python scripts/train_meta.py"),
    ("artifacts/phase7/frozen_policy_baseline.json", 7, "baseline threshold policy (global tau 0.32)", True, "python scripts/calibration_experiments.py"),
    ("artifacts/phase7/frozen_policy_final.json", 7, "FINAL threshold policy (global tau 0.35)", True, "python scripts/calibration_experiments.py"),
    ("artifacts/phase8/freeze_manifest.json", 8, "freeze manifest (hashes of everything evaluated)", True, "python scripts/freeze_manifest.py"),
    ("artifacts/phase8/test_scores.csv", 8, "per-pair test scores from the single test run", False, "python scripts/final_evaluation.py (runs once)"),
    ("artifacts/phase8/test_results.json", 8, "reported test metrics", False, "python scripts/final_evaluation.py (runs once)"),
    ("configs/data.yaml", 1, "split configuration", True, "hand-written"),
    ("configs/clustering.yaml", 4, "clustering configuration", True, "hand-written"),
    ("configs/models.yaml", 10, "versioned description of the final system", True, "hand-written"),
]


def git_tracked(path: str) -> bool:
    return subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=PROJECT_ROOT,
                          capture_output=True).returncode == 0


def feature_config() -> dict:
    import joblib
    from src.features.constraints import FEATURE_GROUPS
    bundle = joblib.load(PROJECT_ROOT / "artifacts" / "phase6" / "meta_model.joblib")
    v = bundle["model"].variant
    return {"variant": v.name, "model": v.model, "uses_spacy": bundle["uses_spacy"], "threshold": bundle["threshold"],
            "feature_groups": {g: FEATURE_GROUPS[g] for g in v.groups}, "design_columns_in_order": v.columns(),
            "canonicalisation": {"base_logit": "logit(float32(base_probability)), eps=1e-6",
                                 "all_inputs": "round to 6 decimals in MetaModel.design()"},
            "text_preprocessing": "src/preprocessing/text.py normalize()/tokenize(); questions whitespace-cleaned first",
            "matching_rule": "item matched if present in the other question's annotation or as whole words in its text"}


def environment() -> dict:
    import numpy, pandas, sklearn, scipy, torch, sentence_transformers, transformers, streamlit  # noqa: E401
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{m.__name__: m.__version__ for m in (numpy, pandas, sklearn, scipy, torch, sentence_transformers, transformers, streamlit)}}


def build() -> dict:
    import yaml
    from huggingface_hub import hf_hub_download
    models = yaml.safe_load((PROJECT_ROOT / "configs" / "models.yaml").read_text())
    enc = models["encoder"]
    manifest = json.loads((PROJECT_ROOT / "artifacts" / "phase8" / "freeze_manifest.json").read_text())
    enc_file = Path(hf_hub_download(enc["name"], enc["weights_file"], revision=enc["revision"]))
    entries = []
    for path, phase, role, final, producer in ARTIFACTS:
        p = PROJECT_ROOT / path
        entries.append({"path": path, "phase": phase, "role": role, "used_by_final_system": final,
                        "exists": p.exists(), "sha256": content_sha256(p) if p.exists() else None,
                        "size_kb": round(p.stat().st_size / 1024, 1) if p.exists() else None,
                        "git_tracked": git_tracked(path), "regenerate_with": producer})
    reg = {"encoder": {"name": enc["name"], "revision": enc["revision"], "weights_sha256": content_sha256(enc_file),
                       "matches_freeze_manifest": content_sha256(enc_file) == manifest["encoder"]["model.safetensors_sha256"]},
           "artifacts": entries, "environment": environment(),
           "note": "Hashes are line-ending-normalised for text files (src/utils/data.py::content_sha256). "
                   "Untracked artifacts are large reference models or data-derived files, reproducible with 'regenerate_with'."}
    return reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        reg = json.loads(REGISTRY.read_text())
        bad = [a["path"] for a in reg["artifacts"] if a["sha256"] and (PROJECT_ROOT / a["path"]).exists()
               and content_sha256(PROJECT_ROOT / a["path"]) != a["sha256"]]
        missing = [a["path"] for a in reg["artifacts"] if a["used_by_final_system"] and not (PROJECT_ROOT / a["path"]).exists()]
        print(f"registry check: {len(reg['artifacts'])} entries, changed={bad}, missing final-system artifacts={missing}")
        sys.exit(1 if bad or missing else 0)
    reg = build()
    REGISTRY.write_text(json.dumps(reg, indent=2))
    FEATURE_CONFIG.write_text(json.dumps(feature_config(), indent=2))
    untracked_final = [a["path"] for a in reg["artifacts"] if a["used_by_final_system"] and not a["git_tracked"]]
    print(f"{len(reg['artifacts'])} artifacts registered; encoder matches freeze manifest: "
          f"{reg['encoder']['matches_freeze_manifest']}; final-system artifacts NOT in git: {untracked_final}")


if __name__ == "__main__":
    main()
