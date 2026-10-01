"""Phase 8a: freeze everything the test evaluation will use, BEFORE the test split is read.

Records SHA-256 hashes of every model artifact, threshold policy, config and the code that turns raw text into a
decision, plus the git commit and library versions. The manifest is committed before final_evaluation.py runs;
final_evaluation.py refuses to run if any hash differs. This script does not read the test split.

Usage:  python scripts/freeze_manifest.py
Writes: artifacts/phase8/freeze_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.data import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "phase8" / "freeze_manifest.json"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

FROZEN = {
    # decision-making artifacts
    "split_metadata": "data/processed/splits/split_metadata.json",
    "base_head_weights": "artifacts/phase3/sbert_all-MiniLM-L6-v2/mlp.pt",
    "base_head_config": "artifacts/phase3/sbert_all-MiniLM-L6-v2/mlp_config.json",
    "cluster_centroids": "artifacts/phase4/cluster_model/centroids.npy",
    "meta_model": "artifacts/phase6/meta_model.joblib",
    "policy_baseline": "artifacts/phase7/frozen_policy_baseline.json",
    "policy_final": "artifacts/phase7/frozen_policy_final.json",
    "config_data": "configs/data.yaml",
    "config_clustering": "configs/clustering.yaml",
    # code on the scoring path
    "code_preprocessing": "src/preprocessing/text.py",
    "code_constraints": "src/features/constraints.py",
    "code_lexical": "src/features/lexical.py",
    "code_meta": "src/models/meta.py",
    "code_sentence_encoder": "src/models/sentence_encoder.py",
    "code_predictors": "src/models/predictors.py",
    "code_pairs": "src/clustering/pairs.py",
    "code_cluster_core": "src/clustering/core.py",
    "code_thresholds": "src/calibration/thresholds.py",
    "code_metrics": "src/evaluation/metrics.py",
    "code_cluster_analysis": "src/evaluation/cluster_analysis.py",
    "code_final_evaluation": "scripts/final_evaluation.py",
    "code_run_sbert_head": "scripts/run_sbert.py",
    # reference systems (reported for context, not used to choose anything)
    "ref_phase2_tfidf_lexical": "artifacts/phase2/lr_tfidf_pair_lexical.joblib",
    "ref_phase3_bilstm_weights": "artifacts/phase3/bilstm/model.pt",
    "ref_phase3_bilstm_vocab": "artifacts/phase3/bilstm/vocab.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout
    dirty = [l for l in status.splitlines() if "artifacts/phase8/" not in l and "freeze_manifest.py" not in l]
    if dirty:
        raise SystemExit("Working tree has uncommitted changes; commit them before freezing:\n" + "\n".join(dirty))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()

    from huggingface_hub import hf_hub_download
    encoder_file = Path(hf_hub_download(ENCODER, "model.safetensors"))
    import numpy, pandas, sklearn, torch, sentence_transformers, transformers  # noqa: E401
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "systems": {
            "baseline": {"description": "Phase 3 frozen MiniLM + MLP head, global threshold",
                         "score": "sbert_mlp", "policy": FROZEN["policy_baseline"]},
            "final": {"description": "Phase 6 entity/constraint-aware meta-model (C_no_spacy_hgb), global threshold "
                                     "(Phase 7 rejected cluster thresholds)",
                      "score": "C_no_spacy_hgb", "policy": FROZEN["policy_final"]},
        },
        "encoder": {"name": ENCODER, "model.safetensors_sha256": sha256(encoder_file)},
        "files": {k: {"path": v, "sha256": sha256(PROJECT_ROOT / v)} for k, v in FROZEN.items()},
        "library_versions": {m.__name__: m.__version__ for m in (numpy, pandas, sklearn, torch, sentence_transformers, transformers)},
        "rules": ["test split read exactly once by scripts/final_evaluation.py",
                  "no model, feature, cluster, threshold or calibration change after this manifest",
                  "reference systems are reported for context only"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("frozen_at_utc", "git_commit", "systems")}, indent=1))
    print(f"{len(FROZEN)} files hashed -> {OUT}")


if __name__ == "__main__":
    main()
