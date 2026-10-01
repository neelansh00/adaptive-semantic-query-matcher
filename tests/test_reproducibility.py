"""Phase 10: versioned artifacts, model loading, deterministic inference (golden outputs), reproducible results."""
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.utils.data import PROJECT_ROOT, content_sha256

ART = PROJECT_ROOT / "artifacts"
REGISTRY = ART / "MODEL_REGISTRY.json"
GOLDEN = PROJECT_ROOT / "tests" / "golden" / "expected_scores.json"


# ---------------------------------------------------------------- versioned artifacts

def test_registry_hashes_match_present_artifacts():
    reg = json.loads(REGISTRY.read_text())
    for a in reg["artifacts"]:
        p = PROJECT_ROOT / a["path"]
        if p.exists() and a["sha256"]:
            assert content_sha256(p) == a["sha256"], a["path"]


def test_final_system_artifacts_are_present_and_git_tracked():
    reg = json.loads(REGISTRY.read_text())
    for a in reg["artifacts"]:
        if a["used_by_final_system"] and not a["path"].startswith("data/"):
            assert (PROJECT_ROOT / a["path"]).exists(), a["path"]
            assert a["git_tracked"], f"{a['path']} must be versioned in git so a fresh clone can run the final system"


def test_encoder_is_pinned_to_the_evaluated_weights():
    import yaml
    models = yaml.safe_load((PROJECT_ROOT / "configs" / "models.yaml").read_text())
    reg = json.loads(REGISTRY.read_text())
    assert reg["encoder"]["revision"] == models["encoder"]["revision"]
    assert reg["encoder"]["matches_freeze_manifest"] is True


# ---------------------------------------------------------------- model loading

def test_frozen_models_load():
    import joblib
    from scripts.run_sbert import PairMLP
    from src.calibration.thresholds import ThresholdPolicy
    from src.clustering.core import CentroidModel
    head_dir = ART / "phase3" / "sbert_all-MiniLM-L6-v2"
    cfg = json.loads((head_dir / "mlp_config.json").read_text())
    head = PairMLP(cfg["in_dim"], cfg["hidden"], cfg["dropout"])
    head.load_state_dict(torch.load(head_dir / "mlp.pt", map_location="cpu", weights_only=True))
    assert CentroidModel.load(ART / "phase4" / "cluster_model").k == 12
    bundle = joblib.load(ART / "phase6" / "meta_model.joblib")
    assert bundle["variant"] == "C_no_spacy_hgb" and bundle["uses_spacy"] is False
    assert ThresholdPolicy.load(ART / "phase7" / "frozen_policy_final.json").global_threshold == pytest.approx(0.35)


def test_feature_config_matches_the_frozen_model():
    import joblib
    fc = json.loads((ART / "feature_config.json").read_text())
    bundle = joblib.load(ART / "phase6" / "meta_model.joblib")
    assert fc["design_columns_in_order"] == bundle["model"].variant.columns()
    assert fc["threshold"] == pytest.approx(bundle["threshold"])


@pytest.mark.parametrize("name", ["TfidfLexicalPredictor", "BiLSTMPredictor"])
def test_reference_predictors_load_if_present(name):
    from src.models import predictors
    path = {"TfidfLexicalPredictor": ART / "phase2" / "lr_tfidf_pair_lexical.joblib",
            "BiLSTMPredictor": ART / "phase3" / "bilstm" / "model.pt"}[name]
    if not path.exists():
        pytest.skip(f"{path.name} is a large untracked reference model (regenerate via scripts/reproduce.py)")
    p = getattr(predictors, name)()
    s1 = p.predict(["Who founded Microsoft?"], ["Who founded Apple?"])
    s2 = p.predict(["Who founded Microsoft?"], ["Who founded Apple?"])
    assert 0 <= s1[0] <= 1 and np.array_equal(s1, s2)


# ---------------------------------------------------------------- deterministic inference (golden outputs)

@pytest.fixture(scope="module")
def matcher():
    from src.service.matcher import DuplicateMatcher
    return DuplicateMatcher()


def test_golden_outputs_reproduce(matcher):
    golden = json.loads(GOLDEN.read_text())
    for row in golden["probes"]:
        r = matcher.match(row["question_a"], row["question_b"])
        assert r.final_probability == pytest.approx(row["final_probability"], abs=golden["tolerance"]), row["question_a"]
        assert r.base_probability == pytest.approx(row["base_probability"], abs=golden["tolerance"])
        assert r.decision == row["decision"] and r.cluster_id == row["cluster_id"]
        assert [s["type"] for s in r.signals] == row["signal_types"]


# ---------------------------------------------------------------- reported results are reproducible

def test_reported_test_metrics_recompute_from_saved_scores():
    from scripts.reproduce import verify_test_results
    res = verify_test_results()
    for system, metrics in res.items():
        for k, (got, reported, ok) in metrics.items():
            assert ok, f"{system}.{k}: recomputed {got} vs reported {reported}"


def test_reproduce_lists_every_pipeline_step_with_existing_scripts():
    from scripts.reproduce import STEPS
    out = subprocess.run([sys.executable, "scripts/reproduce.py", "--list"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True)
    assert out.returncode == 0
    for name, _, _, cmds in STEPS:
        assert name in out.stdout
        for c in cmds:
            for tok in c.split():
                if tok.startswith("scripts/"):
                    assert (PROJECT_ROOT / tok).exists(), tok
