import json

import pytest

from src.utils.data import PROJECT_ROOT, content_sha256

MANIFEST = PROJECT_ROOT / "artifacts" / "phase8" / "freeze_manifest.json"


def test_nothing_changed_since_freeze():
    """Every model, threshold, config and scoring-code file must still match the Phase 8 freeze manifest."""
    if not MANIFEST.exists():
        pytest.skip("Phase 8 freeze manifest not written yet")
    manifest = json.loads(MANIFEST.read_text())
    registry = json.loads((PROJECT_ROOT / "artifacts" / "MODEL_REGISTRY.json").read_text())
    untracked_refs = {a["path"] for a in registry["artifacts"] if not a["git_tracked"] and not a["used_by_final_system"]}
    missing = [v["path"] for v in manifest["files"].values() if not (PROJECT_ROOT / v["path"]).exists()]
    # a fresh clone lacks only the large, deliberately untracked reference models; anything else missing is an error
    assert set(missing) <= untracked_refs, f"frozen files missing: {set(missing) - untracked_refs}"
    changed = [k for k, v in manifest["files"].items()
               if (PROJECT_ROOT / v["path"]).exists() and content_sha256(PROJECT_ROOT / v["path"]) != v["sha256"]]
    assert not changed, f"frozen files changed after the freeze: {changed}"


def test_final_system_is_frozen_global_threshold_meta_model():
    if not MANIFEST.exists():
        pytest.skip("Phase 8 freeze manifest not written yet")
    m = json.loads(MANIFEST.read_text())
    assert m["systems"]["final"]["score"] == "C_no_spacy_hgb"
    pol = json.loads((PROJECT_ROOT / m["systems"]["final"]["policy"]).read_text())
    assert pol["mode"] == "global" and pol["global_threshold"] == pytest.approx(0.35)


def test_test_split_evaluated_at_most_once_and_after_freeze():
    marker = PROJECT_ROOT / "artifacts" / "phase8" / "TEST_EVALUATED.json"
    if not marker.exists():
        pytest.skip("test split not evaluated yet")
    mk, man = json.loads(marker.read_text()), json.loads(MANIFEST.read_text())
    assert mk["manifest_frozen_at_utc"] == man["frozen_at_utc"]
    assert mk["evaluated_at_utc"] >= man["frozen_at_utc"]       # evaluation happened after freezing


def test_content_hash_ignores_line_endings_for_text_only(tmp_path):
    crlf, lf = bytes([120, 13, 10, 121, 13, 10]), bytes([120, 10, 121, 10])   # "x\r\ny\r\n" vs "x\ny\n"
    (tmp_path / "a.py").write_bytes(crlf)
    (tmp_path / "b.py").write_bytes(lf)
    assert content_sha256(tmp_path / "a.py") == content_sha256(tmp_path / "b.py")
    (tmp_path / "a.pt").write_bytes(crlf)
    (tmp_path / "b.pt").write_bytes(lf)
    assert content_sha256(tmp_path / "a.pt") != content_sha256(tmp_path / "b.pt")   # binary: byte-exact
