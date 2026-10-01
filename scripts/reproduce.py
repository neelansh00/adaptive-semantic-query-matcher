"""Phase 10: one entry point to reproduce the project from the raw Kaggle archive to the final evaluation.

  python scripts/reproduce.py --list                 # every step, in order, with rough CPU time
  python scripts/reproduce.py --run splits baselines # run selected steps
  python scripts/reproduce.py --from encode --to calibration
  python scripts/reproduce.py --verify               # fast checks of the committed frozen state (no training)

Re-running training steps regenerates models. Bit-identical weights are NOT guaranteed for neural / K-Means steps
(multithreaded float reductions differ between runs; see docs/clustering.md and docs/reproducibility.md), and
re-training would change the hashes recorded in the Phase 8 freeze manifest. The committed frozen artifacts are
the source of truth; --verify checks them. The test evaluation (final_eval) runs only once by design.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.data import PROJECT_ROOT  # noqa: E402

PY = sys.executable
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
NLI = "sentence-transformers/nli-distilroberta-base-v2"

# (name, phase, approx CPU time, commands)
STEPS = [
    ("extract", 1, "<1 min", ["__extract__"]),
    ("audit", 1, "2 min", [f"{PY} scripts/audit_dataset.py"]),
    ("splits", 1, "2 min", [f"{PY} scripts/make_splits.py"]),
    ("baselines", 2, "12 min", [f"{PY} scripts/run_baselines.py", f"{PY} scripts/analyze_baseline_errors.py"]),
    ("encode", 3, "30 min", [f"{PY} scripts/encode_questions.py --model {MINILM}"]),
    ("sbert_heads", 3, "3 min", [f"{PY} scripts/run_sbert.py"]),
    ("bilstm", 3, "1-2 h", [f"{PY} scripts/train_bilstm.py"]),
    ("encoder_control", 3, "1-2 h", [f"{PY} scripts/encode_questions.py --model {NLI} --control-subset",
                                      f"{PY} scripts/run_encoder_control.py"]),
    ("compare_deep", 3, "5 min", [f"{PY} scripts/compare_phase3.py"]),
    ("clustering", 4, "1.5 h", [f"{PY} scripts/compare_clustering.py",
                                f"{PY} scripts/stability_recheck.py --k 5 8 10 12 15 20",
                                f"{PY} scripts/build_clusters.py"]),
    ("cluster_errors", 5, "6 min", [f"{PY} scripts/analyze_clusters.py", f"{PY} scripts/threshold_reliability.py",
                                    f"{PY} scripts/summarize_manual_errors.py"]),
    ("constraints", 6, "12 min", [f"{PY} -m spacy download en_core_web_sm", f"{PY} scripts/build_constraint_features.py"]),
    ("crossfit", 6, "9 min", [f"{PY} scripts/crossfit_base.py"]),
    ("meta", 6, "15 min", [f"{PY} scripts/train_meta.py"]),
    ("calibration", 7, "10 min", [f"{PY} scripts/calibration_experiments.py"]),
    ("freeze", 8, "1 min", [f"{PY} scripts/freeze_manifest.py"]),
    ("dry_run", 8, "2 min", [f"{PY} scripts/final_evaluation.py --dry-run-on-val"]),
    ("final_eval", 8, "6 min, ONCE", [f"{PY} scripts/final_evaluation.py"]),
    ("registry", 10, "1 min", [f"{PY} scripts/build_registry.py"]),
]


def extract() -> None:
    archive = PROJECT_ROOT / "quora-question-pairs.zip"
    raw = PROJECT_ROOT / "data" / "raw"
    if (raw / "train.csv").exists():
        print("[extract] data/raw/train.csv already present")
        return
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extract("train.csv.zip", raw)
    with zipfile.ZipFile(raw / "train.csv.zip") as z:
        z.extractall(raw)
    (raw / "train.csv.zip").unlink()
    print("[extract] data/raw/train.csv extracted (the archive itself is untouched)")


def run_step(name: str) -> None:
    step = next(s for s in STEPS if s[0] == name)
    print(f"\n=== {name} (phase {step[1]}, ~{step[2]}) ===", flush=True)
    for cmd in step[3]:
        if cmd == "__extract__":
            extract()
            continue
        print(f"$ {cmd}", flush=True)
        if subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT).returncode != 0:
            raise SystemExit(f"step {name} failed: {cmd}")


def verify_test_results() -> dict:
    """Recompute the reported test metrics from the saved per-pair test scores and the frozen thresholds."""
    import pandas as pd
    from src.calibration.thresholds import ThresholdPolicy
    from src.evaluation.cluster_analysis import cluster_table
    from src.evaluation.metrics import classification_metrics
    art = PROJECT_ROOT / "artifacts"
    scores = pd.read_csv(art / "phase8" / "test_scores.csv")
    reported = json.loads((art / "phase8" / "test_results.json").read_text())["results"]
    out = {}
    for name, policy in (("baseline", "frozen_policy_baseline.json"), ("final", "frozen_policy_final.json")):
        t = ThresholdPolicy.load(art / "phase7" / policy).global_threshold
        m = classification_metrics(scores.is_duplicate, scores[name], t)
        tab = cluster_table(scores.is_duplicate, scores[name], scores.cluster, t)
        got = {"f1": m["f1"], "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "macro_cluster_f1": tab.f1.mean(),
               "worst_cluster_f1": tab.f1.min()}
        # scores were saved rounded to 6 decimals: allow tiny differences in threshold-free metrics
        out[name] = {k: (round(v, 5), round(reported[name][k], 5), abs(v - reported[name][k]) < 5e-4) for k, v in got.items()}
    return out


def verify() -> None:
    checks = {}
    print("[verify] frozen artifacts vs Phase 8 freeze manifest + registry", flush=True)
    checks["registry"] = subprocess.run([PY, "scripts/build_registry.py", "--check"], cwd=PROJECT_ROOT).returncode == 0
    from src.utils.data import content_sha256
    manifest = json.loads((PROJECT_ROOT / "artifacts" / "phase8" / "freeze_manifest.json").read_text())
    changed = [k for k, v in manifest["files"].items()
               if (PROJECT_ROOT / v["path"]).exists() and content_sha256(PROJECT_ROOT / v["path"]) != v["sha256"]]
    checks["freeze_manifest"] = not changed
    print(f"[verify] freeze manifest: {'OK' if not changed else 'CHANGED ' + str(changed)}")
    res = verify_test_results()
    checks["test_results_recomputed"] = all(ok for sysm in res.values() for (_, _, ok) in sysm.values())
    print(f"[verify] test metrics recomputed from saved scores: {res}")
    if (PROJECT_ROOT / "data" / "raw" / "train.csv").exists():
        checks["split_reproduces"] = subprocess.run([PY, "scripts/make_splits.py"], cwd=PROJECT_ROOT).returncode == 0
    else:
        print("[verify] raw data absent: skipping split regeneration (run --run extract splits)")
    checks["tests"] = subprocess.run([PY, "-m", "pytest", "-q"], cwd=PROJECT_ROOT).returncode == 0
    print("\n[verify] summary:", checks)
    sys.exit(0 if all(checks.values()) else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", nargs="+")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    names = [s[0] for s in STEPS]
    if args.list or not (args.run or args.start or args.end or args.verify):
        for name, phase, cost, cmds in STEPS:
            print(f"{name:16s} phase {phase:<2d} ~{cost:12s} " + " && ".join(c.replace(PY, 'python') for c in cmds))
        return
    if args.verify:
        verify()
        return
    selected = args.run or names[names.index(args.start or names[0]): names.index(args.end or names[-1]) + 1]
    for n in selected:
        if n not in names:
            raise SystemExit(f"unknown step {n}; see --list")
        run_step(n)


if __name__ == "__main__":
    main()
