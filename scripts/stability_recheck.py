"""Phase 4a (follow-up): robust stability estimates for shortlisted K.

The sweep measured stability with ONE pair of fits per K, and the values jumped non-monotonically
(seed ARI 0.98 at K=12 but 0.64 at K=10), so a single estimate is too noisy to decide K on.
Here, per K: 4 full-data fits with different seeds (6 pairwise ARIs) and 3 independent split-half draws.

Usage:  python scripts/stability_recheck.py [--k 8 10 12 15]
Writes: artifacts/phase4/stability_recheck.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_clustering import load_cfg  # noqa: E402
from src.clustering.core import fit_kmeans, stability  # noqa: E402
from src.models.sentence_encoder import EmbeddingCache  # noqa: E402
from src.utils.data import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "phase4" / "stability_recheck.json"


def summary(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values)),
            "values": [round(v, 4) for v in values]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[8, 10, 12, 15])
    ap.add_argument("--n-init", type=int, default=3)
    args = ap.parse_args()
    cfg = load_cfg()
    X, _ = EmbeddingCache(cfg["encoder"], "train").load()
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for k in args.k:
        if str(k) in results:
            print(f"[skip] K={k} already done", flush=True)
            continue
        t = time.time()
        labels = [fit_kmeans(X, k, seed, args.n_init)[1] for seed in (100, 200, 300, 400)]
        seed_aris = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labels, 2)]
        split = [stability(X, k, seed, include_seed=False)["split_half_ari"] for seed in (11, 22, 33)]
        results[str(k)] = {"seed_ari": summary(seed_aris), "split_half_ari": summary(split),
                           "n_init": args.n_init, "seconds": round(time.time() - t, 1)}
        print(f"K={k}: seed ARI {summary(seed_aris)['mean']:.3f} [{min(seed_aris):.3f}, {max(seed_aris):.3f}]  "
              f"split-half ARI {np.mean(split):.3f} [{min(split):.3f}, {max(split):.3f}]", flush=True)
        OUT.write_text(json.dumps(results, indent=2))  # incremental


if __name__ == "__main__":
    main()
