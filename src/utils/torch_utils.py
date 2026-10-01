"""Device selection, seeding and size helpers. Everything runs on CPU; CUDA is used only if present."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_threads(n: int | None = None) -> int:
    """Default: all cores but two. Override with the ASQM_THREADS environment variable."""
    n = n or int(os.environ.get("ASQM_THREADS", 0)) or max(1, (os.cpu_count() or 2) - 2)
    torch.set_num_threads(n)
    # Subnormal floats (tiny weights/activations) are extremely slow on x86 CPUs; observed as a
    # 4s -> 130s per-epoch slowdown of the SBERT MLP head. Flushing them to zero fixes it.
    torch.set_flush_denormal(True)
    return n


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def dir_size_mb(path: str | Path) -> float:
    path = Path(path)
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(f.stat().st_size for f in files) / 2**20
