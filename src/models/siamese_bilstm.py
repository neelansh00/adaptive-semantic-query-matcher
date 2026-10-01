"""Siamese BiLSTM for question-pair matching (PyTorch).

    question -> token ids -> embedding -> BiLSTM -> masked max-pool -> sentence vector (2H)
    One encoder with shared weights encodes both questions, so identical questions get identical
    vectors, and the pair head sees only order-invariant comparisons:
        [|u - v|, u * v, cos(u, v)] -> MLP -> logit
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK = 0, 1


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    @classmethod
    def build(cls, token_lists, min_freq: int = 2) -> "Vocab":
        counts = Counter(t for toks in token_lists for t in toks)
        kept = sorted((t for t, c in counts.items() if c >= min_freq), key=lambda t: (-counts[t], t))
        return cls(["<pad>", "<unk>", *kept])

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        ids = [self.stoi.get(t, UNK) for t in tokens[:max_len]]
        return ids or [UNK]  # never an empty sequence

    def __len__(self) -> int:
        return len(self.itos)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class BiLSTMConfig:
    vocab_size: int
    embed_dim: int = 200
    hidden_dim: int = 128
    head_dim: int = 256
    dropout: float = 0.2
    max_len: int = 40


class SiameseBiLSTM(nn.Module):
    def __init__(self, cfg: BiLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=PAD)
        self.dropout = nn.Dropout(cfg.dropout)
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.hidden_dim, batch_first=True, bidirectional=True)
        rep = 2 * cfg.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(2 * rep + 1, cfg.head_dim), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.head_dim, 1))

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        mask = (ids != PAD).unsqueeze(-1)
        return out.masked_fill(~mask, float("-inf")).max(dim=1).values  # masked max-pool over time

    def forward(self, a_ids, a_len, b_ids, b_len) -> torch.Tensor:
        u, v = self.encode(a_ids, a_len), self.encode(b_ids, b_len)
        cos = nn.functional.cosine_similarity(u, v, dim=1).unsqueeze(1)
        return self.head(torch.cat([(u - v).abs(), u * v, cos], dim=1)).squeeze(1)


# --------------------------------------------------------------------------- batching

def pad_batch(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in seqs])
    out = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out, lengths


def bucketed_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator | None,
                     bucket_factor: int = 50) -> list[np.ndarray]:
    """Group similar-length pairs to minimise padding. With rng: shuffled chunks, then shuffled batches."""
    idx = rng.permutation(len(lengths)) if rng is not None else np.arange(len(lengths))
    chunk = batch_size * bucket_factor
    batches = []
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    if rng is not None:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches


@torch.no_grad()
def predict_proba(model: SiameseBiLSTM, a: list[list[int]], b: list[list[int]],
                  batch_size: int = 1024, device: torch.device | str = "cpu") -> np.ndarray:
    model.eval()
    lengths = np.array([max(len(x), len(y)) for x, y in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    for batch in bucketed_batches(lengths, batch_size, rng=None):
        ai, al = pad_batch([a[i] for i in batch])
        bi, bl = pad_batch([b[i] for i in batch])
        logits = model(ai.to(device), al, bi.to(device), bl)
        out[batch] = torch.sigmoid(logits).cpu().numpy()
    return out


def save_model(model: SiameseBiLSTM, vocab: Vocab, directory: Path, extra: dict | None = None) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt")
    vocab.save(directory / "vocab.json")
    (directory / "config.json").write_text(json.dumps({**asdict(model.cfg), **(extra or {})}, indent=2))


def load_model(directory: Path, device: str = "cpu") -> tuple[SiameseBiLSTM, Vocab, dict]:
    directory = Path(directory)
    meta = json.loads((directory / "config.json").read_text())
    cfg = BiLSTMConfig(**{k: meta[k] for k in BiLSTMConfig.__dataclass_fields__})
    model = SiameseBiLSTM(cfg)
    model.load_state_dict(torch.load(directory / "model.pt", map_location=device, weights_only=True))
    model.eval()
    return model, Vocab.load(directory / "vocab.json"), meta
