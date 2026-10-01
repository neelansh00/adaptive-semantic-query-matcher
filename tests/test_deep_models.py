import numpy as np
import pytest
import torch

from src.evaluation.metrics import threshold_transfer, validation_report
from src.models.sentence_encoder import clean, embedding_pair_features
from src.models.siamese_bilstm import (UNK, BiLSTMConfig, SiameseBiLSTM, Vocab, bucketed_batches, load_model,
                                       pad_batch, predict_proba, save_model)
from src.utils.data import identity_key
from src.utils.splits import train_dev_split
from src.utils.torch_utils import get_device, set_seed


# ---------------------------------------------------------------- vocabulary

def test_vocab_min_freq_and_unknowns():
    v = Vocab.build([["who", "founded", "apple"], ["who", "founded", "microsoft"]], min_freq=2)
    assert v.itos[:2] == ["<pad>", "<unk>"]
    assert set(v.itos[2:]) == {"who", "founded"}
    assert v.encode(["who", "founded", "apple"], max_len=10) == [v.stoi["who"], v.stoi["founded"], UNK]
    assert v.encode([], max_len=10) == [UNK]            # never an empty sequence
    assert len(v.encode(["who"] * 50, max_len=40)) == 40  # truncation


# ---------------------------------------------------------------- Siamese BiLSTM

@pytest.fixture
def tiny_model():
    set_seed(0)
    return SiameseBiLSTM(BiLSTMConfig(vocab_size=50, embed_dim=8, hidden_dim=6, head_dim=8)).eval()


def test_bilstm_is_symmetric_and_padding_invariant(tiny_model):
    a, b = [[5, 6, 7], [8, 9]], [[3, 4], [10, 11, 12, 13]]
    ai, al = pad_batch(a)
    bi, bl = pad_batch(b)
    with torch.no_grad():
        ab, ba = tiny_model(ai, al, bi, bl), tiny_model(bi, bl, ai, al)
        assert torch.allclose(ab, ba, atol=1e-6)
        # the same sequence padded to a different length must give the same encoding
        short, _ = pad_batch([[5, 6, 7]])
        long_ = torch.tensor([[5, 6, 7, 0, 0, 0]])
        assert torch.allclose(tiny_model.encode(short, torch.tensor([3])),
                              tiny_model.encode(long_, torch.tensor([3])), atol=1e-6)


def test_bilstm_save_load_is_deterministic(tiny_model, tmp_path):
    vocab = Vocab(["<pad>", "<unk>"] + [f"w{i}" for i in range(48)])
    a, b = [[2, 3, 4], [5, 6]], [[2, 3], [7, 8, 9]]
    p1 = predict_proba(tiny_model, a, b)
    save_model(tiny_model, vocab, tmp_path)
    loaded, v2, _ = load_model(tmp_path)
    np.testing.assert_allclose(p1, predict_proba(loaded, a, b), atol=1e-6)
    np.testing.assert_allclose(p1, predict_proba(loaded, a, b), atol=0)  # repeated inference identical
    assert v2.itos == vocab.itos


def test_bucketed_batches_cover_every_index_once():
    lengths = np.random.default_rng(0).integers(1, 30, size=1003)
    for rng in (None, np.random.default_rng(1)):
        batches = bucketed_batches(lengths, 64, rng)
        flat = np.concatenate(batches)
        assert sorted(flat.tolist()) == list(range(1003))


# ---------------------------------------------------------------- sentence-embedding features

def test_embedding_pair_features_symmetric_and_cosine():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(4, 8)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = rng.normal(size=(4, 8)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    f_uv, f_vu = embedding_pair_features(u, v), embedding_pair_features(v, u)
    np.testing.assert_allclose(f_uv, f_vu, atol=1e-6)
    np.testing.assert_allclose(f_uv[:, -1], np.sum(u * v, axis=1), atol=1e-6)
    assert f_uv.shape == (4, 17)
    np.testing.assert_allclose(embedding_pair_features(u, u)[:, -1], 1.0, atol=1e-6)


def test_clean_text_for_encoder():
    assert clean("  What   is\nAI? ") == "What is AI?"  # case kept for the transformer tokenizer
    assert clean(None) == "" and clean(float("nan")) == ""


# ---------------------------------------------------------------- training/eval plumbing

def test_train_dev_split_is_question_disjoint(toy_pairs):
    is_fit, is_dev = train_dev_split(toy_pairs, dev_frac=0.2, seed=3)
    assert (is_fit ^ is_dev).all() and is_dev.any()
    q = lambda d: set(d.question1.map(identity_key)) | set(d.question2.map(identity_key))  # noqa: E731
    assert not (q(toy_pairs[is_fit]) & q(toy_pairs[is_dev]))


def test_device_falls_back_to_cpu():
    assert get_device(prefer_cuda=False).type == "cpu"
    assert get_device().type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_validation_report_fields():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    s = np.array([0.1, 0.8, 0.3, 0.6, 0.4, 0.2, 0.7, 0.9])
    rep = validation_report(y, s)
    assert {"tuned_threshold", "val_at_tuned_threshold", "val_at_0.5", "threshold_transfer", "calibration"} <= set(rep)
    assert len(threshold_transfer(y, s)["cross_half_f1"]) == 2


@pytest.mark.parametrize("module", ["scripts.train_bilstm", "scripts.run_sbert", "scripts.encode_questions"])
def test_phase3_scripts_cannot_read_test_split(module):
    import importlib
    with pytest.raises(AssertionError):
        importlib.import_module(module).load_split("test")
