import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.run_baselines import TFIDF_PARAMS, load_split, pair_matrix, rowwise_cosine
from src.preprocessing.text import normalize


@pytest.fixture(scope="module")
def tfidf():
    corpus = [normalize(q) for q in ["Who founded Microsoft?", "Who founded Apple?",
                                     "How do I learn Python?", "How can I learn Python fast?"]]
    return TfidfVectorizer(**{**TFIDF_PARAMS, "min_df": 1}).fit(corpus)


def test_pair_matrix_is_order_invariant(tfidf):
    a = tfidf.transform([normalize("Who founded Microsoft?")])
    b = tfidf.transform([normalize("How do I learn Python?")])
    assert (pair_matrix(a, b) != pair_matrix(b, a)).nnz == 0


def test_rowwise_cosine(tfidf):
    a = tfidf.transform([normalize("Who founded Microsoft?"), normalize("Who founded Microsoft?")])
    b = tfidf.transform([normalize("Who founded Microsoft?"), normalize("How do I learn Python?")])
    cos = rowwise_cosine(a, b)
    assert cos[0] == pytest.approx(1.0, abs=1e-6)
    assert cos[1] == pytest.approx(0.0, abs=1e-6)


def test_phase2_cannot_read_test_split():
    with pytest.raises(AssertionError):
        load_split("test")
