"""Tests for SBERT EmbeddingModel."""

import numpy as np
import sys
import types
from sklearn.feature_extraction.text import HashingVectorizer
from leredd.embeddings import EmbeddingModel


def _force_offline_fallback(monkeypatch):
    def initialize_fallback(embedder):
        embedder._fallback_hasher = HashingVectorizer(
            n_features=384,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
        )

    monkeypatch.setattr(EmbeddingModel, "_init_model", initialize_fallback)


def test_embedding_encode_shape(tmp_path, monkeypatch):
    _force_offline_fallback(monkeypatch)
    embedder = EmbeddingModel(cache_dir=str(tmp_path))
    texts = [
        "The system shall brake automatically.",
        "The subsystem shall monitor camera inputs."
    ]
    embs = embedder.encode(texts)
    assert isinstance(embs, np.ndarray)
    assert embs.shape[0] == 2
    assert embs.shape[1] == 384


def test_embedding_caching(tmp_path, monkeypatch):
    _force_offline_fallback(monkeypatch)
    embedder = EmbeddingModel(cache_dir=str(tmp_path))
    text = ["Test requirement text for caching."]
    
    emb1 = embedder.encode(text)
    emb2 = embedder.encode(text)
    
    np.testing.assert_array_almost_equal(emb1, emb2)


def test_hash_fallback_uses_consistent_features_across_calls(tmp_path):
    """Corpus and query embeddings must share a stable feature space."""
    embedder = EmbeddingModel.__new__(EmbeddingModel)
    embedder.model_name = "test-model"
    embedder.cache_dir = str(tmp_path)
    embedder._model = None
    embedder._fallback_hasher = HashingVectorizer(
        n_features=384,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
    )

    corpus = embedder.encode(["brake controller monitors wheel speed"])
    query = embedder.encode(["brake controller checks wheel speed"])

    assert corpus.shape == (1, 384)
    assert query.shape == (1, 384)
    assert float(np.dot(corpus[0], query[0])) > 0
    assert np.linalg.norm(corpus[0]) > 0


def test_embedding_model_initialization_error_uses_hash_fallback(tmp_path, monkeypatch):
    """A model initialization error must fall back without network or shared-cache use."""
    sentence_transformers = types.ModuleType("sentence_transformers")

    class FailingSentenceTransformer:
        def __init__(self, model_name):
            raise RuntimeError("controlled model initialization failure")

    sentence_transformers.SentenceTransformer = FailingSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    embedder = EmbeddingModel(cache_dir=str(tmp_path))
    corpus = embedder.encode(["brake controller monitors wheel speed"])
    query = embedder.encode(["brake controller checks wheel speed"])

    assert embedder._model is None
    assert embedder._fallback_hasher is not None
    assert corpus.shape == query.shape == (1, 384)
    assert float(np.dot(corpus[0], query[0])) > 0
