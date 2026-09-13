"""Tests for SBERT EmbeddingModel."""

import numpy as np
from leredd.embeddings import EmbeddingModel


def test_embedding_encode_shape(tmp_path):
    embedder = EmbeddingModel(cache_dir=str(tmp_path))
    texts = [
        "The system shall brake automatically.",
        "The subsystem shall monitor camera inputs."
    ]
    embs = embedder.encode(texts)
    assert isinstance(embs, np.ndarray)
    assert embs.shape[0] == 2
    assert embs.shape[1] == 384


def test_embedding_caching(tmp_path):
    embedder = EmbeddingModel(cache_dir=str(tmp_path))
    text = ["Test requirement text for caching."]
    
    emb1 = embedder.encode(text)
    emb2 = embedder.encode(text)
    
    np.testing.assert_array_almost_equal(emb1, emb2)
