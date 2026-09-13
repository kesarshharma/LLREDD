"""Tests for ExampleRetriever and RAGRetriever."""

import numpy as np
from leredd.retrieval import (
    compute_vector_similarity,
    compute_pair_similarity,
    ExampleRetriever,
    RAGRetriever,
)
from leredd.data_models import DependencyType


def test_similarity_math_formulas():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Euclidean distance = 0 -> sim = 1 / (1 + 0) = 1.0
    sim_euc = compute_vector_similarity(v1, v2, metric="euclidean")
    assert abs(sim_euc - 1.0) < 1e-5

    # Cosine similarity = 1.0
    sim_cos = compute_vector_similarity(v1, v2, metric="cosine")
    assert abs(sim_cos - 1.0) < 1e-5

    # Eq. 1 Average vs Eq. 2 Max
    score_avg = compute_pair_similarity(v1, v2, v1, v2, metric="euclidean", aggregation="average")
    score_max = compute_pair_similarity(v1, v2, v1, v2, metric="euclidean", aggregation="max")
    assert score_avg == 1.0
    assert score_max == 1.0


def test_example_retriever(mock_config, sample_pair, sample_annotated_pool):
    retriever = ExampleRetriever(config=mock_config)
    examples = retriever.retrieve(sample_pair, sample_annotated_pool, k=2)
    
    assert isinstance(examples, dict)
    assert DependencyType.REQUIRES in examples or DependencyType.IS_SIMILAR in examples


def test_rag_retriever_chunking(mock_config, sample_pair):
    rag = RAGRetriever(config=mock_config)
    srs_text = "Section 1. " + ("The system shall verify sensor inputs. " * 50)
    chunks = rag.chunk_document(srs_text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 1
    retrieved = rag.retrieve(sample_pair, srs_text, k=2)
    assert len(retrieved) <= 2
