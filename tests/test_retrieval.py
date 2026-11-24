"""Tests for ExampleRetriever and RAGRetriever."""

import numpy as np
import pytest
from leredd.retrieval import (
    compute_vector_similarity,
    compute_pair_similarity,
    ExampleRetriever,
    RAGRetriever,
)
from leredd.data_models import (
    AnnotationProvenance, DependencyType, Requirement, RequirementPair,
    VerificationStatus,
)


class _OfflineEmbedder:
    """Deterministic test vectors; retrieval unit tests must not download models."""

    def encode(self, texts):
        return np.asarray(
            [[len(text), sum(map(ord, text)) % 997, text.lower().count("system")]
             for text in texts],
            dtype=np.float32,
        )


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
    retriever = ExampleRetriever(embedder=_OfflineEmbedder(), config=mock_config)
    examples = retriever.retrieve(sample_pair, sample_annotated_pool, k=2)
    
    assert isinstance(examples, dict)
    assert DependencyType.REQUIRES in examples or DependencyType.IS_SIMILAR in examples


def test_example_retriever_excludes_unlabeled_and_provisional_pairs(mock_config):
    class TinyEmbedder:
        @staticmethod
        def encode(texts):
            return np.array([[len(text), 1.0] for text in texts], dtype=np.float32)

    req_a = Requirement(id="A", text="brake control", system="S")
    req_b = Requirement(id="B", text="sensor input", system="S")
    target = RequirementPair(id="TARGET", req_a=req_a, req_b=req_b)
    unlabeled = RequirementPair(
        id="UNLABELED", req_a=req_a, req_b=req_b, dependency_type=None,
    )
    provisional = RequirementPair(
        id="PROVISIONAL", req_a=req_a, req_b=req_b,
        dependency_type=DependencyType.NO_DEPENDENCY,
        provenance=AnnotationProvenance(status=VerificationStatus.PROVISIONAL),
    )
    retriever = ExampleRetriever(embedder=TinyEmbedder(), config=mock_config)
    examples = retriever.retrieve(target, [unlabeled, provisional], k=1)
    assert all(not pairs for pairs in examples.values())


def test_rag_retriever_chunking(mock_config, sample_pair):
    rag = RAGRetriever(embedder=_OfflineEmbedder(), config=mock_config)
    srs_text = "Section 1. " + ("The system shall verify sensor inputs. " * 50)
    chunks = rag.chunk_document(srs_text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 1
    retrieved = rag.retrieve(sample_pair, srs_text, k=2)
    assert len(retrieved) <= 2


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_rag_chunking_rejects_invalid_chunk_sizes(mock_config, chunk_size):
    rag = RAGRetriever(embedder=_OfflineEmbedder(), config=mock_config)

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        rag.chunk_document("x" * 20, chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [-1, 10, 11])
def test_rag_chunking_rejects_invalid_overlap(mock_config, overlap):
    rag = RAGRetriever(embedder=_OfflineEmbedder(), config=mock_config)

    with pytest.raises(ValueError, match="overlap must be an integer"):
        rag.chunk_document("x" * 20, chunk_size=10, overlap=overlap)


def test_rag_zero_requested_chunks_returns_without_embedding(mock_config, sample_pair):
    class NoEncode:
        def encode(self, texts):
            raise AssertionError("zero chunks must not initialize retrieval embeddings")

    rag = RAGRetriever(embedder=NoEncode(), config=mock_config)
    assert rag.retrieve(sample_pair, "Relevant SRS text." * 20, k=0) == []


@pytest.mark.parametrize("k", [-1, True, 1.5])
def test_rag_retrieve_rejects_invalid_chunk_count(mock_config, sample_pair, k):
    rag = RAGRetriever(embedder=_OfflineEmbedder(), config=mock_config)

    with pytest.raises(ValueError, match="k must be a non-negative integer"):
        rag.retrieve(sample_pair, "Relevant SRS text." * 20, k=k)
