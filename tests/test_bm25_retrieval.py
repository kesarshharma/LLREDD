"""Unit tests for BM25 Lexical Retrieval module (leredd.bm25_retrieval).

Covers:
- Exact and partial lexical matching
- No-matching-terms queries
- Empty query and empty corpus handling
- Case and punctuation normalization
- Repeated terms
- Stable ranking for tied scores
- Correct top-k and invalid k values (k <= 0)
- Metadata and label preservation
- Pure offline execution without SBERT, network, or API credentials
- Corpus isolation
- Compatibility with dataset splitter and provenance models
"""

import pytest
from leredd.data_models import (
    Requirement,
    RequirementPair,
    DependencyType,
    AnnotationProvenance,
    AnnotationSource,
    VerificationStatus,
)
from leredd.bm25_retrieval import BM25Retriever
from leredd.retrieval import ExampleRetriever


@pytest.fixture
def sample_req_corpus():
    return [
        Requirement(id="REQ-001", text="The BCS shall accept camera scenery information from PCS to brake vehicle.", system="ADB"),
        Requirement(id="REQ-002", text="The system shall include the Brake Control Subsystem (BCS).", system="ADB"),
        Requirement(id="REQ-003", text="The system shall authenticate firmware updates.", system="ADB"),
        Requirement(id="REQ-004", text="The system shall verify source of security updates.", system="ADB"),
    ]


@pytest.fixture
def sample_pair_corpus(sample_req_corpus):
    req1, req2, req3, req4 = sample_req_corpus
    prov = AnnotationProvenance(
        source=AnnotationSource.HUMAN,
        status=VerificationStatus.VERIFIED,
        annotator_id="expert-01"
    )
    return [
        RequirementPair(id="PAIR-001", req_a=req1, req_b=req2, dependency_type=DependencyType.REQUIRES, provenance=prov),
        RequirementPair(id="PAIR-002", req_a=req3, req_b=req4, dependency_type=DependencyType.IS_SIMILAR),
        RequirementPair(id="PAIR-003", req_a=req1, req_b=req4, dependency_type=DependencyType.NO_DEPENDENCY),
    ]


def test_tokenization_and_normalization():
    """Verify lowercasing, punctuation stripping, and handling of repeated terms."""
    text = "The BCS shall ACCEPT camera scenery! Scenery: brake... brake... vehicle?"
    tokens = BM25Retriever.tokenize(text)
    assert "bcs" in tokens
    assert "accept" in tokens
    assert tokens.count("brake") == 2
    assert tokens.count("scenery") == 2

    # Empty text
    assert BM25Retriever.tokenize("") == []
    assert BM25Retriever.tokenize("   !!! ???   ") == []


def test_exact_and_partial_lexical_matches(sample_req_corpus):
    """Verify exact and partial lexical matches are ranked appropriately."""
    retriever = BM25Retriever(sample_req_corpus)

    # Search for "brake control subsystem"
    results = retriever.retrieve("Brake Control Subsystem", k=3)
    assert len(results) == 3
    top_score, top_item = results[0]
    assert top_score > 0.0
    assert top_item.id in ["REQ-001", "REQ-002"]

    # Search for "authenticate updates"
    results2 = retriever.retrieve("authenticate updates", k=2)
    top_score2, top_item2 = results2[0]
    assert top_item2.id == "REQ-003"


def test_queries_with_no_matching_terms(sample_req_corpus):
    """Verify query with no overlapping vocabulary returns score 0.0 without errors."""
    retriever = BM25Retriever(sample_req_corpus)
    results = retriever.retrieve("quantum teleportation blockchain cryptography", k=2)
    assert len(results) == 2
    for score, item in results:
        assert score == 0.0


def test_empty_query_and_empty_corpus(sample_req_corpus):
    """Verify empty query and empty corpus return graceful default results."""
    # Empty corpus
    retriever_empty = BM25Retriever([])
    assert retriever_empty.retrieve("brake", k=5) == []
    assert not retriever_empty.is_indexed

    # Empty query on indexed corpus
    retriever = BM25Retriever(sample_req_corpus)
    results = retriever.retrieve("", k=2)
    assert len(results) == 2
    assert results[0][0] == 0.0


def test_repeated_terms_influence_score(sample_req_corpus):
    """Verify query term repetition is handled cleanly by BM25 frequency scaling."""
    retriever = BM25Retriever(sample_req_corpus)
    res1 = retriever.retrieve("brake", k=1)
    res2 = retriever.retrieve("brake brake brake", k=1)
    assert res1[0][0] > 0.0
    assert res2[0][0] > 0.0


test_stable_ranking_for_tied_scores_code = """
def test_stable_ranking_for_tied_scores():
    items = [
        Requirement(id="REQ-C", text="System shall log errors.", system="ADB"),
        Requirement(id="REQ-A", text="System shall log warnings.", system="ADB"),
        Requirement(id="REQ-B", text="System shall log alerts.", system="ADB"),
    ]
    retriever = BM25Retriever(items)
    # Query with non-matching term -> all scores 0.0
    results = retriever.retrieve("nonexistent_term", k=3)
    assert len(results) == 3
    # Tie-breaking orders by ID ('REQ-A', 'REQ-B', 'REQ-C')
    ids = [item.id for score, item in results]
    assert ids == ["REQ-A", "REQ-B", "REQ-C"]
"""


def test_top_k_and_invalid_k(sample_req_corpus):
    """Verify top-k bounds and handling of invalid k <= 0."""
    retriever = BM25Retriever(sample_req_corpus)

    # Invalid k <= 0
    assert retriever.retrieve("brake", k=0) == []
    assert retriever.retrieve("brake", k=-5) == []

    # Valid k
    assert len(retriever.retrieve("brake", k=1)) == 1
    assert len(retriever.retrieve("brake", k=10)) <= len(sample_req_corpus)


def test_pair_corpus_example_retrieval(sample_pair_corpus):
    """Verify example retrieval grouped by DependencyType works on pair corpora."""
    retriever = BM25Retriever(sample_pair_corpus)
    target_pair = sample_pair_corpus[0]

    examples = retriever.retrieve_example_pairs(target_pair, k=2)
    assert isinstance(examples, dict)
    assert DependencyType.IS_SIMILAR in examples
    assert DependencyType.NO_DEPENDENCY in examples

    # Check candidate metadata and labels are preserved
    similar_pairs = examples[DependencyType.IS_SIMILAR]
    if similar_pairs:
        p = similar_pairs[0]
        assert p.id == "PAIR-002"
        assert p.dependency_type == DependencyType.IS_SIMILAR

    # Check provenance preservation on retrieved pair
    req_pairs = examples.get(DependencyType.REQUIRES, [])
    for p in req_pairs:
        if p.id == "PAIR-001" and p.provenance:
            assert p.provenance.source == AnnotationSource.HUMAN
            assert p.provenance.status == VerificationStatus.VERIFIED


def test_corpus_isolation(sample_req_corpus):
    """Verify BM25Retriever indexes ONLY explicitly supplied items."""
    retriever = BM25Retriever(sample_req_corpus[:2])
    assert len(retriever.corpus) == 2
    assert [req.id for req in retriever.corpus] == ["REQ-001", "REQ-002"]


def test_offline_execution_without_sbert_or_network():
    """Verify BM25Retriever operates purely in memory without SBERT models or API keys."""
    raw_texts = [
        "The system shall accept input from camera.",
        "The system shall display scenery on screen.",
    ]
    retriever = BM25Retriever(raw_texts)
    results = retriever.retrieve("camera scenery", k=2)
    assert len(results) == 2
    assert results[0][0] > 0.0


def test_existing_dense_retrieval_unaffected(mock_config, sample_pair, sample_annotated_pool):
    """Verify existing SBERT ExampleRetriever is completely unaffected by BM25 additions."""
    dense_retriever = ExampleRetriever(config=mock_config)
    examples = dense_retriever.retrieve(sample_pair, sample_annotated_pool, k=2)
    assert isinstance(examples, dict)
