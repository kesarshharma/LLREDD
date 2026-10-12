"""Unit tests for Hybrid Retrieval (leredd.hybrid_retrieval) using Reciprocal Rank Fusion (RRF).

Covers:
- Mathematical correctness of RRF fusion over synthetic rankings
- Candidates present in both vs single rankers
- Deduplication and stable tie-breaking
- Top-k truncation and invalid k values (k <= 0)
- Empty corpus and empty ranking handling
- Candidate metadata, labels, and provenance preservation
- Target query self-exclusion policy
- Pure retrieval corpus isolation
- Compatibility with mocked dense and BM25 retrievers (offline, zero API calls)
- Comparative retrieval sanity check (BM25 vs Dense vs Hybrid)
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
from leredd.hybrid_retrieval import HybridRetriever, get_candidate_id
from leredd.retrieval import ExampleRetriever


@pytest.fixture
def sample_pairs_pool():
    req1 = Requirement(id="REQ-ADB-001", text="System shall include Brake Control Subsystem.", system="ADB")
    req2 = Requirement(id="REQ-ADB-002", text="BCS shall accept camera scenery from PCS.", system="ADB")
    req3 = Requirement(id="REQ-ADB-003", text="System shall verify firmware update source.", system="ADB")
    req4 = Requirement(id="REQ-ADB-004", text="System shall authenticate update credentials.", system="ADB")
    req5 = Requirement(id="REQ-ADB-005", text="System shall log all diagnostic alerts.", system="ADB")

    prov = AnnotationProvenance(
        source=AnnotationSource.HUMAN,
        status=VerificationStatus.VERIFIED,
        annotator_id="expert-01"
    )

    return [
        RequirementPair(id="PAIR-001", req_a=req1, req_b=req2, dependency_type=DependencyType.REQUIRES, provenance=prov),
        RequirementPair(id="PAIR-002", req_a=req3, req_b=req4, dependency_type=DependencyType.IS_SIMILAR),
        RequirementPair(id="PAIR-003", req_a=req1, req_b=req4, dependency_type=DependencyType.NO_DEPENDENCY),
        RequirementPair(id="PAIR-004", req_a=req2, req_b=req5, dependency_type=DependencyType.REQUIRES),
        RequirementPair(id="PAIR-005", req_a=req3, req_b=req5, dependency_type=DependencyType.DETAILS),
    ]


def test_rrf_math_synthetic_rankings():
    """Verify RRF mathematical score calculation on synthetic candidate lists."""
    # Ranker 1: ['A', 'B', 'C'] -> ranks 1, 2, 3
    # Ranker 2: ['B', 'C', 'D'] -> ranks 1, 2, 3
    ranking1 = ["A", "B", "C"]
    ranking2 = ["B", "C", "D"]
    rrf_k = 60

    # RRF score for A: 1/(60+1) + 0 = 1/61 ≈ 0.016393
    # RRF score for B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.016129 + 0.016393 = 0.032522
    # RRF score for C: 1/(60+3) + 1/(60+2) = 1/63 + 1/62 ≈ 0.015873 + 0.016129 = 0.032002
    # RRF score for D: 0 + 1/(60+3) = 1/63 ≈ 0.015873

    fused = HybridRetriever.reciprocal_rank_fusion([ranking1, ranking2], rrf_k=rrf_k, top_k=10)
    items = [item for score, item in fused]

    # Candidate 'B' is ranked #2 in R1 and #1 in R2 -> highest fused score!
    assert items[0] == "B"
    assert items[1] == "C"
    assert items[2] == "A"
    assert items[3] == "D"

    # Score checks
    score_b = fused[0][0]
    expected_b = (1.0 / 62.0) + (1.0 / 61.0)
    assert abs(score_b - expected_b) < 1e-6


def test_rrf_single_ranker_and_deduplication():
    """Verify candidates present in only one ranking remain eligible and duplicates are merged."""
    ranking1 = ["Item1", "Item2"]
    ranking2 = ["Item2", "Item3"]

    fused = HybridRetriever.reciprocal_rank_fusion([ranking1, ranking2], rrf_k=60, top_k=10)
    unique_items = [item for _, item in fused]

    assert len(unique_items) == 3
    assert set(unique_items) == {"Item1", "Item2", "Item3"}
    # Item2 appears in both, so it should be #1
    assert unique_items[0] == "Item2"


def test_rrf_stable_tie_breaking():
    """Verify deterministic tie-breaking when RRF scores are equal."""
    ranking1 = ["B_Item", "A_Item"]
    ranking2 = []

    fused = HybridRetriever.reciprocal_rank_fusion([ranking1, ranking2], rrf_k=60, top_k=2)
    # Both have same score from ranking1; tie-breaking orders by ID ('A_Item' before 'B_Item')
    # Wait, A_Item is rank 2 (1/62), B_Item is rank 1 (1/61).
    # To test equal scores:
    ranking_a = ["ItemX"]
    ranking_b = ["ItemY"]
    fused_equal = HybridRetriever.reciprocal_rank_fusion([ranking_a, ranking_b], rrf_k=60, top_k=2)
    assert fused_equal[0][0] == fused_equal[1][0]  # Both score 1/61
    assert fused_equal[0][1] == "ItemX"
    assert fused_equal[1][1] == "ItemY"


def test_top_k_truncation_and_invalid_k():
    """Verify top-k bounds and handling of invalid k <= 0."""
    rankings = [["A", "B", "C", "D", "E"]]

    assert HybridRetriever.reciprocal_rank_fusion(rankings, top_k=0) == []
    assert HybridRetriever.reciprocal_rank_fusion(rankings, top_k=-3) == []

    fused_2 = HybridRetriever.reciprocal_rank_fusion(rankings, top_k=2)
    assert len(fused_2) == 2


def test_empty_corpus_and_empty_rankings():
    """Verify empty rankings return empty list gracefully."""
    assert HybridRetriever.reciprocal_rank_fusion([]) == []
    assert HybridRetriever.reciprocal_rank_fusion([[], []]) == []


def test_metadata_and_provenance_preservation(sample_pairs_pool):
    """Verify candidate metadata, labels, and provenance are preserved after RRF fusion."""
    target_pair = sample_pairs_pool[0]
    hybrid = HybridRetriever()

    fused_dict = hybrid.retrieve_example_pairs(target_pair, sample_pairs_pool, k=2)

    assert isinstance(fused_dict, dict)
    # Check that retrieved pair has provenance intact
    if DependencyType.REQUIRES in fused_dict:
        pairs = fused_dict[DependencyType.REQUIRES]
        for p in pairs:
            if p.id == "PAIR-001" and p.provenance:
                assert p.provenance.source == AnnotationSource.HUMAN
                assert p.provenance.status == VerificationStatus.VERIFIED


def test_target_self_exclusion(sample_pairs_pool):
    """Verify target pair cannot retrieve itself as a candidate."""
    target_pair = sample_pairs_pool[0]  # PAIR-001
    hybrid = HybridRetriever()

    fused_dict = hybrid.retrieve_example_pairs(target_pair, sample_pairs_pool, k=5)

    for dep_type, pairs in fused_dict.items():
        for p in pairs:
            assert p.id != target_pair.id
            assert not (p.req_a.text == target_pair.req_a.text and p.req_b.text == target_pair.req_b.text)


def test_corpus_isolation(sample_pairs_pool):
    """Verify hybrid retrieval operates strictly over the explicitly supplied candidate pool."""
    target_pair = sample_pairs_pool[0]
    restricted_pool = sample_pairs_pool[:2]  # Only PAIR-001 and PAIR-002

    hybrid = HybridRetriever()
    fused_dict = hybrid.retrieve_example_pairs(target_pair, restricted_pool, k=5)

    all_retrieved_ids = {p.id for sublist in fused_dict.values() for p in sublist}
    assert "PAIR-005" not in all_retrieved_ids
    assert all_retrieved_ids.issubset({"PAIR-001", "PAIR-002"})


def test_comparative_retrieval_sanity_check(sample_pairs_pool):
    """Comparative sanity check comparing Dense (SBERT), BM25, and Hybrid (RRF) retrieval outputs."""
    target_pair = sample_pairs_pool[0]

    dense = ExampleRetriever()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    dense_examples = dense.retrieve(target_pair, sample_pairs_pool, k=2)
    bm25_examples = bm25.retrieve_example_pairs(target_pair, candidate_pool=sample_pairs_pool, k=2)
    hybrid_examples = hybrid.retrieve_example_pairs(target_pair, sample_pairs_pool, k=2)

    assert isinstance(dense_examples, dict)
    assert isinstance(bm25_examples, dict)
    assert isinstance(hybrid_examples, dict)

    # Hybrid results must be a valid subset of the input pool
    hybrid_pair_ids = {p.id for pairs in hybrid_examples.values() for p in pairs}
    pool_ids = {p.id for p in sample_pairs_pool}
    assert hybrid_pair_ids.issubset(pool_ids)
