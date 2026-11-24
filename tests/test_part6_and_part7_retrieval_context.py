"""Comprehensive tests for Part 6 (Metadata-Aware Hybrid Retrieval) and Part 7 (Context Selection)."""

from __future__ import annotations

import numpy as np
import pytest

from leredd.bm25_retrieval import BM25Retriever
from leredd.config import LEREDDConfig
from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.hybrid_retrieval import HybridRetriever
from leredd.retrieval import ExampleRetriever, RAGRetriever
from leredd.review_queue import QueueStatus, ReviewQueueConfig, ReviewQueueManager


class _KeywordEmbedder:
    """Deterministic offline embedder sensitive to specific domain terms."""

    VOCAB = ("brake", "radar", "camera", "battery", "thermal", "can", "speed", "fault")

    def encode(self, texts):
        rows = []
        for text in texts:
            lower = text.lower()
            vec = [float(lower.count(term)) for term in self.VOCAB]
            vec.append(float(len(text) % 17 + 1))
            rows.append(vec)
        return np.asarray(rows, dtype=np.float32)


class _TieEmbedder:
    """Returns identical unit vectors to force exact score ties."""

    def encode(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


# ---------------------------------------------------------------------------
# Part 6: Metadata-Aware Hybrid Retrieval
# ---------------------------------------------------------------------------

def test_part6_requirement_module_and_metadata_roundtrip_through_review_queue():
    pair = RequirementPair(
        id="P-MOD-1",
        req_a=Requirement(
            id="REQ-M1",
            text="The brake supervisor shall arbitrate torque requests.",
            system="Chassis",
            component="BrakeSupervisor",
            subsystem="Braking",
            module="TorqueArbitration",
            source="chassis_srs.md",
            metadata={"asil": "D", "asil_verified": True},
        ),
        req_b=Requirement(
            id="REQ-M2",
            text="Wheel slip controller shall cap regenerative braking torque.",
            system="Chassis",
            component="SlipControl",
            subsystem="Braking",
            module="RegenControl",
            source="chassis_srs.md",
            metadata={"asil": "C"},
        ),
        dependency_type=DependencyType.REQUIRES,
        provenance=AnnotationProvenance(
            source=AnnotationSource.LLM,
            status=VerificationStatus.PROVISIONAL,
            llm_label=DependencyType.REQUIRES,
            confidence_score=2,
        ),
    )
    mgr = ReviewQueueManager(config=ReviewQueueConfig(minimum_requirement_characters=0))
    item = mgr.create_review_item(pair)
    assert item is not None
    _, verified = mgr.record_decision(
        item,
        QueueStatus.ACCEPTED,
        final_label=DependencyType.REQUIRES,
        reviewer_id="architect-reviewer",
    )
    assert verified.req_a.model_dump() == pair.req_a.model_dump()
    assert verified.req_b.model_dump() == pair.req_b.model_dump()


def test_part6_configurable_weights_and_module_dict_metadata_signal():
    target = RequirementPair(
        id="TARGET-01",
        req_a=Requirement(
            id="T-1",
            text="brake radar controller processes input",
            module="BMS_Core",
            metadata={"safety_level": "ASIL-D"},
        ),
        req_b=Requirement(id="T-2", text="brake output signal dispatched", module="BMS_Core"),
    )
    meta_match = RequirementPair(
        id="CAND-META",
        req_a=Requirement(
            id="M-1",
            text="unrelated wording alpha",
            module="bms_core",
            metadata={"safety_level": "asil-d"},
        ),
        req_b=Requirement(id="M-2", text="unrelated wording beta"),
        dependency_type=DependencyType.REQUIRES,
    )
    lex_match = RequirementPair(
        id="CAND-LEX",
        req_a=Requirement(
            id="L-1",
            text="brake radar controller processes telemetry",
            module="Infotainment",
            metadata={"safety_level": "QM"},
        ),
        req_b=Requirement(id="L-2", text="brake output signal transmitted"),
        dependency_type=DependencyType.REQUIRES,
    )

    embedder = _KeywordEmbedder()
    dense = ExampleRetriever(embedder=embedder)
    bm25 = BM25Retriever()

    # With metadata_weight=0.0 (existing hybrid), lexical/dense match wins
    cfg_no_meta = LEREDDConfig(rrf_k=10, dense_weight=1.0, bm25_weight=1.0, metadata_weight=0.0)
    hybrid_no_meta = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, config=cfg_no_meta)
    res_no_meta = hybrid_no_meta.retrieve_example_pairs(target, [meta_match, lex_match], k=1)
    assert res_no_meta[DependencyType.REQUIRES] == [lex_match]

    # With high metadata_weight in config, structured module + metadata overlap promotes CAND-META
    cfg_meta = LEREDDConfig(rrf_k=10, dense_weight=0.5, bm25_weight=0.5, metadata_weight=5.0)
    hybrid_meta = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, config=cfg_meta)
    res_meta = hybrid_meta.retrieve_example_pairs(target, [meta_match, lex_match], k=1)
    assert res_meta[DependencyType.REQUIRES] == [meta_match]


def test_part6_excludes_reversed_equivalent_self_pairs_across_all_retrievers():
    req_x = Requirement(id="REQ-X", text="The radar unit shall measure range.", component="Radar", system="ADAS")
    req_y = Requirement(id="REQ-Y", text="The brake unit shall read radar range.", component="Brake", system="ADAS")

    target = RequirementPair(id="PAIR-TARGET", req_a=req_x, req_b=req_y)
    reversed_self = RequirementPair(
        id="PAIR-REVERSED-SELF",
        req_a=req_y,
        req_b=req_x,
        dependency_type=DependencyType.REQUIRES,
    )
    text_reversed_self = RequirementPair(
        id="PAIR-TEXT-REV",
        req_a=Requirement(id="OTHER-1", text=req_y.text, component="Brake", system="ADAS"),
        req_b=Requirement(id="OTHER-2", text=req_x.text, component="Radar", system="ADAS"),
        dependency_type=DependencyType.REQUIRES,
    )
    valid_other = RequirementPair(
        id="PAIR-VALID",
        req_a=Requirement(id="REQ-V1", text="Camera module shall detect lane markings.", component="Radar", system="ADAS"),
        req_b=Requirement(id="REQ-V2", text="Steering assist requires lane markings.", component="Brake", system="ADAS"),
        dependency_type=DependencyType.REQUIRES,
    )
    pool = [reversed_self, text_reversed_self, valid_other]

    dense = ExampleRetriever(embedder=_KeywordEmbedder())
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, metadata_weight=5.0)

    assert dense.retrieve(target, pool, k=5)[DependencyType.REQUIRES] == [valid_other]
    assert bm25.retrieve_example_pairs(target, candidate_pool=pool, k=5)[DependencyType.REQUIRES] == [valid_other]
    assert hybrid.retrieve_example_pairs(target, pool, k=5)[DependencyType.REQUIRES] == [valid_other]
    assert [p for _, p in hybrid.retrieve(target, pool, k=5)] == [valid_other]


def test_part6_hybrid_metadata_ranking_strictly_excludes_unverified_candidates():
    """Unverified/provisional/rejected pairs must never leak through _metadata_ranking."""
    target = RequirementPair(
        id="T-VER",
        req_a=Requirement(id="TA", text="Radar tracks target", component="Radar", subsystem="Sensing", system="ADB"),
        req_b=Requirement(id="TB", text="Brake applies pressure", component="Brake", subsystem="Actuation", system="ADB"),
    )
    provisional_match = RequirementPair(
        id="P-PROV",
        req_a=Requirement(id="PA", text="Radar tracks target", component="Radar", subsystem="Sensing", system="ADB"),
        req_b=Requirement(id="PB", text="Other statement", component="Brake", subsystem="Actuation", system="ADB"),
        dependency_type=DependencyType.REQUIRES,
        provenance=AnnotationProvenance(source=AnnotationSource.LLM, status=VerificationStatus.PROVISIONAL),
    )
    rejected_match = provisional_match.model_copy(
        update={
            "id": "P-REJ",
            "provenance": AnnotationProvenance(source=AnnotationSource.HUMAN, status=VerificationStatus.REJECTED),
        }
    )
    verified_match = RequirementPair(
        id="P-VER",
        req_a=Requirement(id="VA", text="Verified radar requirement", component="Radar", system="ADB"),
        req_b=Requirement(id="VB", text="Verified brake requirement", component="Brake", system="ADB"),
        dependency_type=DependencyType.REQUIRES,
        provenance=AnnotationProvenance(
            source=AnnotationSource.HUMAN,
            status=VerificationStatus.VERIFIED,
            reviewer_id="expert-1",
            accepted_label=DependencyType.REQUIRES,
        ),
    )

    hybrid = HybridRetriever(
        dense_retriever=ExampleRetriever(embedder=_KeywordEmbedder()),
        bm25_retriever=BM25Retriever(),
        metadata_weight=10.0,
    )
    res = hybrid.retrieve_example_pairs(
        target, [provisional_match, rejected_match, verified_match], k=5
    )
    assert res[DependencyType.REQUIRES] == [verified_match]


# ---------------------------------------------------------------------------
# Part 7: Measurable RAG Context Selection (Baseline + Optional Enhancements)
# ---------------------------------------------------------------------------

def test_part7_baseline_empty_and_whitespace_documents():
    rag = RAGRetriever(embedder=_KeywordEmbedder())
    pair = RequirementPair(
        id="P-RAG-EMPTY",
        req_a=Requirement(id="R1", text="Brake system shall respond."),
        req_b=Requirement(id="R2", text="Radar system shall scan."),
    )
    assert rag.chunk_document("") == []
    assert rag.chunk_document("   \n\t  ") == []
    chunks, diag = rag.retrieve_with_diagnostics(pair, "   \n  ", k=3)
    assert chunks == []
    assert diag.total_chunks == 0
    assert diag.selected_count == 0


def test_part7_baseline_fixed_chunk_overlap_and_boundary_behavior():
    rag = RAGRetriever(embedder=_KeywordEmbedder())
    exact_boundary = "A" * 50
    assert rag.chunk_document(exact_boundary, chunk_size=50, overlap=10) == [exact_boundary]

    just_over = ("A" * 40) + ("B" * 10) + "C"  # len = 51, step = 40
    chunks = rag.chunk_document(just_over, chunk_size=50, overlap=10)
    assert len(chunks) == 2
    assert chunks[0] == ("A" * 40) + ("B" * 10)
    # Second chunk starts at index 40 -> contains the 10 'B' overlap characters + 'C'
    assert chunks[1] == ("B" * 10) + "C"


def test_part7_baseline_query_pairing_and_deterministic_tie_ranking():
    pair = RequirementPair(
        id="P-TIE",
        req_a=Requirement(id="RA", text="Brake actuator pressure."),
        req_b=Requirement(id="RB", text="Radar obstacle distance."),
    )
    assert RAGRetriever.build_pair_query(pair) == "Brake actuator pressure. Radar obstacle distance."

    rag = RAGRetriever(embedder=_TieEmbedder())
    srs = ("ChunkZero " * 6) + ("ChunkOne_ " * 6) + ("ChunkTwo_ " * 6)
    run_1 = rag.retrieve(pair, srs, k=2, chunk_size=60, overlap=0)
    run_2 = rag.retrieve(pair, srs, k=2, chunk_size=60, overlap=0)
    assert run_1 == run_2
    assert len(run_1) == 2
    assert "ChunkZero" in run_1[0]


def test_part7_duplicate_and_near_duplicate_chunks_are_measurable_and_filterable():
    rag = RAGRetriever(embedder=_KeywordEmbedder())
    pair = RequirementPair(
        id="P-DUP",
        req_a=Requirement(id="R1", text="Brake controller fault handling."),
        req_b=Requirement(id="R2", text="Thermal sensor fault threshold."),
    )
    repeated_block = "The brake thermal controller shall log fault codes at 100ms intervals."
    distinct_block = "The camera optics heater shall activate below zero degrees Celsius."
    srs = f"{repeated_block}\n\n{repeated_block}\n\n{distinct_block}"

    # Baseline (deduplicate=False): duplicates are measured in diagnostics but retained if top-ranked
    base_chunks, base_diag = rag.retrieve_with_diagnostics(
        pair, srs, k=3, chunk_size=80, overlap=0, chunking_mode="requirement_boundary", deduplicate=False
    )
    assert base_diag.total_chunks == 3
    assert base_diag.exact_duplicate_count == 1
    assert base_diag.duplicate_ratio == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert len(base_chunks) == 3

    # Enhanced (deduplicate=True): duplicate chunks are removed before top-k selection
    dedup_chunks, dedup_diag = rag.retrieve_with_diagnostics(
        pair, srs, k=3, chunk_size=80, overlap=0, chunking_mode="requirement_boundary", deduplicate=True
    )
    assert dedup_diag.candidate_chunks_after_dedup == 2
    assert len(dedup_chunks) == 2


def test_part7_requirement_boundary_chunking_and_cited_id_boosting():
    rag = RAGRetriever(embedder=_TieEmbedder())
    pair = RequirementPair(
        id="P-CITE",
        req_a=Requirement(
            id="REQ-ADB-200",
            text="The warning lamp controller shall trigger alerts defined in REQ-ADB-099.",
        ),
        req_b=Requirement(
            id="REQ-ADB-201",
            text="Audible chime shall accompany dashboard alerts.",
        ),
    )
    srs = (
        "[REQ-ADB-001] The headlamp beam pattern shall adjust to oncoming vehicles.\n\n"
        "[REQ-ADB-050] The steering angle sensor shall calibrate on ignition start.\n\n"
        "[REQ-ADB-099] Fault severity level 3 shall latch diagnostic alert indicators."
    )

    # Requirement-boundary chunking keeps each [REQ-...] statement intact
    boundary_chunks = rag.chunk_document(
        srs, chunk_size=90, overlap=0, mode="requirement_boundary"
    )
    assert len(boundary_chunks) == 3
    assert all(ch.startswith("[REQ-ADB-") for ch in boundary_chunks)

    # Under tied embeddings, baseline (cite_id_boost=0.0) picks first chunk [REQ-ADB-001]
    unboosted, diag_unboosted = rag.retrieve_with_diagnostics(
        pair,
        srs,
        k=1,
        chunk_size=90,
        overlap=0,
        chunking_mode="requirement_boundary",
        cite_id_boost=0.0,
    )
    assert diag_unboosted.cited_ids == ["REQ-ADB-099"]
    assert diag_unboosted.cited_id_chunks_available == 1
    assert diag_unboosted.cited_id_chunks_selected == 0
    assert "[REQ-ADB-001]" in unboosted[0]

    # With cite_id_boost=0.25, chunk defining [REQ-ADB-099] is deterministically promoted to #1
    boosted, diag_boosted = rag.retrieve_with_diagnostics(
        pair,
        srs,
        k=1,
        chunk_size=90,
        overlap=0,
        chunking_mode="requirement_boundary",
        cite_id_boost=0.25,
    )
    assert diag_boosted.cited_id_chunks_selected == 1
    assert "[REQ-ADB-099]" in boosted[0]
