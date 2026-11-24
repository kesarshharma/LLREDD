"""Comprehensive tests for Parts 2, 3, 4, and 5 of the LEREDD specification."""

from __future__ import annotations

import json
import numpy as np
import pytest

from api import routes
from leredd.baselines.bert_classifier import BERTBaseline
from leredd.baselines.tfidf_lsa import TFIDF_LSABaseline
from leredd.bm25_retrieval import BM25Retriever
from leredd.config import LEREDDConfig
from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Prediction,
    PredictionValidationStatus,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.dataset_release import create_versioned_dataset_release
from leredd.detector import ResponseParser
from leredd.evaluation import (
    EvaluationGuardrailError,
    GroundTruthViolationError,
    SplitOverlapError,
    build_safe_retrieval_pool,
    tune_confidence_threshold_on_dev,
    validate_document_disjoint_splits,
    validate_ground_truth,
)
from leredd.hybrid_retrieval import HybridRetriever
from leredd.llm import MockLLMClient
from leredd.provisional_labeler import (
    LabelingMode,
    LLMLabeler,
    ProvisionalLabeler,
)
from leredd.retrieval import ExampleRetriever
from leredd.review_queue import (
    QueueStatus,
    ReviewQueueConfig,
    ReviewQueueManager,
    ReviewReasonCode,
)
from leredd.synthetic import (
    SYNTHETIC_SOURCE_IDENTITY,
    is_synthetic,
    load_synthetic_fixture,
    load_synthetic_pairs,
)


class _DeterministicEmbedder:
    def encode(self, texts):
        return np.asarray(
            [[len(t), sum(map(ord, t)) % 251, t.lower().count("shall") + 1.0] for t in texts],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Part 2: Synthetic Review Fixtures & Isolation Across All Real Flows
# ---------------------------------------------------------------------------

def test_part2_synthetic_fixture_rejects_simulated_human_decisions(tmp_path):
    """Fixtures containing simulated human decisions or verified status must fail validation."""
    bad_decision_file = tmp_path / "bad_human_decision.json"
    bad_decision_file.write_text(
        json.dumps(
            {
                "synthetic": True,
                "source_identity": SYNTHETIC_SOURCE_IDENTITY,
                "pairs": [
                    {
                        "id": "SYN-BAD-1",
                        "req_a": {"id": "A", "text": "Req A"},
                        "req_b": {"id": "B", "text": "Req B"},
                        "human_decision": {"decision_type": "accepted", "reviewer_id": "sim"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="simulated human_decision is forbidden"):
        load_synthetic_fixture(str(bad_decision_file))

    bad_verified_file = tmp_path / "bad_verified_prov.json"
    bad_verified_file.write_text(
        json.dumps(
            {
                "synthetic": True,
                "source_identity": SYNTHETIC_SOURCE_IDENTITY,
                "pairs": [
                    {
                        "id": "SYN-BAD-2",
                        "req_a": {"id": "A", "text": "Req A"},
                        "req_b": {"id": "B", "text": "Req B"},
                        "provenance": {"source": "human", "status": "verified"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="simulated human verification provenance is forbidden"):
        load_synthetic_fixture(str(bad_verified_file))


def test_part2_synthetic_fixtures_excluded_from_all_retrievers_and_baselines(tmp_path):
    """Synthetic fixtures must never enter Dense, BM25, Hybrid retrieval, or baseline fitting."""
    synth_pairs = load_synthetic_pairs("six_categories_pairs.json")
    assert len(synth_pairs) == 6
    assert all(is_synthetic(p) for p in synth_pairs)

    # Even if provenance is stripped, pair.synthetic=True must block retrieval & training
    stripped_synth = [
        p.model_copy(
            update={
                "provenance": None,
                "req_a": p.req_a.model_copy(update={"component": "Radar", "system": "ADB"}),
            }
        )
        for p in synth_pairs
    ]
    target = RequirementPair(
        id="REAL-TARGET-01",
        req_a=Requirement(id="REAL-A", text="Front radar shall transmit object vectors.", component="Radar", system="ADB"),
        req_b=Requirement(id="REAL-B", text="Brake controller requires radar telemetry.", system="ADB"),
    )

    embedder = _DeterministicEmbedder()
    dense = ExampleRetriever(embedder=embedder)
    bm25 = BM25Retriever()
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, metadata_weight=10.0)

    assert dense.retrieve(target, stripped_synth, k=3) == {}
    assert bm25.retrieve_example_pairs(target, candidate_pool=stripped_synth, k=3) == {}
    assert hybrid.retrieve_example_pairs(target, stripped_synth, k=3) == {}
    assert hybrid.retrieve(target, stripped_synth, k=3) == []

    safe_pool, audit = build_safe_retrieval_pool(stripped_synth, [target], strict=True)
    assert safe_pool == []
    assert audit["excluded_synthetic_count"] == 6

    with pytest.raises(ValueError, match="synthetic"):
        BERTBaseline(embedder=embedder).fit(stripped_synth)

    with pytest.raises(ValueError, match="synthetic"):
        TFIDF_LSABaseline().fit(stripped_synth)

    src_json = tmp_path / "synth.json"
    src_json.write_text(
        json.dumps([p.model_dump(mode="json") for p in synth_pairs]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="synthetic"):
        create_versioned_dataset_release([src_json], "v1.0.0", tmp_path / "rel")


# ---------------------------------------------------------------------------
# Part 3: Ground-Truth & Document-Disjoint Guardrails
# ---------------------------------------------------------------------------

def test_part3_human_reviewed_but_not_verified_is_rejected_with_invalid_ids():
    """VerificationStatus.HUMAN_REVIEWED is distinct from VERIFIED and must be rejected."""
    pair = RequirementPair(
        id="PAIR-HR-01",
        req_a=Requirement(id="R1", text="System shall log events.", system="S1"),
        req_b=Requirement(id="R2", text="System shall archive logs.", system="S1"),
        dependency_type=DependencyType.REQUIRES,
        provenance=AnnotationProvenance(
            source=AnnotationSource.HUMAN,
            status=VerificationStatus.HUMAN_REVIEWED,
            reviewer_id="alice",
            accepted_label=DependencyType.REQUIRES,
        ),
    )
    with pytest.raises(GroundTruthViolationError) as exc_info:
        validate_ground_truth([pair], strict=True)
    assert exc_info.value.invalid_ids == ["PAIR-HR-01"]


def test_part3_document_disjoint_split_guardrail_reports_invalid_ids():
    train = [
        RequirementPair(
            id="P-DOC-1",
            req_a=Requirement(id="RA-1", text="Req 1", source="brake_srs.md"),
            req_b=Requirement(id="RA-2", text="Req 2", source="brake_srs.md"),
            dependency_type=DependencyType.REQUIRES,
        )
    ]
    clean_test = [
        RequirementPair(
            id="P-DOC-2",
            req_a=Requirement(id="RB-1", text="Req 3", source="vision_srs.md"),
            req_b=Requirement(id="RB-2", text="Req 4", source="vision_srs.md"),
            dependency_type=DependencyType.CONFLICTS,
        )
    ]
    audit = validate_document_disjoint_splits(train, clean_test)
    assert audit["is_disjoint"] is True

    overlapping_test = [
        RequirementPair(
            id="P-DOC-3",
            req_a=Requirement(id="RC-1", text="Req 5", source="brake_srs.md"),
            req_b=Requirement(id="RC-2", text="Req 6", source="brake_srs.md"),
            dependency_type=DependencyType.DETAILS,
        )
    ]
    with pytest.raises(SplitOverlapError) as exc_info:
        validate_document_disjoint_splits(train, overlapping_test)
    assert "brake_srs.md" in exc_info.value.invalid_ids


# ---------------------------------------------------------------------------
# Part 4: Confidence-Aware Validation & Malformed-Response Guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("score", "threshold", "expected_status"),
    [
        (0, 3, PredictionValidationStatus.REQUIRES_REVIEW),
        (2, 3, PredictionValidationStatus.REQUIRES_REVIEW),
        (3, 3, PredictionValidationStatus.REQUIRES_REVIEW),  # Boundary <= threshold
        (4, 3, PredictionValidationStatus.PASSED_THRESHOLD),
        (5, 3, PredictionValidationStatus.PASSED_THRESHOLD),
        (0, 0, PredictionValidationStatus.REQUIRES_REVIEW),
        (1, 0, PredictionValidationStatus.PASSED_THRESHOLD),
        (5, 5, PredictionValidationStatus.REQUIRES_REVIEW),
        (2, -1, PredictionValidationStatus.THRESHOLD_DISABLED),
    ],
)
def test_part4_confidence_boundary_values_preserve_predicted_type(
    score: int, threshold: int, expected_status: PredictionValidationStatus
):
    raw = (
        "**Dependency_Type: Conflicts**\n"
        "**Rationale: Boundary verification.**\n"
        f"**Confidence Score: {score}**"
    )
    pred = ResponseParser.parse(raw, pair_id="P-BOUND", confidence_threshold=threshold)
    assert pred.dependency_type == DependencyType.CONFLICTS
    assert pred.confidence == score
    assert pred.validation_status == expected_status
    assert pred.validation_threshold == threshold


def test_part4_malformed_conversational_response_does_not_become_valid_label():
    """Free-text refusals containing words like 'requires' or 'conflicts' must raise ValueError."""
    conversational_refusal = (
        "I cannot classify this requirement pair because it requires additional "
        "architectural details and conflicts with my formatting instructions."
    )
    with pytest.raises(ValueError, match="Malformed LLM response"):
        ResponseParser.parse(conversational_refusal, pair_id="P-REFUSAL")


def test_part4_missing_or_invalid_confidence_routes_to_review_not_max_score():
    """Missing Confidence Score must never default to 5/5 PASSED_THRESHOLD."""
    missing_conf = "**Dependency_Type: Implements**\n**Rationale: Explicit tag but no score.**"
    pred = ResponseParser.parse(missing_conf, pair_id="P-NOCONF", confidence_threshold=4)
    assert pred.dependency_type == DependencyType.IMPLEMENTS
    assert pred.confidence == 0
    assert pred.validation_status == PredictionValidationStatus.REQUIRES_REVIEW

    invalid_conf = (
        "**Dependency_Type: Details**\n"
        "**Rationale: Non-numeric confidence.**\n"
        "**Confidence Score: high**"
    )
    pred_bad = ResponseParser.parse(invalid_conf, pair_id="P-BADCONF", confidence_threshold=2)
    assert pred_bad.dependency_type == DependencyType.DETAILS
    assert pred_bad.confidence == 0
    assert pred_bad.validation_status == PredictionValidationStatus.REQUIRES_REVIEW


def test_part4_likert_confidence_cannot_be_represented_as_calibrated_probability():
    with pytest.raises(ValueError, match="calibrated_probability"):
        Prediction(
            pair_id="P-PROB",
            dependency_type=DependencyType.REQUIRES,
            rationale="Test",
            confidence=4,
            calibrated_probability=0.8,  # type: ignore[call-arg]
        )


def test_part4_deprecated_legacy_threshold_compatibility_emits_warning():
    raw = "**Dependency_Type: Requires**\n**Rationale: Low score.**\n**Confidence Score: 2**"
    with pytest.warns(DeprecationWarning, match="deprecated"):
        pred = ResponseParser.parse(
            raw,
            pair_id="P-LEG",
            confidence_threshold=3,
            legacy_rewrite_low_confidence=True,
        )
    assert pred.dependency_type == DependencyType.NO_DEPENDENCY
    assert pred.validation_status == PredictionValidationStatus.REQUIRES_REVIEW


def test_part4_threshold_tuning_runs_on_dev_and_forbids_test():
    dev_pairs = [
        RequirementPair(
            id=f"DEV-{i}",
            req_a=Requirement(id=f"DA-{i}", text="Req A", system="DEV"),
            req_b=Requirement(id=f"DB-{i}", text="Req B", system="DEV"),
            dependency_type=DependencyType.REQUIRES,
        )
        for i in range(4)
    ]
    dev_preds = [
        Prediction(pair_id="DEV-0", dependency_type=DependencyType.CONFLICTS, rationale="wrong low conf", confidence=1),
        Prediction(pair_id="DEV-1", dependency_type=DependencyType.CONFLICTS, rationale="wrong low conf", confidence=2),
        Prediction(pair_id="DEV-2", dependency_type=DependencyType.REQUIRES, rationale="right high conf", confidence=5),
        Prediction(pair_id="DEV-3", dependency_type=DependencyType.REQUIRES, rationale="right high conf", confidence=5),
    ]
    tuned = tune_confidence_threshold_on_dev(dev_pairs, dev_preds, partition_label="DEV")
    assert tuned["status"] == "tuned_on_dev_only"
    assert tuned["test_set_touched"] is False
    # Threshold >= 2 filters out the 2 low-confidence errors, achieving 1.0 selective accuracy at 0.5 coverage
    assert tuned["selected_threshold"] >= 2
    assert tuned["selected_metrics"]["selective_accuracy"] == 1.0

    with pytest.raises(EvaluationGuardrailError, match="strictly forbidden"):
        tune_confidence_threshold_on_dev(dev_pairs, dev_preds, partition_label="TEST")


# ---------------------------------------------------------------------------
# Part 5: Structured Disagreement & Review Queue Routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("fixture_name", "expected_reason"),
    [
        ("ambiguous_pairs.json", ReviewReasonCode.EXPLICIT_FLAG),
        ("disagreement_pairs.json", ReviewReasonCode.DISAGREEMENT),
        ("abstention_pairs.json", ReviewReasonCode.BOTH_ABSTAINED),
        ("low_confidence_pairs.json", ReviewReasonCode.LOW_CONFIDENCE),
        ("parsing_failure_pairs.json", ReviewReasonCode.LLM_FAILURE),
        ("missing_metadata_pairs.json", ReviewReasonCode.MISSING_METADATA),
    ],
)
def test_part5_every_synthetic_scenario_produces_exact_queue_reason_via_structured_fields(
    fixture_name: str, expected_reason: ReviewReasonCode
):
    pairs = load_synthetic_pairs(fixture_name)
    # Clear free-text review_notes to prove routing uses structured fields rather than phrase parsing
    stripped_notes_pairs = [
        p.model_copy(
            update={
                "provenance": p.provenance.model_copy(update={"review_notes": ""})
                if p.provenance
                else None
            }
        )
        for p in pairs
    ]
    manager = ReviewQueueManager(
        config=ReviewQueueConfig(minimum_requirement_characters=0)
    )
    queue = manager.build_queue(stripped_notes_pairs)
    assert len(queue) == len(pairs)
    for item in queue:
        assert expected_reason in item.review_reasons, (
            f"Fixture {fixture_name} item {item.pair_id} missing {expected_reason}; got {item.review_reasons}"
        )


def test_part5_provisional_labeler_propagates_structured_fields_and_synthetic_markers():
    """Verify ProvisionalLabeler records structured fields without losing synthetic identity."""
    req_a = Requirement(
        id="REQ-S-1",
        text="The brake module requires REQ-S-2 telemetry before actuation.",
        system="BrakeSys",
        synthetic=True,
        source_identity=SYNTHETIC_SOURCE_IDENTITY,
    )
    req_b = Requirement(
        id="REQ-S-2",
        text="Wheel speed telemetry shall be sampled every 10ms.",
        system="BrakeSys",
        synthetic=True,
        source_identity=SYNTHETIC_SOURCE_IDENTITY,
    )
    pair = RequirementPair(
        id="PAIR-STRUCT-1",
        req_a=req_a,
        req_b=req_b,
        synthetic=True,
        source_identity=SYNTHETIC_SOURCE_IDENTITY,
    )

    # Disagreement: Heuristic -> Requires (cites REQ-S-2), LLM -> Details (confidence 2)
    llm_json = json.dumps(
        {"dependency_type": "Details", "confidence": 2, "explanation": "Sampling rate refines telemetry."}
    )
    coord = ProvisionalLabeler(llm_labeler=LLMLabeler(llm_client=MockLLMClient(response_text=llm_json)))
    labeled = coord.process_pair(pair, mode=LabelingMode.HYBRID)

    assert labeled.synthetic is True
    assert labeled.source_identity == SYNTHETIC_SOURCE_IDENTITY
    assert labeled.dependency_type is None
    assert labeled.provenance is not None
    assert labeled.provenance.heuristic_label == DependencyType.REQUIRES
    assert labeled.provenance.heuristic_rule_id == "RULE-REF-ID-DIRECT"
    assert labeled.provenance.llm_label == DependencyType.DETAILS
    assert labeled.provenance.confidence_score == 2
    assert "disagreement" in labeled.provenance.structured_review_reasons

    # Route into queue even with empty review_notes
    labeled_clean_notes = labeled.model_copy(
        update={"provenance": labeled.provenance.model_copy(update={"review_notes": ""})}
    )
    manager = ReviewQueueManager(config=ReviewQueueConfig(minimum_requirement_characters=0))
    item = manager.create_review_item(labeled_clean_notes)
    assert item is not None
    assert ReviewReasonCode.DISAGREEMENT in item.review_reasons
    assert ReviewReasonCode.LOW_CONFIDENCE in item.review_reasons
    assert item.heuristic_prediction == DependencyType.REQUIRES
    assert item.llm_prediction == DependencyType.DETAILS
    assert item.llm_confidence == 2
    assert "Sampling rate refines telemetry." in item.llm_rationale


def test_part5_llm_failure_in_hybrid_mode_routes_as_llm_failure_not_both_abstained():
    r1 = Requirement(id="R-A", text="The cabin display shall adjust brightness smoothly.", system="HMI")
    r2 = Requirement(id="R-B", text="Ambient light sensor shall publish lux readings.", system="HMI")
    pair = RequirementPair(id="P-FAIL-1", req_a=r1, req_b=r2)

    bad_llm = LLMLabeler(llm_client=MockLLMClient(response_text="Unparseable garbage output"))
    coord = ProvisionalLabeler(llm_labeler=bad_llm)
    labeled, manifest = coord.process_batch([pair], mode=LabelingMode.HYBRID)

    assert manifest.failed_count == 1
    assert labeled[0].provenance is not None
    assert labeled[0].provenance.llm_failed is True

    item = ReviewQueueManager(config=ReviewQueueConfig(minimum_requirement_characters=0)).create_review_item(labeled[0])
    assert item is not None
    assert ReviewReasonCode.LLM_FAILURE in item.review_reasons
    assert ReviewReasonCode.BOTH_ABSTAINED not in item.review_reasons
    assert ReviewReasonCode.LLM_ONLY not in item.review_reasons


def test_part5_deferred_rejected_and_reviewed_items_cannot_become_verified():
    pair = RequirementPair(
        id="P-DECIDE-1",
        req_a=Requirement(id="RA", text="The controller shall validate CAN checksums.", system="CAN"),
        req_b=Requirement(id="RB", text="Invalid frames shall be discarded within 2ms.", system="CAN"),
        dependency_type=DependencyType.DETAILS,
        provenance=AnnotationProvenance(
            source=AnnotationSource.LLM,
            status=VerificationStatus.PROVISIONAL,
            llm_label=DependencyType.DETAILS,
            confidence_score=2,
        ),
    )
    manager = ReviewQueueManager(config=ReviewQueueConfig(minimum_requirement_characters=0))

    for non_verifying_status, expected_prov_status in (
        (QueueStatus.DEFERRED, VerificationStatus.PENDING_REVIEW),
        (QueueStatus.REJECTED, VerificationStatus.REJECTED),
        (QueueStatus.REVIEWED, VerificationStatus.HUMAN_REVIEWED),
    ):
        item = manager.create_review_item(pair)
        assert item is not None
        updated_item, out_pair = manager.record_decision(
            item,
            decision_type=non_verifying_status,
            final_label=DependencyType.DETAILS,
            reviewer_id="domain-expert-01",
            reviewer_notes="Not verified yet.",
        )
        assert out_pair.provenance is not None
        assert out_pair.provenance.status == expected_prov_status
        assert out_pair.provenance.status != VerificationStatus.VERIFIED
        assert out_pair.dependency_type is None
        export_mgr = ReviewQueueManager(items=[updated_item])
        assert export_mgr.export_verified_pairs() == []


def test_api_config_redacts_both_openai_and_gemini_secrets(monkeypatch):
    cfg = LEREDDConfig(
        openai_api_key="sk-secret-openai-value",
        gemini_api_key="AIza-secret-gemini-value",
    )
    monkeypatch.setattr(routes, "get_config", lambda: cfg)
    public = routes.get_system_config()
    serialized = json.dumps(public)
    assert "sk-secret-openai-value" not in serialized
    assert "AIza-secret-gemini-value" not in serialized
    assert public["openai_api_key_configured"] is True
    assert public["gemini_api_key_configured"] is True
