"""tests/test_evaluation_guardrails.py

Comprehensive tests for Part 2 & Part 3 deliverables:
1. Synthetic Review Fixtures integrity and loading.
2. Ground-Truth Guardrails (synthetic rejection, unlabeled rejection, unverified rejection).
3. Prediction Alignment Guardrails (duplicate prediction IDs, missing coverage).
4. System-Disjoint Split Guardrails (system, requirement, and pair ID overlap detection).
5. Review Queue isolation ensuring synthetic pairs cannot be exported as verified ground truth.
"""

from __future__ import annotations

import pytest

from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Prediction,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.evaluation import (
    Evaluator,
    GroundTruthViolationError,
    PredictionAlignmentError,
    SplitOverlapError,
    validate_ground_truth,
    validate_predictions,
    validate_system_disjoint_splits,
)
from leredd.review_queue import (
    HumanDecision,
    QueueStatus,
    ReviewItem,
    ReviewQueueManager,
)
from leredd.synthetic import (
    SYNTHETIC_SOURCE_IDENTITY,
    is_synthetic,
    list_synthetic_fixtures,
    load_synthetic_fixture,
    load_synthetic_pairs,
    load_synthetic_review_items,
)


# ---------------------------------------------------------------------------
# Test Fixture Helpers
# ---------------------------------------------------------------------------

def _req(req_id: str, system: str = "SYS") -> Requirement:
    return Requirement(id=req_id, text=f"Requirement text for {req_id}", system=system)


def _pair(
    pair_id: str,
    req_a_id: str = "REQ-01",
    req_b_id: str = "REQ-02",
    dep_type: DependencyType | None = DependencyType.REQUIRES,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    synthetic: bool = False,
    source_identity: str | None = None,
    system: str = "SYS",
) -> RequirementPair:
    prov = None
    if status is not None or synthetic or source_identity:
        prov = AnnotationProvenance(
            source=AnnotationSource.HUMAN if not synthetic else AnnotationSource.LLM,
            status=status or VerificationStatus.VERIFIED,
            synthetic=synthetic,
            source_identity=source_identity,
        )
    return RequirementPair(
        id=pair_id,
        req_a=_req(req_a_id, system=system),
        req_b=_req(req_b_id, system=system),
        dependency_type=dep_type,
        provenance=prov,
        synthetic=synthetic,
        source_identity=source_identity,
    )


# ---------------------------------------------------------------------------
# 1. Synthetic Fixture Integrity Tests
# ---------------------------------------------------------------------------

class TestSyntheticFixtureIntegrity:
    """Verify synthetic fixture files conform to deterministic specification."""

    def test_list_synthetic_fixtures(self):
        fixtures = list_synthetic_fixtures()
        assert len(fixtures) >= 7
        expected_names = [
            "ambiguous_pairs.json",
            "disagreement_pairs.json",
            "abstention_pairs.json",
            "low_confidence_pairs.json",
            "parsing_failure_pairs.json",
            "missing_metadata_pairs.json",
            "six_categories_pairs.json",
        ]
        for name in expected_names:
            assert name in fixtures

    def test_all_fixtures_have_required_metadata(self):
        fixtures = list_synthetic_fixtures()
        for fix_name in fixtures:
            data = load_synthetic_fixture(fix_name)
            assert data.get("synthetic") is True, f"{fix_name} missing synthetic: true"
            assert (
                data.get("source_identity") == SYNTHETIC_SOURCE_IDENTITY
            ), f"{fix_name} invalid source_identity"
            assert "pairs" in data and len(data["pairs"]) > 0

    def test_load_synthetic_pairs_marks_models(self):
        pairs = load_synthetic_pairs("ambiguous_pairs.json")
        assert len(pairs) > 0
        for p in pairs:
            assert p.synthetic is True
            assert p.source_identity == SYNTHETIC_SOURCE_IDENTITY
            assert is_synthetic(p) is True

    def test_load_synthetic_review_items_marks_items(self):
        items = load_synthetic_review_items("disagreement_pairs.json")
        assert len(items) > 0
        for item in items:
            assert item.synthetic is True
            assert item.source_identity == SYNTHETIC_SOURCE_IDENTITY
            assert is_synthetic(item) is True

    def test_six_categories_fixture_covers_all_types(self):
        pairs = load_synthetic_pairs("six_categories_pairs.json")
        types_covered = {p.dependency_type for p in pairs}
        for dt in DependencyType:
            assert dt in types_covered


# ---------------------------------------------------------------------------
# 2. Ground-Truth Guardrail Tests
# ---------------------------------------------------------------------------

class TestGroundTruthGuardrails:
    """Verify validate_ground_truth enforces strict fail-closed safety."""

    def test_empty_ground_truth_raises_strict(self):
        with pytest.raises(GroundTruthViolationError, match="Ground truth dataset is empty"):
            validate_ground_truth([], strict=True)

    def test_empty_ground_truth_returns_empty_non_strict(self):
        eligible, warnings = validate_ground_truth([], strict=False)
        assert len(eligible) == 0
        assert any("empty" in w for w in warnings)

    def test_synthetic_pair_rejected_with_invalid_ids(self):
        synth_pair = _pair("P-SYNTH-01", synthetic=True, source_identity=SYNTHETIC_SOURCE_IDENTITY)
        clean_pair = _pair("P-CLEAN-01", status=VerificationStatus.VERIFIED)

        with pytest.raises(GroundTruthViolationError) as exc_info:
            validate_ground_truth([clean_pair, synth_pair], strict=True)

        err = exc_info.value
        assert "P-SYNTH-01" in err.invalid_ids
        assert "Synthetic records rejected" in str(err)

    def test_unlabeled_pair_rejected_with_invalid_ids(self):
        unlabeled = _pair("P-UNLABELED-01", dep_type=None)
        clean_pair = _pair("P-CLEAN-01", status=VerificationStatus.VERIFIED)

        with pytest.raises(GroundTruthViolationError) as exc_info:
            validate_ground_truth([clean_pair, unlabeled], strict=True, reject_unlabeled=True)

        err = exc_info.value
        assert "P-UNLABELED-01" in err.invalid_ids
        assert "Unlabeled pairs rejected" in str(err)

    def test_provisional_pair_rejected(self):
        prov_pair = _pair("P-PROV-01", status=VerificationStatus.PROVISIONAL)
        with pytest.raises(GroundTruthViolationError) as exc_info:
            validate_ground_truth([prov_pair], strict=True)
        assert "P-PROV-01" in exc_info.value.invalid_ids

    def test_pending_review_pair_rejected(self):
        pending_pair = _pair("P-PENDING-01", status=VerificationStatus.PENDING_REVIEW)
        with pytest.raises(GroundTruthViolationError) as exc_info:
            validate_ground_truth([pending_pair], strict=True)
        assert "P-PENDING-01" in exc_info.value.invalid_ids

    def test_rejected_pair_rejected(self):
        rej_pair = _pair("P-REJ-01", status=VerificationStatus.REJECTED)
        with pytest.raises(GroundTruthViolationError) as exc_info:
            validate_ground_truth([rej_pair], strict=True)
        assert "P-REJ-01" in exc_info.value.invalid_ids

    def test_verified_pair_accepted(self):
        pair = _pair("P-VER-01", status=VerificationStatus.VERIFIED)
        eligible, warnings = validate_ground_truth([pair], strict=True)
        assert len(eligible) == 1
        assert eligible[0].id == "P-VER-01"

    def test_legacy_pair_without_provenance_accepted_by_default(self):
        legacy_pair = RequirementPair(
            id="P-LEGACY-01",
            req_a=_req("REQ-01"),
            req_b=_req("REQ-02"),
            dependency_type=DependencyType.REQUIRES,
            provenance=None,
        )
        eligible, warnings = validate_ground_truth([legacy_pair], strict=True, allow_legacy=True)
        assert len(eligible) == 1
        assert any("legacy annotated record" in w for w in warnings)

    def test_legacy_pair_rejected_when_allow_legacy_false(self):
        legacy_pair = RequirementPair(
            id="P-LEGACY-02",
            req_a=_req("REQ-01"),
            req_b=_req("REQ-02"),
            dependency_type=DependencyType.REQUIRES,
            provenance=None,
        )
        with pytest.raises(GroundTruthViolationError) as exc_info:
            validate_ground_truth([legacy_pair], strict=True, allow_legacy=False)
        assert "P-LEGACY-02" in exc_info.value.invalid_ids


# ---------------------------------------------------------------------------
# 3. Prediction Alignment Guardrail Tests
# ---------------------------------------------------------------------------

class TestPredictionAlignmentGuardrails:
    """Verify validate_predictions enforces duplicate checks and complete coverage."""

    def test_duplicate_prediction_ids_rejected(self):
        gt = [_pair("P1"), _pair("P2")]
        preds = [
            Prediction(pair_id="P1", dependency_type=DependencyType.REQUIRES, rationale="", confidence=5),
            Prediction(pair_id="P1", dependency_type=DependencyType.REQUIRES, rationale="", confidence=4),
            Prediction(pair_id="P2", dependency_type=DependencyType.REQUIRES, rationale="", confidence=5),
        ]
        with pytest.raises(PredictionAlignmentError) as exc_info:
            validate_predictions(gt, preds)

        assert "P1" in exc_info.value.invalid_ids
        assert "Duplicate prediction IDs" in str(exc_info.value)

    def test_missing_prediction_coverage_rejected(self):
        gt = [_pair("P1"), _pair("P2"), _pair("P3")]
        preds = [
            Prediction(pair_id="P1", dependency_type=DependencyType.REQUIRES, rationale="", confidence=5),
        ]
        with pytest.raises(PredictionAlignmentError) as exc_info:
            validate_predictions(gt, preds)

        assert "P2" in exc_info.value.invalid_ids
        assert "P3" in exc_info.value.invalid_ids
        assert "Missing prediction coverage" in str(exc_info.value)

    def test_aligned_predictions_pass(self):
        gt = [_pair("P1"), _pair("P2")]
        preds = [
            Prediction(pair_id="P1", dependency_type=DependencyType.REQUIRES, rationale="", confidence=5),
            Prediction(pair_id="P2", dependency_type=DependencyType.CONFLICTS, rationale="", confidence=4),
        ]
        validate_predictions(gt, preds)  # Does not raise


# ---------------------------------------------------------------------------
# 4. System-Disjoint Split Guardrail Tests
# ---------------------------------------------------------------------------

class TestSystemDisjointSplitGuardrails:
    """Verify validate_system_disjoint_splits checks systems, reqs, and pair IDs."""

    def test_clean_disjoint_splits_pass(self):
        train = [_pair("P-T1", "REQ-A-1", "REQ-A-2", system="ADB")]
        test = [_pair("P-E1", "REQ-B-1", "REQ-B-2", system="TJA")]
        dev = [_pair("P-D1", "REQ-C-1", "REQ-C-2", system="APA")]

        audit = validate_system_disjoint_splits(train, test, dev)
        assert audit["is_disjoint"] is True
        assert len(audit["overlapping_systems"]) == 0

    def test_overlapping_system_detected(self):
        train = [_pair("P-T1", "REQ-A-1", "REQ-A-2", system="ADB")]
        test = [_pair("P-E1", "REQ-B-1", "REQ-B-2", system="ADB")]

        with pytest.raises(SplitOverlapError) as exc_info:
            validate_system_disjoint_splits(train, test)

        assert "ADB" in exc_info.value.invalid_ids
        assert "systems" in str(exc_info.value)

    def test_overlapping_req_id_detected(self):
        train = [_pair("P-T1", "SHARED-REQ", "REQ-A-2", system="ADB")]
        test = [_pair("P-E1", "SHARED-REQ", "REQ-B-2", system="TJA")]

        with pytest.raises(SplitOverlapError) as exc_info:
            validate_system_disjoint_splits(train, test)

        assert "SHARED-REQ" in exc_info.value.invalid_ids
        assert "req_ids" in str(exc_info.value)

    def test_overlapping_pair_id_detected(self):
        train = [_pair("SAME-PAIR-ID", "REQ-A-1", "REQ-A-2", system="ADB")]
        test = [_pair("SAME-PAIR-ID", "REQ-B-1", "REQ-B-2", system="TJA")]

        with pytest.raises(SplitOverlapError) as exc_info:
            validate_system_disjoint_splits(train, test)

        assert "SAME-PAIR-ID" in exc_info.value.invalid_ids
        assert "pair_ids" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Review Queue Synthetic Isolation Tests
# ---------------------------------------------------------------------------

class TestReviewQueueSyntheticIsolation:
    """Verify synthetic pairs in review workflows cannot leak into verified export."""

    def test_export_verified_pairs_excludes_synthetic_items(self):
        real_item = ReviewItem(
            queue_id="QITEM-REAL-01",
            pair_id="P-REAL-01",
            req_a_id="REQ-01",
            req_b_id="REQ-02",
            req_a_text="Requirement 1",
            req_b_text="Requirement 2",
            orientation=("REQ-01", "REQ-02"),
            system_name="SYS",
            proposed_label=DependencyType.REQUIRES,
            status=QueueStatus.ACCEPTED,
            human_decision=HumanDecision(
                item_id="QITEM-REAL-01",
                pair_id="P-REAL-01",
                decision_type=QueueStatus.ACCEPTED,
                final_label=DependencyType.REQUIRES,
                reviewer_id="reviewer-fixture",
            ),
            synthetic=False,
        )
        synth_item = ReviewItem(
            queue_id="QITEM-SYNTH-01",
            pair_id="P-SYNTH-01",
            req_a_id="REQ-S1",
            req_b_id="REQ-S2",
            req_a_text="Synthetic Req 1",
            req_b_text="Synthetic Req 2",
            orientation=("REQ-S1", "REQ-S2"),
            system_name="SYS",
            proposed_label=DependencyType.CONFLICTS,
            status=QueueStatus.ACCEPTED,
            human_decision=HumanDecision(
                item_id="QITEM-SYNTH-01",
                pair_id="P-SYNTH-01",
                decision_type=QueueStatus.ACCEPTED,
                final_label=DependencyType.CONFLICTS,
            ),
            synthetic=True,
            source_identity=SYNTHETIC_SOURCE_IDENTITY,
        )

        manager = ReviewQueueManager(items=[real_item, synth_item])
        exported = manager.export_verified_pairs(allow_synthetic=False)

        assert len(exported) == 1
        assert exported[0].id == "P-REAL-01"
        assert exported[0].provenance.status == VerificationStatus.VERIFIED

    def test_evaluator_evaluate_integrates_guardrails(self):
        """Evaluator.evaluate rejects synthetic pairs by default."""
        synth_pair = _pair("P-SYNTH-01", synthetic=True, source_identity=SYNTHETIC_SOURCE_IDENTITY)
        preds = [
            Prediction(pair_id="P-SYNTH-01", dependency_type=DependencyType.REQUIRES, rationale="", confidence=5)
        ]
        with pytest.raises(GroundTruthViolationError):
            Evaluator.evaluate([synth_pair], preds)
