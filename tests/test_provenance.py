"""Tests for provenance schema extensions in leredd.data_models.

Covers:
- Legacy dataset loading (backward compatibility)
- New provenance metadata on RequirementPair
- New provenance fields on Prediction
- Missing optional fields
- Invalid verification states
- Serialization/deserialization round trips
- Preservation of existing dependency categories
- Compatibility with existing prediction and API models
"""

import json
import pytest
from pydantic import ValidationError

from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Prediction,
    Requirement,
    RequirementPair,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legacy_pair_dict():
    """Return a dict matching the existing JSON dataset format (no provenance)."""
    return {
        "id": "PAIR-ADB-001",
        "req_a": {
            "id": "REQ-ADB-002",
            "text": "The BCS shall accept camera scenery information.",
            "system": "ADB",
        },
        "req_b": {
            "id": "REQ-ADB-001",
            "text": "The system shall include the Brake Control Subsystem (BCS).",
            "system": "ADB",
        },
        "dependency_type": "Requires",
    }


def _legacy_prediction_dict():
    """Return a dict matching the existing Prediction JSON format (no provenance)."""
    return {
        "pair_id": "PAIR-ADB-001",
        "dependency_type": "Requires",
        "rationale": "BCS depends on system inclusion.",
        "confidence": 5,
    }


def _full_provenance_dict():
    """Return a fully-populated provenance metadata dict."""
    return {
        "source": "llm",
        "status": "human_reviewed",
        "annotator_id": "system-leredd-v1",
        "model_id": "gpt-4",
        "model_version": "gpt-4-0613",
        "config_hash": "abc123def456",
        "created_at": "2026-10-09T12:00:00Z",
        "reviewed_at": "2026-10-09T14:00:00Z",
        "reviewer_id": "human-reviewer-1",
        "predicted_label": "Requires",
        "accepted_label": "Implements",
        "confidence_score": 4,
        "calibrated_probability": 0.85,
        "evidence": "Subsystem BCS is referenced in requirement A.",
        "review_notes": "Changed from Requires to Implements after domain review.",
        "dataset_version": "v2.0-provisional",
    }


# ===========================================================================
# 1. Legacy Dataset Loading (Backward Compatibility)
# ===========================================================================

class TestLegacyBackwardCompatibility:
    """Verify that existing JSON datasets without provenance fields load correctly."""

    def test_legacy_pair_loads_without_provenance(self):
        """Legacy pair dict (no 'provenance' key) must load with provenance=None."""
        pair = RequirementPair.model_validate(_legacy_pair_dict())
        assert pair.id == "PAIR-ADB-001"
        assert pair.dependency_type == DependencyType.REQUIRES
        assert pair.provenance is None

    def test_legacy_prediction_loads_without_provenance(self):
        """Legacy prediction dict (no provenance keys) must load with defaults."""
        pred = Prediction.model_validate(_legacy_prediction_dict())
        assert pred.pair_id == "PAIR-ADB-001"
        assert pred.dependency_type == DependencyType.REQUIRES
        assert pred.annotation_source is None
        assert pred.model_id is None
        assert pred.config_hash is None
        assert pred.created_at is None

    def test_legacy_pair_roundtrip_json(self):
        """Load legacy dict -> dump -> reload must preserve all fields."""
        pair = RequirementPair.model_validate(_legacy_pair_dict())
        dumped = pair.model_dump(mode="json")
        reloaded = RequirementPair.model_validate(dumped)
        assert reloaded.id == pair.id
        assert reloaded.dependency_type == pair.dependency_type
        assert reloaded.req_a.id == pair.req_a.id
        assert reloaded.req_b.text == pair.req_b.text
        assert reloaded.provenance is None

    def test_legacy_prediction_roundtrip_json(self):
        """Load legacy prediction dict -> dump -> reload must preserve all fields."""
        pred = Prediction.model_validate(_legacy_prediction_dict())
        dumped = pred.model_dump(mode="json")
        reloaded = Prediction.model_validate(dumped)
        assert reloaded.pair_id == pred.pair_id
        assert reloaded.dependency_type == pred.dependency_type
        assert reloaded.confidence == pred.confidence
        assert reloaded.annotation_source is None

    def test_load_actual_dataset_file(self):
        """Load the real adb_pairs.json dataset and verify all pairs parse."""
        import os
        dataset_path = os.path.join("data", "annotated", "adb_pairs.json")
        if not os.path.exists(dataset_path):
            pytest.skip("Dataset file not found")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = [RequirementPair.model_validate(item) for item in data]
        assert len(pairs) > 0
        for pair in pairs:
            assert pair.provenance is None
            assert pair.dependency_type is not None


# ===========================================================================
# 2. New Provenance Metadata on RequirementPair
# ===========================================================================

class TestRequirementPairProvenance:
    """Test AnnotationProvenance attached to RequirementPair."""

    def test_pair_with_full_provenance(self):
        """RequirementPair with fully populated provenance must validate."""
        data = _legacy_pair_dict()
        data["provenance"] = _full_provenance_dict()
        pair = RequirementPair.model_validate(data)

        assert pair.provenance is not None
        assert pair.provenance.source == AnnotationSource.LLM
        assert pair.provenance.status == VerificationStatus.HUMAN_REVIEWED
        assert pair.provenance.model_id == "gpt-4"
        assert pair.provenance.predicted_label == DependencyType.REQUIRES
        assert pair.provenance.accepted_label == DependencyType.IMPLEMENTS
        assert pair.provenance.confidence_score == 4
        assert pair.provenance.calibrated_probability == 0.85
        assert pair.provenance.reviewer_id == "human-reviewer-1"
        assert pair.provenance.dataset_version == "v2.0-provisional"

    def test_pair_with_minimal_provenance(self):
        """RequirementPair with empty provenance dict uses defaults."""
        data = _legacy_pair_dict()
        data["provenance"] = {}
        pair = RequirementPair.model_validate(data)

        assert pair.provenance is not None
        assert pair.provenance.source == AnnotationSource.IMPORTED
        assert pair.provenance.status == VerificationStatus.PROVISIONAL
        assert pair.provenance.annotator_id is None
        assert pair.provenance.model_id is None
        assert pair.provenance.predicted_label is None

    def test_pair_provenance_roundtrip(self):
        """Full provenance survives JSON serialization/deserialization."""
        data = _legacy_pair_dict()
        data["provenance"] = _full_provenance_dict()
        pair = RequirementPair.model_validate(data)
        dumped = pair.model_dump(mode="json")
        reloaded = RequirementPair.model_validate(dumped)

        assert reloaded.provenance is not None
        assert reloaded.provenance.source == AnnotationSource.LLM
        assert reloaded.provenance.model_id == "gpt-4"
        assert reloaded.provenance.predicted_label == DependencyType.REQUIRES
        assert reloaded.provenance.accepted_label == DependencyType.IMPLEMENTS
        assert reloaded.provenance.calibrated_probability == 0.85

    def test_pair_provenance_label_normalization(self):
        """Provenance label fields should normalize strings like DependencyType."""
        prov = AnnotationProvenance(
            predicted_label="requires",
            accepted_label="is_similar",
        )
        assert prov.predicted_label == DependencyType.REQUIRES
        assert prov.accepted_label == DependencyType.IS_SIMILAR


# ===========================================================================
# 3. New Provenance Fields on Prediction
# ===========================================================================

class TestPredictionProvenance:
    """Test provenance fields on Prediction model."""

    def test_prediction_with_provenance_fields(self):
        """Prediction with provenance fields must validate."""
        pred = Prediction(
            pair_id="PAIR-001",
            dependency_type=DependencyType.REQUIRES,
            rationale="Test rationale",
            confidence=5,
            annotation_source=AnnotationSource.LLM,
            model_id="gpt-4",
            config_hash="abc123",
            created_at="2026-10-09T12:00:00Z",
        )
        assert pred.annotation_source == AnnotationSource.LLM
        assert pred.model_id == "gpt-4"
        assert pred.config_hash == "abc123"
        assert pred.created_at == "2026-10-09T12:00:00Z"

    def test_prediction_provenance_defaults_to_none(self):
        """Prediction without provenance fields must default to None."""
        pred = Prediction(
            pair_id="PAIR-001",
            dependency_type=DependencyType.NO_DEPENDENCY,
            rationale="No dependency found",
            confidence=3,
        )
        assert pred.annotation_source is None
        assert pred.model_id is None
        assert pred.config_hash is None
        assert pred.created_at is None

    def test_prediction_provenance_roundtrip(self):
        """Prediction provenance fields survive JSON round trip."""
        pred = Prediction(
            pair_id="PAIR-001",
            dependency_type=DependencyType.CONFLICTS,
            rationale="Conflicting requirements",
            confidence=4,
            annotation_source=AnnotationSource.HEURISTIC,
            model_id="rule-engine-v2",
            config_hash="def789",
            created_at="2026-10-09T15:30:00Z",
        )
        dumped = pred.model_dump(mode="json")
        reloaded = Prediction.model_validate(dumped)
        assert reloaded.annotation_source == AnnotationSource.HEURISTIC
        assert reloaded.model_id == "rule-engine-v2"
        assert reloaded.config_hash == "def789"
        assert reloaded.created_at == "2026-10-09T15:30:00Z"


# ===========================================================================
# 4. AnnotationProvenance Standalone Tests
# ===========================================================================

class TestAnnotationProvenance:
    """Test AnnotationProvenance model independently."""

    def test_defaults(self):
        """Empty AnnotationProvenance has safe defaults."""
        prov = AnnotationProvenance()
        assert prov.source == AnnotationSource.IMPORTED
        assert prov.status == VerificationStatus.PROVISIONAL
        assert prov.annotator_id is None
        assert prov.model_id is None
        assert prov.predicted_label is None
        assert prov.accepted_label is None
        assert prov.confidence_score is None
        assert prov.calibrated_probability is None

    def test_confidence_score_bounds(self):
        """Confidence score must be 0-5."""
        prov = AnnotationProvenance(confidence_score=0)
        assert prov.confidence_score == 0
        prov = AnnotationProvenance(confidence_score=5)
        assert prov.confidence_score == 5
        with pytest.raises(ValidationError):
            AnnotationProvenance(confidence_score=6)
        with pytest.raises(ValidationError):
            AnnotationProvenance(confidence_score=-1)

    def test_calibrated_probability_bounds(self):
        """Calibrated probability must be 0.0-1.0."""
        prov = AnnotationProvenance(calibrated_probability=0.0)
        assert prov.calibrated_probability == 0.0
        prov = AnnotationProvenance(calibrated_probability=1.0)
        assert prov.calibrated_probability == 1.0
        with pytest.raises(ValidationError):
            AnnotationProvenance(calibrated_probability=1.5)
        with pytest.raises(ValidationError):
            AnnotationProvenance(calibrated_probability=-0.1)

    def test_full_provenance_construction(self):
        """Fully populated provenance must validate all fields."""
        prov = AnnotationProvenance.model_validate(_full_provenance_dict())
        assert prov.source == AnnotationSource.LLM
        assert prov.status == VerificationStatus.HUMAN_REVIEWED
        assert prov.annotator_id == "system-leredd-v1"
        assert prov.model_id == "gpt-4"
        assert prov.model_version == "gpt-4-0613"
        assert prov.config_hash == "abc123def456"
        assert prov.created_at == "2026-10-09T12:00:00Z"
        assert prov.reviewed_at == "2026-10-09T14:00:00Z"
        assert prov.reviewer_id == "human-reviewer-1"
        assert prov.predicted_label == DependencyType.REQUIRES
        assert prov.accepted_label == DependencyType.IMPLEMENTS
        assert prov.confidence_score == 4
        assert prov.calibrated_probability == 0.85
        assert prov.evidence == "Subsystem BCS is referenced in requirement A."
        assert prov.review_notes == "Changed from Requires to Implements after domain review."
        assert prov.dataset_version == "v2.0-provisional"

    def test_provenance_json_roundtrip(self):
        """Full provenance JSON round trip preserves all fields."""
        prov = AnnotationProvenance.model_validate(_full_provenance_dict())
        dumped = json.loads(prov.model_dump_json())
        reloaded = AnnotationProvenance.model_validate(dumped)
        assert reloaded == prov


# ===========================================================================
# 5. Enum Tests
# ===========================================================================

class TestAnnotationSourceEnum:
    """Verify AnnotationSource enum values and serialization."""

    def test_all_values_present(self):
        expected = {"human", "llm", "heuristic", "imported", "ensemble", "candidate_generator"}
        actual = {item.value for item in AnnotationSource}
        assert actual == expected

    def test_string_serialization(self):
        assert str(AnnotationSource.LLM) == "AnnotationSource.LLM"
        assert AnnotationSource.LLM.value == "llm"

    def test_from_string(self):
        assert AnnotationSource("human") == AnnotationSource.HUMAN
        assert AnnotationSource("llm") == AnnotationSource.LLM

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError):
            AnnotationSource("unknown_source")


class TestVerificationStatusEnum:
    """Verify VerificationStatus enum values and serialization."""

    def test_all_values_present(self):
        expected = {"provisional", "pending_review", "human_reviewed", "verified", "rejected"}
        actual = {item.value for item in VerificationStatus}
        assert actual == expected

    def test_from_string(self):
        assert VerificationStatus("provisional") == VerificationStatus.PROVISIONAL
        assert VerificationStatus("verified") == VerificationStatus.VERIFIED

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            VerificationStatus("invalid_status")


# ===========================================================================
# 6. DependencyType Preservation
# ===========================================================================

class TestDependencyTypePreservation:
    """Confirm existing DependencyType values are unchanged."""

    def test_all_original_values_preserved(self):
        expected = {
            "Requires", "Implements", "Conflicts",
            "Details", "Is similar", "No_dependency",
        }
        actual = {item.value for item in DependencyType}
        assert actual == expected

    def test_normalize_still_works(self):
        assert DependencyType.normalize("requires") == DependencyType.REQUIRES
        assert DependencyType.normalize("Is Similar") == DependencyType.IS_SIMILAR
        assert DependencyType.normalize("no_dependency") == DependencyType.NO_DEPENDENCY
        assert DependencyType.normalize("implements") == DependencyType.IMPLEMENTS
        with pytest.raises(ValueError, match="Unsupported dependency type"):
            DependencyType.normalize("Notable")
        with pytest.raises(ValueError, match="Unsupported dependency type"):
            DependencyType.normalize("Implemented by")


# ===========================================================================
# 7. Invalid / Edge Case Validation
# ===========================================================================

class TestInvalidInputHandling:
    """Test that invalid provenance data is properly rejected."""

    def test_invalid_annotation_source_in_provenance(self):
        """Invalid annotation source string must raise ValidationError."""
        with pytest.raises(ValidationError):
            AnnotationProvenance(source="invalid_source")

    def test_invalid_verification_status_in_provenance(self):
        """Invalid verification status string must raise ValidationError."""
        with pytest.raises(ValidationError):
            AnnotationProvenance(status="not_a_real_status")

    def test_provenance_confidence_out_of_range(self):
        """Confidence score outside 0-5 must raise ValidationError."""
        with pytest.raises(ValidationError):
            AnnotationProvenance(confidence_score=10)

    def test_provenance_probability_out_of_range(self):
        """Calibrated probability outside 0.0-1.0 must raise ValidationError."""
        with pytest.raises(ValidationError):
            AnnotationProvenance(calibrated_probability=2.0)

    def test_pair_ignores_unknown_fields(self):
        """Extra unknown fields in pair JSON should not cause errors."""
        data = _legacy_pair_dict()
        data["unknown_future_field"] = "some_value"
        # Pydantic V2 default behavior: extra fields are ignored
        pair = RequirementPair.model_validate(data)
        assert pair.id == "PAIR-ADB-001"


# ===========================================================================
# 8. API and Existing Model Compatibility
# ===========================================================================

class TestExistingModelCompatibility:
    """Ensure existing creation patterns for Prediction and RequirementPair still work."""

    def test_existing_prediction_construction_pattern(self):
        """Prediction created with only original required fields must work."""
        pred = Prediction(
            pair_id="PAIR-001",
            dependency_type=DependencyType.REQUIRES,
            rationale="Test",
            confidence=5,
        )
        assert pred.pair_id == "PAIR-001"
        assert pred.raw_response is None
        assert pred.prompt_tokens == 0
        assert pred.annotation_source is None

    def test_existing_pair_construction_pattern(self):
        """RequirementPair created with only original fields must work."""
        pair = RequirementPair(
            id="PAIR-001",
            req_a=Requirement(id="REQ-A", text="Text A", system="ADB"),
            req_b=Requirement(id="REQ-B", text="Text B", system="ADB"),
            dependency_type=DependencyType.CONFLICTS,
        )
        assert pair.provenance is None
        assert pair.dependency_type == DependencyType.CONFLICTS

    def test_model_dump_includes_provenance_when_set(self):
        """model_dump output includes provenance when it is set."""
        pair = RequirementPair(
            id="PAIR-001",
            req_a=Requirement(id="REQ-A", text="Text A"),
            req_b=Requirement(id="REQ-B", text="Text B"),
            provenance=AnnotationProvenance(
                source=AnnotationSource.LLM,
                model_id="gpt-4",
            ),
        )
        dumped = pair.model_dump(mode="json")
        assert "provenance" in dumped
        assert dumped["provenance"]["source"] == "llm"
        assert dumped["provenance"]["model_id"] == "gpt-4"

    def test_model_dump_provenance_none_when_unset(self):
        """model_dump output has provenance=None when not set."""
        pair = RequirementPair(
            id="PAIR-001",
            req_a=Requirement(id="REQ-A", text="Text A"),
            req_b=Requirement(id="REQ-B", text="Text B"),
        )
        dumped = pair.model_dump(mode="json")
        assert dumped["provenance"] is None
