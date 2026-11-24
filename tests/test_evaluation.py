"""Tests for Evaluation metrics and statistical significance tests."""

from leredd.evaluation import (
    Evaluator,
    PredictionAlignmentError,
    SplitOverlapError,
    validate_system_disjoint_splits,
)
import pytest
from leredd.data_models import (
    AnnotationProvenance, Requirement, RequirementPair, Prediction,
    DependencyType, VerificationStatus,
)


def test_evaluator_metrics(sample_pair):
    gt = [sample_pair]
    preds = [
        Prediction(
            pair_id=sample_pair.id,
            dependency_type=DependencyType.REQUIRES,
            rationale="Test",
            confidence=5
        )
    ]
    report = Evaluator.evaluate(gt, preds, system_name="TestSystem")

    assert report.overall_accuracy == 1.0
    assert report.per_class_metrics["Requires"].precision == 1.0


def test_evaluator_mcnemar():
    req1 = Requirement(id="REQ-1", text="Requirement 1", system="Test")
    req2 = Requirement(id="REQ-2", text="Requirement 2", system="Test")
    gt = [
        RequirementPair(id="P1", req_a=req1, req_b=req2, dependency_type=DependencyType.REQUIRES),
        RequirementPair(id="P2", req_a=req1, req_b=req2, dependency_type=DependencyType.CONFLICTS)
    ]
    preds_a = [
        Prediction(pair_id="P1", dependency_type=DependencyType.REQUIRES, rationale="", confidence=5),
        Prediction(pair_id="P2", dependency_type=DependencyType.CONFLICTS, rationale="", confidence=5)
    ]
    preds_b = [
        Prediction(pair_id="P1", dependency_type=DependencyType.REQUIRES, rationale="", confidence=5),
        Prediction(pair_id="P2", dependency_type=DependencyType.NO_DEPENDENCY, rationale="", confidence=5)
    ]

    stat, p_val = Evaluator.run_mcnemar_test(gt, preds_a, preds_b)
    assert isinstance(stat, float)
    assert isinstance(p_val, float)


def test_evaluator_rejects_unlabeled_ground_truth(sample_pair):
    sample_pair.dependency_type = None
    prediction = Prediction(
        pair_id=sample_pair.id, dependency_type=DependencyType.NO_DEPENDENCY,
        rationale="", confidence=5,
    )
    with pytest.raises(ValueError, match="unlabeled or unverified"):
        Evaluator.evaluate([sample_pair], [prediction])


def test_evaluator_rejects_provisional_ground_truth(sample_pair):
    sample_pair.provenance = AnnotationProvenance(status=VerificationStatus.PROVISIONAL)
    prediction = Prediction(
        pair_id=sample_pair.id, dependency_type=DependencyType.REQUIRES,
        rationale="", confidence=5,
    )
    with pytest.raises(ValueError, match="unlabeled or unverified"):
        Evaluator.evaluate([sample_pair], [prediction])


def test_evaluator_rejects_missing_predictions(sample_pair):
    with pytest.raises(ValueError, match="missing ground-truth pair IDs"):
        Evaluator.evaluate([sample_pair], [])


def test_evaluator_duplicate_prediction_error_reports_pair_ids(sample_pair):
    prediction = Prediction(
        pair_id=sample_pair.id,
        dependency_type=DependencyType.REQUIRES,
        rationale="controlled test",
        confidence=5,
    )

    with pytest.raises(PredictionAlignmentError, match=sample_pair.id) as exc_info:
        Evaluator.evaluate([sample_pair], [prediction, prediction])

    assert exc_info.value.invalid_ids == [sample_pair.id]


def test_system_disjoint_validator_accepts_separate_partitions():
    def pair(pair_id, system, req_a_id, req_b_id):
        return RequirementPair(
            id=pair_id,
            req_a=Requirement(id=req_a_id, text="A", system=system),
            req_b=Requirement(id=req_b_id, text="B", system=system),
        )

    audit = validate_system_disjoint_splits(
        train_pairs=[pair("P-TRAIN", "TRAIN", "TR-A", "TR-B")],
        test_pairs=[pair("P-TEST", "TEST", "TE-A", "TE-B")],
        dev_pairs=[pair("P-DEV", "DEV", "DE-A", "DE-B")],
    )

    assert audit["is_disjoint"] is True
    assert audit["overlapping_systems"] == []


def test_system_disjoint_validator_reports_overlapping_identity_ids():
    train = RequirementPair(
        id="P-SHARED",
        req_a=Requirement(id="REQ-SHARED", text="A", system="ADB"),
        req_b=Requirement(id="REQ-TRAIN", text="B", system="ADB"),
    )
    test = RequirementPair(
        id="P-SHARED",
        req_a=Requirement(id="REQ-SHARED", text="A", system="ADB"),
        req_b=Requirement(id="REQ-TEST", text="B", system="ADB"),
    )

    with pytest.raises(SplitOverlapError) as exc_info:
        validate_system_disjoint_splits([train], [test])

    assert exc_info.value.invalid_ids == ["ADB", "P-SHARED", "REQ-SHARED"]
