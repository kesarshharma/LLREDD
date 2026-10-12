"""Tests for Evaluation metrics and statistical significance tests."""

from leredd.evaluation import Evaluator
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
