"""Tests for ResponseParser and LEREDDDetector."""

from leredd.detector import ResponseParser, LEREDDDetector
from leredd.data_models import DependencyType


def test_response_parser_threshold_reannotation():
    raw_response = """**Dependency_Type: Requires**
**Rationale: Test rationale explanation.**
**Confidence Score: 3**"""

    # With threshold = 4, confidence 3 <= 4 -> should be reclassified to No_dependency
    pred = ResponseParser.parse(raw_response, pair_id="PAIR-001", confidence_threshold=4)
    assert pred.dependency_type == DependencyType.NO_DEPENDENCY
    assert pred.confidence == 3

    # With threshold = 2, confidence 3 > 2 -> should remain Requires
    pred2 = ResponseParser.parse(raw_response, pair_id="PAIR-001", confidence_threshold=2)
    assert pred2.dependency_type == DependencyType.REQUIRES


def test_detector_detect_pair(mock_config, sample_pair, sample_annotated_pool):
    detector = LEREDDDetector(config=mock_config)
    pred = detector.detect_pair(sample_pair, annotated_pool=sample_annotated_pool)

    assert pred.pair_id == sample_pair.id
    assert isinstance(pred.dependency_type, DependencyType)
    assert pred.confidence >= 0 and pred.confidence <= 5
