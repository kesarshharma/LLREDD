"""Tests for ResponseParser and LEREDDDetector."""

from leredd.detector import ResponseParser, LEREDDDetector
from leredd.data_models import (
    DependencyType,
    PredictionStabilityStatus,
    PredictionValidationStatus,
)
from leredd.config import LEREDDConfig
from leredd.llm import LLMClient
import pytest
import numpy as np


class _OfflineEmbedder:
    def encode(self, texts):
        return np.asarray(
            [[len(text), sum(map(ord, text)) % 997, text.lower().count("system")] for text in texts],
            dtype=np.float32,
        )


def test_response_parser_confidence_threshold_flags_review_without_changing_label():
    raw_response = """**Dependency_Type: Requires**
**Rationale: Test rationale explanation.**
**Confidence Score: 3**"""

    # Low confidence requests review but must preserve the model's predicted class.
    pred = ResponseParser.parse(raw_response, pair_id="PAIR-001", confidence_threshold=4)
    assert pred.dependency_type == DependencyType.REQUIRES
    assert pred.confidence == 3
    assert pred.validation_status == PredictionValidationStatus.REQUIRES_REVIEW
    assert pred.validation_threshold == 4
    assert pred.rationale == "Test rationale explanation."

    # Confidence above the configured cutoff passes the screen without relabeling.
    pred2 = ResponseParser.parse(raw_response, pair_id="PAIR-001", confidence_threshold=2)
    assert pred2.dependency_type == DependencyType.REQUIRES
    assert pred2.validation_status == PredictionValidationStatus.PASSED_THRESHOLD
    assert pred2.validation_threshold == 2
    assert pred2.rationale == "Test rationale explanation."

    disabled = ResponseParser.parse(raw_response, pair_id="PAIR-001", confidence_threshold=-1)
    assert disabled.dependency_type == DependencyType.REQUIRES
    assert disabled.validation_status == PredictionValidationStatus.THRESHOLD_DISABLED
    assert disabled.validation_threshold == -1


def test_response_parser_does_not_map_unknown_label_to_no_dependency():
    with pytest.raises(ValueError, match="Unsupported dependency type"):
        ResponseParser.parse(
            "Dependency_Type: Notable\nRationale: unknown\nConfidence Score: 5",
            pair_id="PAIR-UNKNOWN",
        )


def test_response_parser_multiline_rationale_preserved():
    """Verify ResponseParser preserves rationales spanning at least three lines (ISSUE-013)."""
    raw_response = (
        "**Dependency_Type: Requires**\n"
        "**Rationale: Requirement A explicitly depends on the functionality of Requirement B.\n"
        "Specifically, the brake control subsystem requires sensor input from the radar unit.\n"
        "Without these inputs, deceleration routines cannot be safely scheduled.**\n"
        "**Confidence Score: 5**"
    )
    pred = ResponseParser.parse(raw_response, pair_id="PAIR-MULTI-1")
    expected_lines = [
        "Requirement A explicitly depends on the functionality of Requirement B.",
        "Specifically, the brake control subsystem requires sensor input from the radar unit.",
        "Without these inputs, deceleration routines cannot be safely scheduled.",
    ]
    for line in expected_lines:
        assert line in pred.rationale
    assert pred.rationale == "\n".join(expected_lines)
    assert pred.confidence == 5
    assert pred.dependency_type == DependencyType.REQUIRES
    assert "Confidence Score" not in pred.rationale
    assert "Dependency_Type" not in pred.rationale


def test_response_parser_rationale_after_confidence():
    """Verify ResponseParser correctly parses multiline rationale when it appears after the confidence field."""
    raw_response = (
        "**Dependency_Type: Conflicts**\n"
        "**Confidence Score: 5**\n"
        "**Rationale: Line 1 of the explanation.\n"
        "Line 2 describing the conflict.\n"
        "Line 3 detailing safety restrictions.**"
    )
    pred = ResponseParser.parse(raw_response, pair_id="PAIR-AFTER-CONF")
    assert "Line 1 of the explanation." in pred.rationale
    assert "Line 2 describing the conflict." in pred.rationale
    assert "Line 3 detailing safety restrictions." in pred.rationale
    assert pred.confidence == 5
    assert pred.dependency_type == DependencyType.CONFLICTS
    assert "Confidence Score" not in pred.rationale
    assert "Dependency_Type" not in pred.rationale


def test_response_parser_rationale_colon_inside_bold_variant():
    """Verify ResponseParser handles the `**Rationale:** text` formatting variant with multiline content."""
    raw_response = (
        "**Dependency_Type: Implements**\n"
        "**Rationale:** Requirement B specifies implementation details for Requirement A.\n"
        "Additional execution parameters are defined on this line.\n"
        "**Confidence Score: 5**"
    )
    pred = ResponseParser.parse(raw_response, pair_id="PAIR-COLON-BOLD")
    assert "Requirement B specifies implementation details for Requirement A." in pred.rationale
    assert "Additional execution parameters are defined on this line." in pred.rationale
    assert pred.confidence == 5
    assert pred.dependency_type == DependencyType.IMPLEMENTS
    assert "Confidence Score" not in pred.rationale
    assert "Dependency_Type" not in pred.rationale


def test_detector_detect_pair(mock_config, sample_pair, sample_annotated_pool):
    detector = LEREDDDetector(config=mock_config, embedder=_OfflineEmbedder())
    pred = detector.detect_pair(sample_pair, annotated_pool=sample_annotated_pool)

    assert pred.pair_id == sample_pair.id
    assert isinstance(pred.dependency_type, DependencyType)
    assert pred.confidence >= 0 and pred.confidence <= 5


def test_target_ground_truth_label_does_not_change_prediction_prompt(sample_pair):
    """Prediction input must depend on requirement text, never its held-out label."""
    class RecordingLLM(LLMClient):
        def __init__(self):
            self.prompts = []

        def generate(self, prompt: str, temperature: float = 0.2):
            self.prompts.append(prompt)
            return (
                "Dependency_Type: Requires\nRationale: controlled prediction.\nConfidence Score: 5",
                {},
            )

    llm = RecordingLLM()
    detector = LEREDDDetector(
        config=LEREDDConfig(llm_provider="mock", k_examples=0),
        embedder=_OfflineEmbedder(),
        llm=llm,
    )
    unlabeled_target = sample_pair.model_copy(update={"dependency_type": None, "provenance": None})

    labeled_prediction = detector.detect_pair(sample_pair)
    unlabeled_prediction = detector.detect_pair(unlabeled_target)

    assert llm.prompts[0] == llm.prompts[1]
    assert labeled_prediction.dependency_type == unlabeled_prediction.dependency_type


def test_zero_shot_detector_does_not_initialize_embedding_model(sample_pair, monkeypatch):
    """API-style zero-shot prediction must not cold-load an unused SBERT model."""
    from leredd.embeddings import EmbeddingModel
    from leredd.llm import MockLLMClient

    def unexpected_embedding_init(*args, **kwargs):
        raise AssertionError("zero-shot prediction must not initialize embeddings")

    monkeypatch.setattr(EmbeddingModel, "__init__", unexpected_embedding_init)
    detector = LEREDDDetector(
        config=LEREDDConfig(llm_provider="mock"),
        llm=MockLLMClient(),
    )

    prediction = detector.detect_pair(sample_pair)

    assert prediction.pair_id == sample_pair.id
    assert prediction.validation_status == PredictionValidationStatus.PASSED_THRESHOLD
    assert detector._embedder is None
    assert detector._example_retriever is None


def test_detector_passes_in_context_examples_to_llm_prompt(sample_pair):
    """Regression test for ISSUE-009: Verify LEREDDDetector includes retrieved in-context examples

    in the prompt forwarded to the LLM client when k_examples > 0, and omits them when k_examples = 0.
    """
    from unittest.mock import MagicMock
    from leredd.config import LEREDDConfig
    from leredd.data_models import DependencyType, Requirement, RequirementPair
    from leredd.llm import LLMClient

    class RecordingLLMClient(LLMClient):
        """Controlled test double that records the prompts received by generate()."""
        def __init__(self):
            self.prompts: list[str] = []

        def generate(self, prompt: str, temperature: float = 0.2):
            self.prompts.append(prompt)
            raw = (
                "**Dependency_Type: Requires**\n"
                "**Rationale: Controlled test double response.**\n"
                "**Confidence Score: 5**"
            )
            usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": 10, "cost_usd": 0.0}
            return raw, usage

    # Setup controlled example pair
    example_pair = RequirementPair(
        id="PAIR-EX-CONFLICTS",
        req_a=Requirement(id="REQ-E1", text="The vehicle must disengage autopilot immediately.", system="TEST"),
        req_b=Requirement(id="REQ-E2", text="The vehicle must alert the driver to resume control.", system="TEST"),
        dependency_type=DependencyType.CONFLICTS,
    )
    pool = [example_pair]

    # Case 1: k_examples = 0 (zero-shot)
    # The detector must not include any retrieved-example block in the generated prompt
    llm_zero = RecordingLLMClient()
    config_zero = LEREDDConfig(k_examples=0)
    detector_zero = LEREDDDetector(config=config_zero, embedder=MagicMock(), llm=llm_zero)
    detector_zero.detect_pair(sample_pair, annotated_pool=pool)

    assert len(llm_zero.prompts) == 1
    prompt_zero = llm_zero.prompts[0]
    assert "[Conflicts Examples]" not in prompt_zero
    assert "The vehicle must disengage autopilot immediately." not in prompt_zero
    assert "No in-context examples provided." in prompt_zero

    # Case 2: k_examples = 1 (in-context learning)
    # The detector must forward a prompt containing the retrieved example text and type section
    llm_icl = RecordingLLMClient()
    config_icl = LEREDDConfig(k_examples=1)
    detector_icl = LEREDDDetector(config=config_icl, embedder=MagicMock(), llm=llm_icl)
    # Deterministically stub retriever to return the supplied example without relying on embedding model
    detector_icl.example_retriever.retrieve = MagicMock(
        return_value={DependencyType.CONFLICTS: [example_pair]}
    )
    detector_icl.detect_pair(sample_pair, annotated_pool=pool)

    assert len(llm_icl.prompts) == 1
    prompt_icl = llm_icl.prompts[0]
    assert "[Conflicts Examples]" in prompt_icl
    assert 'Requirement A: "The vehicle must disengage autopilot immediately."' in prompt_icl
    assert 'Requirement B: "The vehicle must alert the driver to resume control."' in prompt_icl
    assert "Dependency_Type: Conflicts" in prompt_icl


@pytest.mark.parametrize(
    ("labels", "confidence_scores", "expected_status", "expected_reasons", "expected_review"),
    [
        (["Requires", "Requires"], [5, 5], PredictionStabilityStatus.CONSISTENT, [], False),
        (
            ["Requires", "No_dependency"],
            [5, 5],
            PredictionStabilityStatus.INCONSISTENT,
            ["dependency_label_disagreement"],
            True,
        ),
        (
            ["Requires", "Requires"],
            [5, 2],
            PredictionStabilityStatus.INCONSISTENT,
            ["self_reported_confidence_variation"],
            True,
        ),
    ],
)
def test_repeated_inference_reports_stability_and_preserves_first_label(
    sample_pair, labels, confidence_scores, expected_status, expected_reasons, expected_review
):
    class SequenceLLM(LLMClient):
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, temperature=0.2):
            index = self.calls
            self.calls += 1
            response = (
                f"Dependency_Type: {labels[index]}\n"
                f"Rationale: controlled run {index + 1}.\n"
                f"Confidence Score: {confidence_scores[index]}"
            )
            return response, {"prompt_tokens": 1, "completion_tokens": 2, "cost_usd": 0.01}

    llm = SequenceLLM()
    detector = LEREDDDetector(
        config=LEREDDConfig(k_examples=0, confidence_threshold=4),
        llm=llm,
    )

    prediction = detector.detect_pair(sample_pair, stability_runs=2)

    assert llm.calls == 2
    assert prediction.dependency_type == DependencyType.REQUIRES
    assert prediction.stability_status == expected_status
    assert prediction.stability_run_count == 2
    assert [label.value for label in prediction.stability_labels] == labels
    assert prediction.stability_confidences == confidence_scores
    assert prediction.stability_reasons == expected_reasons
    assert prediction.requires_review is expected_review
    assert prediction.prompt_tokens == 2
    assert prediction.completion_tokens == 4
    assert prediction.estimated_cost_usd == pytest.approx(0.02)


@pytest.mark.parametrize("run_count", [0, 6, True, 2.5])
def test_repeated_inference_rejects_invalid_run_count(sample_pair, run_count):
    from leredd.llm import MockLLMClient

    detector = LEREDDDetector(
        config=LEREDDConfig(k_examples=0),
        llm=MockLLMClient(),
    )
    with pytest.raises(ValueError, match="stability_runs must be an integer from 1 to 5"):
        detector.detect_pair(sample_pair, stability_runs=run_count)
