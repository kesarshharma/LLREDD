"""Data models for SRS Requirements, Pairs, LLM Predictions, and Evaluation Metrics."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


class AnnotationSource(str, Enum):
    """Origin of a dependency annotation or prediction.

    Tracks how a label was produced to support provenance auditing
    in the semi-automated dataset generation pipeline.
    """
    HUMAN = "human"
    LLM = "llm"
    HEURISTIC = "heuristic"
    IMPORTED = "imported"
    ENSEMBLE = "ensemble"
    CANDIDATE_GENERATOR = "candidate_generator"


class VerificationStatus(str, Enum):
    """Lifecycle state of an annotation within the review pipeline.

    Transitions: provisional -> pending_review -> human_reviewed -> verified
                                                                 -> rejected
    """
    PROVISIONAL = "provisional"
    PENDING_REVIEW = "pending_review"
    HUMAN_REVIEWED = "human_reviewed"
    VERIFIED = "verified"
    REJECTED = "rejected"


class DependencyType(str, Enum):
    """Supported dependency types defined in paper Table I."""
    REQUIRES = "Requires"
    IMPLEMENTS = "Implements"
    CONFLICTS = "Conflicts"
    DETAILS = "Details"
    IS_SIMILAR = "Is similar"
    NO_DEPENDENCY = "No_dependency"

    @classmethod
    def normalize(cls, val: str) -> "DependencyType":
        """Normalize supported label spellings without guessing unknown labels."""
        clean = " ".join(val.strip().replace("_", " ").replace("-", " ").lower().split())
        aliases = {
            "requires": cls.REQUIRES,
            "require": cls.REQUIRES,
            "implements": cls.IMPLEMENTS,
            "implement": cls.IMPLEMENTS,
            "conflicts": cls.CONFLICTS,
            "conflict": cls.CONFLICTS,
            "details": cls.DETAILS,
            "detail": cls.DETAILS,
            "is similar": cls.IS_SIMILAR,
            "similar": cls.IS_SIMILAR,
            "no": cls.NO_DEPENDENCY,
            "no dependency": cls.NO_DEPENDENCY,
            "independent": cls.NO_DEPENDENCY,
        }
        try:
            return aliases[clean]
        except KeyError as exc:
            raise ValueError(f"Unsupported dependency type: {val!r}") from exc


class Requirement(BaseModel):
    """Representation of an individual Software Requirement statement."""
    id: str = Field(..., description="Unique requirement ID (e.g. REQ-ADB-001)")
    text: str = Field(..., description="Natural language requirement statement text")
    system: str = Field(default="GenericSystem", description="System or subsystem name (e.g. ADB, TJA, APA)")
    component: str | None = Field(
        default=None,
        description="Optional component metadata used by metadata-aware retrieval",
    )
    subsystem: str | None = Field(
        default=None,
        description="Optional subsystem metadata used by metadata-aware retrieval",
    )
    module: str | None = Field(
        default=None,
        description="Optional module metadata used by metadata-aware retrieval and document splitting",
    )
    source: str | None = Field(default=None, description="Source document or section reference")
    source_location: str | None = Field(
        default=None,
        description="Location within the source document, such as an extracted-text line number",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured key-value requirement metadata",
    )
    synthetic: bool = Field(
        default=False,
        description="Whether this requirement originates from synthetic/demo data",
    )
    source_identity: str | None = Field(
        default=None,
        description="Identity of the dataset or synthetic source that produced this requirement",
    )


class AnnotationProvenance(BaseModel):
    """Tracks origin, lifecycle, and review state of a dependency annotation.

    Used on RequirementPair to record how a ground-truth or provisional label
    was produced, reviewed, and finalized. Supports the semi-automated dataset
    generation pipeline (Phase 4+) and human review queue (Phase 5+).
    """
    source: AnnotationSource = Field(
        default=AnnotationSource.IMPORTED,
        description="How this annotation was produced"
    )
    status: VerificationStatus = Field(
        default=VerificationStatus.PROVISIONAL,
        description="Current lifecycle state of the annotation"
    )
    annotator_id: str | None = Field(
        default=None,
        description="Identifier of the original annotator (human or system)"
    )
    model_id: str | None = Field(
        default=None,
        description="LLM model identifier (e.g. 'gpt-4', 'ollama/llama3')"
    )
    model_version: str | None = Field(
        default=None,
        description="Specific model version or checkpoint (e.g. 'gpt-4-0613')"
    )
    config_hash: str | None = Field(
        default=None,
        description="Hash of the prompt template and configuration used"
    )
    created_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when the annotation was created"
    )
    reviewed_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when the annotation was reviewed"
    )
    reviewer_id: str | None = Field(
        default=None,
        description="Identifier of the human reviewer, if reviewed"
    )
    predicted_label: DependencyType | None = Field(
        default=None,
        description="Raw predicted label before human review or re-annotation"
    )
    accepted_label: DependencyType | None = Field(
        default=None,
        description="Final accepted label after review (may differ from predicted_label)"
    )
    confidence_score: int | None = Field(
        default=None, ge=0, le=5,
        description="LLM self-reported Likert confidence score (0-5)"
    )
    calibrated_probability: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Calibrated probability estimate (future use, distinct from Likert confidence)"
    )
    evidence: str | None = Field(
        default=None,
        description="Supporting rationale or evidence text for the annotation"
    )
    review_notes: str | None = Field(
        default=None,
        description="Human reviewer notes or decision justification"
    )
    dataset_version: str | None = Field(
        default=None,
        description="Dataset version identifier (e.g. 'v1.0', 'v2.0-provisional')"
    )
    heuristic_label: DependencyType | None = Field(
        default=None,
        description="Structured raw prediction from HeuristicLabeler",
    )
    heuristic_rule_id: str | None = Field(
        default=None,
        description="Deterministic rule identifier from HeuristicLabeler",
    )
    heuristic_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Rule heuristic score [0.0, 1.0]",
    )
    heuristic_evidence: str | None = Field(
        default=None,
        description="Structured rule evidence from HeuristicLabeler",
    )
    llm_label: DependencyType | None = Field(
        default=None,
        description="Structured raw prediction from LLMLabeler",
    )
    llm_rationale: str | None = Field(
        default=None,
        description="Structured rationale from LLMLabeler",
    )
    llm_raw_response: str | None = Field(
        default=None,
        description="Raw LLM output text captured during provisional labeling",
    )
    llm_failed: bool = Field(
        default=False,
        description="Whether LLM invocation or response parsing failed",
    )
    llm_error: str | None = Field(
        default=None,
        description="Structured error message when LLM labeling failed",
    )
    explicit_review_flag: bool = Field(
        default=False,
        description="Explicit ambiguity or manual flag routing this pair to human review",
    )
    structured_review_reasons: list[str] = Field(
        default_factory=list,
        description="Structured machine-readable review reason codes",
    )
    synthetic: bool = Field(
        default=False,
        description="Whether this annotation was created for synthetic/demo data",
    )
    source_identity: str | None = Field(
        default=None,
        description="Identity of the dataset or synthetic source for this annotation",
    )

    @field_validator("predicted_label", "accepted_label", "heuristic_label", "llm_label", mode="before")
    @classmethod
    def parse_provenance_labels(cls, v: Any) -> DependencyType | None:
        if v is None:
            return None
        if isinstance(v, DependencyType):
            return v
        if isinstance(v, str):
            return DependencyType.normalize(v)
        return None


class RequirementPair(BaseModel):
    """A pair of requirement statements evaluated for dependency."""
    id: str = Field(..., description="Unique requirement pair ID")
    req_a: Requirement = Field(..., description="First requirement statement (R1 / Requirement A)")
    req_b: Requirement = Field(..., description="Second requirement statement (R2 / Requirement B)")
    dependency_type: DependencyType | None = Field(
        default=None,
        description="Ground truth or annotated dependency type"
    )
    provenance: AnnotationProvenance | None = Field(
        default=None,
        description="Annotation provenance metadata tracking label origin and review state"
    )
    synthetic: bool = Field(
        default=False,
        description="Whether this pair originates from synthetic/demo data",
    )
    source_identity: str | None = Field(
        default=None,
        description="Identity of the dataset or synthetic source for this pair",
    )

    @field_validator("dependency_type", mode="before")
    @classmethod
    def parse_dependency_type(cls, v: Any) -> DependencyType | None:
        if v is None:
            return None
        if isinstance(v, DependencyType):
            return v
        if isinstance(v, str):
            return DependencyType.normalize(v)
        return None


class PredictionValidationStatus(str, Enum):
    """Confidence-screening outcome; never a substitute for annotation status."""

    NOT_EVALUATED = "not_evaluated"
    PASSED_THRESHOLD = "passed_threshold"
    REQUIRES_REVIEW = "requires_review"
    THRESHOLD_DISABLED = "threshold_disabled"


class PredictionStabilityStatus(str, Enum):
    """Outcome of optional repeated inference with a fixed prompt."""

    NOT_CHECKED = "not_checked"
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"


class Prediction(BaseModel):
    """Output prediction generated by LEREDD detection engine."""
    pair_id: str = Field(..., description="Target requirement pair ID")
    dependency_type: DependencyType = Field(..., description="Predicted dependency type")
    rationale: str = Field(..., description="Detailed explanation/reasoning generated by LLM")
    confidence: int = Field(..., ge=0, le=5, description="Self-assessed Likert confidence score (0 to 5)")
    validation_status: PredictionValidationStatus = Field(
        default=PredictionValidationStatus.NOT_EVALUATED,
        description=(
            "Confidence-screening outcome; this does not verify the predicted label "
            "or treat self-reported confidence as calibrated probability"
        ),
    )
    validation_threshold: int | None = Field(
        default=None,
        ge=-1,
        le=5,
        description="Configured self-reported confidence cutoff used for screening, if evaluated",
    )
    stability_status: PredictionStabilityStatus = Field(
        default=PredictionStabilityStatus.NOT_CHECKED,
        description="Whether optional repeated inferences agreed on label and self-reported confidence",
    )
    stability_run_count: int = Field(default=1, ge=1, le=5)
    stability_labels: list[DependencyType] = Field(default_factory=list)
    stability_confidences: list[int] = Field(
        default_factory=list,
        description="Self-reported confidence scores from repeated inferences",
    )
    stability_reasons: list[str] = Field(default_factory=list)
    raw_response: str | None = Field(default=None, description="Raw unparsed output from LLM")
    prompt_tokens: int = Field(default=0, description="Tokens used for prompt")
    completion_tokens: int = Field(default=0, description="Tokens used for completion")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated API cost in USD")
    # Provenance fields for prediction traceability
    annotation_source: AnnotationSource | None = Field(
        default=None,
        description="Source method that produced this prediction (e.g. llm, heuristic)"
    )
    model_id: str | None = Field(
        default=None,
        description="LLM model identifier used for this prediction"
    )
    config_hash: str | None = Field(
        default=None,
        description="Hash of prompt template and config used for this prediction"
    )
    created_at: str | None = Field(
        default=None,
        description="ISO 8601 timestamp when this prediction was generated"
    )

    @model_validator(mode="before")
    @classmethod
    def reject_calibrated_probability_representation(cls, data: Any) -> Any:
        """Forbid representing uncalibrated Likert confidence as calibrated probability."""
        if isinstance(data, dict) and data.get("calibrated_probability") is not None:
            raise ValueError(
                "Self-reported Likert confidence (0-5) cannot be represented as "
                "calibrated_probability on Prediction."
            )
        return data

    @property
    def requires_review(self) -> bool:
        """Whether confidence or repeated-run evidence recommends human review."""
        if self.validation_status == PredictionValidationStatus.REQUIRES_REVIEW:
            return True
        if self.stability_status == PredictionStabilityStatus.INCONSISTENT:
            return True
        return bool(
            self.validation_threshold is not None
            and self.validation_threshold >= 0
            and any(score <= self.validation_threshold for score in self.stability_confidences)
        )


class EvaluationMetric(BaseModel):
    """Precision, Recall, F1 score and support for a single class or overall average."""
    precision: float = Field(..., description="Precision score [0.0, 1.0]")
    recall: float = Field(..., description="Recall score [0.0, 1.0]")
    f1_score: float = Field(..., description="F1 score [0.0, 1.0]")
    support: int = Field(..., description="Number of true instances")


class EvaluationReport(BaseModel):
    """Comprehensive evaluation summary matching Paper Tables III-IX structure."""
    system_name: str = Field(..., description="Dataset or evaluation system name (e.g. D1, D2, D3)")
    model_name: str = Field(..., description="Model/Strategy name evaluated")
    overall_accuracy: float = Field(..., description="Overall accuracy")
    macro_f1: float = Field(..., description="Macro average F1 score")
    weighted_f1: float = Field(..., description="Weighted average F1 score")
    macro_precision: float | None = Field(default=None, description="Macro average precision score")
    macro_recall: float | None = Field(default=None, description="Macro average recall score")
    per_class_metrics: dict[str, EvaluationMetric] = Field(..., description="Metrics for each dependency class")
    confusion_matrix: dict[str, dict[str, int]] | None = Field(
        default=None,
        description="Confusion matrix dictionary mapping ground_truth -> predicted -> count"
    )
