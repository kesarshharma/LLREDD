"""Data models for SRS Requirements, Pairs, LLM Predictions, and Evaluation Metrics."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


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
    source: str | None = Field(default=None, description="Source document or section reference")


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

    @field_validator("predicted_label", "accepted_label", mode="before")
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


class Prediction(BaseModel):
    """Output prediction generated by LEREDD detection engine."""
    pair_id: str = Field(..., description="Target requirement pair ID")
    dependency_type: DependencyType = Field(..., description="Predicted dependency type")
    rationale: str = Field(..., description="Detailed explanation/reasoning generated by LLM")
    confidence: int = Field(..., ge=0, le=5, description="Self-assessed Likert confidence score (0 to 5)")
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
    per_class_metrics: dict[str, EvaluationMetric] = Field(..., description="Metrics for each dependency class")
    confusion_matrix: dict[str, dict[str, int]] | None = Field(
        default=None,
        description="Confusion matrix dictionary mapping ground_truth -> predicted -> count"
    )
