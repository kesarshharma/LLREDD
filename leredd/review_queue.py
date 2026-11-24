"""Multi-Method Disagreement and Uncertainty Review Queue module (Phase 5, TASK-P5-01 & Part 5).

Implements a persistent, auditable human-review queue for provisional requirement dependency annotations:
1. ReviewReasonCode & QueueStatus enums.
2. ReviewItem & HumanDecision Pydantic models.
3. ReviewQueueManager: Handles structured eligibility routing, deterministic prioritization,
   idempotent queue refresh, and human decision recording.
4. Ground-Truth Protection: Machine predictions remain provisional until explicitly verified by a human reviewer.

CORE PRINCIPLE: Machine predictions NEVER silently become verified annotations.
Only explicit human ACCEPT or CORRECT decisions produce VerificationStatus.VERIFIED records.
"""

import datetime
from enum import Enum
import hashlib
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.synthetic import is_synthetic, synthetic_source_identity


class ReviewReasonCode(str, Enum):
    """Reason codes for routing a provisional annotation to the review queue."""
    DISAGREEMENT = "disagreement"  # Heuristic and LLM produced conflicting valid labels
    BOTH_ABSTAINED = "both_abstained"  # Neither method produced a valid prediction
    LLM_FAILURE = "llm_failure"  # LLM execution error, timeout, or malformed response
    HEURISTIC_ONLY = "heuristic_only"  # Heuristic predicted, LLM abstained/not run
    LLM_ONLY = "llm_only"  # LLM predicted, Heuristic abstained/not run
    LOW_CONFIDENCE = "low_confidence"  # Self-reported confidence at or below the configured cutoff
    SHORT_REQUIREMENT = "short_requirement"  # Requirement text is below the configured minimum length
    MISSING_METADATA = "missing_metadata"  # Requirement or provenance metadata incomplete
    EXPLICIT_FLAG = "explicit_flag"  # Explicitly flagged as ambiguous or requiring human review


class QueueStatus(str, Enum):
    """Lifecycle status of an item in the review queue."""
    PENDING = "pending"  # Awaiting human review
    IN_REVIEW = "in_review"  # Currently assigned or under review
    ACCEPTED = "accepted"  # Proposed provisional label accepted as-is by reviewer
    CORRECTED = "corrected"  # Proposed label corrected to a different valid class by reviewer
    REJECTED = "rejected"  # Candidate pair rejected as invalid/unusable
    DEFERRED = "deferred"  # Decision deferred / unresolved pending more evidence
    REVIEWED = "reviewed"  # Generic completed review note without ground-truth verification


class ReviewPriority(int, Enum):
    """Deterministic priority levels for review items (1 = Highest, 5 = Lowest)."""
    CRITICAL = 1  # Disagreement between methods
    HIGH = 2      # LLM or execution failures
    MEDIUM = 3    # Missing metadata, short requirement text, or both abstained
    LOW = 4       # Single-method predictions or low confidence
    INFO = 5      # Explicitly flagged items with no higher-severity condition


class HumanDecision(BaseModel):
    """Recorded audit record of a human reviewer decision."""
    decision_id: str = Field(default_factory=lambda: f"DEC-{uuid.uuid4().hex[:8].upper()}")
    item_id: str = Field(description="Target queue item ID")
    pair_id: str = Field(description="Target requirement pair ID")
    decision_type: QueueStatus = Field(description="Decision status (accepted, corrected, rejected, deferred)")
    final_label: Optional[DependencyType] = Field(default=None, description="Final label assigned by human reviewer")
    reviewer_id: str = Field(default="", description="Identifier of human reviewer")
    reviewer_notes: str = Field(default="", description="Detailed justification or review notes")
    decided_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    previous_label: Optional[DependencyType] = Field(default=None, description="Provisional label before decision")

    @field_validator("final_label", mode="before")
    @classmethod
    def parse_final_label(cls, v: Any) -> Optional[DependencyType]:
        if v is None:
            return None
        if isinstance(v, DependencyType):
            return v
        if isinstance(v, str):
            return DependencyType.normalize(v)
        return None


class ReviewItem(BaseModel):
    """Individual item in the human review queue representing a provisional requirement pair."""
    queue_id: str = Field(description="Deterministic stable queue item identifier")
    pair_id: str = Field(description="Target requirement pair ID")
    req_a_id: str = Field(description="Requirement A identifier")
    req_b_id: str = Field(description="Requirement B identifier")
    req_a_text: str = Field(description="Requirement A full text")
    req_b_text: str = Field(description="Requirement B full text")
    system_name: Optional[str] = Field(default=None, description="Source system/SRS identifier")
    req_a_system: Optional[str] = Field(default=None, description="Requirement A system identifier")
    req_b_system: Optional[str] = Field(default=None, description="Requirement B system identifier")
    req_a_component: Optional[str] = Field(default=None, description="Requirement A component metadata")
    req_b_component: Optional[str] = Field(default=None, description="Requirement B component metadata")
    req_a_subsystem: Optional[str] = Field(default=None, description="Requirement A subsystem metadata")
    req_b_subsystem: Optional[str] = Field(default=None, description="Requirement B subsystem metadata")
    req_a_module: Optional[str] = Field(default=None, description="Requirement A module metadata")
    req_b_module: Optional[str] = Field(default=None, description="Requirement B module metadata")
    req_a_source: Optional[str] = Field(default=None, description="Requirement A source document")
    req_b_source: Optional[str] = Field(default=None, description="Requirement B source document")
    req_a_source_location: Optional[str] = Field(default=None, description="Requirement A source location")
    req_b_source_location: Optional[str] = Field(default=None, description="Requirement B source location")
    req_a_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Requirement A structured metadata")
    req_b_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Requirement B structured metadata")
    synthetic: bool = Field(default=False, description="Whether this review item is synthetic/demo data")
    source_identity: Optional[str] = Field(default=None, description="Identifier of the dataset or fixture source")
    orientation: Tuple[str, str] = Field(description="Preserved directional orientation (req_a_id, req_b_id)")
    heuristic_prediction: Optional[DependencyType] = Field(default=None, description="Raw prediction from Heuristic")
    heuristic_evidence: str = Field(default="", description="Supporting evidence / rule ID from Heuristic")
    llm_prediction: Optional[DependencyType] = Field(default=None, description="Raw prediction from LLM")
    llm_rationale: str = Field(default="", description="Supporting rationale / explanation from LLM")
    llm_confidence: Optional[int] = Field(default=None, ge=0, le=5, description="LLM Likert confidence score (0-5)")
    proposed_label: Optional[DependencyType] = Field(default=None, description="Proposed provisional label")
    provenance: Optional[AnnotationProvenance] = Field(default=None, description="Original annotation provenance")
    review_reasons: List[ReviewReasonCode] = Field(default_factory=list, description="Machine-readable review reason codes")
    review_notes: str = Field(default="", description="Machine-generated explanation of why item was queued")
    status: QueueStatus = Field(default=QueueStatus.PENDING, description="Current queue status")
    priority: int = Field(default=ReviewPriority.LOW, ge=1, le=5, description="Priority rank (1=Highest, 5=Lowest)")
    human_decision: Optional[HumanDecision] = Field(default=None, description="Recorded human decision if reviewed")
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    @field_validator("heuristic_prediction", "llm_prediction", "proposed_label", mode="before")
    @classmethod
    def parse_enum_labels(cls, v: Any) -> Optional[DependencyType]:
        if v is None:
            return None
        if isinstance(v, DependencyType):
            return v
        if isinstance(v, str):
            return DependencyType.normalize(v)
        return None


class ReviewQueueConfig(BaseModel):
    """Configuration for review queue eligibility and prioritization policies."""
    include_disagreements: bool = True
    include_both_abstained: bool = True
    include_failures: bool = True
    include_single_method: bool = True  # Include Heuristic-only or LLM-only predictions
    include_low_confidence: bool = True
    include_missing_metadata: bool = True
    include_explicit_flags: bool = True
    confidence_threshold: int = 3  # Ratings <= 3 trigger review if enabled
    minimum_requirement_characters: int = Field(
        default=20,
        ge=0,
        description="Non-empty requirement text shorter than this length is routed for review; zero disables the check.",
    )


class ReviewQueueManifest(BaseModel):
    """Summary manifest for a review queue state or refresh run."""
    manifest_id: str = Field(default_factory=lambda: f"QMAN-{uuid.uuid4().hex[:8].upper()}")
    source_identity: str = Field(description="Input dataset or system identifier")
    total_items: int = Field(description="Total review items in queue")
    pending_count: int = Field(description="Items awaiting review")
    accepted_count: int = Field(description="Items accepted by human reviewer")
    corrected_count: int = Field(description="Items corrected by human reviewer")
    rejected_count: int = Field(description="Items rejected by human reviewer")
    deferred_count: int = Field(description="Items deferred by human reviewer")
    reasons_breakdown: Dict[str, int] = Field(default_factory=dict, description="Item counts by review reason")
    eligibility_config: ReviewQueueConfig = Field(
        default_factory=ReviewQueueConfig,
        description="Eligibility thresholds and switches used to build this queue.",
    )
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class ReviewQueueManager:
    """Manages creation, persistence, filtering, prioritization, and human decision recording for review queues."""

    def __init__(
        self,
        config: Optional[ReviewQueueConfig] = None,
        items: Optional[List[ReviewItem]] = None,
    ):
        self.config = config or ReviewQueueConfig()
        self.items = list(items or [])

    @staticmethod
    def generate_item_id(pair_id: str, req_a_id: str, req_b_id: str) -> str:
        """Generate a deterministic, stable queue item ID preserving directional orientation."""
        content = f"{pair_id}:{req_a_id}->{req_b_id}"
        hash_str = hashlib.md5(content.encode("utf-8")).hexdigest()[:10].upper()
        return f"QITEM-{hash_str}"

    @staticmethod
    def _extract_structured_predictions(
        pair: RequirementPair,
    ) -> Tuple[Optional[DependencyType], str, Optional[DependencyType], str, Optional[int]]:
        """Extract heuristic & LLM predictions/evidence preferring structured fields over legacy notes."""
        prov = pair.provenance
        if prov is None:
            return None, "", None, "", None

        h_pred = prov.heuristic_label
        l_pred = prov.llm_label
        h_ev = prov.heuristic_evidence or ""
        l_rat = prov.llm_rationale or ""
        l_conf = prov.confidence_score

        if prov.heuristic_rule_id and h_ev and prov.heuristic_rule_id not in h_ev:
            h_ev = f"{prov.heuristic_rule_id}: {h_ev}"
        elif prov.heuristic_rule_id and not h_ev:
            h_ev = prov.heuristic_rule_id

        # Legacy compatibility fallbacks when structured provenance fields were not populated
        ev_str = prov.evidence or ""
        notes_str = prov.review_notes or ""

        if h_pred is None and l_pred is None:
            if "AGREEMENT" in ev_str and pair.dependency_type is not None:
                h_pred = pair.dependency_type
                l_pred = pair.dependency_type
            elif "DISAGREEMENT" in notes_str and "'" in notes_str:
                parts = notes_str.split("'")
                if len(parts) >= 4:
                    try:
                        h_pred = DependencyType.normalize(parts[1])
                        l_pred = DependencyType.normalize(parts[3])
                    except ValueError:
                        pass
            elif prov.source == AnnotationSource.HEURISTIC and not prov.llm_failed and "UNRESOLVED" not in notes_str and "FAILED" not in notes_str:
                h_pred = pair.dependency_type or prov.predicted_label
            elif prov.source == AnnotationSource.LLM and not prov.llm_failed and "UNRESOLVED" not in notes_str and "FAILED" not in notes_str:
                l_pred = pair.dependency_type or prov.predicted_label

        if not l_rat and "LLM Rationale:" in ev_str:
            l_rat = ev_str.split("LLM Rationale:")[-1].strip()
        if not h_ev and "Heuristic [" in ev_str:
            h_ev = ev_str

        return h_pred, h_ev, l_pred, l_rat, l_conf

    @classmethod
    def determine_eligibility(
        cls, pair: RequirementPair, config: ReviewQueueConfig
    ) -> Tuple[bool, List[ReviewReasonCode], int, str]:
        """Determine whether a requirement pair is eligible for human review.

        Prefers structured provenance fields over parsing phrases from review_notes,
        with non-destructive legacy fallback for older persisted artifacts.
        """
        if pair.provenance and pair.provenance.status == VerificationStatus.VERIFIED:
            return False, [], ReviewPriority.INFO.value, ""

        reasons: List[ReviewReasonCode] = []
        explanation_parts: List[str] = []

        prov = pair.provenance
        h_label, _, l_label, _, l_conf = cls._extract_structured_predictions(pair)
        structured_codes = set(prov.structured_review_reasons) if prov else set()
        notes_str = (prov.review_notes or "") if prov else ""

        def add_reason(code: ReviewReasonCode, detail: str) -> None:
            if code not in reasons:
                reasons.append(code)
                if detail:
                    explanation_parts.append(detail)

        # 1. Heuristic / LLM Disagreement (structured first, legacy fallback second)
        has_disagreement = (
            (h_label is not None and l_label is not None and h_label != l_label)
            or ReviewReasonCode.DISAGREEMENT.value in structured_codes
            or "DISAGREEMENT" in notes_str
            or "MULTI_LLM_DISAGREEMENT:" in notes_str
        )
        if has_disagreement:
            detail = (
                notes_str
                if notes_str
                else f"Structured disagreement: Heuristic='{h_label.value if h_label else 'N/A'}', "
                f"LLM='{l_label.value if l_label else 'N/A'}'."
            )
            add_reason(ReviewReasonCode.DISAGREEMENT, detail)

        # 2. LLM Failure (provider error or malformed output)
        has_llm_failure = bool(
            (prov and (prov.llm_failed or prov.llm_error))
            or ReviewReasonCode.LLM_FAILURE.value in structured_codes
            or "FAILED" in notes_str
        )
        if has_llm_failure:
            err_detail = (
                (prov.llm_error if prov and prov.llm_error else "")
                or notes_str
                or "LLM execution or response parsing failed."
            )
            add_reason(ReviewReasonCode.LLM_FAILURE, err_detail)

        # 3. Both Methods Abstained
        has_both_abstained = bool(
            ReviewReasonCode.BOTH_ABSTAINED.value in structured_codes
            or (
                not has_disagreement
                and not has_llm_failure
                and (
                    "UNRESOLVED" in notes_str
                    or (
                        prov is not None
                        and h_label is None
                        and l_label is None
                        and pair.dependency_type is None
                        and prov.predicted_label is None
                    )
                )
            )
        )
        if has_both_abstained:
            add_reason(
                ReviewReasonCode.BOTH_ABSTAINED,
                "Both Heuristic and LLM abstained from asserting a dependency label.",
            )

        # 4. Single-Method Predictions (only when genuinely predicted by one method)
        if not has_disagreement and not has_both_abstained:
            if (
                ReviewReasonCode.LLM_ONLY.value in structured_codes
                or (l_label is not None and h_label is None and not has_llm_failure)
            ):
                add_reason(
                    ReviewReasonCode.LLM_ONLY,
                    "Single-method prediction from LLM (Heuristic abstained).",
                )
            elif (
                ReviewReasonCode.HEURISTIC_ONLY.value in structured_codes
                or (h_label is not None and l_label is None)
            ):
                add_reason(
                    ReviewReasonCode.HEURISTIC_ONLY,
                    "Single-method prediction from Heuristic (LLM abstained or failed).",
                )

        # 5. Low Confidence
        if (
            ReviewReasonCode.LOW_CONFIDENCE.value in structured_codes
            or (l_conf is not None and l_conf <= config.confidence_threshold)
        ):
            score_display = f"{l_conf}/5" if l_conf is not None else "at or below cutoff"
            add_reason(
                ReviewReasonCode.LOW_CONFIDENCE,
                f"LLM confidence score {score_display} is at or below review cutoff {config.confidence_threshold}.",
            )

        # 6. Missing Metadata (empty text, missing IDs, empty system identifier, or structured flag)
        missing_fields: List[str] = []
        if not pair.req_a.text or not pair.req_a.text.strip():
            missing_fields.append("req_a.text")
        if not pair.req_b.text or not pair.req_b.text.strip():
            missing_fields.append("req_b.text")
        if not pair.req_a.id or not pair.req_a.id.strip():
            missing_fields.append("req_a.id")
        if not pair.req_b.id or not pair.req_b.id.strip():
            missing_fields.append("req_b.id")
        if pair.req_a.system is None or not pair.req_a.system.strip():
            missing_fields.append("req_a.system")
        if pair.req_b.system is None or not pair.req_b.system.strip():
            missing_fields.append("req_b.system")

        if missing_fields or ReviewReasonCode.MISSING_METADATA.value in structured_codes:
            detail_suffix = f" ({', '.join(missing_fields)})" if missing_fields else ""
            add_reason(
                ReviewReasonCode.MISSING_METADATA,
                f"Missing requirement text or essential metadata{detail_suffix}.",
            )

        # 7. Short Requirement Screening
        minimum_length = config.minimum_requirement_characters
        if minimum_length:
            short_requirements = [
                (label, len(requirement.text.strip()))
                for label, requirement in (("A", pair.req_a), ("B", pair.req_b))
                if requirement.text and len(requirement.text.strip()) < minimum_length
            ]
            if short_requirements:
                short_details = ", ".join(
                    f"Requirement {label} ({length} character{'s' if length != 1 else ''})"
                    for label, length in short_requirements
                )
                add_reason(
                    ReviewReasonCode.SHORT_REQUIREMENT,
                    f"{short_details} {'is' if len(short_requirements) == 1 else 'are'} "
                    f"below the configured review minimum of "
                    f"{minimum_length} characters; inspect for incomplete or ambiguous text.",
                )

        # 8. Explicit Ambiguity / Review Flags
        if (
            (prov and prov.explicit_review_flag)
            or ReviewReasonCode.EXPLICIT_FLAG.value in structured_codes
        ):
            add_reason(
                ReviewReasonCode.EXPLICIT_FLAG,
                "Explicitly flagged as ambiguous or requiring human expert inspection.",
            )

        # Filter against config policy
        filtered_reasons: List[ReviewReasonCode] = []
        for r in reasons:
            if r == ReviewReasonCode.DISAGREEMENT and config.include_disagreements:
                filtered_reasons.append(r)
            elif r == ReviewReasonCode.BOTH_ABSTAINED and config.include_both_abstained:
                filtered_reasons.append(r)
            elif r == ReviewReasonCode.LLM_FAILURE and config.include_failures:
                filtered_reasons.append(r)
            elif r in (ReviewReasonCode.HEURISTIC_ONLY, ReviewReasonCode.LLM_ONLY) and config.include_single_method:
                filtered_reasons.append(r)
            elif r == ReviewReasonCode.LOW_CONFIDENCE and config.include_low_confidence:
                filtered_reasons.append(r)
            elif r == ReviewReasonCode.SHORT_REQUIREMENT:
                filtered_reasons.append(r)
            elif r == ReviewReasonCode.MISSING_METADATA and config.include_missing_metadata:
                filtered_reasons.append(r)
            elif r == ReviewReasonCode.EXPLICIT_FLAG and config.include_explicit_flags:
                filtered_reasons.append(r)

        if not filtered_reasons:
            return False, [], ReviewPriority.INFO.value, ""

        # Determine priority ranking based on highest-severity reason
        if ReviewReasonCode.DISAGREEMENT in filtered_reasons:
            priority = ReviewPriority.CRITICAL
        elif ReviewReasonCode.LLM_FAILURE in filtered_reasons:
            priority = ReviewPriority.HIGH
        elif any(
            reason in filtered_reasons
            for reason in (
                ReviewReasonCode.BOTH_ABSTAINED,
                ReviewReasonCode.MISSING_METADATA,
                ReviewReasonCode.SHORT_REQUIREMENT,
            )
        ):
            priority = ReviewPriority.MEDIUM
        elif any(
            reason in filtered_reasons
            for reason in (
                ReviewReasonCode.HEURISTIC_ONLY,
                ReviewReasonCode.LLM_ONLY,
                ReviewReasonCode.LOW_CONFIDENCE,
            )
        ):
            priority = ReviewPriority.LOW
        else:
            priority = ReviewPriority.INFO

        explanation = " | ".join(explanation_parts)
        return True, filtered_reasons, priority.value, explanation

    def create_review_item(self, pair: RequirementPair) -> Optional[ReviewItem]:
        """Convert a RequirementPair to a ReviewItem if eligible under queue policy."""
        is_eligible, reasons, priority, explanation = self.determine_eligibility(pair, self.config)
        if not is_eligible:
            return None

        req_a_id = pair.req_a.id or "A"
        req_b_id = pair.req_b.id or "B"
        queue_id = self.generate_item_id(pair.id, req_a_id, req_b_id)

        h_pred, h_ev, l_pred, l_rat, l_conf = self._extract_structured_predictions(pair)
        prov = pair.provenance

        req_a_system = getattr(pair.req_a, "system", None)
        req_b_system = getattr(pair.req_b, "system", None)

        return ReviewItem(
            queue_id=queue_id,
            pair_id=pair.id,
            req_a_id=req_a_id,
            req_b_id=req_b_id,
            req_a_text=pair.req_a.text,
            req_b_text=pair.req_b.text,
            system_name=req_a_system or req_b_system,
            req_a_system=req_a_system,
            req_b_system=req_b_system,
            req_a_component=pair.req_a.component,
            req_b_component=pair.req_b.component,
            req_a_subsystem=pair.req_a.subsystem,
            req_b_subsystem=pair.req_b.subsystem,
            req_a_module=pair.req_a.module,
            req_b_module=pair.req_b.module,
            req_a_source=pair.req_a.source,
            req_b_source=pair.req_b.source,
            req_a_source_location=pair.req_a.source_location,
            req_b_source_location=pair.req_b.source_location,
            req_a_metadata=pair.req_a.metadata,
            req_b_metadata=pair.req_b.metadata,
            synthetic=is_synthetic(pair),
            source_identity=synthetic_source_identity(pair),
            orientation=(req_a_id, req_b_id),
            heuristic_prediction=h_pred,
            heuristic_evidence=h_ev,
            llm_prediction=l_pred,
            llm_rationale=l_rat,
            llm_confidence=l_conf,
            proposed_label=pair.dependency_type,
            provenance=prov,
            review_reasons=reasons,
            review_notes=explanation,
            status=QueueStatus.PENDING,
            priority=priority,
        )

    def build_queue(self, pairs: List[RequirementPair]) -> List[ReviewItem]:
        """Build a prioritized review queue from a list of requirement pairs."""
        items: List[ReviewItem] = []
        for pair in pairs:
            item = self.create_review_item(pair)
            if item:
                items.append(item)

        items.sort(key=lambda x: (x.priority, x.pair_id, x.queue_id))
        logger.info(f"Built review queue with {len(items)} eligible items from {len(pairs)} input pairs.")
        return items

    def refresh_queue(
        self, existing_items: List[ReviewItem], new_pairs: List[RequirementPair]
    ) -> List[ReviewItem]:
        """Idempotently refresh review queue with new candidate pairs.

        Guarantees that previously recorded human decisions are preserved and not reset to PENDING.
        """
        existing_map = {item.queue_id: item for item in existing_items}
        new_queue = self.build_queue(new_pairs)

        refreshed: List[ReviewItem] = []
        seen_ids = set()

        for item in new_queue:
            seen_ids.add(item.queue_id)
            if item.queue_id in existing_map:
                prev_item = existing_map[item.queue_id]
                if prev_item.status != QueueStatus.PENDING or prev_item.human_decision is not None:
                    refreshed.append(prev_item)
                else:
                    refreshed.append(item)
            else:
                refreshed.append(item)

        for q_id, prev_item in existing_map.items():
            if q_id not in seen_ids and (prev_item.status != QueueStatus.PENDING or prev_item.human_decision is not None):
                refreshed.append(prev_item)

        refreshed.sort(key=lambda x: (x.priority, x.pair_id, x.queue_id))
        logger.info(f"Refreshed review queue: {len(refreshed)} items total (preserved {len(existing_items)} existing).")
        return refreshed

    @classmethod
    def record_decision(
        cls,
        item: ReviewItem,
        decision_type: QueueStatus,
        final_label: Optional[Union[DependencyType, str]] = None,
        reviewer_id: Optional[str] = None,
        reviewer_notes: str = "",
    ) -> Tuple[ReviewItem, RequirementPair]:
        """Record a human decision on a review item and produce an updated RequirementPair.

        GROUND-TRUTH PROTECTION:
        - Only explicit human ACCEPTED or CORRECTED decisions produce VerificationStatus.VERIFIED.
        - DEFERRED produces VerificationStatus.PENDING_REVIEW (unlabeled).
        - REJECTED produces VerificationStatus.REJECTED (unlabeled).
        - Raw heuristic and LLM predictions, confidence, rationale, and human decision are preserved.
        """
        if not reviewer_id or not reviewer_id.strip():
            raise ValueError("A nonblank reviewer_id is required for a human decision.")
        reviewer_id = reviewer_id.strip()
        if reviewer_id.casefold() in {"human_reviewer_1", "human_cli"}:
            raise ValueError("A generic placeholder cannot identify a human reviewer.")

        if decision_type not in (
            QueueStatus.ACCEPTED,
            QueueStatus.CORRECTED,
            QueueStatus.REJECTED,
            QueueStatus.DEFERRED,
            QueueStatus.REVIEWED,
        ):
            raise ValueError(f"Invalid decision status '{decision_type}'.")

        parsed_label: Optional[DependencyType] = None
        if final_label is not None:
            parsed_label = DependencyType.normalize(final_label) if isinstance(final_label, str) else final_label

        if decision_type == QueueStatus.ACCEPTED:
            parsed_label = parsed_label or item.proposed_label or item.llm_prediction or item.heuristic_prediction
            if parsed_label is None:
                raise ValueError("Cannot ACCEPT an item with no proposed or specified final label.")

        elif decision_type == QueueStatus.CORRECTED:
            if parsed_label is None:
                raise ValueError("CORRECTED decision requires specifying a valid non-None final_label.")

        is_verified_decision = decision_type in (QueueStatus.ACCEPTED, QueueStatus.CORRECTED)

        decision = HumanDecision(
            item_id=item.queue_id,
            pair_id=item.pair_id,
            decision_type=decision_type,
            final_label=parsed_label if is_verified_decision else None,
            reviewer_id=reviewer_id,
            reviewer_notes=reviewer_notes,
            previous_label=item.proposed_label,
        )

        item.status = decision_type
        item.human_decision = decision
        item.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if is_verified_decision:
            new_status = VerificationStatus.VERIFIED
        elif decision_type == QueueStatus.DEFERRED:
            new_status = VerificationStatus.PENDING_REVIEW
        elif decision_type == QueueStatus.REVIEWED:
            new_status = VerificationStatus.HUMAN_REVIEWED
        else:
            new_status = VerificationStatus.REJECTED

        orig_prov = item.provenance
        verified_prov = AnnotationProvenance(
            source=AnnotationSource.HUMAN,
            status=new_status,
            annotator_id=reviewer_id,
            model_id=orig_prov.model_id if orig_prov else None,
            created_at=(orig_prov.created_at if orig_prov and orig_prov.created_at else datetime.datetime.now(datetime.timezone.utc).isoformat()),
            reviewed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            reviewer_id=reviewer_id,
            predicted_label=item.proposed_label or item.llm_prediction or item.heuristic_prediction,
            accepted_label=parsed_label if new_status == VerificationStatus.VERIFIED else None,
            confidence_score=item.llm_confidence,
            evidence=f"Human Reviewer ({reviewer_id}) decision: {decision_type.value.upper()}. Notes: {reviewer_notes}",
            review_notes=reviewer_notes,
            dataset_version="human-verified-v1.0" if new_status == VerificationStatus.VERIFIED else "human-reviewed-unverified-v1.0",
            heuristic_label=item.heuristic_prediction,
            heuristic_rule_id=orig_prov.heuristic_rule_id if orig_prov else None,
            heuristic_confidence=orig_prov.heuristic_confidence if orig_prov else None,
            heuristic_evidence=item.heuristic_evidence or (orig_prov.heuristic_evidence if orig_prov else None),
            llm_label=item.llm_prediction,
            llm_rationale=item.llm_rationale or (orig_prov.llm_rationale if orig_prov else None),
            llm_raw_response=orig_prov.llm_raw_response if orig_prov else None,
            llm_failed=orig_prov.llm_failed if orig_prov else False,
            llm_error=orig_prov.llm_error if orig_prov else None,
            explicit_review_flag=orig_prov.explicit_review_flag if orig_prov else False,
            structured_review_reasons=[r.value for r in item.review_reasons],
            synthetic=item.synthetic,
            source_identity=item.source_identity,
        )

        verified_pair = RequirementPair(
            id=item.pair_id,
            req_a=Requirement(
                id=item.req_a_id,
                text=item.req_a_text,
                system=item.req_a_system if item.req_a_system is not None else (item.system_name or "GenericSystem"),
                component=item.req_a_component,
                subsystem=item.req_a_subsystem,
                module=item.req_a_module,
                source=item.req_a_source,
                source_location=item.req_a_source_location,
                metadata=item.req_a_metadata,
                synthetic=item.synthetic,
                source_identity=item.source_identity,
            ),
            req_b=Requirement(
                id=item.req_b_id,
                text=item.req_b_text,
                system=item.req_b_system if item.req_b_system is not None else (item.system_name or "GenericSystem"),
                component=item.req_b_component,
                subsystem=item.req_b_subsystem,
                module=item.req_b_module,
                source=item.req_b_source,
                source_location=item.req_b_source_location,
                metadata=item.req_b_metadata,
                synthetic=item.synthetic,
                source_identity=item.source_identity,
            ),
            dependency_type=parsed_label if new_status == VerificationStatus.VERIFIED else None,
            provenance=verified_prov,
            synthetic=item.synthetic,
            source_identity=item.source_identity,
        )

        logger.info(f"Recorded decision {decision_type.value} for item {item.queue_id} by {reviewer_id}.")
        return item, verified_pair

    @staticmethod
    def filter_queue(
        items: List[ReviewItem],
        status: Optional[QueueStatus] = None,
        reason: Optional[ReviewReasonCode] = None,
        system_name: Optional[str] = None,
        priority: Optional[int] = None,
    ) -> List[ReviewItem]:
        """Filter review items by status, reason, system name, or priority."""
        filtered = items
        if status:
            filtered = [i for i in filtered if i.status == status]
        if reason:
            filtered = [i for i in filtered if reason in i.review_reasons]
        if system_name:
            filtered = [i for i in filtered if (i.system_name or "").lower() == system_name.lower()]
        if priority:
            filtered = [i for i in filtered if i.priority == priority]
        return filtered

    @staticmethod
    def create_manifest(
        items: List[ReviewItem],
        source_identity: str = "Provisional_Dataset",
        config: Optional[ReviewQueueConfig] = None,
    ) -> ReviewQueueManifest:
        """Create summary manifest audit record for the review queue."""
        reasons_breakdown: Dict[str, int] = {}
        for item in items:
            for r in item.review_reasons:
                reasons_breakdown[r.value] = reasons_breakdown.get(r.value, 0) + 1

        return ReviewQueueManifest(
            source_identity=source_identity,
            total_items=len(items),
            pending_count=sum(1 for i in items if i.status == QueueStatus.PENDING),
            accepted_count=sum(1 for i in items if i.status == QueueStatus.ACCEPTED),
            corrected_count=sum(1 for i in items if i.status == QueueStatus.CORRECTED),
            rejected_count=sum(1 for i in items if i.status == QueueStatus.REJECTED),
            deferred_count=sum(1 for i in items if i.status == QueueStatus.DEFERRED),
            reasons_breakdown=reasons_breakdown,
            eligibility_config=config or ReviewQueueConfig(),
        )

    @staticmethod
    def save_queue(
        items: List[ReviewItem],
        manifest: ReviewQueueManifest,
        output_queue_file: str,
        output_manifest_file: Optional[str] = None,
    ) -> None:
        """Save review queue items and manifest via same-directory atomic replacements."""
        if output_manifest_file is None:
            base, _ = os.path.splitext(output_queue_file)
            output_manifest_file = f"{base}_manifest.json"

        outputs = (
            (output_queue_file, [i.model_dump(mode="json") for i in items]),
            (output_manifest_file, manifest.model_dump(mode="json")),
        )
        temporary_files: List[Tuple[str, str]] = []
        try:
            for output_path, payload in outputs:
                output_path = os.fspath(output_path)
                parent = os.path.dirname(os.path.abspath(output_path))
                os.makedirs(parent, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=parent,
                    prefix=f".{os.path.basename(output_path)}.",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temporary_files.append((temp_file.name, output_path))
                    json.dump(payload, temp_file, indent=2)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())

            for temporary_path, output_path in temporary_files:
                os.replace(temporary_path, output_path)

            logger.info(f"Saved {len(items)} review items to '{output_queue_file}'.")
        finally:
            for temporary_path, _ in temporary_files:
                if os.path.exists(temporary_path):
                    try:
                        os.unlink(temporary_path)
                    except OSError as exc:
                        logger.warning(f"Could not remove temporary queue file '{temporary_path}': {exc}")

    @staticmethod
    def load_queue(queue_file: str) -> List[ReviewItem]:
        """Load review queue items from a JSON file."""
        if not os.path.exists(queue_file):
            raise FileNotFoundError(f"Review queue file '{queue_file}' not found.")

        with open(queue_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [ReviewItem.model_validate(item) for item in data]

    def export_verified_pairs(
        self,
        allow_synthetic: bool = False,
    ) -> List[RequirementPair]:
        """Export human-verified RequirementPairs from items with ACCEPTED or CORRECTED decisions.

        If allow_synthetic is False (default), synthetic test items are strictly excluded from
        the exported verified dataset. Deferred, rejected, pending, and generic reviewed items
        can never be exported as verified ground truth.
        """
        verified_pairs: List[RequirementPair] = []
        excluded_synthetic_count = 0

        for item in self.items:
            if (
                item.status in (QueueStatus.ACCEPTED, QueueStatus.CORRECTED)
                and item.human_decision is not None
                and item.human_decision.final_label is not None
            ):
                if (item.synthetic or is_synthetic(item)) and not allow_synthetic:
                    excluded_synthetic_count += 1
                    continue
                _, pair = self.record_decision(
                    item,
                    item.status,
                    final_label=item.human_decision.final_label,
                    reviewer_id=item.human_decision.reviewer_id,
                    reviewer_notes=item.human_decision.reviewer_notes
                )
                verified_pairs.append(pair)

        if excluded_synthetic_count > 0:
            logger.warning(
                f"Excluded {excluded_synthetic_count} synthetic review items from verified export (allow_synthetic=False)."
            )

        return verified_pairs
