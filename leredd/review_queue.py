"""Multi-Method Disagreement and Uncertainty Review Queue module (Phase 5, TASK-P5-01).

Implements a persistent, auditable human-review queue for provisional requirement dependency annotations:
1. ReviewReasonCode & QueueStatus enums.
2. ReviewItem & HumanDecision Pydantic models.
3. ReviewQueueManager: Handles eligibility filtering, deterministic prioritization, idempotent queue refresh, and human decision recording.
4. Ground-Truth Protection: Machine predictions remain provisional until explicitly verified by a human reviewer.

CORE PRINCIPLE: Machine predictions NEVER silently become verified annotations. Only explicit human decisions produce VerificationStatus.VERIFIED records.
"""

import datetime
import hashlib
import json
import os
import tempfile
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from leredd.config import LEREDDConfig, get_config
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
    HEURISTIC_ONLY = "heuristic_only"  # Heuristic predicted, LLM abstained/failed
    LLM_ONLY = "llm_only"  # LLM predicted, Heuristic abstained
    LOW_CONFIDENCE = "low_confidence"  # Self-reported confidence at or below the configured cutoff
    SHORT_REQUIREMENT = "short_requirement"  # Requirement text is below the configured minimum length
    MISSING_METADATA = "missing_metadata"  # Requirement or provenance metadata incomplete
    EXPLICIT_FLAG = "explicit_flag"  # Explicitly flagged for human review


class QueueStatus(str, Enum):
    """Lifecycle status of an item in the review queue."""
    PENDING = "pending"  # Awaiting human review
    IN_REVIEW = "in_review"  # Currently assigned or under review
    ACCEPTED = "accepted"  # Proposed provisional label accepted as-is by reviewer
    CORRECTED = "corrected"  # Proposed label corrected to a different valid class by reviewer
    REJECTED = "rejected"  # Candidate pair rejected as invalid/unusable
    DEFERRED = "deferred"  # Decision deferred / unresolved pending more evidence
    REVIEWED = "reviewed"  # Generic completed review state


class ReviewPriority(int, Enum):
    """Deterministic priority levels for review items (1 = Highest, 5 = Lowest)."""
    CRITICAL = 1  # Disagreement between methods
    HIGH = 2      # LLM or execution failures
    MEDIUM = 3    # Missing metadata, short requirement text, or both abstained
    LOW = 4       # Single-method predictions or low confidence
    INFO = 5      # Other explicitly flagged items


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
    req_a_source: Optional[str] = Field(default=None, description="Requirement A source document")
    req_b_source: Optional[str] = Field(default=None, description="Requirement B source document")
    req_a_source_location: Optional[str] = Field(default=None, description="Requirement A source location")
    req_b_source_location: Optional[str] = Field(default=None, description="Requirement B source location")
    synthetic: bool = Field(default=False, description="Whether this review item is synthetic/demo data")
    source_identity: Optional[str] = Field(default=None, description="Source identity for this review item")
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
    reasons_breakdown: Dict[str, int] = Field(default_factory=dict, description="Item counts by primary review reason")
    eligibility_config: ReviewQueueConfig = Field(
        default_factory=ReviewQueueConfig,
        description="Eligibility thresholds and switches used to build this queue.",
    )
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class ReviewQueueManager:
    """Manages the creation, persistence, filtering, prioritization, and human decision recording for review queues."""

    def __init__(self, config: Optional[ReviewQueueConfig] = None):
        self.config = config or ReviewQueueConfig()

    @staticmethod
    def generate_item_id(pair_id: str, req_a_id: str, req_b_id: str) -> str:
        """Generate a deterministic, stable queue item ID preserving directional orientation."""
        content = f"{pair_id}:{req_a_id}->{req_b_id}"
        hash_str = hashlib.md5(content.encode("utf-8")).hexdigest()[:10].upper()
        return f"QITEM-{hash_str}"

    @classmethod
    def determine_eligibility(
        cls, pair: RequirementPair, config: ReviewQueueConfig
    ) -> Tuple[bool, List[ReviewReasonCode], int, str]:
        """Determine whether a requirement pair is eligible for human review.

        Returns Tuple of (is_eligible, review_reasons, priority_level, explanation).
        """
        reasons: List[ReviewReasonCode] = []
        explanation_parts: List[str] = []
        priority = ReviewPriority.INFO

        prov = pair.provenance
        h_label: Optional[DependencyType] = None
        l_label: Optional[DependencyType] = None
        l_conf: Optional[int] = None
        l_rat: str = ""
        h_ev: str = ""

        if prov:
            l_conf = prov.confidence_score
            # Extract raw predictions if available
            if prov.predicted_label:
                # If source is LLM, predicted_label came from LLM
                if prov.source == AnnotationSource.LLM:
                    l_label = prov.predicted_label
                elif prov.source == AnnotationSource.HEURISTIC:
                    h_label = prov.predicted_label

            # Parse evidence string for heuristics/LLM info
            ev_str = prov.evidence or ""
            if "LLM Rationale:" in ev_str:
                l_rat = ev_str.split("LLM Rationale:")[-1].strip()
            if "Heuristic [" in ev_str:
                h_ev = ev_str.split("Heuristic [")[-1].split("]")[0]

            if "DISAGREEMENT" in (prov.review_notes or ""):
                # Heuristic and LLM had conflicting predictions
                reasons.append(ReviewReasonCode.DISAGREEMENT)
                explanation_parts.append(prov.review_notes)
                # Extract conflicting labels from review notes if possible
                if "'" in prov.review_notes:
                    parts = prov.review_notes.split("'")
                    if len(parts) >= 4:
                        h_label = DependencyType.normalize(parts[1])
                        l_label = DependencyType.normalize(parts[3])

            elif "UNRESOLVED" in (prov.review_notes or ""):
                reasons.append(ReviewReasonCode.BOTH_ABSTAINED)
                explanation_parts.append("Both Heuristic and LLM abstained or failed to produce a label.")

            elif "FAILED" in (prov.review_notes or ""):
                reasons.append(ReviewReasonCode.LLM_FAILURE)
                explanation_parts.append(prov.review_notes)

        # Explicit check on pair contents if provenance didn't explicitly note disagreement
        if prov and prov.source == AnnotationSource.LLM and config.include_single_method:
            reasons.append(ReviewReasonCode.LLM_ONLY)
            explanation_parts.append("Single-method prediction from LLM (Heuristic abstained).")
        elif prov and prov.source == AnnotationSource.HEURISTIC and config.include_single_method:
            reasons.append(ReviewReasonCode.HEURISTIC_ONLY)
            explanation_parts.append("Single-method prediction from Heuristic (LLM abstained).")

        if l_conf is not None and l_conf <= config.confidence_threshold and config.include_low_confidence:
            reasons.append(ReviewReasonCode.LOW_CONFIDENCE)
            explanation_parts.append(
                f"LLM confidence score {l_conf}/5 is at or below review cutoff {config.confidence_threshold}."
            )

        # Required metadata check
        if not pair.req_a.text or not pair.req_b.text:
            reasons.append(ReviewReasonCode.MISSING_METADATA)
            explanation_parts.append("Missing requirement text or essential metadata.")

        minimum_length = config.minimum_requirement_characters
        if minimum_length:
            short_requirements = [
                (label, len(requirement.text.strip()))
                for label, requirement in (("A", pair.req_a), ("B", pair.req_b))
                if requirement.text and len(requirement.text.strip()) < minimum_length
            ]
            if short_requirements:
                reasons.append(ReviewReasonCode.SHORT_REQUIREMENT)
                short_details = ", ".join(
                    f"Requirement {label} ({length} character{'s' if length != 1 else ''})"
                    for label, length in short_requirements
                )
                explanation_parts.append(
                    f"{short_details} {'is' if len(short_requirements) == 1 else 'are'} "
                    f"below the configured review minimum of "
                    f"{minimum_length} characters; inspect for incomplete or ambiguous text."
                )

        # Filter against config policy
        filtered_reasons = []
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
            elif r in (ReviewReasonCode.MISSING_METADATA, ReviewReasonCode.EXPLICIT_FLAG):
                filtered_reasons.append(r)

        if not filtered_reasons:
            return False, [], ReviewPriority.INFO, ""

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
        else:
            priority = ReviewPriority.LOW

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

        prov = pair.provenance
        l_conf = prov.confidence_score if prov else None
        h_pred: Optional[DependencyType] = None
        l_pred: Optional[DependencyType] = None
        h_ev = ""
        l_rat = ""

        if prov:
            ev_str = prov.evidence or ""
            if "AGREEMENT" in ev_str:
                h_pred = pair.dependency_type
                l_pred = pair.dependency_type
            if prov.source == AnnotationSource.HEURISTIC:
                h_pred = pair.dependency_type or prov.predicted_label
            elif prov.source == AnnotationSource.LLM:
                l_pred = pair.dependency_type or prov.predicted_label

            if "DISAGREEMENT" in (prov.review_notes or "") and "'" in prov.review_notes:
                parts = prov.review_notes.split("'")
                if len(parts) >= 4:
                    h_pred = DependencyType.normalize(parts[1])
                    l_pred = DependencyType.normalize(parts[3])

            if "LLM Rationale:" in ev_str:
                l_rat = ev_str.split("LLM Rationale:")[-1].strip()
            if "Heuristic [" in ev_str:
                h_ev = ev_str

        req_a_system = getattr(pair.req_a, "system", getattr(pair.req_a, "system_name", None))
        req_b_system = getattr(pair.req_b, "system", getattr(pair.req_b, "system_name", None))

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
            req_a_source=pair.req_a.source,
            req_b_source=pair.req_b.source,
            req_a_source_location=pair.req_a.source_location,
            req_b_source_location=pair.req_b.source_location,
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

        # Deterministic sorting: priority rank (1=highest), then creation_at, then pair_id
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

        # 1. Update or retain existing items
        for item in new_queue:
            seen_ids.add(item.queue_id)
            if item.queue_id in existing_map:
                prev_item = existing_map[item.queue_id]
                # If item has been reviewed or decided, preserve its status and decision!
                if prev_item.status != QueueStatus.PENDING or prev_item.human_decision is not None:
                    refreshed.append(prev_item)
                else:
                    # Update pending item details
                    refreshed.append(item)
            else:
                refreshed.append(item)

        # 2. Retain any existing items that were already reviewed even if omitted from new_pairs
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
        """Record a human decision on a review item and produce a verified RequirementPair.

        GROUND-TRUTH PROTECTION:
        - Only explicit human decisions produce VerificationStatus.VERIFIED records.
        - Accepts/corrects distinguish label promotion.
        - Raw heuristic and LLM predictions are preserved in audit trail.
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

        decision = HumanDecision(
            item_id=item.queue_id,
            pair_id=item.pair_id,
            decision_type=decision_type,
            final_label=parsed_label if decision_type in (QueueStatus.ACCEPTED, QueueStatus.CORRECTED) else None,
            reviewer_id=reviewer_id,
            reviewer_notes=reviewer_notes,
            previous_label=item.proposed_label,
        )

        item.status = decision_type
        item.human_decision = decision
        item.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Produce verified RequirementPair
        new_status = VerificationStatus.VERIFIED if decision_type in (QueueStatus.ACCEPTED, QueueStatus.CORRECTED) else VerificationStatus.REJECTED
        if decision_type == QueueStatus.DEFERRED:
            new_status = VerificationStatus.PENDING_REVIEW

        verified_prov = AnnotationProvenance(
            source=AnnotationSource.HUMAN,
            status=new_status,
            annotator_id=reviewer_id,
            model_id=item.provenance.model_id if item.provenance else None,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            reviewed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            reviewer_id=reviewer_id,
            predicted_label=item.proposed_label or item.llm_prediction or item.heuristic_prediction,
            accepted_label=parsed_label if new_status == VerificationStatus.VERIFIED else None,
            confidence_score=item.llm_confidence,
            evidence=f"Human Reviewer ({reviewer_id}) decision: {decision_type.value.upper()}. Notes: {reviewer_notes}",
            review_notes=reviewer_notes,
            dataset_version="human-verified-v1.0",
            synthetic=item.synthetic,
            source_identity=item.source_identity,
        )

        verified_pair = RequirementPair(
            id=item.pair_id,
            req_a=Requirement(
                id=item.req_a_id,
                text=item.req_a_text,
                system=item.req_a_system or item.system_name or "GenericSystem",
                component=item.req_a_component,
                subsystem=item.req_a_subsystem,
                source=item.req_a_source,
                source_location=item.req_a_source_location,
                synthetic=item.synthetic,
                source_identity=item.source_identity,
            ),
            req_b=Requirement(
                id=item.req_b_id,
                text=item.req_b_text,
                system=item.req_b_system or item.system_name or "GenericSystem",
                component=item.req_b_component,
                subsystem=item.req_b_subsystem,
                source=item.req_b_source,
                source_location=item.req_b_source_location,
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
        """Save review queue items and manifest via same-directory atomic replacements.

        Each file is replaced atomically so a failed serialization or initial write
        cannot truncate an existing queue. The two output files cannot be committed
        as one filesystem transaction; the queue is replaced first and the manifest
        second so the queue remains the authoritative artifact if manifest writing
        fails.
        """
        if output_manifest_file is None:
            base, ext = os.path.splitext(output_queue_file)
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
