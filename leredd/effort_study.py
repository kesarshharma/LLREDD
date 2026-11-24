"""Counterbalanced, descriptive manual-versus-assisted annotation effort study."""

from __future__ import annotations

import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from leredd.data_models import (
    AnnotationSource,
    DependencyType,
    RequirementPair,
    VerificationStatus,
)
from leredd.independent_annotation import (
    load_candidate_pairs,
    validate_reviewer_id,
)


EFFORT_STUDY_SCHEMA_VERSION = "1.0"
EffortCondition = Literal["manual", "assisted"]


def ensure_effort_output_path(path: str | Path, *, allow_existing: bool = False) -> Path:
    """Keep study artifacts outside project data trees and preserve existing files."""
    target = Path(path).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    data_root = (project_root / "data").resolve()
    if target == data_root or data_root in target.parents:
        raise ValueError(f"Refusing to write annotation-effort study output under project data: {data_root}")
    if target.exists() and not allow_existing:
        raise ValueError("Refusing to overwrite an existing effort-study artifact.")
    return target


class EffortStudyAssignment(BaseModel):
    pair_id: str = Field(min_length=1)
    manual_reviewer: str = Field(min_length=1)
    assisted_reviewer: str = Field(min_length=1)

    @field_validator("manual_reviewer", "assisted_reviewer")
    @classmethod
    def non_placeholder_reviewer(cls, value: str) -> str:
        return validate_reviewer_id(value)

    @model_validator(mode="after")
    def different_reviewers_per_pair(self):
        if self.manual_reviewer == self.assisted_reviewer:
            raise ValueError("Each pair must be assigned to different manual and assisted reviewers.")
        return self


class EffortStudySchedule(BaseModel):
    schema_version: str = EFFORT_STUDY_SCHEMA_VERSION
    study_id: str = Field(min_length=1)
    seed: int
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reviewer_ids: list[str] = Field(min_length=2, max_length=2)
    assignments: list[EffortStudyAssignment] = Field(min_length=1)
    created_at: str

    @field_validator("reviewer_ids")
    @classmethod
    def distinct_real_reviewers(cls, values: list[str]) -> list[str]:
        normalized = [validate_reviewer_id(value) for value in values]
        if len(set(normalized)) != 2:
            raise ValueError("The effort study requires two distinct reviewer IDs.")
        return normalized

    @field_validator("assignments")
    @classmethod
    def unique_pair_assignments(cls, values: list[EffortStudyAssignment]):
        pair_ids = [assignment.pair_id for assignment in values]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("The effort study schedule contains duplicate pair IDs.")
        return values

    @model_validator(mode="after")
    def assignment_reviewers_match(self):
        allowed = set(self.reviewer_ids)
        if any(
            assignment.manual_reviewer not in allowed
            or assignment.assisted_reviewer not in allowed
            for assignment in self.assignments
        ):
            raise ValueError("Every assignment must use the two declared reviewer IDs.")
        for reviewer_id in self.reviewer_ids:
            manual_count = sum(item.manual_reviewer == reviewer_id for item in self.assignments)
            assisted_count = sum(item.assisted_reviewer == reviewer_id for item in self.assignments)
            if abs(manual_count - assisted_count) > 1:
                raise ValueError("The schedule must balance manual and assisted conditions per reviewer.")
        return self


class EffortStudyRecord(BaseModel):
    pair_id: str = Field(min_length=1)
    condition: EffortCondition
    reviewer_id: str = Field(min_length=1)
    label: DependencyType | None
    elapsed_seconds: float = Field(ge=0.0)
    rationale: str = ""
    recorded_at: str

    @field_validator("reviewer_id")
    @classmethod
    def non_placeholder_reviewer(cls, value: str) -> str:
        return validate_reviewer_id(value)


class EffortStudyManifest(BaseModel):
    schema_version: str = EFFORT_STUDY_SCHEMA_VERSION
    study_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    schedule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(min_length=1)
    status: Literal["in_progress", "complete"] = "in_progress"
    created_at: str
    updated_at: str
    records: list[EffortStudyRecord] = Field(default_factory=list)

    @field_validator("reviewer_id")
    @classmethod
    def non_placeholder_reviewer(cls, value: str) -> str:
        return validate_reviewer_id(value)

    @field_validator("records")
    @classmethod
    def unique_pair_records(cls, values: list[EffortStudyRecord]):
        pair_ids = [record.pair_id for record in values]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("An effort-study manifest contains duplicate pair IDs.")
        return values

    @model_validator(mode="after")
    def record_reviewer_matches_manifest(self):
        if any(record.reviewer_id != self.reviewer_id for record in self.records):
            raise ValueError("Record reviewer IDs must match the manifest reviewer ID.")
        return self


def validate_assisted_candidates(pairs: list[RequirementPair]) -> None:
    """Require provisional machine suggestions; reject human/imported labels."""
    if not pairs:
        raise ValueError("The effort study candidate file is empty.")
    allowed_sources = {
        AnnotationSource.LLM,
        AnnotationSource.HEURISTIC,
        AnnotationSource.ENSEMBLE,
    }
    for pair in pairs:
        provenance = pair.provenance
        if (
            provenance is None
            or provenance.status != VerificationStatus.PROVISIONAL
            or provenance.source not in allowed_sources
        ):
            raise ValueError(
                f"Pair {pair.id} must have provisional machine-label provenance; "
                "human, imported, verified, and unprovenanced records are not study inputs."
            )
        suggestion = provenance.accepted_label or provenance.predicted_label or pair.dependency_type
        if suggestion is None:
            raise ValueError(f"Pair {pair.id} has no provisional suggestion for the assisted condition.")


def create_effort_schedule(
    pairs: list[RequirementPair],
    reviewer_a: str,
    reviewer_b: str,
    *,
    seed: int,
    source_sha256: str,
    context_sha256: str | None,
    study_id: str,
) -> EffortStudySchedule:
    """Assign each pair to one manual and one assisted reviewer with balanced arms."""
    validate_assisted_candidates(pairs)
    reviewer_a = validate_reviewer_id(reviewer_a)
    reviewer_b = validate_reviewer_id(reviewer_b)
    if reviewer_a == reviewer_b:
        raise ValueError("The effort study requires two distinct reviewers.")
    if seed < 0:
        raise ValueError("The randomization seed must be nonnegative.")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    assignments = [
        EffortStudyAssignment(
            pair_id=pair.id,
            manual_reviewer=reviewer_a if index % 2 == 0 else reviewer_b,
            assisted_reviewer=reviewer_b if index % 2 == 0 else reviewer_a,
        )
        for index, pair in enumerate(shuffled)
    ]
    return EffortStudySchedule(
        study_id=study_id,
        seed=seed,
        source_sha256=source_sha256,
        context_sha256=context_sha256,
        reviewer_ids=[reviewer_a, reviewer_b],
        assignments=assignments,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def assisted_suggestion(pair: RequirementPair) -> tuple[DependencyType, str, int | None]:
    provenance = pair.provenance
    if provenance is None:
        raise ValueError(f"Pair {pair.id} has no provisional suggestion.")
    label = provenance.accepted_label or provenance.predicted_label or pair.dependency_type
    if label is None:
        raise ValueError(f"Pair {pair.id} has no provisional suggestion.")
    return label, provenance.evidence or "No supporting rationale was recorded.", provenance.confidence_score


def validate_effort_schedule_for_source(
    schedule: EffortStudySchedule,
    pairs: list[RequirementPair],
    *,
    source_sha256: str,
    context_sha256: str | None,
) -> None:
    if schedule.schema_version != EFFORT_STUDY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported effort-study schedule schema: {schedule.schema_version}")
    validate_assisted_candidates(pairs)
    if schedule.source_sha256 != source_sha256:
        raise ValueError("The schedule source hash does not match the candidate file.")
    if schedule.context_sha256 != context_sha256:
        raise ValueError("The schedule context hash does not match the supplied context file.")
    if {assignment.pair_id for assignment in schedule.assignments} != {pair.id for pair in pairs}:
        raise ValueError("The schedule must contain each candidate pair exactly once.")


def compare_effort_manifests(
    manual_and_assisted_a: EffortStudyManifest,
    manual_and_assisted_b: EffortStudyManifest,
    schedule: EffortStudySchedule,
) -> dict[str, Any]:
    manifests = [manual_and_assisted_a, manual_and_assisted_b]
    if any(manifest.status != "complete" for manifest in manifests):
        raise ValueError("Both effort-study manifests must be complete.")
    if manifests[0].reviewer_id == manifests[1].reviewer_id:
        raise ValueError("Effort comparison requires two distinct reviewers.")
    if any(
        manifest.study_id != schedule.study_id
        or manifest.source_sha256 != schedule.source_sha256
        or manifest.context_sha256 != schedule.context_sha256
        for manifest in manifests
    ):
        raise ValueError("Effort manifests and schedule must refer to the same study/source/context.")
    if not manifests[0].schedule_sha256 or any(
        manifest.schedule_sha256 != manifests[0].schedule_sha256
        for manifest in manifests
    ):
        raise ValueError("Both effort manifests must reference the same nonempty schedule hash.")

    records_by_reviewer = {
        manifest.reviewer_id: {record.pair_id: record for record in manifest.records}
        for manifest in manifests
    }
    if not records_by_reviewer[manifests[0].reviewer_id] or (
        records_by_reviewer[manifests[0].reviewer_id].keys()
        != records_by_reviewer[manifests[1].reviewer_id].keys()
    ):
        raise ValueError("Both reviewers must complete the same nonempty pair sample.")
    reviewer_ids = {manifest.reviewer_id for manifest in manifests}
    if reviewer_ids != set(schedule.reviewer_ids):
        raise ValueError("Effort manifests must belong to the two reviewers named in the schedule.")

    rows: dict[str, list[tuple[str, EffortStudyRecord]]] = {"manual": [], "assisted": []}
    pair_times: list[dict[str, Any]] = []
    assignments = {assignment.pair_id: assignment for assignment in schedule.assignments}
    expected_pair_ids = set(records_by_reviewer[manifests[0].reviewer_id])
    if expected_pair_ids != set(assignments):
        raise ValueError("Effort manifests must include exactly the scheduled pair sample.")

    for pair_id in sorted(expected_pair_ids):
        assignment = assignments[pair_id]
        pair_records: dict[str, EffortStudyRecord] = {}
        for reviewer_id, reviewer_records in records_by_reviewer.items():
            record = reviewer_records[pair_id]
            expected_condition = (
                "manual" if reviewer_id == assignment.manual_reviewer else "assisted"
            )
            if record.condition != expected_condition:
                raise ValueError(f"Pair {pair_id} condition does not match the randomized schedule.")
            if record.reviewer_id != reviewer_id:
                raise ValueError("An effort record reviewer ID does not match its manifest.")
            rows[record.condition].append((reviewer_id, record))
            pair_records[record.condition] = record
        if set(pair_records) != {"manual", "assisted"}:
            raise ValueError(f"Pair {pair_id} must have one record from each condition.")
        pair_times.append({
            "pair_id": pair_id,
            "manual_seconds": pair_records["manual"].elapsed_seconds,
            "assisted_seconds": pair_records["assisted"].elapsed_seconds,
            "manual_minus_assisted_seconds": round(
                pair_records["manual"].elapsed_seconds - pair_records["assisted"].elapsed_seconds,
                3,
            ),
        })

    summaries: dict[str, dict[str, Any]] = {}
    for condition, condition_rows in rows.items():
        durations = [record.elapsed_seconds for _, record in condition_rows]
        reviewer_counts = {
            reviewer_id: sum(owner == reviewer_id for owner, _ in condition_rows)
            for reviewer_id in sorted(reviewer_ids)
        }
        summaries[condition] = {
            "annotation_count": len(durations),
            "total_seconds": round(sum(durations), 3),
            "mean_seconds_per_pair": round(statistics.mean(durations), 3),
            "median_seconds_per_pair": round(statistics.median(durations), 3),
            "reviewer_annotation_counts": reviewer_counts,
        }
    manual_durations = [record.elapsed_seconds for _, record in rows["manual"]]
    assisted_durations = [record.elapsed_seconds for _, record in rows["assisted"]]
    manual_mean = statistics.mean(manual_durations)
    assisted_mean = statistics.mean(assisted_durations)
    mean_difference = manual_mean - assisted_mean
    reduction = (100.0 * mean_difference / manual_mean) if manual_mean > 0 else None
    paired_differences = [row["manual_minus_assisted_seconds"] for row in pair_times]
    label_disagreements = sum(
        records_by_reviewer[assignment.manual_reviewer][pair_id].label
        != records_by_reviewer[assignment.assisted_reviewer][pair_id].label
        for pair_id, assignment in assignments.items()
    )

    return {
        "report_type": "annotation_effort_study",
        "study_id": schedule.study_id,
        "source_sha256": schedule.source_sha256,
        "context_sha256": schedule.context_sha256,
        "schedule_seed": schedule.seed,
        "reviewer_ids": schedule.reviewer_ids,
        "pair_count": len(expected_pair_ids),
        "conditions": summaries,
        "mean_manual_minus_assisted_seconds": round(mean_difference, 3),
        "mean_relative_time_reduction_percent": round(reduction, 3) if reduction is not None else None,
        "mean_pairwise_manual_minus_assisted_seconds": round(statistics.mean(paired_differences), 3),
        "manual_assisted_label_disagreements": label_disagreements,
        "per_pair_times": pair_times,
        "labels_are_ground_truth": False,
        "inferential_significance_tested": False,
        "note": (
            "Descriptive effort comparison only. Each pair was reviewed in both conditions by different "
            "reviewers under a counterbalanced schedule. The sample is small, reviewer identities are "
            "self-reported, timing may include pauses, and this report does not establish annotation quality, "
            "statistical significance, or ground truth."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_effort_study_input(path: str):
    """Load candidates and verify the provisional machine labels required for assisted review."""
    pairs = load_candidate_pairs(path)
    validate_assisted_candidates(pairs)
    return pairs
