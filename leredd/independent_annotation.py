"""Blinded, independent human annotation and agreement-report helpers.

These records are kept separate from the assisted review queue. Completing an
independent annotation file does not verify or promote any dataset labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from leredd.data_models import DependencyType
from leredd.utils import load_pairs_json


ANNOTATION_SCHEMA_VERSION = "1.0"
_PLACEHOLDER_IDENTITIES = {"human_cli", "human_reviewer_1", "reviewer", "annotator"}


def validate_reviewer_id(reviewer_id: str) -> str:
    """Return a normalized reviewer ID or reject missing placeholder values."""
    normalized = reviewer_id.strip()
    if not normalized:
        raise ValueError("A nonblank reviewer ID is required.")
    if normalized.casefold() in _PLACEHOLDER_IDENTITIES:
        raise ValueError("A generic placeholder cannot identify an independent reviewer.")
    return normalized


class IndependentAnnotationRecord(BaseModel):
    pair_id: str = Field(min_length=1)
    label: DependencyType | None
    reviewer_id: str = Field(min_length=1)
    elapsed_seconds: float = Field(ge=0.0)
    recorded_at: str
    rationale: str = ""

    @field_validator("reviewer_id")
    @classmethod
    def non_placeholder_reviewer(cls, value: str) -> str:
        return validate_reviewer_id(value)


class IndependentAnnotationManifest(BaseModel):
    schema_version: str = ANNOTATION_SCHEMA_VERSION
    annotation_mode: str = "blind_independent"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(min_length=1)
    status: str = Field(default="in_progress", pattern=r"^(in_progress|complete)$")
    created_at: str
    updated_at: str
    records: list[IndependentAnnotationRecord] = Field(default_factory=list)

    @field_validator("reviewer_id")
    @classmethod
    def non_placeholder_reviewer(cls, value: str) -> str:
        return validate_reviewer_id(value)

    @field_validator("records")
    @classmethod
    def unique_pair_ids(cls, records: list[IndependentAnnotationRecord]):
        pair_ids = [record.pair_id for record in records]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("Independent annotation contains duplicate pair IDs.")
        return records


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_annotation_manifest(path: str | Path) -> IndependentAnnotationManifest:
    with Path(path).open("r", encoding="utf-8") as source:
        manifest = IndependentAnnotationManifest.model_validate(json.load(source))
    if manifest.schema_version != ANNOTATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported annotation schema version: {manifest.schema_version}")
    if any(record.reviewer_id != manifest.reviewer_id for record in manifest.records):
        raise ValueError("Record reviewer IDs must match the manifest reviewer ID.")
    return manifest


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Publish JSON atomically while preserving an existing destination on error."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temp_path = temporary.name
            json.dump(payload, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def ensure_output_path_is_unprotected(path: str | Path) -> Path:
    """Keep study outputs out of imported source and historical report trees."""
    target = Path(path).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    protected_roots = (
        (project_root / "data" / "replication").resolve(),
        (project_root / "data" / "evaluation").resolve(),
    )
    for protected in protected_roots:
        if target == protected or protected in target.parents:
            raise ValueError(
                f"Refusing to write independent annotation output under protected data path: {protected}"
            )
    return target


def ensure_new_output_path(path: str | Path) -> Path:
    """Reject overwrites and keep new study outputs out of protected data trees."""
    target = ensure_output_path_is_unprotected(path)
    if target.exists():
        raise ValueError("Refusing to overwrite an existing annotation or report file.")
    return target


def compare_independent_annotations(
    manifest_a: IndependentAnnotationManifest,
    manifest_b: IndependentAnnotationManifest,
) -> dict[str, Any]:
    """Compare complete blind annotation vectors without creating ground truth."""
    for manifest in (manifest_a, manifest_b):
        if manifest.status != "complete":
            raise ValueError("Both independent annotation manifests must be complete.")
        if manifest.annotation_mode != "blind_independent":
            raise ValueError("Agreement requires blind_independent annotation manifests.")
    if manifest_a.reviewer_id == manifest_b.reviewer_id:
        raise ValueError("Agreement requires two different reviewer IDs.")
    if manifest_a.source_sha256 != manifest_b.source_sha256:
        raise ValueError("Reviewers must annotate the same source dataset file.")
    if manifest_a.context_sha256 != manifest_b.context_sha256:
        raise ValueError("Reviewers must use the same SRS context file.")

    records_a = {record.pair_id: record for record in manifest_a.records}
    records_b = {record.pair_id: record for record in manifest_b.records}
    if not records_a or records_a.keys() != records_b.keys():
        raise ValueError("Reviewers must label the same nonempty set of pair IDs.")
    pair_ids = sorted(records_a)
    labels_a = [records_a[pair_id].label for pair_id in pair_ids]
    labels_b = [records_b[pair_id].label for pair_id in pair_ids]
    n = len(pair_ids)
    agreements = sum(a == b for a, b in zip(labels_a, labels_b))
    expected_agreement = 0.0
    categories: list[DependencyType | None] = [*DependencyType, None]
    for label in categories:
        prevalence_a = sum(item == label for item in labels_a) / n
        prevalence_b = sum(item == label for item in labels_b) / n
        expected_agreement += prevalence_a * prevalence_b
    kappa = (
        None
        if math.isclose(expected_agreement, 1.0)
        else ((agreements / n) - expected_agreement) / (1.0 - expected_agreement)
    )

    display = lambda label: label.value if label is not None else "ABSTAIN"
    confusion = {
        display(label): {
            display(other): sum(
                a == label and b == other for a, b in zip(labels_a, labels_b)
            )
            for other in categories
        }
        for label in categories
    }
    time_summary: dict[str, dict[str, float]] = {}
    for manifest, records in ((manifest_a, records_a), (manifest_b, records_b)):
        durations = [records[pair_id].elapsed_seconds for pair_id in pair_ids]
        time_summary[manifest.reviewer_id] = {
            "total_seconds": round(sum(durations), 3),
            "mean_seconds_per_pair": round(statistics.mean(durations), 3),
            "median_seconds_per_pair": round(statistics.median(durations), 3),
        }

    return {
        "report_type": "independent_annotation_agreement",
        "source_sha256": manifest_a.source_sha256,
        "context_sha256": manifest_a.context_sha256,
        "pair_count": n,
        "reviewer_ids": [manifest_a.reviewer_id, manifest_b.reviewer_id],
        "observed_agreement": agreements / n,
        "agreement_count": agreements,
        "abstention_count": {
            manifest_a.reviewer_id: sum(label is None for label in labels_a),
            manifest_b.reviewer_id: sum(label is None for label in labels_b),
        },
        "cohens_kappa": round(kappa, 6) if kappa is not None and math.isfinite(kappa) else None,
        "cohens_kappa_defined": kappa is not None and math.isfinite(kappa),
        "confusion_matrix_reviewer_a_rows_reviewer_b_columns": confusion,
        "annotation_time_seconds": time_summary,
        "labels_are_ground_truth": False,
        "note": (
            "This report measures independent annotation agreement only. "
            "ABSTAIN is included as a separate response category. The report does not "
            "adjudicate disagreements or verify/promote dataset labels."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_candidate_pairs(path: str | Path):
    pairs = load_pairs_json(str(path))
    if not pairs:
        raise ValueError("The candidate dataset is empty.")
    pair_ids = [pair.id for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("The candidate dataset contains duplicate pair IDs.")
    return pairs
