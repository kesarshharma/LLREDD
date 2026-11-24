"""Create immutable, provenance-checked releases of verified pair datasets."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

from pydantic import BaseModel, Field

from leredd.data_models import DependencyType, RequirementPair
from leredd.evaluation import ground_truth_provenance_problem


DATASET_RELEASE_SCHEMA_VERSION = "1.0"
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}\Z")


class DatasetReleaseSource(BaseModel):
    """Content identity for one verified input file included in a release."""

    filename: str
    sha256: str
    record_count: int = Field(ge=0)


class DatasetReleaseManifest(BaseModel):
    """Integrity and provenance summary for one immutable dataset release."""

    schema_version: str = DATASET_RELEASE_SCHEMA_VERSION
    dataset_version: str
    created_at: str
    verification_policy: str = "verified_human_annotations_only"
    sources: List[DatasetReleaseSource]
    dataset_filename: str = "pairs.json"
    dataset_sha256: str
    record_count: int = Field(ge=1)
    class_counts: Dict[str, int]


def _write_json_durable(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def create_versioned_dataset_release(
    input_file: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    version: str,
    output_dir: str | os.PathLike[str],
) -> DatasetReleaseManifest:
    """Publish verified pairs and a manifest as a new, non-overwritable release.

    The release directory appears only after both files are fully written. Any
    provenance-bearing pair must have a coherent human VERIFIED annotation;
    provenance-free legacy records are rejected for release.
    """
    if not isinstance(version, str) or not _VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            "Version must be 1-64 characters, start with a letter or digit, "
            "and contain only letters, digits, '.', '_', '+', or '-'."
        )

    if isinstance(input_file, (str, os.PathLike)):
        raw_sources = [part.strip() for part in os.fspath(input_file).split(",") if part.strip()]
    else:
        raw_sources = [os.fspath(part) for part in input_file]
    if not raw_sources:
        raise ValueError("At least one verified dataset input file is required.")

    sources: List[DatasetReleaseSource] = []
    pairs: List[RequirementPair] = []
    for raw_source in raw_sources:
        source_path = Path(raw_source).expanduser()
        try:
            source_bytes = source_path.read_bytes()
            raw_records = json.loads(source_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read pair dataset '{source_path}': {exc}") from exc

        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(
                f"Verified dataset release input '{source_path}' must be a non-empty JSON array."
            )
        try:
            pairs.extend(RequirementPair.model_validate(record) for record in raw_records)
        except Exception as exc:
            raise ValueError(f"Input '{source_path}' contains an invalid requirement pair: {exc}") from exc
        sources.append(DatasetReleaseSource(
            filename=source_path.name,
            sha256=_sha256(source_bytes),
            record_count=len(raw_records),
        ))

    if not pairs:
        raise ValueError("Verified dataset release input must contain at least one pair.")

    pair_ids = [pair.id for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("Verified dataset release cannot contain duplicate pair IDs.")

    release_pairs: List[RequirementPair] = []
    class_counts = {label.value: 0 for label in DependencyType}
    for pair in pairs:
        if pair.provenance is None:
            raise ValueError(
                f"Pair {pair.id} has no provenance; releases require verified human annotations."
            )
        problem = ground_truth_provenance_problem(pair)
        if problem:
            raise ValueError(f"Pair {pair.id} is not eligible for release: {problem}.")
        if pair.dependency_type is None:
            raise ValueError(f"Pair {pair.id} has no dependency label.")

        release_pair = pair.model_copy(deep=True)
        release_pair.provenance.dataset_version = version
        release_pairs.append(release_pair)
        class_counts[pair.dependency_type.value] += 1

    destination = Path(output_dir).expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"Dataset release destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        dataset_path = staging_dir / "pairs.json"
        _write_json_durable(
            dataset_path,
            [pair.model_dump(mode="json") for pair in release_pairs],
        )
        dataset_bytes = dataset_path.read_bytes()
        manifest = DatasetReleaseManifest(
            dataset_version=version,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            sources=sources,
            dataset_sha256=_sha256(dataset_bytes),
            record_count=len(release_pairs),
            class_counts=class_counts,
        )
        _write_json_durable(staging_dir / "manifest.json", manifest.model_dump(mode="json"))

        if os.path.lexists(destination):
            raise FileExistsError(f"Dataset release destination already exists: {destination}")
        os.rename(staging_dir, destination)
        staging_dir = None
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)

    return manifest
