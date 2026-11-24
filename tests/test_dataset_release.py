"""Tests for immutable, verified dataset release artifacts."""

import hashlib
import json

import pytest
from typer.testing import CliRunner

from leredd.cli import app
from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.dataset_release import create_versioned_dataset_release
from leredd.utils import load_pairs_json


runner = CliRunner()


def _verified_pair(pair_id="PAIR-1", label=DependencyType.REQUIRES):
    return RequirementPair(
        id=pair_id,
        req_a=Requirement(
            id=f"{pair_id}-A",
            text="The authentication service shall issue a signed token.",
            system="DEMO",
            component="Authentication",
            subsystem="Identity",
            source="demo-srs.md",
            source_location="line 8",
        ),
        req_b=Requirement(
            id=f"{pair_id}-B",
            text="The API shall validate the signed token.",
            system="DEMO",
            component="API Gateway",
            subsystem="Security",
            source="demo-srs.md",
            source_location="line 15",
        ),
        dependency_type=label,
        provenance=AnnotationProvenance(
            source=AnnotationSource.HUMAN,
            status=VerificationStatus.VERIFIED,
            annotator_id="reviewer-1",
            reviewer_id="reviewer-1",
            accepted_label=label,
            predicted_label=label,
            dataset_version="human-verified-v1.0",
        ),
    )


def test_dataset_release_cli_publishes_hash_verified_immutable_artifact(tmp_path):
    source = tmp_path / "verified.json"
    destination = tmp_path / "releases" / "v1.0.0"
    pair = _verified_pair()
    source_bytes = json.dumps([pair.model_dump(mode="json")], indent=2).encode("utf-8")
    source.write_bytes(source_bytes)

    result = runner.invoke(app, [
        "dataset-release", "--input", str(source), "--version", "v1.0.0",
        "--output-dir", str(destination),
    ])

    assert result.exit_code == 0, result.output
    assert "Published dataset version v1.0.0" in result.stdout
    dataset_path = destination / "pairs.json"
    manifest_path = destination / "manifest.json"
    released = load_pairs_json(str(dataset_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(released) == 1
    assert released[0].provenance.dataset_version == "v1.0.0"
    assert released[0].req_a.model_dump() == pair.req_a.model_dump()
    assert released[0].req_b.model_dump() == pair.req_b.model_dump()
    assert released[0].dependency_type == pair.dependency_type
    assert manifest["record_count"] == 1
    assert manifest["class_counts"]["Requires"] == 1
    assert manifest["sources"] == [{
        "filename": "verified.json",
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "record_count": 1,
    }]
    assert manifest["dataset_sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert source.read_bytes() == source_bytes

    retry = runner.invoke(app, [
        "dataset-release", "--input", str(source), "--version", "v1.0.0",
        "--output-dir", str(destination),
    ])
    assert retry.exit_code == 1
    assert "already exists" in retry.stdout
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == manifest["dataset_sha256"]


@pytest.mark.parametrize(
    "pair, expected",
    [
        (
            _verified_pair().model_copy(update={
                "provenance": AnnotationProvenance(
                    source=AnnotationSource.HUMAN,
                    status=VerificationStatus.PENDING_REVIEW,
                    annotator_id="reviewer-1",
                ),
            }),
            "explicit VERIFIED status is required",
        ),
        (_verified_pair().model_copy(update={"provenance": None}), "has no provenance"),
        (
            _verified_pair().model_copy(update={"dependency_type": None}),
            "dependency_type is missing",
        ),
        (
            _verified_pair().model_copy(update={
                "provenance": AnnotationProvenance(
                    source=AnnotationSource.HUMAN,
                    status=VerificationStatus.VERIFIED,
                    reviewer_id="reviewer-1",
                    accepted_label=DependencyType.DETAILS,
                ),
            }),
            "does not match dependency_type",
        ),
    ],
)
def test_release_rejects_unverified_or_incoherent_records(tmp_path, pair, expected):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([pair.model_dump(mode="json")]), encoding="utf-8")
    destination = tmp_path / "release"

    with pytest.raises(ValueError, match=expected):
        create_versioned_dataset_release(source, "v1.0", destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".release.*")) == []


def test_release_rejects_empty_duplicate_and_invalid_version_inputs(tmp_path):
    source = tmp_path / "input.json"
    destination = tmp_path / "release"

    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty JSON array"):
        create_versioned_dataset_release(source, "v1.0", destination)

    pair = _verified_pair()
    source.write_text(
        json.dumps([pair.model_dump(mode="json"), pair.model_dump(mode="json")]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate pair IDs"):
        create_versioned_dataset_release(source, "v1.0", destination)

    source.write_text(json.dumps([pair.model_dump(mode="json")]), encoding="utf-8")
    with pytest.raises(ValueError, match="Version must be"):
        create_versioned_dataset_release(source, "../v1", destination)
    assert not destination.exists()


def test_release_combines_and_records_multiple_verified_sources(tmp_path):
    source_a = tmp_path / "verified-a.json"
    source_b = tmp_path / "verified-b.json"
    first = json.dumps([_verified_pair("PAIR-A").model_dump(mode="json")]).encode("utf-8")
    second = json.dumps([_verified_pair("PAIR-B", DependencyType.DETAILS).model_dump(mode="json")]).encode("utf-8")
    source_a.write_bytes(first)
    source_b.write_bytes(second)
    destination = tmp_path / "release"

    manifest = create_versioned_dataset_release(
        [source_a, source_b], "v1.1.0", destination
    )

    released = load_pairs_json(str(destination / "pairs.json"))
    assert {pair.id for pair in released} == {"PAIR-A", "PAIR-B"}
    assert manifest.record_count == 2
    assert manifest.class_counts["Requires"] == 1
    assert manifest.class_counts["Details"] == 1
    assert [source.sha256 for source in manifest.sources] == [
        hashlib.sha256(first).hexdigest(),
        hashlib.sha256(second).hexdigest(),
    ]


def test_release_rename_failure_cleans_staging_directory(tmp_path, monkeypatch):
    import leredd.dataset_release as release_module

    source = tmp_path / "input.json"
    source.write_text(json.dumps([_verified_pair().model_dump(mode="json")]), encoding="utf-8")
    destination = tmp_path / "release"

    def fail_rename(*args, **kwargs):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(release_module.os, "rename", fail_rename)
    with pytest.raises(OSError, match="simulated publish failure"):
        create_versioned_dataset_release(source, "v1.0", destination)

    assert not destination.exists()
    assert list(tmp_path.glob(".release.*")) == []
