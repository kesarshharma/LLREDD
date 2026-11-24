"""Tests for design-based analysis of stratified candidate-coverage pilots."""

from __future__ import annotations

import hashlib
import json

import pytest

from leredd.candidate_coverage import analyze_candidate_recall
from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from scripts import analyze_candidate_coverage as coverage_script


def _pair(pair_id: str, label: DependencyType | None = None, *, verified: bool = False):
    provenance = (
        AnnotationProvenance(
            source=AnnotationSource.HUMAN,
            status=VerificationStatus.VERIFIED,
            annotator_id="reviewer-real-a",
            reviewer_id="reviewer-real-a",
            accepted_label=label,
        )
        if verified
        else AnnotationProvenance(
            source=AnnotationSource.CANDIDATE_GENERATOR,
            status=VerificationStatus.PROVISIONAL,
            annotator_id="pilot-sample",
        )
    )
    return RequirementPair(
        id=pair_id,
        req_a=Requirement(id=f"{pair_id}-A", text=f"Requirement A for {pair_id}."),
        req_b=Requirement(id=f"{pair_id}-B", text=f"Requirement B for {pair_id}."),
        dependency_type=label,
        provenance=provenance,
    )


def _pilot_files(tmp_path, *, verified_labels=None):
    context = tmp_path / "srs.md"
    context.write_text("A shall do a thing.\nB shall do another thing.\n", encoding="utf-8")
    universe = [_pair(f"P-{index}") for index in range(1, 9)]
    candidates = universe[:4]
    pairs = [universe[index] for index in (0, 1, 4, 5)]
    reviewer_file = tmp_path / "blind-pairs.json"
    reviewer_file.write_text(
        json.dumps([pair.model_dump(mode="json") for pair in pairs]), encoding="utf-8"
    )
    strata = [
        {
            "blind_pair_id": pair.id,
            "original_pair_id": pair.id,
            "req_a_id": pair.req_a.id,
            "req_b_id": pair.req_b.id,
            "stratum": "hybrid_selected" if pair.id in {"P-1", "P-2"} else "hybrid_omitted",
            "population_stratum_size": 4,
            "sample_stratum_size": 2,
            "inclusion_probability": 0.5,
            "analysis_weight": 2.0,
        }
        for pair in pairs
    ]
    map_file = tmp_path / "strata.json"
    map_file.write_text(json.dumps(strata), encoding="utf-8")
    candidate_file = tmp_path / "candidate-pairs.json"
    candidate_file.write_text(
        json.dumps([pair.model_dump(mode="json") for pair in candidates]), encoding="utf-8"
    )
    universe_file = tmp_path / "all-pairs.json"
    universe_file.write_text(
        json.dumps([pair.model_dump(mode="json") for pair in universe]), encoding="utf-8"
    )
    manifest = {
        "artifact_type": "unlabeled_blind_candidate_coverage_pilot",
        "status": "prepared_not_reviewed",
        "strategy": "hybrid",
        "top_k": 5,
        "reviewer_input": str(reviewer_file),
        "reviewer_input_sha256": hashlib.sha256(reviewer_file.read_bytes()).hexdigest(),
        "analysis_only_strata_map_sha256": hashlib.sha256(map_file.read_bytes()).hexdigest(),
        "candidate_source_pairs_path": str(candidate_file),
        "candidate_source_pairs_sha256": hashlib.sha256(candidate_file.read_bytes()).hexdigest(),
        "full_pair_universe_path": str(universe_file),
        "full_pair_universe_sha256": hashlib.sha256(universe_file.read_bytes()).hexdigest(),
        "source_srs_path": str(context),
        "source_srs_sha256": hashlib.sha256(context.read_bytes()).hexdigest(),
        "population_pair_count": 8,
        "strategy_candidate_count": 4,
        "strategy_omitted_count": 4,
        "sample_total": 4,
    }
    manifest_file = tmp_path / "pilot-manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    labels = verified_labels or {
        "P-1": DependencyType.REQUIRES,
        "P-2": DependencyType.NO_DEPENDENCY,
        "P-5": DependencyType.CONFLICTS,
        "P-6": DependencyType.NO_DEPENDENCY,
    }
    verified_file = tmp_path / "verified.json"
    verified_file.write_text(
        json.dumps([
            _pair(pair_id, label, verified=True).model_dump(mode="json")
            for pair_id, label in labels.items()
        ]),
        encoding="utf-8",
    )
    return manifest_file, map_file, verified_file


def test_stratified_candidate_recall_uses_verified_labels_and_weights(tmp_path):
    paths = _pilot_files(tmp_path)

    report = analyze_candidate_recall(*paths)

    assert report["estimated_positive_recall"] == pytest.approx(0.5)
    assert report["estimated_selected_positive_count"] == pytest.approx(2.0)
    assert report["estimated_omitted_positive_count"] == pytest.approx(2.0)
    assert report["population_pair_count"] == 8
    assert report["strata"]["hybrid_selected"]["analysis_weight"] == 2.0
    assert report["approx_95_percent_confidence_interval"] is not None
    assert report["status"] == "exploratory_estimate_from_verified_adjudications"


def test_stratified_candidate_recall_rejects_provisional_labels(tmp_path):
    manifest, strata, _ = _pilot_files(tmp_path)
    provisional = tmp_path / "provisional.json"
    provisional.write_text(
        json.dumps([
            _pair(pair_id).model_dump(mode="json")
            for pair_id in ("P-1", "P-2", "P-5", "P-6")
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not verified human ground truth"):
        analyze_candidate_recall(manifest, strata, provisional)


@pytest.mark.parametrize("target", ["strata", "candidate", "universe"])
def test_candidate_recall_rejects_tampered_design_inputs(tmp_path, target):
    manifest, strata, verified = _pilot_files(tmp_path)
    path = {
        "strata": strata,
        "candidate": tmp_path / "candidate-pairs.json",
        "universe": tmp_path / "all-pairs.json",
    }[target]
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match"):
        analyze_candidate_recall(manifest, strata, verified)


def test_candidate_recall_rejects_stratum_that_disagrees_with_pinned_strategy(tmp_path):
    manifest, strata, verified = _pilot_files(tmp_path)
    rows = json.loads(strata.read_text(encoding="utf-8"))
    rows[0]["stratum"] = "hybrid_omitted"
    strata.write_text(json.dumps(rows), encoding="utf-8")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["analysis_only_strata_map_sha256"] = hashlib.sha256(
        strata.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Stratum assignment does not match"):
        analyze_candidate_recall(manifest, strata, verified)


def test_candidate_recall_rejects_verified_provenance_with_wrong_accepted_label(tmp_path):
    manifest, strata, verified = _pilot_files(tmp_path)
    pairs = json.loads(verified.read_text(encoding="utf-8"))
    pairs[0]["provenance"]["accepted_label"] = DependencyType.NO_DEPENDENCY.value
    verified.write_text(json.dumps(pairs), encoding="utf-8")

    with pytest.raises(ValueError, match="not verified human ground truth"):
        analyze_candidate_recall(manifest, strata, verified)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_stratified_candidate_recall_rejects_sample_id_mismatch(tmp_path, mutation):
    manifest, strata, verified = _pilot_files(tmp_path)
    pairs = json.loads(verified.read_text(encoding="utf-8"))
    if mutation == "missing":
        pairs.pop()
    elif mutation == "extra":
        extra = json.loads(json.dumps(pairs[0]))
        extra["id"] = "EXTRA"
        pairs.append(extra)
    else:
        pairs.append(json.loads(json.dumps(pairs[0])))
    verified.write_text(json.dumps(pairs), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate pair IDs|exactly match"):
        analyze_candidate_recall(manifest, strata, verified)


def test_stratified_candidate_recall_refuses_zero_observed_positives(tmp_path):
    manifest, strata, _ = _pilot_files(tmp_path)
    verified = tmp_path / "all-negative.json"
    verified.write_text(
        json.dumps([
            _pair(f"P-{i}", DependencyType.NO_DEPENDENCY, verified=True).model_dump(mode="json")
            for i in (1, 2, 5, 6)
        ]), encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No positive dependency was observed"):
        analyze_candidate_recall(manifest, strata, verified)


def test_candidate_coverage_cli_writes_new_report(tmp_path):
    manifest, strata, verified = _pilot_files(tmp_path)
    output = tmp_path / "coverage-report.json"

    result = coverage_script.main([
        "--pilot-manifest", str(manifest), "--strata-map", str(strata),
        "--verified-pairs", str(verified), "--output", str(output),
    ])

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["estimated_positive_recall"] == pytest.approx(0.5)
    assert any("pilot estimate" in warning for warning in report["warnings"])


def test_candidate_coverage_cli_refuses_existing_output(tmp_path):
    manifest, strata, verified = _pilot_files(tmp_path)
    output = tmp_path / "coverage-report.json"
    output.write_text("keep", encoding="utf-8")

    result = coverage_script.main([
        "--pilot-manifest", str(manifest), "--strata-map", str(strata),
        "--verified-pairs", str(verified), "--output", str(output),
    ])

    assert result == 2
    assert output.read_text(encoding="utf-8") == "keep"
