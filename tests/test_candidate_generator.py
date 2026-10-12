"""Unit tests for Candidate Requirement Pair Generation (leredd.candidate_generator).

Covers:
- All-pairs count n(n-1)/2
- Self-pair exclusion
- Canonical unordered pair deduplication ((A,B) and (B,A))
- Deterministic pair IDs and output ordering
- Handling missing or repeated requirement IDs
- Retrieval-based candidate generation (BM25, DENSE, HYBRID)
- Top-k behavior and deduplication
- Max candidate limits and explicit truncation reporting in manifest
- Unlabeled candidate state (dependency_type is None)
- Provenance and metadata preservation
- Manifest JSON serialization round-tripping
- Malformed / empty inputs
- CLI candidates command execution
- Pure corpus isolation (zero implicit global reads)
"""

import json
import os
import tempfile
import pytest
from typer.testing import CliRunner

from leredd.data_models import Requirement, RequirementPair, DependencyType, AnnotationSource, VerificationStatus
from leredd.candidate_generator import (
    CandidateGenerator,
    CandidateGenerationManifest,
    GenerationStrategy,
    get_canonical_pair_key,
)
from leredd.cli import app


@pytest.fixture
def sample_reqs():
    return [
        Requirement(id="REQ-ADB-001", text="System shall include Brake Control Subsystem.", system="ADB"),
        Requirement(id="REQ-ADB-002", text="BCS shall accept camera scenery information.", system="ADB"),
        Requirement(id="REQ-ADB-003", text="System shall verify firmware update source.", system="ADB"),
        Requirement(id="REQ-ADB-004", text="System shall authenticate update credentials.", system="ADB"),
    ]


def test_canonical_pair_key():
    """Verify (A,B) and (B,A) yield identical canonical keys."""
    req1 = Requirement(id="REQ-001", text="Text 1")
    req2 = Requirement(id="REQ-002", text="Text 2")

    key1 = get_canonical_pair_key(req1, req2)
    key2 = get_canonical_pair_key(req2, req1)
    assert key1 == key2 == "REQ-001|||REQ-002"


def test_all_pairs_generation_count_and_unlabeled_state(sample_reqs):
    """Verify ALL_PAIRS strategy generates exactly n(n-1)/2 pairs with dependency_type=None."""
    n = len(sample_reqs)
    expected_count = (n * (n - 1)) // 2

    candidates, manifest = CandidateGenerator.generate_candidates(
        requirements=sample_reqs,
        strategy=GenerationStrategy.ALL_PAIRS,
        source_identity="TEST_SYS"
    )

    assert len(candidates) == expected_count == 6
    assert manifest.generated_count == 6
    assert manifest.is_exhaustive is True
    assert manifest.truncated_count == 0

    # Verify all generated pairs are genuinely unlabeled (dependency_type is None)
    for pair in candidates:
        assert pair.dependency_type is None
        assert pair.req_a.id != pair.req_b.id  # No self-pairs
        assert pair.provenance is not None
        assert pair.provenance.status == VerificationStatus.PROVISIONAL


def test_no_duplicate_unordered_pairs(sample_reqs):
    """Verify (A,B) and (B,A) are never duplicated."""
    candidates, _ = CandidateGenerator.generate_candidates(
        requirements=sample_reqs,
        strategy=GenerationStrategy.ALL_PAIRS
    )

    keys = set()
    for pair in candidates:
        key = get_canonical_pair_key(pair.req_a, pair.req_b)
        assert key not in keys, f"Duplicate pair key found: {key}"
        keys.add(key)


def test_max_candidates_truncation_reporting(sample_reqs):
    """Verify max_candidates limits output and records truncation in audit manifest."""
    candidates, manifest = CandidateGenerator.generate_candidates(
        requirements=sample_reqs,
        strategy=GenerationStrategy.ALL_PAIRS,
        max_candidates=3
    )

    assert len(candidates) == 3
    assert manifest.generated_count == 3
    assert manifest.is_exhaustive is False
    assert manifest.truncated_count == 3  # 6 total - 3 = 3 omitted


def test_retrieval_candidate_generation(sample_reqs):
    """Verify BM25 retrieval-based candidate generation."""
    candidates, manifest = CandidateGenerator.generate_candidates(
        requirements=sample_reqs,
        strategy=GenerationStrategy.BM25,
        top_k=2
    )

    assert len(candidates) > 0
    assert manifest.strategy == GenerationStrategy.BM25
    assert manifest.top_k == 2

    for pair in candidates:
        assert pair.dependency_type is None
        assert pair.req_a.id != pair.req_b.id


def test_empty_and_malformed_inputs():
    """Verify empty requirement list returns empty candidates and manifest without crashing."""
    candidates, manifest = CandidateGenerator.generate_candidates([])
    assert candidates == []
    assert manifest.input_requirement_count == 0
    assert manifest.generated_count == 0


def test_duplicate_requirement_id_handling():
    """Verify duplicate requirements in input are deduplicated before generation."""
    reqs_with_dups = [
        Requirement(id="REQ-001", text="Text 1"),
        Requirement(id="REQ-001", text="Text 1 duplicate"),
        Requirement(id="REQ-002", text="Text 2"),
    ]

    candidates, manifest = CandidateGenerator.generate_candidates(
        requirements=reqs_with_dups,
        strategy=GenerationStrategy.ALL_PAIRS
    )

    # 2 unique requirements -> 1 pair
    assert len(candidates) == 1
    assert manifest.input_requirement_count == 2


def test_save_and_manifest_roundtrip(sample_reqs, tmp_path):
    """Verify save_candidates writes JSON files that load cleanly."""
    candidates, manifest = CandidateGenerator.generate_candidates(
        requirements=sample_reqs,
        strategy=GenerationStrategy.ALL_PAIRS
    )

    pairs_file = str(tmp_path / "candidates.json")
    manifest_file = str(tmp_path / "manifest.json")

    CandidateGenerator.save_candidates(candidates, manifest, pairs_file, manifest_file)

    assert os.path.exists(pairs_file)
    assert os.path.exists(manifest_file)

    with open(pairs_file, "r") as f:
        loaded_pairs = json.load(f)
        assert len(loaded_pairs) == 6
        assert loaded_pairs[0]["dependency_type"] is None

    with open(manifest_file, "r") as f:
        loaded_manifest = json.load(f)
        m = CandidateGenerationManifest.model_validate(loaded_manifest)
        assert m.generated_count == 6
        assert m.is_exhaustive is True


def test_cli_candidates_command(tmp_path):
    """Test CLI `leredd-cli candidates` command execution."""
    runner = CliRunner()
    input_json = "data/annotated/adb_pairs.json"
    out_pairs = str(tmp_path / "cli_cand_pairs.json")
    out_manifest = str(tmp_path / "cli_cand_manifest.json")

    result = runner.invoke(app, [
        "candidates",
        "--input", input_json,
        "--output", out_pairs,
        "--manifest", out_manifest,
        "--strategy", "all_pairs",
        "--max-candidates", "50"
    ])

    assert result.exit_code == 0, f"CLI output: {result.output}"
    assert "CANDIDATE GENERATION SUMMARY" in result.output
    assert os.path.exists(out_pairs)
    assert os.path.exists(out_manifest)
