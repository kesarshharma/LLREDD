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
import numpy as np
import pytest
from typer.testing import CliRunner

from leredd.data_models import Requirement, VerificationStatus
from leredd.candidate_generator import (
    CandidateGenerator,
    CandidateGenerationManifest,
    GenerationStrategy,
    get_canonical_pair_key,
)
from leredd.cli import app
from leredd.retrieval import RequirementRetriever


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
    assert manifest.pair_universe_count == 6
    assert manifest.pair_space_recall == 1.0
    assert manifest.uncovered_pair_count == 0
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
    assert manifest.pair_universe_count == 6
    assert manifest.pair_space_recall == pytest.approx(0.5)
    assert manifest.uncovered_pair_count == 3


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
    assert manifest.is_exhaustive is False
    assert manifest.pair_space_recall == pytest.approx(len(candidates) / 6)
    assert manifest.uncovered_pair_count == 6 - len(candidates)

    for pair in candidates:
        assert pair.dependency_type is None
        assert pair.req_a.id != pair.req_b.id


class _FixedEmbedder:
    def __init__(self, requirements):
        self.vectors = {
            requirements[0].text: np.array([1.0, 0.0]),
            requirements[1].text: np.array([0.95, 0.0]),
            requirements[2].text: np.array([0.0, 1.0]),
            requirements[3].text: np.array([-1.0, 0.0]),
        }

    def encode(self, texts):
        return np.vstack([self.vectors[text] for text in texts])


def test_dense_candidate_generation_uses_dense_requirement_ranking(sample_reqs):
    dense = RequirementRetriever(sample_reqs, embedder=_FixedEmbedder(sample_reqs))
    candidates, manifest = CandidateGenerator.generate_candidates(
        sample_reqs,
        strategy=GenerationStrategy.DENSE,
        top_k=1,
        retriever=dense,
    )

    assert any(
        {pair.req_a.id, pair.req_b.id} == {sample_reqs[0].id, sample_reqs[1].id}
        for pair in candidates
    )
    assert manifest.strategy == GenerationStrategy.DENSE
    assert manifest.is_exhaustive is False
    assert all(pair.provenance.annotator_id == "candidate_generator_dense" for pair in candidates)


def test_hybrid_candidate_generation_fuses_dense_and_bm25_rankings(sample_reqs):
    dense = RequirementRetriever(sample_reqs, embedder=_FixedEmbedder(sample_reqs))
    candidates, manifest = CandidateGenerator.generate_candidates(
        sample_reqs,
        strategy=GenerationStrategy.HYBRID,
        top_k=1,
        retriever=dense,
    )

    assert candidates
    assert manifest.strategy == GenerationStrategy.HYBRID
    assert manifest.is_exhaustive is False
    assert all(pair.provenance.annotator_id == "candidate_generator_hybrid" for pair in candidates)


@pytest.mark.parametrize("strategy", [GenerationStrategy.BM25, GenerationStrategy.DENSE, GenerationStrategy.HYBRID])
@pytest.mark.parametrize("top_k", [0, -1, None, "2", 1.5, True])
def test_retrieval_candidate_generation_rejects_invalid_top_k(sample_reqs, strategy, top_k):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        CandidateGenerator.generate_candidates(sample_reqs, strategy=strategy, top_k=top_k)


def test_dense_candidates_report_exhaustive_when_top_k_covers_corpus(sample_reqs):
    retriever = RequirementRetriever(sample_reqs, embedder=_FixedEmbedder(sample_reqs))

    candidates, manifest = CandidateGenerator.generate_candidates(
        sample_reqs,
        strategy=GenerationStrategy.DENSE,
        top_k=len(sample_reqs) - 1,
        retriever=retriever,
    )

    assert len(candidates) == len(sample_reqs) * (len(sample_reqs) - 1) // 2
    assert manifest.is_exhaustive is True


def test_empty_and_malformed_inputs():
    """Verify empty requirement list returns empty candidates and manifest without crashing."""
    candidates, manifest = CandidateGenerator.generate_candidates([])
    assert candidates == []
    assert manifest.input_requirement_count == 0
    assert manifest.generated_count == 0
    assert manifest.pair_universe_count == 0
    assert manifest.pair_space_recall is None
    assert manifest.uncovered_pair_count == 0


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
        assert m.pair_space_recall == 1.0
        assert m.uncovered_pair_count == 0
        assert m.is_exhaustive is True


def test_cli_candidates_command(tmp_path):
    """Test CLI `leredd-cli candidates` command execution."""
    runner = CliRunner()
    input_json = tmp_path / "requirements.json"
    input_json.write_text(json.dumps([
        {"id": "REQ-A", "text": "The brake controller shall monitor wheel speed.", "system": "ADB"},
        {"id": "REQ-B", "text": "The brake controller shall report wheel speed faults.", "system": "ADB"},
        {"id": "REQ-C", "text": "The parking system shall detect an available space.", "system": "APA"},
    ]), encoding="utf-8")
    out_pairs = str(tmp_path / "cli_cand_pairs.json")
    out_manifest = str(tmp_path / "cli_cand_manifest.json")

    result = runner.invoke(app, [
        "candidates",
        "--input", str(input_json),
        "--output", out_pairs,
        "--manifest", out_manifest,
        "--strategy", "all_pairs",
        "--max-candidates", "50"
    ])

    assert result.exit_code == 0, f"CLI output: {result.output}"
    assert "CANDIDATE GENERATION SUMMARY" in result.output
    assert "Pair-Space Recall   : 1.0" in result.output
    assert "Uncovered Pair Space: 0" in result.output
    assert os.path.exists(out_pairs)
    assert os.path.exists(out_manifest)
