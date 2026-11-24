"""Synthetic/demo provenance must survive normal workflows and stay out of research outputs."""

from __future__ import annotations

import json

import pytest

from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.dataset_release import create_versioned_dataset_release
from leredd.candidate_generator import CandidateGenerator, GenerationStrategy
from leredd.evaluation import Evaluator, ground_truth_provenance_problem
from leredd.review_queue import QueueStatus, ReviewQueueConfig, ReviewQueueManager
from leredd.retrieval import ExampleRetriever
from leredd.synthetic import is_synthetic
from leredd.utils import (
    generate_requirement_pairs,
    load_pairs_csv,
    save_pairs_csv,
)
from typer.testing import CliRunner
from leredd.cli import app
from scripts.run_full_evaluation import (
    GroundTruthViolationError,
    build_safe_retrieval_pool,
    check_ground_truth_eligibility,
)


def _pair(*, synthetic: bool, labeled: bool = True) -> RequirementPair:
    label = DependencyType.REQUIRES if labeled else None
    return RequirementPair(
        id="PAIR-SYN-01" if synthetic else "PAIR-REAL-01",
        req_a=Requirement(
            id="REQ-A", text="Requirement A shall do the first action.",
            synthetic=synthetic,
            source_identity="synthetic:fixture" if synthetic else None,
        ),
        req_b=Requirement(
            id="REQ-B", text="Requirement B shall do the second action.",
            synthetic=synthetic,
            source_identity="synthetic:fixture" if synthetic else None,
        ),
        dependency_type=label,
        provenance=(
            AnnotationProvenance(
                source=AnnotationSource.HUMAN,
                status=VerificationStatus.VERIFIED,
                annotator_id="reviewer-for-controlled-test",
                reviewer_id="reviewer-for-controlled-test",
                accepted_label=label,
            )
            if labeled else None
        ),
    )


def test_synthetic_provenance_is_detected_recursively():
    pair = _pair(synthetic=True)

    assert is_synthetic(pair)
    assert is_synthetic(pair.model_dump(mode="json"))
    assert ground_truth_provenance_problem(pair) == (
        "synthetic/demo records cannot be used as ground truth"
    )


def test_pair_generation_preserves_synthetic_origin():
    requirements = [
        Requirement(
            id="REQ-A", text="Synthetic requirement A shall act.",
            synthetic=True, source_identity="synthetic:cli-extract",
        ),
        Requirement(id="REQ-B", text="Synthetic requirement B shall act."),
    ]

    pair = generate_requirement_pairs(requirements)[0]

    assert pair.synthetic is True
    assert pair.source_identity == "synthetic:cli-extract"
    assert pair.provenance.synthetic is True
    assert is_synthetic(pair)


def test_retrieval_candidate_generation_preserves_synthetic_origin():
    requirements = [
        Requirement(
            id="REQ-A", text="Synthetic requirement A shall act.",
            synthetic=True, source_identity="synthetic:retrieval-test",
        ),
        Requirement(id="REQ-B", text="Requirement B shall act."),
    ]

    pairs, _ = CandidateGenerator.generate_candidates(
        requirements, strategy=GenerationStrategy.ALL_PAIRS
    )

    assert pairs[0].synthetic is True
    assert pairs[0].provenance.synthetic is True
    assert is_synthetic(pairs[0])


def test_csv_round_trip_preserves_synthetic_origin(tmp_path):
    pair = generate_requirement_pairs([
        Requirement(
            id="REQ-A", text="Synthetic requirement A shall act.",
            synthetic=True, source_identity="synthetic:csv-round-trip",
        ),
        Requirement(id="REQ-B", text="Requirement B shall act."),
    ])[0]
    csv_path = tmp_path / "pairs.csv"

    save_pairs_csv([pair], str(csv_path))
    restored = load_pairs_csv(str(csv_path))[0]

    assert restored.synthetic is True
    assert restored.req_a.synthetic is True
    assert restored.req_a.source_identity == "synthetic:csv-round-trip"
    assert is_synthetic(restored)


def test_cli_extract_synthetic_option_marks_source_records_and_pairs(tmp_path):
    source = tmp_path / "demo.txt"
    requirements_path = tmp_path / "requirements.json"
    pairs_path = tmp_path / "pairs.json"
    source.write_text(
        "[REQ-A] The synthetic system shall perform an action.\n"
        "[REQ-B] The synthetic system shall report the action.",
        encoding="utf-8",
    )
    runner = CliRunner()

    extracted = runner.invoke(app, [
        "extract", "--input", str(source), "--output", str(requirements_path),
        "--system", "DEMO", "--synthetic",
    ])
    assert extracted.exit_code == 0, extracted.stdout
    records = json.loads(requirements_path.read_text(encoding="utf-8"))
    assert all(record["synthetic"] for record in records)
    assert all(record["source_identity"] == "synthetic:cli-extract" for record in records)

    generated = runner.invoke(app, [
        "pairs", "--input", str(requirements_path), "--output", str(pairs_path),
    ])
    assert generated.exit_code == 0, generated.stdout
    pair = json.loads(pairs_path.read_text(encoding="utf-8"))[0]
    assert pair["synthetic"] is True
    assert pair["provenance"]["synthetic"] is True


def test_evaluation_and_verified_export_reject_synthetic_pairs(tmp_path):
    pair = _pair(synthetic=True)

    with pytest.raises(ValueError, match="synthetic/demo records"):
        Evaluator.evaluate([pair], [])
    with pytest.raises(ValueError, match="synthetic/demo records"):
        create_versioned_dataset_release(
            _write_pairs(tmp_path, [pair]), "demo-v1", tmp_path / "release"
        )


def test_strict_runner_rejects_unlabeled_synthetic_legacy_records():
    with pytest.raises(GroundTruthViolationError, match="synthetic/demo records"):
        check_ground_truth_eligibility([_pair(synthetic=True, labeled=False)], "test")


def test_safe_retrieval_pool_excludes_synthetic_training_examples():
    synthetic = _pair(synthetic=True)

    safe, audit = build_safe_retrieval_pool([synthetic], [], strict=True)

    assert safe == []
    assert audit["excluded_synthetic_pair_ids"] == [synthetic.id]
    assert audit["excluded_synthetic_count"] == 1


def test_example_retriever_excludes_synthetic_examples_without_encoding():
    class UnexpectedEmbedder:
        def encode(self, texts):
            raise AssertionError("synthetic-only example pool must be filtered before encoding")

    retriever = ExampleRetriever(embedder=UnexpectedEmbedder())
    target = _pair(synthetic=False)

    assert retriever.retrieve(target, [_pair(synthetic=True)], k=1) == {}


def test_review_queue_preserves_synthetic_marker_after_human_decision():
    source = _pair(synthetic=True, labeled=False)
    source.provenance = AnnotationProvenance(
        source=AnnotationSource.LLM,
        status=VerificationStatus.PROVISIONAL,
        predicted_label=DependencyType.REQUIRES,
        confidence_score=2,
    )
    manager = ReviewQueueManager(
        config=ReviewQueueConfig(minimum_requirement_characters=0)
    )
    item = manager.create_review_item(source)
    assert item is not None

    _, reviewed = manager.record_decision(
        item,
        QueueStatus.ACCEPTED,
        final_label=DependencyType.REQUIRES,
        reviewer_id="actual-reviewer-for-test",
    )

    assert item.synthetic is True
    assert reviewed.synthetic is True
    assert reviewed.provenance.synthetic is True
    assert is_synthetic(reviewed)
    assert ground_truth_provenance_problem(reviewed) == (
        "synthetic/demo records cannot be used as ground truth"
    )


def _write_pairs(tmp_path, pairs):
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps([pair.model_dump(mode="json") for pair in pairs]),
        encoding="utf-8",
    )
    return path
