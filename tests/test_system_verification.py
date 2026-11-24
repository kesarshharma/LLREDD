"""tests/test_system_verification.py
End-to-end multi-interface system verification and integration tests for TASK-P8-01.
"""

from __future__ import annotations

import json
from pathlib import Path

from leredd.candidate_generator import CandidateGenerator, GenerationStrategy
from leredd.config import LEREDDConfig
from leredd.data_models import (
    VerificationStatus,
)
from leredd.dataset_splitter import DatasetSplitter, SplitConfig, SplitStrategy
from leredd.detector import LEREDDDetector
from leredd.evaluation import Evaluator
from leredd.llm import MockLLMClient
from leredd.provisional_labeler import LabelingMode, LLMLabeler, ProvisionalLabeler
from leredd.review_queue import QueueStatus, ReviewQueueConfig, ReviewQueueManager
from leredd.utils import (
    extract_requirements_from_text,
    load_pairs_json,
    save_pairs_json,
)
from scripts.run_full_evaluation import build_safe_retrieval_pool


class TestEndToEndSystemVerification:
    """End-to-end system verification tests covering the full pipeline."""

    def test_full_pipeline_end_to_end(self, tmp_path):
        """Execute complete pipeline from SRS text to verified dataset evaluation."""
        # Every artifact in this integration test is isolated under tmp_path.
        review_dir = tmp_path / "review"
        split_dir = tmp_path / "splits"
        review_dir.mkdir()

        srs_documents = {
            "ADB": "[REQ-ADB-001] The brake controller shall use the wheel speed signal.\n"
                  "[REQ-ADB-002] The brake controller shall report wheel speed faults.",
            "APA": "[REQ-APA-001] The parking system shall detect an available space.\n"
                  "[REQ-APA-002] The parking system shall report space detection status.",
            "TJA": "[REQ-TJA-001] The traffic assist shall maintain a safe distance.\n"
                  "[REQ-TJA-002] The traffic assist shall report distance sensor faults.",
        }

        # SRS -> extracted requirements -> unlabeled exhaustive candidates.
        requirements = {
            system: extract_requirements_from_text(text, system_name=system)
            for system, text in srs_documents.items()
        }
        assert all(len(items) == 2 for items in requirements.values())
        candidate_pairs = []
        for system, items in requirements.items():
            pairs, manifest = CandidateGenerator.generate_candidates(
                items, strategy=GenerationStrategy.ALL_PAIRS, source_identity=system,
            )
            assert len(pairs) == manifest.generated_count == 1
            assert pairs[0].dependency_type is None
            assert pairs[0].provenance.status == VerificationStatus.PROVISIONAL
            candidate_pairs.extend(pairs)
            candidate_path = review_dir / f"candidates_{system}.json"
            CandidateGenerator.save_candidates(pairs, manifest, str(candidate_path))
            assert len(json.loads(candidate_path.read_text(encoding="utf-8"))) == 1

        # Controlled mock labeling creates provisional annotations only.
        config = LEREDDConfig(llm_provider="mock", cache_dir=str(tmp_path / "cache"))
        provisional_labeler = ProvisionalLabeler(
            llm_labeler=LLMLabeler(
                llm_client=MockLLMClient(response_text=json.dumps({
                    "dependency_type": "Requires",
                    "confidence": 5,
                    "explanation": "Controlled synthetic test annotation.",
                })),
                config=config,
            ),
            config=config,
        )
        provisional_pairs, labeling_manifest = provisional_labeler.process_batch(
            candidate_pairs, mode=LabelingMode.LLM, source_identity="synthetic-fixture",
        )
        provisional_path = review_dir / "provisional.json"
        ProvisionalLabeler.save_provisional_dataset(
            provisional_pairs, labeling_manifest, str(provisional_path),
        )
        assert all(pair.provenance.status == VerificationStatus.PROVISIONAL for pair in provisional_pairs)
        saved_provisional = json.loads(provisional_path.read_text(encoding="utf-8"))
        assert all(pair["provenance"]["status"] == "provisional" for pair in saved_provisional)

        # Exercise queue persistence and the explicit decision transition.
        queue_manager = ReviewQueueManager(ReviewQueueConfig(include_single_method=True))
        queue_items = queue_manager.build_queue(provisional_pairs)
        assert len(queue_items) == len(provisional_pairs)
        queue_path = review_dir / "queue.json"
        queue_manager.save_queue(
            queue_items, queue_manager.create_manifest(queue_items, "synthetic-fixture"),
            str(queue_path),
        )
        loaded_items = queue_manager.load_queue(str(queue_path))
        verified_pairs = []
        for item in loaded_items:
            _, verified_pair = queue_manager.record_decision(
                item=item,
                decision_type=QueueStatus.ACCEPTED,
                final_label=item.proposed_label,
                reviewer_id="synthetic-test-reviewer",
                reviewer_notes="Controlled fixture decision; not human annotation evidence.",
            )
            verified_pairs.append(verified_pair)
        queue_manager.save_queue(
            loaded_items, queue_manager.create_manifest(loaded_items, "synthetic-fixture"),
            str(queue_path),
        )
        assert all(pair.provenance.status == VerificationStatus.VERIFIED for pair in verified_pairs)
        assert all(pair.provenance.reviewer_id == "synthetic-test-reviewer" for pair in verified_pairs)

        verified_path = review_dir / "verified.json"
        save_pairs_json(verified_pairs, str(verified_path))
        exported_verified = load_pairs_json(str(verified_path))
        assert len(exported_verified) == len(provisional_pairs)
        assert all(pair.provenance.status == VerificationStatus.VERIFIED for pair in exported_verified)

        # System-disjoint split and explicit test/dev exclusion from retrieval.
        split = DatasetSplitter.split(
            exported_verified,
            config=SplitConfig(
                strategy=SplitStrategy.SYSTEM,
                explicit_system_splits={"train": ["ADB"], "dev": ["APA"], "test": ["TJA"]},
            ),
        )
        DatasetSplitter.save_split(split, str(split_dir))
        meta = json.loads((split_dir / "split_metadata.json").read_text(encoding="utf-8"))
        assert meta["leakage_report"]["has_leakage"] is False
        assert [pair.req_a.system for pair in split.test_pairs] == ["TJA"]
        safe_pool, audit = build_safe_retrieval_pool(
            split.train_pairs, split.test_pairs, split.dev_pairs, strict=True,
        )
        assert audit["contamination_detected"] is False
        heldout_ids = {
            requirement.id
            for pair in [*split.dev_pairs, *split.test_pairs]
            for requirement in (pair.req_a, pair.req_b)
        }
        assert not any(
            requirement.id in heldout_ids
            for pair in safe_pool
            for requirement in (pair.req_a, pair.req_b)
        )

        class OfflineEmbedder:
            """Small deterministic embedder that cannot access the network or disk."""
            def encode(self, texts):
                import numpy as np
                return np.asarray([
                    [float(len(text)), float(sum(map(ord, text)) % 997)]
                    for text in texts
                ], dtype=np.float32)

        detector = LEREDDDetector(
            config=LEREDDConfig(
                llm_provider="mock", retrieval_mode="bm25", k_examples=2,
                confidence_threshold=-1, cache_dir=str(tmp_path / "cache"),
            ),
            embedder=OfflineEmbedder(),
            llm=MockLLMClient(response_text=(
                "Dependency_Type: Requires\nRationale: controlled test prediction.\nConfidence Score: 5"
            )),
        )
        predictions = detector.detect_batch(
            split.test_pairs, annotated_pool=safe_pool, max_workers=1,
        )
        assert {prediction.pair_id for prediction in predictions} == {pair.id for pair in split.test_pairs}
        report = Evaluator.evaluate(split.test_pairs, predictions, system_name="TJA", model_name="mock")
        report_path = tmp_path / "report.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        saved_report = json.loads(report_path.read_text(encoding="utf-8"))
        assert saved_report["overall_accuracy"] == 1.0
        assert saved_report["model_name"] == "mock"

    def test_cli_and_api_resilience_on_invalid_inputs(self, tmp_path):
        """Verify CLI and API fail safely when provided invalid paths or malformed JSON."""
        # Non-existent input file
        bad_file = str(tmp_path / "non_existent.json")
        out_file = str(tmp_path / "out.json")

        from typer.testing import CliRunner
        from leredd.cli import app
        runner = CliRunner()

        result = runner.invoke(app, ["pairs", "--input", bad_file, "--output", out_file])
        assert result.exit_code != 0

    def test_docker_configuration_file_integrity(self):
        """Validate Dockerfile and docker-compose.yml content and environment specifications."""
        root = Path(__file__).resolve().parent.parent
        dockerfile = root / "Dockerfile"
        compose_file = root / "docker-compose.yml"

        assert dockerfile.exists()
        assert compose_file.exists()

        df_text = dockerfile.read_text(encoding="utf-8")
        assert "FROM python:3.10-slim as builder" in df_text
        assert "EXPOSE 8000 8501" in df_text

        dc_text = compose_file.read_text(encoding="utf-8")
        assert "services:" in dc_text
        assert "api:" in dc_text
        assert "ui:" in dc_text
        assert "8000:8000" in dc_text
        assert "8501:8501" in dc_text
