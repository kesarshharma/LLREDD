"""tests/test_system_verification.py
End-to-end multi-interface system verification and integration tests for TASK-P8-01.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest

from leredd.candidate_generator import CandidateGenerator, GenerationStrategy
from leredd.config import LEREDDConfig
from leredd.data_models import (
    DependencyType,
    Prediction,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.dataset_splitter import DatasetSplitter, SplitConfig, SplitStrategy
from leredd.detector import LEREDDDetector
from leredd.llm import MockLLMClient
from leredd.provisional_labeler import LabelingMode, ProvisionalLabeler
from leredd.review_queue import HumanDecision, QueueStatus, ReviewQueueConfig, ReviewQueueManager
from leredd.utils import (
    extract_requirements_from_text,
    load_pairs_json,
    save_pairs_json,
)


class TestEndToEndSystemVerification:
    """End-to-end system verification tests covering the full pipeline."""

    def test_full_pipeline_end_to_end(self, tmp_path):
        """Execute complete pipeline from SRS text to verified dataset evaluation."""
        # 1. Extraction
        srs_adb = (
            "[REQ-ADB-001] The ADB system shall control high beam headlights.\n"
            "[REQ-ADB-002] The ADB system shall disengage high beams when oncoming camera detects vehicles.\n"
            "[REQ-ADB-003] The ADB system shall include the camera sensor subsystem."
        )
        srs_tja = (
            "[REQ-TJA-001] The TJA system shall control longitudinal acceleration based on front radar target distance.\n"
            "[REQ-TJA-002] The TJA system shall bring the vehicle to a full stop when preceding vehicle stops."
        )

        reqs_adb = extract_requirements_from_text(srs_adb, system_name="ADB")
        reqs_tja = extract_requirements_from_text(srs_tja, system_name="TJA")
        assert len(reqs_adb) >= 3
        assert len(reqs_tja) >= 2

        # 2. Candidate Generation (static method — no instantiation)
        cand_adb_pairs, _ = CandidateGenerator.generate_candidates(
            reqs_adb, strategy=GenerationStrategy.ALL_PAIRS, source_identity="ADB_Test"
        )
        cand_tja_pairs, _ = CandidateGenerator.generate_candidates(
            reqs_tja, strategy=GenerationStrategy.ALL_PAIRS, source_identity="TJA_Test"
        )
        assert len(cand_adb_pairs) >= 3

        # 3. Provisional Labeling
        config = LEREDDConfig(llm_provider="mock")
        labeler = ProvisionalLabeler(config=config)
        lab_adb_pairs, _ = labeler.process_batch(cand_adb_pairs, mode=LabelingMode.HYBRID)
        lab_tja_pairs, _ = labeler.process_batch(cand_tja_pairs, mode=LabelingMode.HYBRID)

        # Promote to verified for downstream split / evaluation
        for p in lab_adb_pairs:
            p.dependency_type = DependencyType.REQUIRES
            if p.provenance:
                p.provenance.status = VerificationStatus.VERIFIED
        for p in lab_tja_pairs:
            p.dependency_type = DependencyType.REQUIRES
            if p.provenance:
                p.provenance.status = VerificationStatus.VERIFIED

        # 4. Review Queue Creation & Human Decision Submission
        q_path = tmp_path / "queue.json"
        q_mgr = ReviewQueueManager()             # no constructor args
        items = q_mgr.build_queue(lab_adb_pairs)
        assert len(items) > 0

        # record_decision is a @classmethod — pass ReviewItem + QueueStatus enum
        decided_item, verified_pair = ReviewQueueManager.record_decision(
            item=items[0],
            decision_type=QueueStatus.ACCEPTED,
            final_label=DependencyType.REQUIRES,
            reviewer_id="expert-annotator-1",
        )
        assert verified_pair.provenance.status == VerificationStatus.VERIFIED

        # Persist queue
        manifest = ReviewQueueManager.create_manifest(items, source_identity="ADB_Queue")
        ReviewQueueManager.save_queue(items, manifest, output_queue_file=str(q_path))
        assert q_path.exists()

        # 5. Verified Ground-Truth Export
        v_path = tmp_path / "verified.json"
        save_pairs_json([verified_pair], str(v_path))
        assert v_path.exists()

        # 6. Dataset Splitting (all-static class — pass config as kwarg)
        all_annotated = lab_adb_pairs + lab_tja_pairs
        split_dir = tmp_path / "splits"
        split_config = SplitConfig(strategy=SplitStrategy.SYSTEM)
        split = DatasetSplitter.split(all_annotated, config=split_config)
        DatasetSplitter.save_split(split, str(split_dir))
        assert (split_dir / "train_pairs.json").exists()
        assert (split_dir / "split_metadata.json").exists()

        # Verify split metadata structure
        meta = json.loads((split_dir / "split_metadata.json").read_text())
        assert "summary" in meta
        assert "leakage_report" in meta
        assert meta["leakage_report"]["has_leakage"] is False

        # 7. Smoke-test run_evaluation against the real pre-built system-disjoint split
        #    (the synthetic 2-system tiny dataset cannot generate a non-empty test partition
        #    when only 2 systems exist, so we only run against the real project split data)
        from scripts.run_full_evaluation import run_evaluation

        real_split_dir = Path(__file__).resolve().parent.parent / "data" / "splits" / "system_disjoint"
        if real_split_dir.exists():
            out_rep = tmp_path / "report.json"
            report = run_evaluation(
                split_dir=str(real_split_dir),
                output_path=str(out_rep),
                strict_contamination=False,
                strict_gt=False,
            )
            assert out_rep.exists()
            assert "model_results" in report
            assert "ablation_results" in report
        else:
            pytest.skip("Real system-disjoint split not present; skipping run_evaluation smoke-test.")

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
