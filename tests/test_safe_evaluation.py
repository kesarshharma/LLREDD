"""tests/test_safe_evaluation.py
Tests for scripts/run_full_evaluation.py safe evaluation pipeline.

Coverage:
  - Contamination guard: test/dev req IDs excluded from retrieval pool
  - Baseline training isolation: test pairs never used during baseline training
  - Ground-truth eligibility: only VERIFIED provenance is accepted
  - Ground-truth eligibility: UI-test fixture records rejected
  - Report schema completeness
  - Leakage raises ContaminationError with strict=True
  - Leakage logged (not raised) with strict=False
  - Empty test partition raises ValueError
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure scripts/ and project root are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import modules under test
from scripts.run_full_evaluation import (
    ContaminationError,
    GroundTruthViolationError,
    build_safe_retrieval_pool,
    check_ground_truth_eligibility,
    run_evaluation,
    _serialize_report,
)
from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Prediction,
    PredictionValidationStatus,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.evaluation import Evaluator
from leredd.evaluation import SplitOverlapError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_req(req_id: str, system: str = "SYS") -> Requirement:
    return Requirement(id=req_id, text=f"The system shall do {req_id}.", system=system)


def _make_pair(
    pair_id: str,
    req_a_id: str,
    req_b_id: str,
    dep_type: DependencyType = DependencyType.REQUIRES,
    system: str = "SYS",
    provenance: AnnotationProvenance | None = None,
) -> RequirementPair:
    return RequirementPair(
        id=pair_id,
        req_a=_make_req(req_a_id, system=system),
        req_b=_make_req(req_b_id, system=system),
        dependency_type=dep_type,
        provenance=provenance,
    )


def _provisional_provenance() -> AnnotationProvenance:
    return AnnotationProvenance(
        source=AnnotationSource.LLM,
        status=VerificationStatus.PROVISIONAL,
    )


def _verified_provenance(
    accepted_label: DependencyType = DependencyType.REQUIRES,
) -> AnnotationProvenance:
    return AnnotationProvenance(
        source=AnnotationSource.HUMAN,
        status=VerificationStatus.VERIFIED,
        annotator_id="synthetic-test-reviewer",
        accepted_label=accepted_label,
    )


def test_full_runner_rechecks_actual_system_disjoint_partitions(tmp_path):
    split_dir = tmp_path / "system-split"
    split_dir.mkdir()

    def pair(pair_id, req_a_id, req_b_id):
        return {
            "id": pair_id,
            "req_a": {"id": req_a_id, "text": "Requirement A", "system": "ADB"},
            "req_b": {"id": req_b_id, "text": "Requirement B", "system": "ADB"},
            "dependency_type": "Requires",
            "provenance": None,
        }

    (split_dir / "train_pairs.json").write_text(
        json.dumps([pair("PAIR-TRAIN", "TRAIN-A", "TRAIN-B")]), encoding="utf-8"
    )
    (split_dir / "dev_pairs.json").write_text("[]", encoding="utf-8")
    (split_dir / "test_pairs.json").write_text(
        json.dumps([pair("PAIR-TEST", "TEST-A", "TEST-B")]), encoding="utf-8"
    )
    (split_dir / "split_metadata.json").write_text(
        json.dumps({"summary": {"strategy": "system"}}), encoding="utf-8"
    )

    with pytest.raises(SplitOverlapError, match="ADB") as exc_info:
        run_evaluation(
            split_dir=str(split_dir),
            output_path=str(tmp_path / "must-not-be-created.json"),
        )

    assert exc_info.value.invalid_ids == ["ADB"]
    assert not (tmp_path / "must-not-be-created.json").exists()


def test_evaluation_report_records_confidence_review_coverage():
    pairs = [
        _make_pair("PAIR-SCREEN-1", "REQ-SCREEN-1A", "REQ-SCREEN-1B"),
        _make_pair("PAIR-SCREEN-2", "REQ-SCREEN-2A", "REQ-SCREEN-2B"),
    ]
    predictions = [
        Prediction(
            pair_id=pairs[0].id,
            dependency_type=DependencyType.REQUIRES,
            rationale="synthetic low-confidence prediction",
            confidence=2,
            validation_status=PredictionValidationStatus.REQUIRES_REVIEW,
            validation_threshold=4,
        ),
        Prediction(
            pair_id=pairs[1].id,
            dependency_type=DependencyType.REQUIRES,
            rationale="synthetic high-confidence prediction",
            confidence=5,
            validation_status=PredictionValidationStatus.PASSED_THRESHOLD,
            validation_threshold=4,
        ),
    ]
    metrics = Evaluator.evaluate(
        pairs,
        predictions,
        system_name="SYNTHETIC",
        model_name="controlled",
    )
    serialized = _serialize_report(
        metrics,
        model_tag="controlled",
        note="synthetic test",
        predictions=predictions,
        n_test=len(pairs),
    )

    assert serialized["predictions"][pairs[0].id] == "Requires"
    assert serialized["confidence_screening"] == {
        "threshold": 4,
        "screened_count": 2,
        "passed_threshold_count": 1,
        "review_required_count": 1,
        "review_required_rate": 0.5,
        "not_evaluated_count": 0,
        "threshold_disabled_count": 0,
        "interpretation": (
            "Review routing is based on self-reported Likert confidence. "
            "It does not change predicted classes, establish annotation truth, "
            "or represent calibrated probability."
        ),
    }


@pytest.fixture(scope="module")
def cached_evaluation_report_path():
    """Module-scoped cached evaluation report for pure-inspection integration tests."""
    import tempfile, json
    from pathlib import Path
    from scripts.run_full_evaluation import run_evaluation
    with tempfile.TemporaryDirectory() as tmpdir:
        split_dir = Path(tmpdir) / "system_disjoint"
        split_dir.mkdir()
        adb = [
            {
                "id": f"PAIR-DEMO-TRAIN-{i:03d}",
                "req_a": {"id": f"REQ-DEMO-TRAIN-{i*2:03d}", "text": f"The system shall do action {i}.", "system": "DEMO-TRAIN"},
                "req_b": {"id": f"REQ-DEMO-TRAIN-{i*2+1:03d}", "text": f"The system shall support feature {i}.", "system": "DEMO-TRAIN"},
                "dependency_type": dep,
                "provenance": None,
            }
            for i, dep in enumerate([
                "Requires", "Implements", "Conflicts", "Details",
                "Is similar", "No_dependency", "No_dependency",
                "Requires", "Conflicts", "No_dependency"
            ], start=1)
        ]
        (split_dir / "train_pairs.json").write_text(json.dumps(adb), encoding="utf-8")
        tja = [
            {
                "id": f"PAIR-DEMO-TEST-{i:03d}",
                "req_a": {"id": f"REQ-DEMO-TEST-{i*2:03d}", "text": f"Test demo shall perform task {i}.", "system": "DEMO-TEST"},
                "req_b": {"id": f"REQ-DEMO-TEST-{i*2+1:03d}", "text": f"Test demo shall handle event {i}.", "system": "DEMO-TEST"},
                "dependency_type": dep,
                "provenance": None,
            }
            for i, dep in enumerate(["Requires", "Details", "Conflicts", "No_dependency", "Requires"], start=1)
        ]
        (split_dir / "test_pairs.json").write_text(json.dumps(tja), encoding="utf-8")
        apa = [
            {
                "id": f"PAIR-DEMO-DEV-{i:03d}",
                "req_a": {"id": f"REQ-DEMO-DEV-{i*2:03d}", "text": f"Dev demo shall comply with rule {i}.", "system": "DEMO-DEV"},
                "req_b": {"id": f"REQ-DEMO-DEV-{i*2+1:03d}", "text": f"Dev demo shall verify condition {i}.", "system": "DEMO-DEV"},
                "dependency_type": dep,
                "provenance": None,
            }
            for i, dep in enumerate(["Details", "Requires", "Implements", "No_dependency", "Requires"], start=1)
        ]
        (split_dir / "dev_pairs.json").write_text(json.dumps(apa), encoding="utf-8")
        meta = {
            "summary": {
                "strategy": "system",
                "seed": 42,
                "counts": {
                    "train": {"pairs": 10, "requirements": 15},
                    "dev": {"pairs": 5, "requirements": 10},
                    "test": {"pairs": 5, "requirements": 9},
                },
            },
            "leakage_report": {
                "has_leakage": False,
                "req_overlap_train_test": [],
                "system_overlap_train_test": [],
            },
        }
        (split_dir / "split_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        out_path = Path(tmpdir) / "cached_report.json"
        run_evaluation(
            split_dir=str(split_dir),
            output_path=str(out_path),
            seed=42,
            bert_model_type="rf",
            strict_contamination=True,
            strict_gt=True,
        )
        yield str(out_path)


# ---------------------------------------------------------------------------
# Contamination Guard Tests
# ---------------------------------------------------------------------------

class TestBuildSafeRetrievalPool:
    """Tests for build_safe_retrieval_pool()."""

    def test_clean_pool_no_contamination(self):
        """Train pairs with completely disjoint req IDs pass without removal."""
        train = [
            _make_pair("P-T1", "REQ-A-001", "REQ-A-002"),
            _make_pair("P-T2", "REQ-A-003", "REQ-A-004"),
        ]
        test = [_make_pair("P-E1", "REQ-B-001", "REQ-B-002")]

        pool, audit = build_safe_retrieval_pool(train, test, strict=True)

        assert len(pool) == 2
        assert audit["contaminated_count"] == 0
        assert audit["contamination_detected"] is False
        assert "REQ-B-001" in audit["excluded_req_ids"]
        assert "REQ-B-002" in audit["excluded_req_ids"]

    def test_test_req_ids_excluded_from_pool(self):
        """A train pair sharing a req ID with a test pair is removed from pool."""
        shared_req = "REQ-SHARED-001"
        train = [
            _make_pair("P-T1", shared_req, "REQ-A-002"),  # contaminated
            _make_pair("P-T2", "REQ-A-003", "REQ-A-004"),  # clean
        ]
        test = [_make_pair("P-E1", shared_req, "REQ-B-002")]

        pool, audit = build_safe_retrieval_pool(train, test, strict=False)

        assert len(pool) == 1
        assert pool[0].id == "P-T2"
        assert audit["contaminated_count"] == 1
        assert audit["contamination_detected"] is True
        assert any("P-T1" in c for c in audit["contaminated_pairs"])

    def test_dev_req_ids_also_excluded(self):
        """Dev partition req IDs are also excluded from the retrieval pool."""
        dev_req = "REQ-DEV-001"
        train = [
            _make_pair("P-T1", dev_req, "REQ-A-002"),  # contaminated via dev
            _make_pair("P-T2", "REQ-A-003", "REQ-A-004"),  # clean
        ]
        test = [_make_pair("P-E1", "REQ-B-001", "REQ-B-002")]
        dev = [_make_pair("P-D1", dev_req, "REQ-D-002")]

        pool, audit = build_safe_retrieval_pool(train, test, dev_pairs=dev, strict=False)

        assert len(pool) == 1
        assert pool[0].id == "P-T2"
        assert dev_req in audit["excluded_req_ids"]

    def test_contamination_raises_error_when_strict(self):
        """ContaminationError raised in strict mode when leakage detected."""
        shared_req = "REQ-SHARED-99"
        train = [_make_pair("P-T1", shared_req, "REQ-A-002")]
        test = [_make_pair("P-E1", shared_req, "REQ-B-002")]

        with pytest.raises(ContaminationError, match="CONTAMINATION DETECTED"):
            build_safe_retrieval_pool(train, test, strict=True)

    def test_no_raise_when_not_strict(self):
        """No exception raised in non-strict mode; contaminated pairs are removed."""
        shared_req = "REQ-SHARED-99"
        train = [_make_pair("P-T1", shared_req, "REQ-A-002")]
        test = [_make_pair("P-E1", shared_req, "REQ-B-002")]

        pool, audit = build_safe_retrieval_pool(train, test, strict=False)
        assert len(pool) == 0
        assert audit["contaminated_count"] == 1

    def test_test_pair_id_itself_excluded(self):
        """A train pair with the same ID as a test pair is excluded."""
        train = [_make_pair("SAME-ID", "REQ-A-001", "REQ-A-002")]
        test = [_make_pair("SAME-ID", "REQ-B-001", "REQ-B-002")]

        pool, audit = build_safe_retrieval_pool(train, test, strict=False)
        assert len(pool) == 0


# ---------------------------------------------------------------------------
# Ground-Truth Eligibility Tests
# ---------------------------------------------------------------------------

class TestCheckGroundTruthEligibility:
    """Tests for check_ground_truth_eligibility()."""

    def test_legacy_pair_accepted_with_warning(self):
        """Pairs with no provenance (legacy annotated) are accepted with a warning."""
        pair = _make_pair("P-001", "REQ-A-001", "REQ-A-002", provenance=None)
        eligible, warnings = check_ground_truth_eligibility([pair], "TEST", strict=True)

        assert len(eligible) == 1
        assert eligible[0].id == "P-001"
        assert any("NO provenance" in w for w in warnings)

    def test_provisional_record_rejected_strict(self):
        """PROVISIONAL pairs raise GroundTruthViolationError in strict mode."""
        pair = _make_pair(
            "P-PROV", "REQ-A-001", "REQ-A-002",
            provenance=_provisional_provenance()
        )
        with pytest.raises(GroundTruthViolationError, match="explicit VERIFIED status"):
            check_ground_truth_eligibility([pair], "TEST", strict=True)

    def test_provisional_record_excluded_non_strict(self):
        """PROVISIONAL pairs are excluded and warned (not raised) in non-strict mode."""
        pair = _make_pair(
            "P-PROV", "REQ-A-001", "REQ-A-002",
            provenance=_provisional_provenance()
        )
        eligible, warnings = check_ground_truth_eligibility([pair], "TEST", strict=False)

        assert len(eligible) == 0
        assert any("REJECTED" in w and "provisional" in w and "VERIFIED" in w for w in warnings)

    @pytest.mark.parametrize("status", [VerificationStatus.PENDING_REVIEW, VerificationStatus.HUMAN_REVIEWED])
    def test_unverified_record_rejected(self, status):
        """Labels imported or reviewed without explicit verification are not ground truth."""
        pair = _make_pair(
            f"P-{status.value}", "REQ-A-001", "REQ-A-002",
            provenance=AnnotationProvenance(source=AnnotationSource.IMPORTED, status=status),
        )

        with pytest.raises(GroundTruthViolationError, match="explicit VERIFIED status"):
            check_ground_truth_eligibility([pair], "TEST", strict=True)

        eligible, warnings = check_ground_truth_eligibility([pair], "TEST", strict=False)
        assert eligible == []
        assert any("REJECTED" in warning and status.value in warning for warning in warnings)

    def test_verified_pair_accepted(self):
        """Human-verified records with an attributable, matching label are accepted."""
        pair = _make_pair(
            "P-VER", "REQ-A-001", "REQ-A-002",
            provenance=_verified_provenance()
        )
        eligible, warnings = check_ground_truth_eligibility([pair], "TEST", strict=True)

        assert len(eligible) == 1

    @pytest.mark.parametrize(
        ("provenance", "problem"),
        [
            (
                AnnotationProvenance(
                    source=AnnotationSource.LLM,
                    status=VerificationStatus.VERIFIED,
                    annotator_id="machine",
                    accepted_label=DependencyType.REQUIRES,
                ),
                "not human",
            ),
            (
                AnnotationProvenance(
                    source=AnnotationSource.HUMAN,
                    status=VerificationStatus.VERIFIED,
                    accepted_label=DependencyType.REQUIRES,
                ),
                "missing reviewer_id and annotator_id",
            ),
            (
                AnnotationProvenance(
                    source=AnnotationSource.HUMAN,
                    status=VerificationStatus.VERIFIED,
                    annotator_id="   ",
                    reviewer_id="",
                    accepted_label=DependencyType.REQUIRES,
                ),
                "missing reviewer_id and annotator_id",
            ),
            (
                AnnotationProvenance(
                    source=AnnotationSource.HUMAN,
                    status=VerificationStatus.VERIFIED,
                    reviewer_id="human_reviewer_1",
                    accepted_label=DependencyType.REQUIRES,
                ),
                "generic placeholder identity",
            ),
            (
                AnnotationProvenance(
                    source=AnnotationSource.HUMAN,
                    status=VerificationStatus.VERIFIED,
                    annotator_id="reviewer",
                ),
                "missing accepted_label",
            ),
            (
                AnnotationProvenance(
                    source=AnnotationSource.HUMAN,
                    status=VerificationStatus.VERIFIED,
                    annotator_id="reviewer",
                    accepted_label=DependencyType.CONFLICTS,
                ),
                "does not match dependency_type",
            ),
        ],
    )
    def test_contradictory_verified_provenance_is_rejected(self, provenance, problem):
        pair = _make_pair(
            "P-CONTRADICTORY", "REQ-A-001", "REQ-A-002", provenance=provenance
        )

        with pytest.raises(GroundTruthViolationError, match=problem):
            check_ground_truth_eligibility([pair], "TEST", strict=True)

        eligible, warnings = check_ground_truth_eligibility(
            [pair], "TEST", strict=False
        )
        assert eligible == []
        assert any("REJECTED" in warning and problem in warning for warning in warnings)

        from leredd.data_models import Prediction

        prediction = Prediction(
            pair_id=pair.id,
            dependency_type=DependencyType.REQUIRES,
            rationale="Controlled regression assertion.",
            confidence=5,
        )
        with pytest.raises(ValueError, match=problem):
            Evaluator.evaluate([pair], [prediction])

    def test_none_dependency_type_skipped(self):
        """Pairs with dependency_type=None are skipped (cannot evaluate)."""
        pair = _make_pair("P-NONE", "REQ-A-001", "REQ-A-002")
        pair = pair.model_copy(update={"dependency_type": None})
        eligible, warnings = check_ground_truth_eligibility([pair], "TEST", strict=True)

        assert len(eligible) == 0
        assert any("dependency_type=None" in w for w in warnings)

    def test_ui_test_fixture_rejected_if_provisional(self):
        """
        Simulated ui_test fixture: pair with PROVISIONAL status is rejected
        even if it looks like it has reviewer notes.
        """
        prov = AnnotationProvenance(
            source=AnnotationSource.HUMAN,
            status=VerificationStatus.PROVISIONAL,
            review_notes="UI persistence test. Label selected only to verify decision recording.",
        )
        pair = _make_pair("P-UI", "REQ-A-001", "REQ-A-002", provenance=prov)
        with pytest.raises(GroundTruthViolationError):
            check_ground_truth_eligibility([pair], "TEST", strict=True)


# ---------------------------------------------------------------------------
# Integration Test: run_evaluation with temp split directory
# ---------------------------------------------------------------------------

class TestRunEvaluationIntegration:
    """Integration tests using a minimal fake split directory."""

    def _write_fake_split(self, split_dir: Path, strategy: str = "system"):
        """Write minimal JSON files for train, dev, test and split_metadata."""
        # ADB train pairs (system=ADB)
        adb_pairs = [
            {
                "id": f"PAIR-ADB-{i:03d}",
                "req_a": {"id": f"REQ-ADB-{i*2:03d}", "text": f"The system shall do action {i}.", "system": "ADB"},
                "req_b": {"id": f"REQ-ADB-{i*2+1:03d}", "text": f"The system shall support feature {i}.", "system": "ADB"},
                "dependency_type": dep,
                "provenance": None,
            }
            for i, dep in enumerate([
                "Requires", "Implements", "Conflicts", "Details",
                "Is similar", "No_dependency", "No_dependency",
                "Requires", "Conflicts", "No_dependency"
            ], start=1)
        ]
        (split_dir / "train_pairs.json").write_text(json.dumps(adb_pairs), encoding="utf-8")

        # TJA test pairs (system=TJA)
        tja_pairs = [
            {
                "id": f"PAIR-TJA-{i:03d}",
                "req_a": {"id": f"REQ-TJA-{i*2:03d}", "text": f"TJA shall perform task {i}.", "system": "TJA"},
                "req_b": {"id": f"REQ-TJA-{i*2+1:03d}", "text": f"TJA shall handle event {i}.", "system": "TJA"},
                "dependency_type": dep,
                "provenance": None,
            }
            for i, dep in enumerate(["Requires", "Details", "Conflicts", "No_dependency", "Requires"], start=1)
        ]
        (split_dir / "test_pairs.json").write_text(json.dumps(tja_pairs), encoding="utf-8")

        # APA dev pairs (system=APA)
        apa_pairs = [
            {
                "id": f"PAIR-APA-{i:03d}",
                "req_a": {"id": f"REQ-APA-{i*2:03d}", "text": f"APA shall comply with rule {i}.", "system": "APA"},
                "req_b": {"id": f"REQ-APA-{i*2+1:03d}", "text": f"APA shall verify condition {i}.", "system": "APA"},
                "dependency_type": dep,
                "provenance": None,
            }
            for i, dep in enumerate(["Details", "Requires", "Implements", "No_dependency", "Requires"], start=1)
        ]
        (split_dir / "dev_pairs.json").write_text(json.dumps(apa_pairs), encoding="utf-8")

        metadata = {
            "summary": {
                "strategy": strategy,
                "seed": 42,
                "counts": {
                    "train": {"pairs": 10, "requirements": 15},
                    "dev": {"pairs": 5, "requirements": 10},
                    "test": {"pairs": 5, "requirements": 9},
                },
            },
            "leakage_report": {
                "has_leakage": False,
                "req_overlap_train_test": [],
                "system_overlap_train_test": [],
            },
        }
        (split_dir / "split_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_report_written_with_required_fields(self, cached_evaluation_report_path):
        """run_evaluation writes a JSON report containing all required top-level fields."""
        assert Path(cached_evaluation_report_path).exists(), "Cached report file was not written."
        with open(cached_evaluation_report_path, "r", encoding="utf-8") as f:
            saved = json.load(f)

            required_fields = [
                "schema_version", "script_version", "run_timestamp",
                "configuration", "split_integrity", "split_inventory", "ground_truth_eligibility",
                "leakage_audit", "model_results", "statistical_tests",
                "evaluation_limitations", "split_metadata",
            ]
            for field in required_fields:
                assert field in saved, f"Missing required field '{field}' in report."
            assert saved["split_integrity"]["checked"] is True
            assert saved["split_integrity"]["is_disjoint"] is True

            assert "BERTBaseline" in saved["model_results"]
            assert "TFIDF_LSA" in saved["model_results"]
            assert "LEREDD_Hybrid" in saved["model_results"]
            # ISSUE-011: LEREDD_MockLLM was a phantom duplicate of LEREDD_Hybrid and has been removed.
            assert "LEREDD_MockLLM" not in saved["model_results"], (
                "LEREDD_MockLLM must not appear in model_results — it was a copy-paste "
                "duplicate of LEREDD_Hybrid (ISSUE-011)."
            )

            # The report must describe this supplied fixture, not legacy hard-coded systems.
            assert saved["split_inventory"]["train"]["systems"] == ["DEMO-TRAIN"]
            assert saved["split_inventory"]["dev"]["systems"] == ["DEMO-DEV"]
            assert saved["split_inventory"]["test"]["systems"] == ["DEMO-TEST"]
            assert saved["ground_truth_eligibility"]["raw_counts"] == {
                "train": 10, "dev": 5, "test": 5
            }
            assert saved["ground_truth_eligibility"]["eligible_counts"] == {
                "train": 10, "dev": 5, "test": 5
            }
            assert saved["ground_truth_eligibility"]["provenance_free_count"] == 20
            assert saved["ground_truth_eligibility"]["eligible_provenance_count"] == 0
            assert "accepted conditionally" in saved["ground_truth_eligibility"]["provenance_note"]
            assert "DEMO-TRAIN" in saved["model_results"]["BERTBaseline"]["note"]
            assert "DEMO-TEST" in saved["evaluation_limitations"][0]

    def test_no_test_req_ids_in_retrieval_pool_logged(self, cached_evaluation_report_path):
        """Leakage audit in report confirms 0 contaminated pairs for disjoint split."""
        with open(cached_evaluation_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

            leakage = report["leakage_audit"]
            assert leakage["contaminated_count"] == 0, (
                f"Expected 0 contaminated pairs but got: {leakage['contaminated_pairs']}"
            )
            assert leakage["contamination_detected"] is False

    def test_test_pairs_not_in_retrieval_pool(self, cached_evaluation_report_path):
        """Explicitly verify that test req IDs appear in excluded_req_ids in the audit."""
        with open(cached_evaluation_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

            excluded = set(report["leakage_audit"]["excluded_req_ids"])
            # Test fixture requirement IDs are excluded from the retrieval pool.
            for i in range(1, 6):
                assert f"REQ-DEMO-TEST-{i*2:03d}" in excluded
                assert f"REQ-DEMO-TEST-{i*2+1:03d}" in excluded

    def test_provisional_pairs_in_test_raise_error(self):
        """If a PROVISIONAL pair is written into the test partition, run_evaluation raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "system_disjoint"
            split_dir.mkdir()
            self._write_fake_split(split_dir)

            # Overwrite test partition with a PROVISIONAL record
            bad_test = [
                {
                    "id": "PAIR-PROV-001",
                    "req_a": {"id": "REQ-TJA-002", "text": "Provisional req A.", "system": "TJA"},
                    "req_b": {"id": "REQ-TJA-003", "text": "Provisional req B.", "system": "TJA"},
                    "dependency_type": "Requires",
                    "provenance": {"source": "llm", "status": "provisional"},
                }
            ]
            (split_dir / "test_pairs.json").write_text(json.dumps(bad_test), encoding="utf-8")

            output_path = Path(tmpdir) / "report.json"
            with pytest.raises(GroundTruthViolationError, match="explicit VERIFIED status"):
                run_evaluation(
                    split_dir=str(split_dir),
                    output_path=str(output_path),
                    strict_gt=True,
                )

    def test_empty_test_partition_raises_value_error(self):
        """run_evaluation raises ValueError when the test partition file is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "system_disjoint"
            split_dir.mkdir()
            self._write_fake_split(split_dir)

            # Empty the test file
            (split_dir / "test_pairs.json").write_text("[]", encoding="utf-8")

            output_path = Path(tmpdir) / "report.json"
            with pytest.raises(ValueError, match="Test partition is empty"):
                run_evaluation(split_dir=str(split_dir), output_path=str(output_path))

    def test_contamination_raises_in_strict_mode(self):
        """Overlapping req IDs between train and test raise ContaminationError in strict mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "system_disjoint"
            split_dir.mkdir()
            self._write_fake_split(split_dir, strategy="pair")

            # Inject a shared req ID between train and test
            contaminated_train = [
                {
                    "id": "PAIR-CONTAM-001",
                    "req_a": {"id": "REQ-TJA-002", "text": "Shared req leaks to train.", "system": "ADB"},
                    "req_b": {"id": "REQ-ADB-001", "text": "Normal ADB req.", "system": "ADB"},
                    "dependency_type": "Requires",
                    "provenance": None,
                }
            ]
            (split_dir / "train_pairs.json").write_text(json.dumps(contaminated_train), encoding="utf-8")

            output_path = Path(tmpdir) / "report.json"
            with pytest.raises(ContaminationError, match="CONTAMINATION DETECTED"):
                run_evaluation(
                    split_dir=str(split_dir),
                    output_path=str(output_path),
                    strict_contamination=True,
                )

    def test_contamination_log_only_in_non_strict_mode(self):
        """In non-strict mode, contaminated pairs are removed and evaluation continues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "system_disjoint"
            split_dir.mkdir()
            self._write_fake_split(split_dir, strategy="pair")

            # Inject a shared req ID between train and test
            contaminated_train = [
                {
                    "id": "PAIR-CONTAM-001",
                    "req_a": {"id": "REQ-TJA-002", "text": "Shared req leaks to train.", "system": "ADB"},
                    "req_b": {"id": "REQ-ADB-001", "text": "Normal ADB req.", "system": "ADB"},
                    "dependency_type": "Requires",
                    "provenance": None,
                },
                {
                    "id": "PAIR-CLEAN-001",
                    "req_a": {"id": "REQ-ADB-100", "text": "The system shall operate.", "system": "ADB"},
                    "req_b": {"id": "REQ-ADB-101", "text": "The system shall respond.", "system": "ADB"},
                    "dependency_type": "Requires",
                    "provenance": None,
                },
            ]
            (split_dir / "train_pairs.json").write_text(json.dumps(contaminated_train), encoding="utf-8")

            output_path = Path(tmpdir) / "report.json"
            # Should NOT raise even with contamination
            report = run_evaluation(
                split_dir=str(split_dir),
                output_path=str(output_path),
                strict_contamination=False,
                strict_gt=True,
            )

            assert report["leakage_audit"]["contaminated_count"] == 1
            assert report["leakage_audit"]["safe_pool_size"] == 1

    def test_report_model_metrics_have_correct_schema(self, cached_evaluation_report_path):
        """Each model result dict contains the required metric keys."""
        with open(cached_evaluation_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

            for model_key, result in report["model_results"].items():
                for field in ["model", "note", "overall_accuracy", "macro_f1",
                              "weighted_f1", "per_class_metrics", "confusion_matrix"]:
                    assert field in result, f"Model '{model_key}' missing field '{field}'"
                assert isinstance(result["overall_accuracy"], float)
                assert isinstance(result["macro_f1"], float)

    def test_bert_classifier_type_metadata_rf_and_lr(self):
        """Passing bert_model_type='rf' or 'lr' updates configuration and model identity tag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "system_disjoint"
            split_dir.mkdir()
            self._write_fake_split(split_dir)

            out_rf = Path(tmpdir) / "report_rf.json"
            rep_rf = run_evaluation(
                split_dir=str(split_dir), output_path=str(out_rf), bert_model_type="rf"
            )
            assert rep_rf["configuration"]["bert_classifier_backend"] == "rf"
            assert "RandomForestClassifier" in rep_rf["configuration"]["bert_classifier_description"]
            assert rep_rf["model_results"]["BERTBaseline"]["model"] == "BERTBaseline-RF"

            out_lr = Path(tmpdir) / "report_lr.json"
            rep_lr = run_evaluation(
                split_dir=str(split_dir), output_path=str(out_lr), bert_model_type="lr"
            )
            assert rep_lr["configuration"]["bert_classifier_backend"] == "lr"
            assert "LogisticRegression" in rep_lr["configuration"]["bert_classifier_description"]
            assert rep_lr["model_results"]["BERTBaseline"]["model"] == "BERTBaseline-LR"

    def test_terminology_supervised_cross_system(self, cached_evaluation_report_path):
        """BERTBaseline note must use 'Supervised cross-system evaluation', not 'zero-shot'."""
        with open(cached_evaluation_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

            bert_note = report["model_results"]["BERTBaseline"]["note"]
            assert "Supervised cross-system evaluation" in bert_note
            assert "zero-shot" not in bert_note.lower() or "Not a zero-shot model" in bert_note

    def test_mcnemar_exploratory_notice(self, cached_evaluation_report_path):
        """McNemar statistical test contains exploratory status and notice."""
        with open(cached_evaluation_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

            mcnemar = report["statistical_tests"]["mcnemar"]
            assert mcnemar["status"] == "exploratory"
            assert "exploratory_notice" in mcnemar
            assert "leredd_vs_bert" in mcnemar
            assert "leredd_vs_tfidf" in mcnemar

    def test_mock_llm_demarcation_in_report(self, cached_evaluation_report_path):
        """Report correctly identifies MockLLM execution via configuration and LEREDD_Hybrid note.

        After ISSUE-011 fix: LEREDD_MockLLM duplicate entry has been removed.
        Demarcation is carried by configuration.leredd_provider and LEREDD_Hybrid note.
        """
        with open(cached_evaluation_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

            # Primary demarcation: top-level configuration field
            assert report["configuration"]["leredd_provider"] == "mock"
            assert "NOT a real LLM evaluation" in report["configuration"]["leredd_note"]

            ablation_validity = report["ablation_validity"]
            assert ablation_validity["is_empirical_model_evidence"] is False
            assert ablation_validity["retrieval_strategy"]["status"] == "not_informative_with_mock_backend"
            assert ablation_validity["in_context_examples"]["status"] == "not_informative_with_mock_backend"
            assert ablation_validity["confidence_thresholding"]["status"] == "not_informative_with_mock_backend"
            assert "fixed confidence 5" in ablation_validity["confidence_thresholding"]["reason"]
            ablation_report = Path(cached_evaluation_report_path).parent / "ablation_report.json"
            saved_ablation = json.loads(ablation_report.read_text(encoding="utf-8"))
            assert saved_ablation["ablation_validity"] == ablation_validity
            assert "does not interpret the prompt's retrieved-example block" in saved_ablation["ablation_validity"]["in_context_examples"]["reason"]

            # LEREDD_Hybrid note must identify the proposed system and mock backend context
            hybrid_note = report["model_results"]["LEREDD_Hybrid"]["note"]
            assert "Hybrid RRF retrieval" in hybrid_note

            # ISSUE-011 regression: the redundant duplicate key must be absent
            assert "LEREDD_MockLLM" not in report["model_results"], (
                "LEREDD_MockLLM must not appear — it was a phantom duplicate of "
                "LEREDD_Hybrid (ISSUE-011)."
            )

    def test_saved_predictions_and_confidence_interval_in_report(self, cached_evaluation_report_path):
        """Report contains explicit predictions and 95% Wilson confidence intervals for each model."""
        with open(cached_evaluation_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

            for model_key, result in report["model_results"].items():
                assert "predictions" in result, f"Missing 'predictions' in {model_key}"
                assert len(result["predictions"]) == 5
                assert "confidence_interval_95" in result, f"Missing 'confidence_interval_95' in {model_key}"
                ci = result["confidence_interval_95"]
                assert ci["method"] == "Wilson score interval"
                assert "lower_bound" in ci
                assert "upper_bound" in ci
                assert "margin_pp" in ci

            mcnemar = report["statistical_tests"]["mcnemar"]
            assert "leredd_vs_bert" in mcnemar
            analysis = mcnemar["leredd_vs_bert"]
            assert "contingency_table" in analysis
            assert "discordant_counts" in analysis
            assert "discordant_pairs" in analysis
            assert "statistic" in analysis
            assert "p_value" in analysis

    def test_deterministic_reproducibility_seed_42(self):
        """Repeated runs with seed 42 produce identical predictions and metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "system_disjoint"
            split_dir.mkdir()
            self._write_fake_split(split_dir)

            out1 = Path(tmpdir) / "report1.json"
            out2 = Path(tmpdir) / "report2.json"

            rep1 = run_evaluation(split_dir=str(split_dir), output_path=str(out1), seed=42)
            rep2 = run_evaluation(split_dir=str(split_dir), output_path=str(out2), seed=42)

            # ISSUE-011: LEREDD_MockLLM removed; check LEREDD_Hybrid instead.
            for model_key in ["BERTBaseline", "TFIDF_LSA", "LEREDD_Hybrid"]:
                m1 = rep1["model_results"][model_key]
                m2 = rep2["model_results"][model_key]
                assert m1["overall_accuracy"] == m2["overall_accuracy"]
                assert m1["macro_f1"] == m2["macro_f1"]
                assert m1["predictions"] == m2["predictions"]


def test_evaluator_wilson_ci_and_mcnemar_analysis():
    """Unit tests for Evaluator.compute_wilson_score_interval and Evaluator.analyze_mcnemar."""
    ci = Evaluator.compute_wilson_score_interval(2, 5)
    assert round(ci["point_estimate"], 2) == 0.40
    assert 0.10 <= ci["lower_bound"] <= 0.15
    assert 0.70 <= ci["upper_bound"] <= 0.80

    req_a = Requirement(id="REQ-1", text="Req 1", system="SYS")
    req_b = Requirement(id="REQ-2", text="Req 2", system="SYS")
    pair = RequirementPair(id="P-1", req_a=req_a, req_b=req_b, dependency_type=DependencyType.REQUIRES)

    preds_a = [Prediction(pair_id="P-1", dependency_type=DependencyType.REQUIRES, confidence=1.0, rationale="Test pred A")]
    preds_b = [Prediction(pair_id="P-1", dependency_type=DependencyType.CONFLICTS, confidence=1.0, rationale="Test pred B")]

    res = Evaluator.analyze_mcnemar([pair], preds_a, preds_b)
    assert res["contingency_table"]["model_a_only_n10_b"] == 1
    assert res["contingency_table"]["model_b_only_n01_c"] == 0
    assert res["discordant_counts"]["b"] == 1
    assert res["discordant_counts"]["c"] == 0
    assert len(res["discordant_pairs"]) == 1
