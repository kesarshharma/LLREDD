import json
from datetime import datetime, timezone

import pytest

from leredd.data_models import DependencyType
from leredd.independent_annotation import (
    IndependentAnnotationManifest,
    IndependentAnnotationRecord,
    compare_independent_annotations,
)
from scripts import annotate_independent, compare_independent_annotations as compare_script


def _candidate(path, pair_ids=("P-1", "P-2")):
    labels = ["Conflicts", "Requires"]
    payload = [
        {
            "id": pair_id,
            "req_a": {"id": f"A-{index}", "text": f"Requirement A {index}", "system": "DEMO"},
            "req_b": {"id": f"B-{index}", "text": f"Requirement B {index}", "system": "DEMO"},
            "dependency_type": labels[index % len(labels)],
            "provenance": {
                "source": "llm",
                "status": "provisional",
                "predicted_label": labels[index % len(labels)],
                "evidence": "SECRET MODEL RATIONALE",
            },
        }
        for index, pair_id in enumerate(pair_ids)
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest(reviewer, labels, *, source_hash="a" * 64, context_hash=None, status="complete"):
    now = datetime.now(timezone.utc).isoformat()
    return IndependentAnnotationManifest(
        source_sha256=source_hash,
        context_sha256=context_hash,
        reviewer_id=reviewer,
        status=status,
        created_at=now,
        updated_at=now,
        records=[
            IndependentAnnotationRecord(
                pair_id=pair_id,
                label=label,
                reviewer_id=reviewer,
                elapsed_seconds=seconds,
                recorded_at=now,
            )
            for pair_id, label, seconds in labels
        ],
    )


def test_blind_annotation_does_not_display_machine_labels_or_modify_candidates(tmp_path, monkeypatch, capsys):
    candidates = _candidate(tmp_path / "candidates.json")
    original = candidates.read_bytes()
    answers = iter(["1", "Wheel-speed fault evidence", "2", "Driver notification evidence"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(annotate_independent.time, "perf_counter", iter([1.0, 3.0, 10.0, 14.0]).__next__)
    output = tmp_path / "reviewer-a.json"

    assert annotate_independent.main([
        "--input", str(candidates), "--annotator", "reviewer-a", "--output", str(output)
    ]) == 0

    stdout = capsys.readouterr().out
    assert "SECRET MODEL RATIONALE" not in stdout
    assert "No generated label" in stdout
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["annotation_mode"] == "blind_independent"
    assert [row["label"] for row in saved["records"]] == ["Requires", "Implements"]
    assert [row["elapsed_seconds"] for row in saved["records"]] == [2.0, 4.0]
    assert saved["records"][0]["rationale"] == "Wheel-speed fault evidence"
    assert candidates.read_bytes() == original


def test_blind_annotation_saves_progress_and_can_resume(tmp_path, monkeypatch):
    candidates = _candidate(tmp_path / "candidates.json")
    output = tmp_path / "reviewer-a.json"
    answers = iter(["1", "Fault alert depends on sensor data", KeyboardInterrupt()])

    def stop_after_first(_prompt):
        answer = next(answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr("builtins.input", stop_after_first)
    monkeypatch.setattr(annotate_independent.time, "perf_counter", iter([0.0, 1.0, 2.0]).__next__)
    assert annotate_independent.main([
        "--input", str(candidates), "--annotator", "reviewer-a", "--output", str(output)
    ]) == 130
    partial = json.loads(output.read_text(encoding="utf-8"))
    assert partial["status"] == "in_progress"
    assert len(partial["records"]) == 1

    answers = iter(["2", "The dashboard displays that notification"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(annotate_independent.time, "perf_counter", iter([3.0, 5.0]).__next__)
    assert annotate_independent.main([
        "--input", str(candidates), "--annotator", "reviewer-a", "--output", str(output), "--resume"
    ]) == 0
    completed = json.loads(output.read_text(encoding="utf-8"))
    assert completed["status"] == "complete"
    assert [row["pair_id"] for row in completed["records"]] == ["P-1", "P-2"]


def test_blind_annotation_refuses_overwrite_and_placeholder_identity(tmp_path, capsys):
    candidates = _candidate(tmp_path / "candidates.json")
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    assert annotate_independent.main([
        "--input", str(candidates), "--annotator", "reviewer-a", "--output", str(output)
    ]) == 2
    assert output.read_text(encoding="utf-8") == "preserve"
    assert annotate_independent.main([
        "--input", str(candidates), "--annotator", "human_cli", "--output", str(tmp_path / "new.json")
    ]) == 2
    assert "placeholder" in capsys.readouterr().err


def test_blind_annotation_rejects_empty_candidates(tmp_path, capsys):
    candidates = tmp_path / "empty.json"
    candidates.write_text("[]", encoding="utf-8")
    output = tmp_path / "annotations.json"
    assert annotate_independent.main([
        "--input", str(candidates), "--annotator", "reviewer-a", "--output", str(output)
    ]) == 2
    assert not output.exists()
    assert "empty" in capsys.readouterr().err.lower()


def test_blind_annotation_can_abstain_with_a_reason(tmp_path, monkeypatch):
    candidates = _candidate(tmp_path / "candidates.json", pair_ids=("P-1",))
    answers = iter(["x", "The available text does not establish a relation."])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(annotate_independent.time, "perf_counter", iter([0.0, 2.0]).__next__)
    output = tmp_path / "abstaining-reviewer.json"

    assert annotate_independent.main([
        "--input", str(candidates), "--annotator", "reviewer-a", "--output", str(output)
    ]) == 0

    record = json.loads(output.read_text(encoding="utf-8"))["records"][0]
    assert record["label"] is None
    assert "does not establish" in record["rationale"]


def test_compare_reports_kappa_confusion_and_annotation_time(tmp_path):
    first = _manifest("reviewer-a", [("P-1", DependencyType.REQUIRES, 4), ("P-2", DependencyType.CONFLICTS, 8)])
    second = _manifest("reviewer-b", [("P-1", DependencyType.REQUIRES, 2), ("P-2", DependencyType.DETAILS, 6)])

    report = compare_independent_annotations(first, second)

    assert report["pair_count"] == 2
    assert report["agreement_count"] == 1
    assert report["observed_agreement"] == 0.5
    assert report["cohens_kappa"] == 0.333333
    assert report["confusion_matrix_reviewer_a_rows_reviewer_b_columns"]["Conflicts"]["Details"] == 1
    assert report["annotation_time_seconds"]["reviewer-a"]["total_seconds"] == 12
    assert report["labels_are_ground_truth"] is False


@pytest.mark.parametrize(
    "first,second,message",
    [
        (_manifest("reviewer-a", [("P-1", DependencyType.REQUIRES, 1)]), _manifest("reviewer-a", [("P-1", DependencyType.REQUIRES, 1)]), "different reviewer"),
        (_manifest("reviewer-a", [("P-1", DependencyType.REQUIRES, 1)]), _manifest("reviewer-b", [("P-1", DependencyType.REQUIRES, 1)], source_hash="b" * 64), "same source dataset"),
        (_manifest("reviewer-a", [("P-1", DependencyType.REQUIRES, 1)]), _manifest("reviewer-b", [("P-2", DependencyType.REQUIRES, 1)]), "same nonempty set"),
        (_manifest("reviewer-a", [("P-1", DependencyType.REQUIRES, 1)], status="in_progress"), _manifest("reviewer-b", [("P-1", DependencyType.REQUIRES, 1)]), "must be complete"),
        (_manifest("reviewer-a", [("P-1", DependencyType.REQUIRES, 1)]), _manifest("reviewer-b", [("P-1", DependencyType.REQUIRES, 1)], context_hash="c" * 64), "same SRS context"),
    ],
)
def test_compare_rejects_incompatible_annotations(first, second, message):
    with pytest.raises(ValueError, match=message):
        compare_independent_annotations(first, second)


def test_compare_script_writes_new_report_and_refuses_overwrite(tmp_path, capsys):
    first = _manifest("reviewer-a", [("P-1", DependencyType.REQUIRES, 1)])
    second = _manifest("reviewer-b", [("P-1", DependencyType.REQUIRES, 2)])
    path_a, path_b, output = (tmp_path / name for name in ("a.json", "b.json", "report.json"))
    path_a.write_text(first.model_dump_json(), encoding="utf-8")
    path_b.write_text(second.model_dump_json(), encoding="utf-8")

    assert compare_script.main([
        "--reviewer-a", str(path_a), "--reviewer-b", str(path_b), "--output", str(output)
    ]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pair_count"] == 1
    assert report["labels_are_ground_truth"] is False

    previous = output.read_bytes()
    assert compare_script.main([
        "--reviewer-a", str(path_a), "--reviewer-b", str(path_b), "--output", str(output)
    ]) == 2
    assert output.read_bytes() == previous
    assert "does not adjudicate" in report["note"]


def test_compare_marks_degenerate_kappa_undefined_without_warning():
    first = _manifest("reviewer-a", [("P-1", DependencyType.REQUIRES, 1)])
    second = _manifest("reviewer-b", [("P-1", DependencyType.REQUIRES, 1)])

    report = compare_independent_annotations(first, second)

    assert report["observed_agreement"] == 1.0
    assert report["cohens_kappa"] is None
    assert report["cohens_kappa_defined"] is False


def test_compare_counts_abstentions_as_a_separate_response():
    first = _manifest("reviewer-a", [("P-1", None, 2)])
    second = _manifest("reviewer-b", [("P-1", DependencyType.REQUIRES, 3)])

    report = compare_independent_annotations(first, second)

    assert report["abstention_count"] == {"reviewer-a": 1, "reviewer-b": 0}
    assert report["confusion_matrix_reviewer_a_rows_reviewer_b_columns"]["ABSTAIN"]["Requires"] == 1


def test_blind_annotation_refuses_protected_data_output_path(tmp_path, capsys):
    candidates = _candidate(tmp_path / "candidates.json")
    protected_output = tmp_path.parents[3] / "data" / "replication" / "new-human-labels.json"
    assert annotate_independent.main([
        "--input", str(candidates), "--annotator", "reviewer-a", "--output", str(protected_output)
    ]) == 2
    assert not protected_output.exists()
    assert "protected data path" in capsys.readouterr().err
