import json
from datetime import datetime, timezone

import pytest

from leredd.data_models import DependencyType, RequirementPair
from leredd.effort_study import (
    EffortStudyAssignment,
    EffortStudyManifest,
    EffortStudyRecord,
    EffortStudySchedule,
    compare_effort_manifests,
    create_effort_schedule,
    ensure_effort_output_path,
    load_effort_study_input,
    validate_assisted_candidates,
)
from leredd.independent_annotation import sha256_file
from scripts import (
    compare_annotation_effort,
    prepare_annotation_effort_study,
    run_annotation_effort_study,
)


def _candidate_file(path, pair_count=4, *, source="llm", status="provisional", with_label=True):
    payload = []
    for index in range(pair_count):
        payload.append({
            "id": f"PAIR-{index + 1}",
            "req_a": {"id": f"REQ-A-{index}", "text": f"Requirement A {index} shall act."},
            "req_b": {"id": f"REQ-B-{index}", "text": f"Requirement B {index} shall act."},
            "dependency_type": "Requires" if with_label else None,
            "provenance": {
                "source": source,
                "status": status,
                "predicted_label": "Requires" if with_label else None,
                "accepted_label": "Requires" if with_label else None,
                "confidence_score": 4,
                "evidence": "PRIVATE MACHINE SUGGESTION EVIDENCE",
            },
        })
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _schedule(tmp_path, pair_count=4):
    source = _candidate_file(tmp_path / "candidates.json", pair_count)
    pairs = load_effort_study_input(source)
    schedule = create_effort_schedule(
        pairs,
        "reviewer-A",
        "reviewer-B",
        seed=42,
        source_sha256=sha256_file(source),
        context_sha256=None,
        study_id="study-1",
    )
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(schedule.model_dump_json(indent=2), encoding="utf-8")
    return source, schedule, schedule_path


def _manifest(schedule, schedule_path, reviewer, records):
    now = datetime.now(timezone.utc).isoformat()
    return EffortStudyManifest(
        study_id=schedule.study_id,
        source_sha256=schedule.source_sha256,
        context_sha256=schedule.context_sha256,
        schedule_sha256=sha256_file(schedule_path),
        reviewer_id=reviewer,
        status="complete",
        created_at=now,
        updated_at=now,
        records=records,
    )


def test_schedule_counterbalances_and_binds_fixed_pair_ids(tmp_path):
    source = _candidate_file(tmp_path / "candidates.json", pair_count=5)
    pairs = load_effort_study_input(source)
    schedule = create_effort_schedule(
        pairs,
        "reviewer-A",
        "reviewer-B",
        seed=14,
        source_sha256=sha256_file(source),
        context_sha256=None,
        study_id="study-counterbalanced",
    )

    assert {item.pair_id for item in schedule.assignments} == {pair.id for pair in pairs}
    for reviewer in schedule.reviewer_ids:
        manual = sum(item.manual_reviewer == reviewer for item in schedule.assignments)
        assisted = sum(item.assisted_reviewer == reviewer for item in schedule.assignments)
        assert abs(manual - assisted) <= 1
    assert all(item.manual_reviewer != item.assisted_reviewer for item in schedule.assignments)


@pytest.mark.parametrize(
    ("source", "status", "with_label"),
    [("human", "verified", True), ("imported", "pending_review", True), ("llm", "provisional", False)],
)
def test_assisted_study_rejects_nonprovisional_or_unlabeled_inputs(tmp_path, source, status, with_label):
    candidate_path = _candidate_file(
        tmp_path / "candidates.json", pair_count=1,
        source=source, status=status, with_label=with_label,
    )
    pairs = [RequirementPair.model_validate(row) for row in json.loads(candidate_path.read_text())]

    with pytest.raises(ValueError, match="provisional machine-label provenance|no provisional suggestion"):
        validate_assisted_candidates(pairs)


def test_prepare_cli_writes_counterbalanced_schedule_without_mutating_candidates(tmp_path, capsys):
    source = _candidate_file(tmp_path / "candidates.json", pair_count=3)
    before = source.read_bytes()
    schedule_path = tmp_path / "schedule.json"

    assert prepare_annotation_effort_study.main([
        "--input", str(source), "--reviewer-a", "reviewer-A",
        "--reviewer-b", "reviewer-B", "--seed", "3", "--output", str(schedule_path),
    ]) == 0

    schedule = EffortStudySchedule.model_validate_json(schedule_path.read_text(encoding="utf-8"))
    assert len(schedule.assignments) == 3
    assert "no annotations were created" in capsys.readouterr().out
    assert source.read_bytes() == before


def test_effort_study_clis_complete_synthetic_manual_and_assisted_journey(
    tmp_path, monkeypatch, capsys
):
    source = _candidate_file(tmp_path / "candidates.json", pair_count=2)
    schedule_path = tmp_path / "schedule.json"
    assert prepare_annotation_effort_study.main([
        "--input", str(source), "--reviewer-a", "reviewer-A",
        "--reviewer-b", "reviewer-B", "--seed", "11", "--output", str(schedule_path),
    ]) == 0
    schedule = EffortStudySchedule.model_validate_json(schedule_path.read_text(encoding="utf-8"))

    for reviewer_id in schedule.reviewer_ids:
        answers = iter(["1", "fixture reason"] * 2)
        ticks = iter([float(value) for value in range(20)])
        monkeypatch.setattr("builtins.input", lambda _prompt, answers=answers: next(answers))
        monkeypatch.setattr(run_annotation_effort_study.time, "perf_counter", lambda ticks=ticks: next(ticks))
        assert run_annotation_effort_study.main([
            "--input", str(source), "--schedule", str(schedule_path),
            "--reviewer", reviewer_id, "--output", str(tmp_path / f"{reviewer_id}.json"),
        ]) == 0

    report_path = tmp_path / "effort-report.json"
    assert compare_annotation_effort.main([
        "--reviewer-a", str(tmp_path / "reviewer-A.json"),
        "--reviewer-b", str(tmp_path / "reviewer-B.json"),
        "--schedule", str(schedule_path), "--output", str(report_path),
    ]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pair_count"] == 2
    assert report["conditions"]["manual"]["annotation_count"] == 2
    assert report["conditions"]["assisted"]["annotation_count"] == 2
    assert report["labels_are_ground_truth"] is False
    assert "descriptive only" in capsys.readouterr().out


@pytest.mark.parametrize("condition", ["manual", "assisted"])
def test_run_cli_hides_or_shows_machine_suggestion_by_condition(tmp_path, monkeypatch, capsys, condition):
    source, schedule, schedule_path = _schedule(tmp_path, pair_count=1)
    assignment = schedule.assignments[0]
    reviewer = assignment.manual_reviewer if condition == "manual" else assignment.assisted_reviewer
    answers = iter(["1", "study rationale"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(run_annotation_effort_study.time, "perf_counter", iter([1.0, 4.0]).__next__)
    output = tmp_path / f"{reviewer}.json"
    before = source.read_bytes()

    assert run_annotation_effort_study.main([
        "--input", str(source), "--schedule", str(schedule_path),
        "--reviewer", reviewer, "--output", str(output),
    ]) == 0

    stdout = capsys.readouterr().out
    if condition == "manual":
        assert "PRIVATE MACHINE SUGGESTION EVIDENCE" not in stdout
        assert "No generated label" in stdout
    else:
        assert "Provisional machine suggestion: Requires" in stdout
        assert "PRIVATE MACHINE SUGGESTION EVIDENCE" in stdout
    saved = EffortStudyManifest.model_validate_json(output.read_text(encoding="utf-8"))
    assert saved.records[0].condition == condition
    assert saved.records[0].elapsed_seconds == 3.0
    assert saved.records[0].label == DependencyType.REQUIRES
    assert source.read_bytes() == before


def test_compare_effort_reports_descriptive_per_condition_time_and_label_disagreement(tmp_path):
    source, schedule, schedule_path = _schedule(tmp_path, pair_count=4)
    now = datetime.now(timezone.utc).isoformat()
    records = {reviewer: [] for reviewer in schedule.reviewer_ids}
    for index, assignment in enumerate(schedule.assignments):
        manual_label = DependencyType.REQUIRES
        assisted_label = DependencyType.CONFLICTS if index == 0 else DependencyType.REQUIRES
        manual_seconds = 12.0 + index
        assisted_seconds = 5.0 + index
        records[assignment.manual_reviewer].append(EffortStudyRecord(
            pair_id=assignment.pair_id,
            condition="manual",
            reviewer_id=assignment.manual_reviewer,
            label=manual_label,
            elapsed_seconds=manual_seconds,
            recorded_at=now,
        ))
        records[assignment.assisted_reviewer].append(EffortStudyRecord(
            pair_id=assignment.pair_id,
            condition="assisted",
            reviewer_id=assignment.assisted_reviewer,
            label=assisted_label,
            elapsed_seconds=assisted_seconds,
            recorded_at=now,
        ))
    first = _manifest(schedule, schedule_path, schedule.reviewer_ids[0], records[schedule.reviewer_ids[0]])
    second = _manifest(schedule, schedule_path, schedule.reviewer_ids[1], records[schedule.reviewer_ids[1]])

    report = compare_effort_manifests(first, second, schedule)

    assert report["pair_count"] == 4
    assert report["conditions"]["manual"]["annotation_count"] == 4
    assert report["conditions"]["assisted"]["annotation_count"] == 4
    assert report["mean_manual_minus_assisted_seconds"] == 7.0
    assert report["manual_assisted_label_disagreements"] == 1
    assert report["labels_are_ground_truth"] is False
    assert report["inferential_significance_tested"] is False


def test_compare_script_writes_report_and_refuses_incomplete_manifests(tmp_path, capsys):
    _, schedule, schedule_path = _schedule(tmp_path, pair_count=2)
    now = datetime.now(timezone.utc).isoformat()
    reviewer_a_records = []
    reviewer_b_records = []
    for assignment in schedule.assignments:
        reviewer_a_condition = "manual" if assignment.manual_reviewer == "reviewer-A" else "assisted"
        reviewer_b_condition = "manual" if assignment.manual_reviewer == "reviewer-B" else "assisted"
        reviewer_a_records.append(EffortStudyRecord(
            pair_id=assignment.pair_id,
            condition=reviewer_a_condition,
            reviewer_id="reviewer-A",
            label=DependencyType.REQUIRES,
            elapsed_seconds=10,
            recorded_at=now,
        ))
        reviewer_b_records.append(EffortStudyRecord(
            pair_id=assignment.pair_id,
            condition=reviewer_b_condition,
            reviewer_id="reviewer-B",
            label=DependencyType.REQUIRES,
            elapsed_seconds=7,
            recorded_at=now,
        ))
    first_path, second_path = tmp_path / "a.json", tmp_path / "b.json"
    first_path.write_text(_manifest(schedule, schedule_path, "reviewer-A", reviewer_a_records).model_dump_json())
    second_path.write_text(_manifest(schedule, schedule_path, "reviewer-B", reviewer_b_records).model_dump_json())
    output = tmp_path / "report.json"

    assert compare_annotation_effort.main([
        "--reviewer-a", str(first_path), "--reviewer-b", str(second_path),
        "--schedule", str(schedule_path), "--output", str(output),
    ]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["conditions"]["manual"]["annotation_count"] == 2
    assert result["conditions"]["assisted"]["annotation_count"] == 2
    assert "descriptive only" in capsys.readouterr().out

    incomplete_path = tmp_path / "incomplete.json"
    incomplete = _manifest(schedule, schedule_path, "reviewer-B", reviewer_b_records)
    incomplete.status = "in_progress"
    incomplete_path.write_text(incomplete.model_dump_json())
    assert compare_annotation_effort.main([
        "--reviewer-a", str(first_path), "--reviewer-b", str(incomplete_path),
        "--schedule", str(schedule_path), "--output", str(tmp_path / "bad-report.json"),
    ]) == 2


def test_schedule_rejects_imbalanced_condition_assignment(tmp_path):
    source, schedule, _ = _schedule(tmp_path, pair_count=4)
    all_to_a = [
        EffortStudyAssignment(
            pair_id=assignment.pair_id,
            manual_reviewer="reviewer-A",
            assisted_reviewer="reviewer-B",
        )
        for assignment in schedule.assignments
    ]

    with pytest.raises(ValueError, match="balance manual and assisted"):
        schedule.model_copy(update={"assignments": all_to_a}).model_validate(
            schedule.model_copy(update={"assignments": all_to_a}).model_dump()
        )


def test_effort_artifacts_are_rejected_under_project_data(tmp_path):
    from pathlib import Path

    project_data = Path(__file__).resolve().parents[1] / "data"
    unsafe_path = project_data / "annotated" / "raw" / "study-output.json"

    with pytest.raises(ValueError, match="under project data"):
        ensure_effort_output_path(unsafe_path)

    safe_path = ensure_effort_output_path(tmp_path / "study-output.json")
    assert safe_path == (tmp_path / "study-output.json").resolve()
