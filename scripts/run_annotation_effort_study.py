"""Collect one reviewer's scheduled manual and assisted effort-study records."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from leredd.data_models import DependencyType
from leredd.effort_study import (
    EffortStudyManifest,
    EffortStudyRecord,
    EffortStudySchedule,
    assisted_suggestion,
    ensure_effort_output_path,
    load_effort_study_input,
    validate_effort_schedule_for_source,
)
from leredd.independent_annotation import (
    atomic_write_json,
    ensure_new_output_path,
    sha256_file,
    validate_reviewer_id,
)


LABELS_BY_NUMBER = {str(index): label for index, label in enumerate(DependencyType, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Same fixed provisional candidate JSON used to prepare the schedule")
    parser.add_argument("--schedule", required=True, help="Counterbalanced schedule from prepare_annotation_effort_study.py")
    parser.add_argument("--context", help="Same fixed UTF-8 SRS context supplied when preparing the schedule")
    parser.add_argument("--reviewer", required=True, help="Reviewer ID named in the schedule")
    parser.add_argument("--output", required=True, help="New reviewer-specific effort manifest path")
    parser.add_argument("--resume", action="store_true", help="Resume a matching in-progress manifest")
    args = parser.parse_args(argv)

    try:
        reviewer_id = validate_reviewer_id(args.reviewer)
        input_path = Path(args.input).expanduser().resolve()
        schedule_path = Path(args.schedule).expanduser().resolve()
        output_path = ensure_effort_output_path(args.output, allow_existing=True)
        context_path = Path(args.context).expanduser().resolve() if args.context else None
        if output_path in {input_path, schedule_path, context_path}:
            raise ValueError("Output path must differ from candidate, schedule, and context inputs.")
        pairs = load_effort_study_input(input_path)
        schedule = EffortStudySchedule.model_validate(json.loads(schedule_path.read_text(encoding="utf-8")))
        context_hash = sha256_file(context_path) if context_path else None
        validate_effort_schedule_for_source(
            schedule,
            pairs,
            source_sha256=sha256_file(input_path),
            context_sha256=context_hash,
        )
        if reviewer_id not in schedule.reviewer_ids:
            raise ValueError("Reviewer ID is not listed in the study schedule.")
        schedule_hash = sha256_file(schedule_path)
        now = datetime.now(timezone.utc).isoformat()

        if output_path.exists():
            if not args.resume:
                raise ValueError("Output already exists; use a new path or pass --resume.")
            manifest = EffortStudyManifest.model_validate(json.loads(output_path.read_text(encoding="utf-8")))
            if manifest.status != "in_progress":
                raise ValueError("Only an in-progress effort manifest can be resumed.")
            expected = (
                schedule.study_id,
                schedule.source_sha256,
                schedule.context_sha256,
                schedule_hash,
                reviewer_id,
            )
            actual = (
                manifest.study_id,
                manifest.source_sha256,
                manifest.context_sha256,
                manifest.schedule_sha256,
                manifest.reviewer_id,
            )
            if actual != expected:
                raise ValueError("Resume study, source, context, schedule, and reviewer must match.")
        else:
            if args.resume:
                raise ValueError("Cannot resume because the effort manifest does not exist.")
            output_path = ensure_effort_output_path(output_path)
            manifest = EffortStudyManifest(
                study_id=schedule.study_id,
                source_sha256=schedule.source_sha256,
                context_sha256=schedule.context_sha256,
                schedule_sha256=schedule_hash,
                reviewer_id=reviewer_id,
                created_at=now,
                updated_at=now,
            )
            atomic_write_json(output_path, manifest.model_dump(mode="json"))

        if context_path:
            print("SRS context (same file required for both reviewers and conditions):")
            print(context_path.read_text(encoding="utf-8"))
        print("Labels recorded here remain study observations, not verified ground truth.")
        print("Choose one supported relation or X to abstain:")
        for index, label in LABELS_BY_NUMBER.items():
            print(f"  {index}. {label.value}")
        print("  X. Abstain / defer")
        print("Press Ctrl+C to save progress and stop.")

        pair_by_id = {pair.id: pair for pair in pairs}
        assignments = [
            assignment for assignment in schedule.assignments
            if reviewer_id in {assignment.manual_reviewer, assignment.assisted_reviewer}
        ]
        completed = {record.pair_id for record in manifest.records}
        for assignment in assignments:
            pair = pair_by_id[assignment.pair_id]
            condition = "manual" if assignment.manual_reviewer == reviewer_id else "assisted"
            if pair.id in completed:
                continue
            started = time.perf_counter()
            print(f"\nPair {pair.id} — {condition.upper()} condition")
            print(f"Requirement A: {pair.req_a.text}")
            print(f"Requirement B: {pair.req_b.text}")
            if condition == "assisted":
                suggestion, evidence, confidence = assisted_suggestion(pair)
                print(f"Provisional machine suggestion: {suggestion.value}")
                if confidence is not None:
                    print(f"Model/heuristic confidence: {confidence}")
                print(f"Recorded machine rationale/evidence: {evidence}")
            else:
                print("No generated label, rationale, or other reviewer decision is shown.")

            while True:
                response = input("Label number (1-6): ").strip()
                if response in LABELS_BY_NUMBER or response.casefold() == "x":
                    break
                print("Enter one of the listed numbers from 1 to 6, or X to abstain.")
            rationale = input("Brief evidence or reason for abstention (optional): ").strip()
            manifest.records.append(EffortStudyRecord(
                pair_id=pair.id,
                condition=condition,
                reviewer_id=reviewer_id,
                label=LABELS_BY_NUMBER.get(response),
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                rationale=rationale,
                recorded_at=datetime.now(timezone.utc).isoformat(),
            ))
            manifest.updated_at = datetime.now(timezone.utc).isoformat()
            atomic_write_json(output_path, manifest.model_dump(mode="json"))

        manifest.status = "complete"
        manifest.updated_at = datetime.now(timezone.utc).isoformat()
        atomic_write_json(output_path, manifest.model_dump(mode="json"))
        print(f"Saved {len(manifest.records)} study observations to {output_path}.")
        print("No annotation was adjudicated or promoted to verified ground truth.")
        return 0
    except KeyboardInterrupt:
        print("\nStopped; completed study records remain saved as in_progress.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
