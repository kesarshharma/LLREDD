"""Prepare a counterbalanced manual-versus-assisted annotation schedule."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from leredd.effort_study import (
    create_effort_schedule,
    ensure_effort_output_path,
    load_effort_study_input,
)
from leredd.independent_annotation import (
    atomic_write_json,
    ensure_new_output_path,
    sha256_file,
    validate_reviewer_id,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Fixed candidate JSON with provisional machine labels")
    parser.add_argument("--context", help="Optional fixed UTF-8 SRS context file")
    parser.add_argument("--reviewer-a", required=True, help="First attributable reviewer ID")
    parser.add_argument("--reviewer-b", required=True, help="Second attributable reviewer ID")
    parser.add_argument("--seed", type=int, default=42, help="Nonnegative counterbalancing seed")
    parser.add_argument("--output", required=True, help="New schedule JSON path")
    args = parser.parse_args(argv)

    try:
        reviewer_a = validate_reviewer_id(args.reviewer_a)
        reviewer_b = validate_reviewer_id(args.reviewer_b)
        input_path = Path(args.input).expanduser().resolve()
        output_path = ensure_effort_output_path(ensure_new_output_path(args.output))
        if output_path == input_path:
            raise ValueError("Schedule output must differ from the candidate input.")
        pairs = load_effort_study_input(input_path)
        context_path = Path(args.context).expanduser().resolve() if args.context else None
        context_hash = sha256_file(context_path) if context_path else None
        schedule = create_effort_schedule(
            pairs,
            reviewer_a,
            reviewer_b,
            seed=args.seed,
            source_sha256=sha256_file(input_path),
            context_sha256=context_hash,
            study_id=uuid.uuid4().hex,
        )
        atomic_write_json(output_path, schedule.model_dump(mode="json"))
        counts = {reviewer: {"manual": 0, "assisted": 0} for reviewer in schedule.reviewer_ids}
        for assignment in schedule.assignments:
            counts[assignment.manual_reviewer]["manual"] += 1
            counts[assignment.assisted_reviewer]["assisted"] += 1
        print(f"Prepared {len(schedule.assignments)} fixed pairs for study {schedule.study_id}.")
        print(f"Reviewer condition counts: {counts}")
        print(f"Schedule written to {output_path}; no annotations were created.")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
