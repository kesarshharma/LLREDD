"""Compare completed counterbalanced manual/assisted annotation effort manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from leredd.effort_study import (
    EffortStudyManifest,
    EffortStudySchedule,
    compare_effort_manifests,
    ensure_effort_output_path,
)
from leredd.independent_annotation import (
    atomic_write_json,
    ensure_new_output_path,
    sha256_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", required=True, help="First reviewer effort manifest")
    parser.add_argument("--reviewer-b", required=True, help="Second reviewer effort manifest")
    parser.add_argument("--schedule", required=True, help="Original counterbalanced study schedule")
    parser.add_argument("--output", required=True, help="New report path")
    args = parser.parse_args(argv)

    try:
        first_path = Path(args.reviewer_a).expanduser().resolve()
        second_path = Path(args.reviewer_b).expanduser().resolve()
        schedule_path = Path(args.schedule).expanduser().resolve()
        output_path = ensure_effort_output_path(ensure_new_output_path(args.output))
        if output_path in {first_path, second_path, schedule_path}:
            raise ValueError("Report output must differ from its input manifests and schedule.")
        first = EffortStudyManifest.model_validate(json.loads(first_path.read_text(encoding="utf-8")))
        second = EffortStudyManifest.model_validate(json.loads(second_path.read_text(encoding="utf-8")))
        schedule = EffortStudySchedule.model_validate(json.loads(schedule_path.read_text(encoding="utf-8")))
        expected_schedule_hash = sha256_file(schedule_path)
        if any(manifest.schedule_sha256 != expected_schedule_hash for manifest in (first, second)):
            raise ValueError("Effort manifests do not match the supplied schedule file.")
        report = compare_effort_manifests(first, second, schedule)
        atomic_write_json(output_path, report)
        print(f"Compared {report['pair_count']} pairs across manual and assisted conditions.")
        print(f"Manual mean: {report['conditions']['manual']['mean_seconds_per_pair']:.2f} seconds/pair")
        print(f"Assisted mean: {report['conditions']['assisted']['mean_seconds_per_pair']:.2f} seconds/pair")
        reduction = report["mean_relative_time_reduction_percent"]
        if reduction is not None:
            print(f"Descriptive mean time reduction: {reduction:.2f}%")
        else:
            print("Manual mean time is zero; relative reduction is undefined.")
        print("This report is descriptive only and does not establish annotation quality or ground truth.")
        print(f"Report written to {output_path}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
