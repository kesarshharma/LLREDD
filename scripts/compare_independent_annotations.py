"""Compare two independent annotation manifests and write an agreement report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from leredd.independent_annotation import (
    atomic_write_json,
    compare_independent_annotations,
    ensure_new_output_path,
    load_annotation_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report observed agreement, Cohen's kappa, and per-reviewer time. "
            "This does not adjudicate labels or mark them verified."
        )
    )
    parser.add_argument("--reviewer-a", required=True, help="First independent manifest")
    parser.add_argument("--reviewer-b", required=True, help="Second independent manifest")
    parser.add_argument("--output", required=True, help="New agreement report path")
    args = parser.parse_args(argv)

    try:
        first = load_annotation_manifest(args.reviewer_a)
        second = load_annotation_manifest(args.reviewer_b)
        report = compare_independent_annotations(first, second)
        report_path = ensure_new_output_path(args.output)
        atomic_write_json(report_path, report)
        print(f"Compared {report['pair_count']} pairs.")
        print(f"Observed agreement: {report['observed_agreement']:.4f}")
        if report["cohens_kappa_defined"]:
            print(f"Cohen's kappa: {report['cohens_kappa']:.6f}")
        else:
            print("Cohen's kappa is undefined for these label distributions.")
        print("No labels were adjudicated or promoted to verified ground truth.")
        print(f"Report written to {report_path}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
