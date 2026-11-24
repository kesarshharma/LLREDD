"""Estimate candidate positive recall from a verified stratified pilot sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from leredd.candidate_coverage import analyze_candidate_recall
from leredd.independent_annotation import atomic_write_json, ensure_new_output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate positive dependency recall for a candidate strategy from "
            "a stratified sample and a separately adjudicated VERIFIED pair file."
        )
    )
    parser.add_argument("--pilot-manifest", required=True, help="Pinned pilot manifest JSON")
    parser.add_argument("--strata-map", required=True, help="Analysis-only sample strata map")
    parser.add_argument(
        "--verified-pairs", required=True,
        help="Human-adjudicated VERIFIED pair JSON",
    )
    parser.add_argument("--output", required=True, help="New JSON report path")
    args = parser.parse_args(argv)

    try:
        report = analyze_candidate_recall(
            args.pilot_manifest, args.strata_map, args.verified_pairs
        )
        output_path = ensure_new_output_path(args.output)
        atomic_write_json(output_path, report)
        print(f"Estimated positive dependency recall: {report['estimated_positive_recall']:.4f}")
        if report["approx_95_percent_confidence_interval"] is None:
            print(f"Approximate interval unavailable: {report['uncertainty_note']}")
        else:
            low, high = report["approx_95_percent_confidence_interval"]
            print(f"Approximate 95% interval: [{low:.4f}, {high:.4f}]")
        print("Exploratory pilot estimate only; verified labels and input hashes are recorded.")
        print(f"Report written to {output_path}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
