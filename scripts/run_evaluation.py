"""Backward-compatible entry point for safe system-disjoint evaluation.

The former implementation fit and scored baselines on the same ADB records and
also supplied those records as LEREDD retrieval examples. This alias now uses
the leakage-checked evaluation runner so the historical command is not misleading.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_full_evaluation import (
    ContaminationError,
    GroundTruthViolationError,
    run_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the leakage-checked system-disjoint LEREDD demonstration benchmark."
    )
    parser.add_argument(
        "--split-dir",
        default="data/splits/system_disjoint",
        help="Directory containing train/dev/test pair partitions.",
    )
    parser.add_argument(
        "--output",
        help="Report path. Defaults to a new timestamped directory under data/evaluation/runs/.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bert-classifier-type", choices=["rf", "lr"], default="rf")
    args = parser.parse_args()

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        run_id = (
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        output_path = Path("data/evaluation/runs") / run_id / "system_disjoint_report.json"
        output_path = output_path.resolve()

    if output_path.exists() or (output_path.parent / "ablation_report.json").exists():
        parser.error(
            "Refusing to overwrite an existing report or ablation_report.json; "
            "choose a fresh output directory."
        )
    project_data = (Path.cwd() / "data").resolve()
    protected_roots = [
        project_data / name for name in ("annotated", "raw", "replication", "splits")
    ]
    if any(output_path == root or root in output_path.parents for root in protected_roots):
        parser.error("Report output cannot be written inside source dataset directories.")

    try:
        run_evaluation(
            split_dir=args.split_dir,
            output_path=str(output_path),
            seed=args.seed,
            bert_model_type=args.bert_classifier_type,
            strict_contamination=True,
            strict_gt=True,
        )
    except (ContaminationError, GroundTruthViolationError, FileNotFoundError, ValueError) as exc:
        parser.error(f"Evaluation aborted: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
