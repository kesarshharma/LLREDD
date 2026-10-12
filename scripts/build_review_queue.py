"""Build a pending human-review queue without simulating reviewer decisions."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

from leredd.data_models import RequirementPair
from leredd.review_queue import ReviewQueueConfig, ReviewQueueManager


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/provisional/adb_provisional_hybrid.json",
        help="Provisional pair JSON input.",
    )
    parser.add_argument(
        "--output",
        help="Queue JSON output. Defaults to a unique path under data/queue/runs/.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace both existing queue and manifest files.")
    args = parser.parse_args(argv)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (
            Path("data/queue/runs")
            / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
            / "review_queue.json"
        ).resolve()
    )
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest{output_path.suffix}")
    project_data = (Path.cwd() / "data").resolve()
    protected_roots = [
        project_data / name for name in ("annotated", "raw", "replication", "splits")
    ]
    if any(output_path == root or root in output_path.parents for root in protected_roots):
        parser.error("Queue output cannot be written inside source dataset directories.")
    if not args.overwrite and (output_path.exists() or manifest_path.exists()):
        parser.error("Refusing to overwrite existing queue or manifest files; choose a fresh path or pass --overwrite.")

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"Provisional input file not found: {input_path}")
    try:
        with input_path.open("r", encoding="utf-8") as source:
            pairs = [RequirementPair.model_validate(item) for item in json.load(source)]
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(f"Could not load provisional pairs: {exc}")

    manager = ReviewQueueManager(config=ReviewQueueConfig(include_single_method=True))
    items = manager.build_queue(pairs)
    manifest = manager.create_manifest(items, source_identity=str(input_path.resolve()))
    manager.save_queue(items, manifest, str(output_path), str(manifest_path))
    print(
        f"Created {manifest.total_items} review items ({manifest.pending_count} pending). "
        f"No human decisions were recorded. Queue: {output_path}; manifest: {manifest_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
