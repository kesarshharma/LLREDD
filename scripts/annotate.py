"""Validate and copy pair JSON to a separate staging file.

This utility does not make human annotation decisions. Use the CLI/UI review
workflow to assign labels and VERIFIED provenance explicitly.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

from leredd.utils import load_pairs_json, save_pairs_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", help="Input requirement-pair JSON file.")
    parser.add_argument("output_json", nargs="?")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args(argv)

    output = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else (
            Path("data/processed/pair_copy_runs")
            / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
            / "pairs.json"
        ).resolve()
    )
    project_data = (Path.cwd() / "data").resolve()
    if any(
        output == project_data / name or (project_data / name) in output.parents
        for name in ("annotated", "raw", "replication", "splits")
    ):
        parser.error("Pair-copy output cannot be written inside source dataset directories.")
    if output.exists() and not args.overwrite:
        parser.error(f"Refusing to overwrite existing file: {output}; pass --overwrite to replace it.")
    try:
        pairs = load_pairs_json(args.input_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_pairs_json(pairs, str(output))
    except (OSError, ValueError) as exc:
        parser.error(f"Could not validate or copy pair JSON: {exc}")
    print(f"Validated {len(pairs)} pairs and copied them to '{output}'. No labels were changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
