"""Extract requirement records from an SRS document."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

from leredd.utils import extract_requirements_from_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("srs_path", help="Input SRS document (.txt or supported document format).")
    parser.add_argument("system_name", nargs="?", default="ADB")
    parser.add_argument("output_path", nargs="?")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args(argv)

    source = Path(args.srs_path)
    if not source.is_file():
        parser.error(f"SRS input file not found: {source}")
    output = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else (
            Path("data/processed/extraction_runs")
            / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
            / "requirements.json"
        ).resolve()
    )
    project_data = (Path.cwd() / "data").resolve()
    if any(
        output == project_data / name or (project_data / name) in output.parents
        for name in ("annotated", "raw", "replication", "splits")
    ):
        parser.error("Extracted requirement output cannot be written inside source dataset directories.")
    if output.exists() and not args.overwrite:
        parser.error(f"Refusing to overwrite existing file: {output}; pass --overwrite to replace it.")

    try:
        requirements = extract_requirements_from_file(str(source), system_name=args.system_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps([item.model_dump(mode="json") for item in requirements], indent=2),
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        parser.error(f"Requirement extraction failed: {exc}")
    print(f"Saved {len(requirements)} requirements to '{output}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
