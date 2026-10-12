"""Create system-disjoint and requirement-disjoint pair dataset splits."""

import argparse
from datetime import datetime, timezone
import glob
from pathlib import Path
import sys
import uuid

from leredd.utils import load_pairs_json
from leredd.dataset_splitter import DatasetSplitter, SplitConfig, SplitStrategy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="*", default=None,
        help="Pair JSON input files. Defaults to data/annotated/*_pairs.json.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output base directory. Defaults to a new run under data/split_runs/.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing split output directories.")
    args = parser.parse_args(argv)

    input_files = args.input or sorted(glob.glob("data/annotated/*_pairs.json"))
    input_files = [Path(filename) for filename in input_files]
    if not input_files:
        parser.error("No input pair JSON files were provided or found.")
    missing = [str(filename) for filename in input_files if not filename.is_file()]
    if missing:
        parser.error(f"Input file(s) not found: {', '.join(missing)}")

    output_base = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            Path("data/split_runs")
            / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        ).resolve()
    )
    project_data = (Path.cwd() / "data").resolve()
    protected_roots = [
        project_data / name for name in ("annotated", "raw", "replication")
    ]
    if any(output_base == root or root in output_base.parents for root in protected_roots):
        parser.error("Split output cannot be written inside source dataset directories.")
    split_paths = [output_base / "system_disjoint", output_base / "requirement_disjoint"]
    if not args.overwrite and any(path.exists() for path in split_paths):
        parser.error("Refusing to overwrite existing split directories; choose a fresh output directory or pass --overwrite.")

    try:
        pairs = []
        for filename in input_files:
            pairs.extend(load_pairs_json(str(filename)))
        system_config = SplitConfig(
            strategy=SplitStrategy.SYSTEM,
            explicit_system_splits={"train": ["ADB"], "dev": ["APA"], "test": ["TJA"]},
            random_seed=42,
        )
        system_split = DatasetSplitter.split(pairs, config=system_config)
        requirement_config = SplitConfig(
            strategy=SplitStrategy.REQUIREMENT,
            train_ratio=0.6,
            dev_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
        )
        requirement_split = DatasetSplitter.split(pairs, config=requirement_config)
        DatasetSplitter.save_split(system_split, str(split_paths[0]))
        DatasetSplitter.save_split(requirement_split, str(split_paths[1]))
        system_leakage = DatasetSplitter.check_leakage(system_split)
        requirement_leakage = DatasetSplitter.check_leakage(requirement_split)
    except (OSError, ValueError) as exc:
        parser.error(f"Dataset split failed: {exc}")

    print(
        f"System-disjoint: train={len(system_split.train_pairs)}, "
        f"dev={len(system_split.dev_pairs)}, test={len(system_split.test_pairs)}, "
        f"leakage={system_leakage.has_leakage}; output={split_paths[0]}"
    )
    print(
        f"Requirement-disjoint: train={len(requirement_split.train_pairs)}, "
        f"dev={len(requirement_split.dev_pairs)}, test={len(requirement_split.test_pairs)}, "
        f"leakage={requirement_leakage.has_leakage}; output={split_paths[1]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
