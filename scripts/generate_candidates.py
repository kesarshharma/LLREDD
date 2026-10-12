"""Generate exhaustive and BM25 candidate pairs from an SRS text file."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

from leredd.utils import extract_requirements_from_file
from leredd.candidate_generator import CandidateGenerator, GenerationStrategy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw/adb_srs.txt", help="SRS text input file.")
    parser.add_argument("--system", default="ADB", help="System identifier for extracted requirements.")
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to a unique directory under data/candidates/runs/.",
    )
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Replace expected output files in the selected output directory.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"SRS input file not found: {input_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            Path("data/candidates/runs")
            / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        ).resolve()
    )
    project_data = (Path.cwd() / "data").resolve()
    protected_roots = [
        project_data / name for name in ("annotated", "raw", "replication", "splits")
    ]
    if any(output_dir == root or root in output_dir.parents for root in protected_roots):
        parser.error("Candidate output cannot be written inside source dataset directories.")

    system_token = "".join(
        character.lower() if character.isalnum() else "_" for character in args.system
    ).strip("_") or "system"
    names = [
        f"{system_token}_candidates_all.json",
        f"{system_token}_candidates_all_manifest.json",
        f"{system_token}_candidates_bm25.json",
        f"{system_token}_candidates_bm25_manifest.json",
    ]
    if not args.overwrite and any((output_dir / name).exists() for name in names):
        parser.error("Refusing to overwrite existing candidate files; choose a fresh output directory or pass --overwrite.")
    if args.max_candidates < 1 or args.top_k < 1:
        parser.error("--max-candidates and --top-k must be positive integers.")

    try:
        requirements = extract_requirements_from_file(str(input_path), system_name=args.system)
        all_pairs, all_manifest = CandidateGenerator.generate_candidates(
            requirements=requirements,
            strategy=GenerationStrategy.ALL_PAIRS,
            source_identity=str(input_path.resolve()),
            max_candidates=args.max_candidates,
        )
        bm25_pairs, bm25_manifest = CandidateGenerator.generate_candidates(
            requirements=requirements,
            strategy=GenerationStrategy.BM25,
            source_identity=str(input_path.resolve()),
            top_k=args.top_k,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        CandidateGenerator.save_candidates(
            all_pairs, all_manifest,
            str(output_dir / names[0]), str(output_dir / names[1]),
        )
        CandidateGenerator.save_candidates(
            bm25_pairs, bm25_manifest,
            str(output_dir / names[2]), str(output_dir / names[3]),
        )
    except (OSError, ValueError) as exc:
        parser.error(f"Candidate generation failed: {exc}")

    print(
        f"Generated {all_manifest.generated_count} ALL_PAIRS and "
        f"{bm25_manifest.generated_count} BM25 candidates under {output_dir}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
