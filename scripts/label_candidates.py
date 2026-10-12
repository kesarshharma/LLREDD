"""Create provisional labels; remote inference requires an explicit provider flag."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

from leredd.config import get_config
from leredd.data_models import RequirementPair
from leredd.provisional_labeler import ProvisionalLabeler, LabelingMode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/candidates/adb_candidates_bm25.json",
        help="Candidate-pair JSON input.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to a unique path under data/provisional/runs/.",
    )
    parser.add_argument(
        "--provider", choices=["mock", "openai", "ollama"], default="mock",
        help="LLM provider; defaults to mock even when environment config selects a remote provider.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files in the selected output directory.")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"Candidate input file not found: {input_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            Path("data/provisional/runs")
            / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        ).resolve()
    )
    project_data = (Path.cwd() / "data").resolve()
    protected_roots = [
        project_data / name for name in ("annotated", "raw", "replication", "splits")
    ]
    if any(output_dir == root or root in output_dir.parents for root in protected_roots):
        parser.error("Provisional output cannot be written inside source dataset directories.")

    outputs = [
        output_dir / f"{input_path.stem}_provisional_{mode}.json"
        for mode in ("heuristic", "hybrid")
    ]
    outputs += [path.with_name(f"{path.stem}_manifest.json") for path in list(outputs)]
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error("Refusing to overwrite existing provisional outputs; choose a fresh output directory or pass --overwrite.")

    try:
        with input_path.open("r", encoding="utf-8") as source:
            candidates = [RequirementPair.model_validate(item) for item in json.load(source)]
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(f"Could not load candidate pairs: {exc}")

    config = get_config().model_copy(update={"llm_provider": args.provider})
    labeler = ProvisionalLabeler(config=config)
    output_dir.mkdir(parents=True, exist_ok=True)
    for mode, output_path in zip((LabelingMode.HEURISTIC, LabelingMode.HYBRID), outputs[:2]):
        labeled, manifest = labeler.process_batch(
            candidates=candidates,
            mode=mode,
            source_identity=str(input_path.resolve()),
        )
        manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
        labeler.save_provisional_dataset(
            labeled, manifest, str(output_path), str(manifest_path)
        )
        print(
            f"{mode.value}: {manifest.labeled_count}/{manifest.input_candidate_count} labeled, "
            f"{manifest.unresolved_count} unresolved; output={output_path}"
        )
    print("All generated labels remain provisional; no human verification was recorded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
