"""Collect one reviewer's blind annotations in a separate manifest."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from leredd.data_models import DependencyType
from leredd.independent_annotation import (
    ANNOTATION_SCHEMA_VERSION,
    IndependentAnnotationManifest,
    IndependentAnnotationRecord,
    atomic_write_json,
    ensure_new_output_path,
    ensure_output_path_is_unprotected,
    load_annotation_manifest,
    load_candidate_pairs,
    sha256_file,
    validate_reviewer_id,
)


LABELS_BY_NUMBER = {str(index): label for index, label in enumerate(DependencyType, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a blinded annotation vector. Machine labels and prior reviewer "
            "decisions are never displayed; results are not automatically verified."
        )
    )
    parser.add_argument("--input", required=True, help="JSON candidate-pair file")
    parser.add_argument("--annotator", required=True, help="Real, attributable reviewer ID")
    parser.add_argument("--output", required=True, help="New JSON annotation manifest path")
    parser.add_argument(
        "--context",
        help="Optional UTF-8 SRS context file shown to reviewers (same file for both reviewers)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue an in-progress manifest after validating reviewer and source hashes",
    )
    args = parser.parse_args(argv)

    try:
        reviewer_id = validate_reviewer_id(args.annotator)
        input_path = Path(args.input).expanduser().resolve()
        output_path = ensure_output_path_is_unprotected(args.output)
        if input_path == output_path:
            raise ValueError("Output path must differ from the candidate input path.")
        pairs = load_candidate_pairs(input_path)
        source_hash = sha256_file(input_path)

        context_text = None
        context_hash = None
        if args.context:
            context_path = Path(args.context).expanduser().resolve()
            context_text = context_path.read_text(encoding="utf-8")
            context_hash = sha256_file(context_path)

        if output_path.exists():
            if not args.resume:
                raise ValueError("Output already exists; use a new path or pass --resume.")
            manifest = load_annotation_manifest(output_path)
            if manifest.status != "in_progress":
                raise ValueError("Only an in-progress annotation manifest can be resumed.")
            if manifest.reviewer_id != reviewer_id or manifest.source_sha256 != source_hash:
                raise ValueError("Resume reviewer and source file must match the saved manifest.")
            if manifest.context_sha256 != context_hash:
                raise ValueError("Resume context file must match the saved manifest.")
            existing_ids = {record.pair_id for record in manifest.records}
            candidate_ids = {pair.id for pair in pairs}
            if not existing_ids.issubset(candidate_ids):
                raise ValueError("Saved annotations contain pair IDs absent from the candidate file.")
        else:
            if args.resume:
                raise ValueError("Cannot resume because the output manifest does not exist.")
            output_path = ensure_new_output_path(output_path)
            now = datetime.now(timezone.utc).isoformat()
            manifest = IndependentAnnotationManifest(
                schema_version=ANNOTATION_SCHEMA_VERSION,
                source_sha256=source_hash,
                context_sha256=context_hash,
                reviewer_id=reviewer_id,
                created_at=now,
                updated_at=now,
            )
            atomic_write_json(output_path, manifest.model_dump(mode="json"))

        if context_text:
            print("SRS context (provided identically to both independent reviewers):")
            print(context_text)
        elif not manifest.records:
            print("No SRS context was supplied; decisions will use pair text only.")
        print("\nChoose the single best-supported relation for each pair:")
        for index, label in LABELS_BY_NUMBER.items():
            print(f"  {index}. {label.value}")
        print("  X. Abstain / defer when evidence is insufficient or ambiguous")
        print("  Press Ctrl+C to save progress and stop.")

        completed_ids = {record.pair_id for record in manifest.records}
        for pair in pairs:
            if pair.id in completed_ids:
                continue
            print(f"\nPair {pair.id}")
            print(f"Requirement A: {pair.req_a.text}")
            print(f"Requirement B: {pair.req_b.text}")
            print("No generated label, rationale, or other reviewer decision is shown.")
            started = time.perf_counter()
            while True:
                response = input("Label number (1-6): ").strip()
                if response in LABELS_BY_NUMBER or response.casefold() == "x":
                    break
                print("Enter one of the listed numbers from 1 to 6, or X to abstain.")
            rationale = input("Brief evidence or reason for abstention (optional): ").strip()
            record = IndependentAnnotationRecord(
                pair_id=pair.id,
                label=LABELS_BY_NUMBER.get(response),
                reviewer_id=reviewer_id,
                elapsed_seconds=max(0.0, time.perf_counter() - started),
                recorded_at=datetime.now(timezone.utc).isoformat(),
                rationale=rationale,
            )
            manifest.records.append(record)
            manifest.updated_at = datetime.now(timezone.utc).isoformat()
            atomic_write_json(output_path, manifest.model_dump(mode="json"))

        manifest.status = "complete"
        manifest.updated_at = datetime.now(timezone.utc).isoformat()
        atomic_write_json(output_path, manifest.model_dump(mode="json"))
        print(f"\nSaved {len(manifest.records)} blind annotations to {output_path}")
        print("These annotations remain separate from verified ground truth until adjudicated.")
        return 0
    except KeyboardInterrupt:
        print("\nStopped; completed annotations remain saved as in_progress.", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
