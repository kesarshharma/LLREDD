"""Typer CLI interface for LEREDD tools."""

import os
import json
import datetime
import typer
from typing import Optional
from loguru import logger
from leredd.config import get_config
from leredd.data_models import (
    AnnotationProvenance, AnnotationSource, DependencyType, RequirementPair,
    Prediction, Requirement, VerificationStatus,
)
from leredd.utils import (
    extract_requirements_from_file,
    generate_requirement_pairs,
    load_pairs_json,
    save_pairs_json,
)
from leredd.detector import LEREDDDetector
from leredd.evaluation import Evaluator
from leredd.baselines.bert_classifier import BERTBaseline
from leredd.dataset_splitter import DatasetSplitter, SplitConfig, SplitStrategy
from leredd.candidate_generator import CandidateGenerator, GenerationStrategy
from leredd.provisional_labeler import ProvisionalLabeler, LabelingMode

app = typer.Typer(
    name="leredd-cli",
    help="LEREDD: LLM-Enabled REquirement Dependency Detection CLI Tool",
    add_completion=False
)


@app.command("extract")
def extract_cmd(
    input_file: str = typer.Option(..., "--input", "-i", help="Path to input SRS document (.txt, .pdf, .docx)"),
    output_file: str = typer.Option("requirements.json", "--output", "-o", help="Output JSON file for extracted requirements"),
    system_name: str = typer.Option("ADB", "--system", "-s", help="System name (e.g. ADB, TJA, APA)")
):
    """Extract natural language requirement statements from SRS document using regex."""
    typer.echo(f"Extracting requirements from '{input_file}' for system '{system_name}'...")
    reqs = extract_requirements_from_file(input_file, system_name=system_name)
    
    data = [req.model_dump(mode="json") for req in reqs]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    typer.secho(f"Successfully extracted {len(reqs)} requirements to '{output_file}'.", fg=typer.colors.GREEN)


@app.command("pairs")
def pairs_cmd(
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSON file containing extracted requirements"),
    output_file: str = typer.Option("pairs.json", "--output", "-o", help="Output JSON file for generated pairs")
):
    """Generate unlabeled candidate pairs n(n-1)/2 from extracted requirements."""
    typer.echo(f"Loading requirements from '{input_file}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    reqs = [Requirement.model_validate(item) for item in data]

    pairs = generate_requirement_pairs(reqs)
    save_pairs_json(pairs, output_file)
    typer.secho(f"Successfully generated {len(pairs)} pairs to '{output_file}'.", fg=typer.colors.GREEN)


@app.command("annotate")
def annotate_cmd(
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSON file containing requirement pairs"),
    output_file: str = typer.Option("annotated.json", "--output", "-o", help="Output JSON file for annotated pairs")
):
    """Interactive CLI tool to annotate requirement pairs with dependency types."""
    pairs = load_pairs_json(input_file)
    typer.echo(f"Loaded {len(pairs)} requirement pairs for interactive annotation.\n")

    annotated = []
    options = ["Requires", "Implements", "Conflicts", "Details", "Is similar", "No_dependency"]

    for idx, pair in enumerate(pairs, 1):
        typer.secho(f"\n--- Pair [{idx}/{len(pairs)}] ---", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"Requirement A: {pair.req_a.text}")
        typer.echo(f"Requirement B: {pair.req_b.text}")
        typer.echo("Dependency Types:")
        for opt_idx, opt in enumerate(options, 1):
            typer.echo(f"  [{opt_idx}] {opt}")

        choice = typer.prompt("Select dependency type (1-6)")
        try:
            chosen_opt = options[int(choice) - 1]
        except (ValueError, IndexError) as exc:
            raise typer.BadParameter("Choose a dependency type from 1 to 6") from exc

        previous_label = pair.provenance.predicted_label if pair.provenance else pair.dependency_type
        pair.dependency_type = DependencyType.normalize(chosen_opt)
        provenance = pair.provenance or AnnotationProvenance()
        pair.provenance = provenance.model_copy(update={
            "source": AnnotationSource.HUMAN,
            "status": VerificationStatus.VERIFIED,
            "annotator_id": provenance.annotator_id or "human_cli",
            "reviewed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "predicted_label": previous_label,
            "accepted_label": pair.dependency_type,
        })
        annotated.append(pair)

    save_pairs_json(annotated, output_file)
    typer.secho(f"Annotation complete! Saved to '{output_file}'.", fg=typer.colors.GREEN)


@app.command("detect")
def detect_cmd(
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSON file containing requirement pairs to evaluate"),
    examples_file: Optional[str] = typer.Option(None, "--examples", "-e", help="Optional annotated JSON file for in-context example retrieval"),
    srs_file: Optional[str] = typer.Option(None, "--srs", "-s", help="Optional SRS file for RAG contextual retrieval"),
    output_file: str = typer.Option("predictions.json", "--output", "-o", help="Output JSON file for predictions"),
    use_rag: bool = typer.Option(False, "--use-rag", help="Enable RAG context retrieval")
):
    """Run LEREDD detection pipeline on requirement pairs."""
    config = get_config()
    if use_rag:
        config.use_rag = True

    pairs = load_pairs_json(input_file)
    annotated_pool = load_pairs_json(examples_file) if examples_file and os.path.exists(examples_file) else []

    srs_text = None
    if srs_file and os.path.exists(srs_file):
        with open(srs_file, "r", encoding="utf-8", errors="ignore") as f:
            srs_text = f.read()

    typer.echo(f"Running LEREDD detection on {len(pairs)} pairs (examples pool: {len(annotated_pool)}, RAG: {config.use_rag})...")

    detector = LEREDDDetector(config=config)
    predictions = detector.detect_batch(pairs, annotated_pool=annotated_pool, srs_text=srs_text)

    data = [pred.model_dump(mode="json") for pred in predictions]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    typer.secho(f"Successfully generated predictions and saved to '{output_file}'.", fg=typer.colors.GREEN)


@app.command("evaluate")
def evaluate_cmd(
    predictions_file: str = typer.Option(..., "--predictions", "-p", help="Input JSON file containing predictions"),
    ground_truth_file: str = typer.Option(..., "--ground-truth", "-g", help="Input JSON file containing ground truth pairs"),
    system_name: str = typer.Option("EvaluationSystem", "--system", "-s", help="System name")
):
    """Compute performance metrics (Accuracy, Precision, Recall, F1) against ground truth."""
    with open(predictions_file, "r", encoding="utf-8") as f:
        preds_data = json.load(f)
    predictions = [Prediction.model_validate(item) for item in preds_data]

    ground_truth = load_pairs_json(ground_truth_file)

    report = Evaluator.evaluate(ground_truth, predictions, system_name=system_name)

    typer.secho(f"\n================ EVALUATION REPORT ({report.system_name}) ================", fg=typer.colors.MAGENTA, bold=True)
    typer.echo(f"Overall Accuracy : {report.overall_accuracy:.4f}")
    typer.echo(f"Macro F1 Score   : {report.macro_f1:.4f}")
    typer.echo(f"Weighted F1 Score: {report.weighted_f1:.4f}\n")
    typer.echo("Per-Class Metrics:")
    for cls_name, metrics in report.per_class_metrics.items():
        typer.echo(f"  {cls_name:15s} | P: {metrics.precision:.4f} | R: {metrics.recall:.4f} | F1: {metrics.f1_score:.4f} | Support: {metrics.support}")
    typer.secho("=================================================================", fg=typer.colors.MAGENTA)


@app.command("train-bert")
def train_bert_cmd(
    train_file: str = typer.Option(..., "--train", "-t", help="Input JSON file containing annotated training pairs"),
    output_dir: str = typer.Option("model/", "--output", "-o", help="Output directory for the trained classifier artifact"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing classifier artifact")
):
    """Train BERT/ML baseline classifier on annotated requirement pairs."""
    artifact_path = os.path.join(output_dir, "bert_baseline.joblib")
    if os.path.exists(artifact_path) and not overwrite:
        raise typer.BadParameter(
            f"Model artifact already exists: {artifact_path}. Pass --overwrite to replace it."
        )

    pairs = load_pairs_json(train_file)
    typer.echo(f"Training BERT baseline on {len(pairs)} pairs...")

    baseline = BERTBaseline()
    baseline.fit(pairs)
    os.makedirs(output_dir, exist_ok=True)
    baseline.save(artifact_path, overwrite=overwrite)
    typer.secho(f"Successfully trained BERT baseline model. Saved to '{artifact_path}'.", fg=typer.colors.GREEN)


@app.command("split")
def split_cmd(
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSON file containing requirement pairs (or comma-separated list of files)"),
    output_dir: str = typer.Option("data/splits/", "--output-dir", "-o", help="Output directory to save split partitions"),
    strategy: str = typer.Option("requirement", "--strategy", "-s", help="Split strategy: system, document, requirement, pair"),
    train_ratio: float = typer.Option(0.6, "--train-ratio", help="Train partition ratio"),
    dev_ratio: float = typer.Option(0.2, "--dev-ratio", help="Dev partition ratio"),
    test_ratio: float = typer.Option(0.2, "--test-ratio", help="Test partition ratio"),
    retrieval_ratio: float = typer.Option(0.0, "--retrieval-ratio", help="Example retrieval partition ratio"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility")
):
    """Partition requirement pair dataset using leakage-resistant splitting strategies."""
    files = [f.strip() for f in input_file.split(",") if f.strip()]
    all_pairs = []
    for fpath in files:
        if os.path.exists(fpath):
            all_pairs.extend(load_pairs_json(fpath))
        else:
            typer.secho(f"Warning: File '{fpath}' not found.", fg=typer.colors.YELLOW)

    if not all_pairs:
        typer.secho("Error: No requirement pairs loaded.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        strat_enum = SplitStrategy(strategy.lower())
    except ValueError:
        typer.secho(f"Error: Invalid strategy '{strategy}'. Must be one of: system, document, requirement, pair.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    config = SplitConfig(
        strategy=strat_enum,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        retrieval_ratio=retrieval_ratio,
        random_seed=seed,
    )

    typer.echo(f"Splitting {len(all_pairs)} pairs using '{strat_enum.value}' strategy (seed={seed})...")
    split_result = DatasetSplitter.split(all_pairs, config=config)
    DatasetSplitter.save_split(split_result, output_dir)

    summary = split_result.summary()
    leakage = DatasetSplitter.check_leakage(split_result)

    typer.secho(f"\n================ DATASET SPLIT SUMMARY ({strat_enum.value.upper()}) ================", fg=typer.colors.CYAN, bold=True)
    for part, stats in summary["counts"].items():
        if isinstance(stats, dict) and "pairs" in stats:
            req_info = f", {stats.get('requirements', 0)} reqs" if "requirements" in stats else ""
            typer.echo(f"  {part.upper():12s}: {stats['pairs']:4d} pairs{req_info}")
    typer.echo(f"  Leakage Detected: {leakage.has_leakage} (rate: {leakage.leakage_rate:.2%})")
    typer.secho(f"Saved partitions to '{output_dir}'.", fg=typer.colors.GREEN)


@app.command("candidates")
def candidates_cmd(
    input_file: str = typer.Option(..., "--input", "-i", help="Input file containing requirements (JSON or text/SRS document)"),
    output_file: str = typer.Option("candidate_pairs.json", "--output", "-o", help="Output JSON file for generated candidate pairs"),
    manifest_file: Optional[str] = typer.Option(None, "--manifest", "-m", help="Optional output JSON file for generation manifest"),
    strategy: str = typer.Option("all_pairs", "--strategy", "-s", help="Generation strategy: all_pairs, bm25, dense, hybrid"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Top-k retrieval parameter if retrieval strategy is used"),
    max_candidates: Optional[int] = typer.Option(None, "--max-candidates", "-l", help="Optional maximum candidate limit"),
    system_name: str = typer.Option("ADB", "--system", help="System identifier name")
):
    """Generate candidate requirement pairs for semi-automated dataset labeling."""
    typer.echo(f"Loading requirements from '{input_file}'...")
    if input_file.endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data and "req_a" in data[0]:
            # Input is already a pair list -> extract unique requirements
            req_map = {}
            for item in data:
                p = RequirementPair.model_validate(item)
                req_map[p.req_a.id] = p.req_a
                req_map[p.req_b.id] = p.req_b
            reqs = list(req_map.values())
        else:
            reqs = [Requirement.model_validate(item) for item in data]
    else:
        reqs = extract_requirements_from_file(input_file, system_name=system_name)

    if not reqs:
        typer.secho("Error: No requirements extracted from input.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        strat_enum = GenerationStrategy(strategy.lower())
    except ValueError:
        typer.secho(f"Error: Invalid strategy '{strategy}'. Must be one of: all_pairs, bm25, dense, hybrid.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Generating candidate pairs for {len(reqs)} requirements using '{strat_enum.value}' strategy...")
    candidates, manifest = CandidateGenerator.generate_candidates(
        requirements=reqs,
        strategy=strat_enum,
        source_identity=system_name,
        top_k=top_k,
        max_candidates=max_candidates,
    )

    CandidateGenerator.save_candidates(candidates, manifest, output_file, manifest_file)

    typer.secho(f"\n================ CANDIDATE GENERATION SUMMARY ================", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Input Requirements  : {manifest.input_requirement_count}")
    typer.echo(f"Generated Candidates: {manifest.generated_count}")
    typer.echo(f"Strategy Used       : {manifest.strategy.value}")
    typer.echo(f"Exhaustive Output   : {manifest.is_exhaustive}")
    typer.echo(f"Omitted Candidates  : {manifest.truncated_count}")
    typer.secho(f"Saved candidates to '{output_file}'.", fg=typer.colors.GREEN)


@app.command("label")
def label_cmd(
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSON file containing candidate pairs"),
    output_file: str = typer.Option("provisional_pairs.json", "--output", "-o", help="Output JSON file for provisional labeled pairs"),
    manifest_file: Optional[str] = typer.Option(None, "--manifest", "-m", help="Optional output JSON file for labeling manifest"),
    mode: str = typer.Option("hybrid", "--mode", "-mode", help="Labeling mode: heuristic, llm, hybrid")
):
    """Run provisional dependency labeling on candidate requirement pairs."""
    typer.echo(f"Loading candidate pairs from '{input_file}'...")
    candidates = load_pairs_json(input_file)

    try:
        mode_enum = LabelingMode(mode.lower())
    except ValueError:
        typer.secho(f"Error: Invalid mode '{mode}'. Must be one of: heuristic, llm, hybrid.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Running provisional labeling on {len(candidates)} candidate pairs (mode='{mode_enum.value}')...")
    labeler = ProvisionalLabeler()
    labeled_pairs, manifest = labeler.process_batch(candidates, mode=mode_enum, source_identity=input_file)

    ProvisionalLabeler.save_provisional_dataset(labeled_pairs, manifest, output_file, manifest_file)

    typer.secho(f"\n================ PROVISIONAL LABELING SUMMARY ================", fg=typer.colors.MAGENTA, bold=True)
    typer.echo(f"Input Candidates   : {manifest.input_candidate_count}")
    typer.echo(f"Provisional Labeled: {manifest.labeled_count}")
    typer.echo(f"Agreed (Heur & LLM): {manifest.agreed_count}")
    typer.echo(f"Disagreed (Conflict): {manifest.disagreed_count}")
    typer.echo(f"Unresolved (Abstain): {manifest.unresolved_count}")
    typer.secho(f"Saved provisional dataset to '{output_file}'.", fg=typer.colors.GREEN)


@app.command("queue-create")
def queue_create_cmd(
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSON file containing provisional requirement pairs"),
    output_file: str = typer.Option("review_queue.json", "--output", "-o", help="Output JSON file for review queue"),
    manifest_file: Optional[str] = typer.Option(None, "--manifest", "-m", help="Optional output JSON file for queue manifest")
):
    """Build or refresh a human review queue from provisional requirement pairs."""
    from leredd.review_queue import ReviewQueueManager

    typer.echo(f"Loading provisional pairs from '{input_file}'...")
    pairs = load_pairs_json(input_file)

    manager = ReviewQueueManager()

    existing_items = []
    if os.path.exists(output_file):
        try:
            existing_items = manager.load_queue(output_file)
            typer.echo(f"Loaded {len(existing_items)} existing items from '{output_file}'.")
        except Exception:
            pass

    items = manager.refresh_queue(existing_items, pairs) if existing_items else manager.build_queue(pairs)
    manifest = manager.create_manifest(items, source_identity=input_file)

    manager.save_queue(items, manifest, output_file, manifest_file)

    typer.secho(f"\n================ REVIEW QUEUE SUMMARY ================", fg=typer.colors.MAGENTA, bold=True)
    typer.echo(f"Total Queue Items  : {manifest.total_items}")
    typer.echo(f"Pending Items      : {manifest.pending_count}")
    typer.echo(f"Accepted Items     : {manifest.accepted_count}")
    typer.echo(f"Corrected Items    : {manifest.corrected_count}")
    typer.echo(f"Rejected Items     : {manifest.rejected_count}")
    typer.echo(f"Deferred Items     : {manifest.deferred_count}")
    typer.secho(f"Saved review queue to '{output_file}'.", fg=typer.colors.GREEN)


@app.command("queue-list")
def queue_list_cmd(
    queue_file: str = typer.Option("review_queue.json", "--queue", "-q", help="Input JSON review queue file"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status (pending, accepted, corrected, etc.)"),
    reason: Optional[str] = typer.Option(None, "--reason", "-r", help="Filter by reason code (disagreement, both_abstained, etc.)")
):
    """List pending or filtered items in the human review queue."""
    from leredd.review_queue import ReviewQueueManager, QueueStatus, ReviewReasonCode

    manager = ReviewQueueManager()
    items = manager.load_queue(queue_file)

    status_enum = QueueStatus(status.lower()) if status else None
    reason_enum = ReviewReasonCode(reason.lower()) if reason else None

    filtered = manager.filter_queue(items, status=status_enum, reason=reason_enum)

    typer.secho(f"\nReview Queue Items ({len(filtered)} / {len(items)} shown):", fg=typer.colors.CYAN, bold=True)
    for item in filtered:
        reasons_str = ", ".join(r.value for r in item.review_reasons)
        typer.echo(f"[{item.queue_id}] Priority {item.priority} | Status: {item.status.value.upper()} | Reasons: {reasons_str}")
        typer.echo(f"   Pair: {item.pair_id} ({item.req_a_id} -> {item.req_b_id})")
        typer.echo(f"   Proposed Label: {item.proposed_label.value if item.proposed_label else 'None'}")
        typer.echo(f"   Notes: {item.review_notes}\n")


@app.command("queue-decide")
def queue_decide_cmd(
    queue_file: str = typer.Option("review_queue.json", "--queue", "-q", help="Target review queue file"),
    item_id: str = typer.Option(..., "--item-id", "-id", help="Queue item ID to record decision for"),
    decision: str = typer.Option(..., "--decision", "-d", help="Decision status: accepted, corrected, rejected, deferred"),
    label: Optional[str] = typer.Option(None, "--label", "-l", help="Final dependency label (required for corrected)"),
    reviewer: str = typer.Option("human_reviewer_1", "--reviewer", "-r", help="Reviewer identifier"),
    notes: str = typer.Option("", "--notes", "-n", help="Reviewer notes/justification"),
    verified_output: Optional[str] = typer.Option(None, "--verified-out", "-v", help="Optional output JSON file to append verified pairs")
):
    """Record a human reviewer decision on a review queue item."""
    from leredd.review_queue import ReviewQueueManager, QueueStatus

    manager = ReviewQueueManager()
    items = manager.load_queue(queue_file)

    target_item = next((i for i in items if i.queue_id.upper() == item_id.upper() or i.pair_id.upper() == item_id.upper()), None)
    if not target_item:
        typer.secho(f"Error: Queue item '{item_id}' not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        status_enum = QueueStatus(decision.lower())
    except ValueError:
        typer.secho(f"Error: Invalid decision '{decision}'. Must be one of: accepted, corrected, rejected, deferred.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    updated_item, verified_pair = manager.record_decision(
        target_item,
        decision_type=status_enum,
        final_label=label,
        reviewer_id=reviewer,
        reviewer_notes=notes,
    )

    manifest = manager.create_manifest(items)
    manager.save_queue(items, manifest, queue_file)

    typer.secho(f"Successfully recorded decision '{status_enum.value.upper()}' for item {target_item.queue_id}.", fg=typer.colors.GREEN)

    if verified_output and verified_pair:
        existing_verified = []
        if os.path.exists(verified_output):
            try:
                existing_verified = load_pairs_json(verified_output)
            except Exception:
                pass
        existing_verified.append(verified_pair)
        save_pairs_json(existing_verified, verified_output)
        typer.secho(f"Appended human-verified pair to '{verified_output}'.", fg=typer.colors.GREEN)


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind (use 0.0.0.0 to expose beyond this machine)"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on")
):
    """Launch FastAPI web server using Uvicorn."""
    import uvicorn
    typer.echo(f"Starting LEREDD FastAPI REST API server on http://{host}:{port}...")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
