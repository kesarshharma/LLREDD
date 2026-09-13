"""Typer CLI interface for LEREDD tools."""

import os
import json
import typer
from typing import Optional
from loguru import logger
from leredd.config import get_config
from leredd.data_models import RequirementPair, Prediction, Requirement
from leredd.utils import (
    extract_requirements_from_file,
    generate_requirement_pairs,
    load_pairs_json,
    save_pairs_json,
)
from leredd.detector import LEREDDDetector
from leredd.evaluation import Evaluator
from leredd.baselines.bert_classifier import BERTBaseline

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
    """Generate all unique requirement pairs n(n-1)/2 from extracted requirements."""
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

        choice = typer.prompt("Select dependency type (1-6)", default="6")
        try:
            chosen_opt = options[int(choice) - 1]
        except Exception:
            chosen_opt = "No_dependency"

        pair.dependency_type = chosen_opt
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
    output_dir: str = typer.Option("model/", "--output", "-o", help="Output directory to save trained model")
):
    """Train BERT/ML baseline classifier on annotated requirement pairs."""
    pairs = load_pairs_json(train_file)
    typer.echo(f"Training BERT baseline on {len(pairs)} pairs...")

    baseline = BERTBaseline()
    baseline.fit(pairs)
    os.makedirs(output_dir, exist_ok=True)
    typer.secho(f"Successfully trained BERT baseline model. Saved to '{output_dir}'.", fg=typer.colors.GREEN)


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host address to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on")
):
    """Launch FastAPI web server using Uvicorn."""
    import uvicorn
    typer.echo(f"Starting LEREDD FastAPI REST API server on http://{host}:{port}...")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
