"""Tests for Typer CLI commands."""

import json
from click import unstyle

from typer.testing import CliRunner
from leredd.cli import app

runner = CliRunner()


def test_cli_extract(tmp_path):
    srs_file = str(tmp_path / "srs.txt")
    with open(srs_file, "w", encoding="utf-8") as f:
        f.write("REQ-001: The system shall verify sensor inputs.")
    
    out_file = str(tmp_path / "reqs.json")
    result = runner.invoke(app, ["extract", "--input", srs_file, "--output", out_file, "--system", "ADB"])
    assert result.exit_code == 0
    assert "Successfully extracted" in result.stdout


def test_cli_metadata_sidecar_flows_from_srs_to_candidate_pairs(tmp_path):
    srs_file = tmp_path / "srs.md"
    srs_file.write_text(
        "[REQ-A] The brake controller shall calculate stopping distance.\n"
        "[REQ-B] The display shall report the stopping distance.",
        encoding="utf-8",
    )
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({
        "REQ-A": {"component": "Brake Controller", "subsystem": "Braking"},
        "REQ-B": {"component": "Display", "subsystem": "Driver Interface"},
    }), encoding="utf-8")
    extracted_file = tmp_path / "requirements.json"
    extract = runner.invoke(app, [
        "extract", "--input", str(srs_file), "--system", "DEMO",
        "--metadata-json", str(metadata_file), "--output", str(extracted_file),
    ])
    assert extract.exit_code == 0, extract.output
    extracted = json.loads(extracted_file.read_text(encoding="utf-8"))
    assert extracted[0]["component"] == "Brake Controller"
    assert extracted[0]["subsystem"] == "Braking"

    candidates_file = tmp_path / "candidate_pairs.json"
    candidates = runner.invoke(app, [
        "candidates", "--input", str(srs_file), "--system", "DEMO",
        "--metadata-json", str(metadata_file), "--strategy", "all_pairs",
        "--output", str(candidates_file),
    ])
    assert candidates.exit_code == 0, candidates.output
    pair = json.loads(candidates_file.read_text(encoding="utf-8"))[0]
    assert pair["req_a"]["component"] == "Brake Controller"
    assert pair["req_b"]["subsystem"] == "Driver Interface"

    help_result = runner.invoke(app, ["extract", "--help"])
    assert help_result.exit_code == 0
    assert ".md" in help_result.stdout
    assert ".docx" in help_result.stdout


def test_cli_serve_binds_to_loopback_by_default(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = runner.invoke(app, ["serve", "--port", "8123"])

    assert result.exit_code == 0, result.stdout
    assert calls == [(("api.main:app",), {"host": "127.0.0.1", "port": 8123, "reload": False})]


def test_cli_pairs(tmp_path):
    reqs_file = str(tmp_path / "reqs.json")
    with open(reqs_file, "w", encoding="utf-8") as f:
        f.write('[{"id": "REQ-1", "text": "The system shall brake.", "system": "ADB"}, {"id": "REQ-2", "text": "The system shall accelerate.", "system": "ADB"}]')
    
    out_file = str(tmp_path / "pairs.json")
    result = runner.invoke(app, ["pairs", "--input", reqs_file, "--output", out_file])
    assert result.exit_code == 0
    assert "Successfully generated" in result.stdout
    generated = json.loads(open(out_file, encoding="utf-8").read())
    assert len(generated) == 1
    assert generated[0]["dependency_type"] is None
    assert generated[0]["provenance"]["source"] == "candidate_generator"
    assert generated[0]["provenance"]["status"] == "provisional"


def test_cli_annotate_records_explicit_human_verification(tmp_path):
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text(json.dumps([{
        "id": "PAIR-1",
        "req_a": {"id": "R1", "text": "Requirement one", "system": "S"},
        "req_b": {"id": "R2", "text": "Requirement two", "system": "S"},
        "dependency_type": None,
        "provenance": {"source": "candidate_generator", "status": "provisional"},
    }]), encoding="utf-8")
    output_file = tmp_path / "annotated.json"
    result = runner.invoke(app, ["annotate", "--input", str(pairs_file), "--output", str(output_file), "--annotator", "fixture-reviewer"], input="1\n")
    assert result.exit_code == 0, result.output
    annotated = json.loads(output_file.read_text(encoding="utf-8"))[0]
    assert annotated["dependency_type"] == "Requires"
    assert annotated["provenance"]["source"] == "human"
    assert annotated["provenance"]["status"] == "verified"
    assert annotated["provenance"]["annotator_id"] == "fixture-reviewer"
    assert annotated["provenance"]["accepted_label"] == "Requires"


def test_cli_annotate_requires_explicit_annotator(tmp_path):
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text("[]", encoding="utf-8")
    result = runner.invoke(app, ["annotate", "--input", str(pairs_file)], color=False)
    assert result.exit_code != 0
    assert "--annotator" in unstyle(result.output)


def test_cli_detect(tmp_path, monkeypatch):
    from leredd.data_models import Prediction, DependencyType, Requirement, RequirementPair

    pair = RequirementPair(
        id="CLI-DETECT-1",
        req_a=Requirement(id="CLI-REQ-A", text="The system shall detect obstacles.", system="ADB"),
        req_b=Requirement(id="CLI-REQ-B", text="The system shall report obstacle status.", system="ADB"),
    )
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text(json.dumps([pair.model_dump(mode="json")]), encoding="utf-8")
    out_file = str(tmp_path / "preds.json")
    received_stability_runs = []
    class ControlledDetector:
        def __init__(self, config):
            self.config = config

        def detect_batch(self, pairs, annotated_pool, srs_text, stability_runs=1):
            assert [item.id for item in pairs] == ["CLI-DETECT-1"]
            received_stability_runs.append(stability_runs)
            return [Prediction(
                pair_id=pairs[0].id,
                dependency_type=DependencyType.REQUIRES,
                confidence=5,
                rationale="controlled CLI integration",
            )]

    import leredd.cli as cli
    monkeypatch.setattr(cli, "LEREDDDetector", ControlledDetector)
    result = runner.invoke(app, [
        "detect", "--input", str(pairs_file), "--output", out_file,
        "--stability-runs", "2",
    ])
    assert result.exit_code == 0, result.output
    assert "Successfully generated predictions" in result.stdout
    assert received_stability_runs == [2]
    assert json.loads((tmp_path / "preds.json").read_text(encoding="utf-8"))[0]["pair_id"] == "CLI-DETECT-1"


def test_cli_evaluate(tmp_path):
    from leredd.data_models import AnnotationProvenance, AnnotationSource, DependencyType, Prediction, Requirement, RequirementPair, VerificationStatus
    preds_file = str(tmp_path / "preds.json")
    gt_file = tmp_path / "ground_truth.json"
    pair = RequirementPair(
        id="CLI-EVAL-1",
        req_a=Requirement(id="EVAL-A", text="The system shall authenticate users.", system="ADB"),
        req_b=Requirement(id="EVAL-B", text="The system shall verify credentials.", system="ADB"),
        dependency_type=DependencyType.REQUIRES,
        provenance=AnnotationProvenance(
            source=AnnotationSource.HUMAN, status=VerificationStatus.VERIFIED,
            reviewer_id="synthetic-test-reviewer", accepted_label=DependencyType.REQUIRES,
        ),
    )
    gt_file.write_text(json.dumps([pair.model_dump(mode="json")]), encoding="utf-8")
    predictions = [Prediction(
        pair_id=pair.id,
        dependency_type=DependencyType.REQUIRES,
        rationale="Synthetic controlled test prediction.",
        confidence=5,
    ).model_dump(mode="json")]
    with open(preds_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    result = runner.invoke(app, ["evaluate", "--predictions", preds_file, "--ground-truth", gt_file, "--system", "ADB"])
    assert result.exit_code == 0
    assert "EVALUATION REPORT" in result.stdout


def test_cli_evaluate_rejects_contradictory_verified_provenance(tmp_path):
    from leredd.data_models import (
        AnnotationProvenance, AnnotationSource, DependencyType, Prediction,
        Requirement, RequirementPair, VerificationStatus,
    )

    pair = RequirementPair(
        id="CLI-EVAL-CONTRADICTORY",
        req_a=Requirement(id="A", text="The system shall authenticate users.", system="SYN"),
        req_b=Requirement(id="B", text="The system shall verify credentials.", system="SYN"),
        dependency_type=DependencyType.REQUIRES,
        provenance=AnnotationProvenance(
            source=AnnotationSource.LLM,
            status=VerificationStatus.VERIFIED,
            annotator_id="mock-llm",
            accepted_label=DependencyType.CONFLICTS,
        ),
    )
    truth_file = tmp_path / "truth.json"
    truth_file.write_text(json.dumps([pair.model_dump(mode="json")]), encoding="utf-8")
    predictions = [Prediction(
        pair_id=pair.id,
        dependency_type=DependencyType.REQUIRES,
        rationale="Controlled regression assertion.",
        confidence=5,
    )]
    predictions_file = tmp_path / "predictions.json"
    predictions_file.write_text(
        json.dumps([prediction.model_dump(mode="json") for prediction in predictions]),
        encoding="utf-8",
    )

    result = runner.invoke(app, [
        "evaluate", "--predictions", str(predictions_file),
        "--ground-truth", str(truth_file), "--system", "SYN",
    ])

    assert result.exit_code != 0
    assert "not human" in str(result.exception)


def test_cli_train_bert(tmp_path, monkeypatch):
    import numpy as np
    from leredd.baselines.bert_classifier import BERTBaseline

    class DeterministicEmbedder:
        model_name = "test-embedder"

        def encode(self, texts):
            vectors = []
            for text in texts:
                vector = np.zeros(384, dtype=np.float32)
                for index, byte in enumerate(text.encode("utf-8")):
                    vector[index % 384] += byte
                vectors.append(vector)
            return np.vstack(vectors)

    monkeypatch.setattr(
        "leredd.cli.BERTBaseline",
        lambda: BERTBaseline(embedder=DeterministicEmbedder()),
    )
    from leredd.data_models import (
        AnnotationProvenance, AnnotationSource, DependencyType,
        Requirement, RequirementPair, VerificationStatus,
    )
    train_records = []
    for index, label in enumerate((DependencyType.REQUIRES, DependencyType.NO_DEPENDENCY), 1):
        train_records.append(RequirementPair(
            id=f"CLI-TRAIN-{index}",
            req_a=Requirement(id=f"CLI-TRAIN-A{index}", text=f"Train requirement A {index}.", system="TRAIN"),
            req_b=Requirement(id=f"CLI-TRAIN-B{index}", text=f"Train requirement B {index}.", system="TRAIN"),
            dependency_type=label,
            provenance=AnnotationProvenance(
                source=AnnotationSource.HUMAN,
                status=VerificationStatus.VERIFIED,
                reviewer_id="synthetic-test-reviewer",
                accepted_label=label,
            ),
        ))
    train_file = tmp_path / "training.json"
    train_file.write_text(json.dumps([p.model_dump(mode="json") for p in train_records]), encoding="utf-8")
    out_dir = str(tmp_path / "model")
    result = runner.invoke(app, ["train-bert", "--train", str(train_file), "--output", out_dir])
    assert result.exit_code == 0
    artifact_path = tmp_path / "model" / "bert_baseline.joblib"
    assert artifact_path.is_file()
    assert f"Saved to '{artifact_path}'" in result.stdout

    overwrite_result = runner.invoke(
        app,
        ["train-bert", "--train", str(train_file), "--output", out_dir, "--overwrite"],
    )
    assert overwrite_result.exit_code == 0, overwrite_result.output


def test_cli_train_bert_refuses_to_overwrite_artifact(tmp_path):
    train_file = "missing-training-file.json"
    out_dir = tmp_path / "model"
    out_dir.mkdir()
    artifact_path = out_dir / "bert_baseline.joblib"
    artifact_path.write_bytes(b"preserve existing artifact")

    result = runner.invoke(app, ["train-bert", "--train", train_file, "--output", str(out_dir)])

    assert result.exit_code != 0
    assert "already exists" in result.stderr
    assert artifact_path.read_bytes() == b"preserve existing artifact"
