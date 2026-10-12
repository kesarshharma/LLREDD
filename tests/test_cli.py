"""Tests for Typer CLI commands."""

import json

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
    result = runner.invoke(app, ["annotate", "--input", str(pairs_file), "--output", str(output_file)], input="1\n")
    assert result.exit_code == 0, result.output
    annotated = json.loads(output_file.read_text(encoding="utf-8"))[0]
    assert annotated["dependency_type"] == "Requires"
    assert annotated["provenance"]["source"] == "human"
    assert annotated["provenance"]["status"] == "verified"
    assert annotated["provenance"]["accepted_label"] == "Requires"


def test_cli_detect(tmp_path):
    pairs_file = "data/annotated/adb_pairs.json"
    out_file = str(tmp_path / "preds.json")
    result = runner.invoke(app, ["detect", "--input", pairs_file, "--output", out_file])
    assert result.exit_code == 0
    assert "Successfully generated predictions" in result.stdout


def test_cli_evaluate(tmp_path):
    from leredd.utils import load_pairs_json

    preds_file = str(tmp_path / "preds.json")
    gt_file = "data/annotated/adb_pairs.json"
    predictions = [
        {
            "pair_id": pair.id,
            "dependency_type": pair.dependency_type.value,
            "rationale": "Test fixture prediction covering each ground-truth pair.",
            "confidence": 5,
        }
        for pair in load_pairs_json(gt_file)
    ]
    with open(preds_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f)

    result = runner.invoke(app, ["evaluate", "--predictions", preds_file, "--ground-truth", gt_file, "--system", "ADB"])
    assert result.exit_code == 0
    assert "EVALUATION REPORT" in result.stdout


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
    train_file = "data/annotated/adb_pairs.json"
    out_dir = str(tmp_path / "model")
    result = runner.invoke(app, ["train-bert", "--train", train_file, "--output", out_dir])
    assert result.exit_code == 0
    artifact_path = tmp_path / "model" / "bert_baseline.joblib"
    assert artifact_path.is_file()
    assert f"Saved to '{artifact_path}'" in result.stdout

    overwrite_result = runner.invoke(
        app,
        ["train-bert", "--train", train_file, "--output", out_dir, "--overwrite"],
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
