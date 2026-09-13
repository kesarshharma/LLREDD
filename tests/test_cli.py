"""Tests for Typer CLI commands."""

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


def test_cli_pairs(tmp_path):
    reqs_file = str(tmp_path / "reqs.json")
    with open(reqs_file, "w", encoding="utf-8") as f:
        f.write('[{"id": "REQ-1", "text": "The system shall brake.", "system": "ADB"}, {"id": "REQ-2", "text": "The system shall accelerate.", "system": "ADB"}]')
    
    out_file = str(tmp_path / "pairs.json")
    result = runner.invoke(app, ["pairs", "--input", reqs_file, "--output", out_file])
    assert result.exit_code == 0
    assert "Successfully generated" in result.stdout


def test_cli_detect(tmp_path):
    pairs_file = "data/annotated/adb_pairs.json"
    out_file = str(tmp_path / "preds.json")
    result = runner.invoke(app, ["detect", "--input", pairs_file, "--output", out_file])
    assert result.exit_code == 0
    assert "Successfully generated predictions" in result.stdout


def test_cli_evaluate(tmp_path):
    preds_file = str(tmp_path / "preds.json")
    with open(preds_file, "w", encoding="utf-8") as f:
        f.write('[{"pair_id": "PAIR-ADB-001", "dependency_type": "Requires", "rationale": "Test", "confidence": 5}]')

    gt_file = "data/annotated/adb_pairs.json"
    result = runner.invoke(app, ["evaluate", "--predictions", preds_file, "--ground-truth", gt_file, "--system", "ADB"])
    assert result.exit_code == 0
    assert "EVALUATION REPORT" in result.stdout


def test_cli_train_bert(tmp_path):
    train_file = "data/annotated/adb_pairs.json"
    out_dir = str(tmp_path / "model")
    result = runner.invoke(app, ["train-bert", "--train", train_file, "--output", out_dir])
    assert result.exit_code == 0
    assert "Successfully trained BERT baseline" in result.stdout
