from pathlib import Path
import sys

import pytest

from scripts import run_evaluation


def test_legacy_evaluation_command_delegates_to_safe_split_runner(tmp_path, monkeypatch):
    calls = []

    def fake_run_evaluation(**kwargs):
        calls.append(kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_evaluation, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(sys, "argv", ["run_evaluation.py"])

    assert run_evaluation.main() == 0

    assert len(calls) == 1
    assert calls[0]["split_dir"] == "data/splits/system_disjoint"
    assert calls[0]["strict_contamination"] is True
    assert calls[0]["strict_gt"] is True
    report = Path(calls[0]["output_path"])
    assert report.parent.parent.name == "runs"
    assert report.name == "system_disjoint_report.json"


def test_legacy_evaluation_command_refuses_existing_ablation_report(tmp_path, monkeypatch):
    output = tmp_path / "run" / "report.json"
    output.parent.mkdir()
    ablation = output.parent / "ablation_report.json"
    ablation.write_text("historical report", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_evaluation.py", "--output", str(output)])

    with pytest.raises(SystemExit) as exc_info:
        run_evaluation.main()

    assert exc_info.value.code == 2
    assert ablation.read_text(encoding="utf-8") == "historical report"
