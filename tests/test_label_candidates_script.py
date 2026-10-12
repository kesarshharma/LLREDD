import json

from scripts.label_candidates import main


def test_label_candidates_script_defaults_to_mock_and_keeps_labels_provisional(tmp_path):
    candidate_file = tmp_path / "candidates.json"
    output_dir = tmp_path / "provisional"
    candidate_file.write_text(json.dumps([{
        "id": "LABEL-SCRIPT-1",
        "req_a": {"id": "REQ-1", "text": "The system shall provide braking.", "system": "ADB"},
        "req_b": {"id": "REQ-2", "text": "The BCS shall brake when required.", "system": "ADB"},
        "dependency_type": None,
        "provenance": {"source": "candidate_generator", "status": "provisional"},
    }]), encoding="utf-8")

    assert main([
        "--input", str(candidate_file), "--output-dir", str(output_dir),
    ]) == 0

    heuristic_file = output_dir / "candidates_provisional_heuristic.json"
    hybrid_file = output_dir / "candidates_provisional_hybrid.json"
    for output_file in (heuristic_file, hybrid_file):
        data = json.loads(output_file.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["provenance"]["status"] == "provisional"
        assert data[0]["provenance"]["source"] in {"heuristic", "llm", "ensemble"}
        assert output_file.with_name(f"{output_file.stem}_manifest.json").exists()
