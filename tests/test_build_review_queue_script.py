import json

from scripts.build_review_queue import main


def test_build_review_queue_script_writes_pending_queue_only(tmp_path):
    input_path = tmp_path / "provisional.json"
    output_path = tmp_path / "queue.json"
    input_path.write_text(json.dumps([{
        "id": "QUEUE-SCRIPT-1",
        "req_a": {"id": "REQ-1", "text": "First requirement.", "system": "ADB"},
        "req_b": {"id": "REQ-2", "text": "Second requirement.", "system": "ADB"},
        "dependency_type": "Requires",
        "provenance": {"source": "llm", "status": "provisional"},
    }]), encoding="utf-8")

    assert main(["--input", str(input_path), "--output", str(output_path)]) == 0

    items = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        output_path.with_name("queue_manifest.json").read_text(encoding="utf-8")
    )
    assert len(items) == 1
    assert items[0]["status"] == "pending"
    assert items[0]["human_decision"] is None
    assert items[0]["provenance"]["status"] == "provisional"
    assert manifest["pending_count"] == 1
