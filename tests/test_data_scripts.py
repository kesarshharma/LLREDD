import builtins
import json

import pytest

from scripts.annotate import main as copy_pairs_main
from scripts.extract_requirements import main as extract_main
from scripts.split_dataset import main as split_main
from leredd.utils import extract_requirements_from_file


def test_extract_requirements_script_writes_to_requested_output(tmp_path):
    srs = tmp_path / "synthetic.txt"
    output = tmp_path / "requirements.json"
    srs.write_text(
        "[REQ-A] The system shall save settings.\n"
        "[REQ-B] The system shall restore settings.",
        encoding="utf-8",
    )

    assert extract_main([str(srs), "TEST", str(output)]) == 0

    records = json.loads(output.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert {record["system"] for record in records} == {"TEST"}


def test_pair_copy_script_validates_and_preserves_input_labels(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "copy.json"
    source.write_text(json.dumps([{
        "id": "COPY-1",
        "req_a": {"id": "A", "text": "Requirement A."},
        "req_b": {"id": "B", "text": "Requirement B."},
        "dependency_type": "Details",
        "provenance": {"source": "imported", "status": "pending_review"},
    }]), encoding="utf-8")

    assert copy_pairs_main([str(source), str(output)]) == 0

    copied = json.loads(output.read_text(encoding="utf-8"))
    assert copied[0]["dependency_type"] == "Details"
    assert copied[0]["provenance"]["status"] == "pending_review"


def test_pair_copy_refuses_existing_output_without_overwrite(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "copy.json"
    source.write_text("[]", encoding="utf-8")
    output.write_text("historical", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        copy_pairs_main([str(source), str(output)])

    assert exc_info.value.code == 2
    assert output.read_text(encoding="utf-8") == "historical"


def test_split_script_writes_both_partition_sets_to_requested_directory(tmp_path):
    source = tmp_path / "pairs.json"
    output = tmp_path / "splits"
    records = []
    for system in ("ADB", "APA", "TJA"):
        records.append({
            "id": f"PAIR-{system}",
            "req_a": {"id": f"{system}-A", "text": "The system shall start.", "system": system},
            "req_b": {"id": f"{system}-B", "text": "The system shall stop.", "system": system},
            "dependency_type": "Requires",
        })
    source.write_text(json.dumps(records), encoding="utf-8")

    assert split_main(["--input", str(source), "--output-dir", str(output)]) == 0

    system_dir = output / "system_disjoint"
    requirement_dir = output / "requirement_disjoint"
    for split_dir in (system_dir, requirement_dir):
        assert (split_dir / "train_pairs.json").exists()
        assert (split_dir / "dev_pairs.json").exists()
        assert (split_dir / "test_pairs.json").exists()
        assert (split_dir / "split_metadata.json").exists()


@pytest.mark.parametrize(("suffix", "module_name"), [(".pdf", "pypdf"), (".docx", "docx")])
def test_document_extraction_reports_missing_optional_dependency(
    tmp_path, monkeypatch, suffix, module_name
):
    source = tmp_path / f"document{suffix}"
    source.write_bytes(b"synthetic document placeholder")
    original_import = builtins.__import__

    def import_without_document_package(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(f"simulated missing {module_name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_document_package)
    with pytest.raises(RuntimeError, match="optional document dependencies"):
        extract_requirements_from_file(str(source))
