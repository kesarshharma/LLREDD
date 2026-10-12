import json

from scripts.generate_candidates import main


def test_generate_candidates_script_writes_isolated_json_and_manifests(tmp_path):
    srs_file = tmp_path / "synthetic_srs.txt"
    output_dir = tmp_path / "candidate_run"
    srs_file.write_text(
        "[REQ-1] The system shall provide braking.\n"
        "[REQ-2] The system shall report brake status.\n"
        "[REQ-3] The system shall record sensor health.\n",
        encoding="utf-8",
    )

    assert main([
        "--input", str(srs_file), "--system", "TEST-SYS",
        "--output-dir", str(output_dir), "--max-candidates", "20", "--top-k", "2",
    ]) == 0

    exhaustive_path = output_dir / "test_sys_candidates_all.json"
    bm25_path = output_dir / "test_sys_candidates_bm25.json"
    exhaustive = json.loads(exhaustive_path.read_text(encoding="utf-8"))
    bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))
    assert len(exhaustive) == 3
    assert len(bm25) <= 3
    assert all(item["dependency_type"] is None for item in exhaustive + bm25)
    assert exhaustive_path.with_name("test_sys_candidates_all_manifest.json").exists()
    assert bm25_path.with_name("test_sys_candidates_bm25_manifest.json").exists()
