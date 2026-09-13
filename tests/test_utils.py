"""Tests for utility functions in leredd.utils."""

from leredd.utils import (
    extract_requirements_from_text,
    extract_requirements_from_file,
    generate_requirement_pairs,
    calculate_cohens_kappa,
    load_pairs_json,
    save_pairs_json,
    load_pairs_csv,
    save_pairs_csv,
)
from leredd.data_models import Requirement, DependencyType, RequirementPair


def test_extract_requirements_from_text():
    text = """
    REQ-ADB-001: The system shall include the Brake Control Subsystem (BCS).
    REQ-ADB-002: The BCS must accept camera scenery information.
    This is an introductory non-requirement sentence.
    REQ-ADB-003: The system should log lighting state transitions.
    """
    reqs = extract_requirements_from_text(text, system_name="ADB")
    assert len(reqs) >= 3
    assert reqs[0].id == "REQ-ADB-001"


def test_extract_requirements_from_file(tmp_path):
    srs_file = str(tmp_path / "srs.txt")
    with open(srs_file, "w", encoding="utf-8") as f:
        f.write("REQ-001: The system shall verify sensor inputs.")
    reqs = extract_requirements_from_file(srs_file, system_name="ADB")
    assert len(reqs) == 1


def test_generate_requirement_pairs():
    req1 = Requirement(id="REQ-1", text="Text 1", system="ADB")
    req2 = Requirement(id="REQ-2", text="Text 2", system="ADB")
    req3 = Requirement(id="REQ-3", text="Text 3", system="ADB")

    pairs = generate_requirement_pairs([req1, req2, req3])
    assert len(pairs) == 3


def test_calculate_cohens_kappa():
    ann1 = [DependencyType.REQUIRES, DependencyType.CONFLICTS, DependencyType.NO_DEPENDENCY]
    ann2 = [DependencyType.REQUIRES, DependencyType.CONFLICTS, DependencyType.NO_DEPENDENCY]
    kappa = calculate_cohens_kappa(ann1, ann2)
    assert abs(kappa - 1.0) < 1e-4


def test_save_load_pairs_json(tmp_path, sample_pair):
    file_path = str(tmp_path / "pairs.json")
    save_pairs_json([sample_pair], file_path)
    loaded = load_pairs_json(file_path)

    assert len(loaded) == 1
    assert loaded[0].id == sample_pair.id


def test_save_load_pairs_csv(tmp_path, sample_pair):
    file_path = str(tmp_path / "pairs.csv")
    save_pairs_csv([sample_pair], file_path)
    loaded = load_pairs_csv(file_path)

    assert len(loaded) == 1
    assert loaded[0].id == sample_pair.id
