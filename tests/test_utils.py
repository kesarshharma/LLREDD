"""Tests for utility functions in leredd.utils."""

import pytest

from leredd.utils import (
    apply_requirement_metadata,
    extract_requirements_from_text,
    extract_requirements_from_file,
    generate_requirement_pairs,
    calculate_cohens_kappa,
    load_pairs_json,
    save_pairs_json,
    load_pairs_csv,
    save_pairs_csv,
)
from leredd.data_models import AnnotationSource, Requirement, DependencyType, VerificationStatus


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


def test_generated_requirement_ids_skip_explicit_ids_in_any_order():
    explicit_first = """
    REQ-ADB-001: The system shall verify sensor inputs.
    The system shall retain the verified sensor status.
    """
    explicit_after = """
    The system shall verify sensor inputs.
    REQ-ADB-001: The system shall retain the verified sensor status.
    """

    first_order_reqs = extract_requirements_from_text(explicit_first, system_name="ADB")
    second_order_reqs = extract_requirements_from_text(explicit_after, system_name="ADB")

    assert [req.id for req in first_order_reqs] == ["REQ-ADB-001", "REQ-ADB-002"]
    assert [req.id for req in second_order_reqs] == ["REQ-ADB-002", "REQ-ADB-001"]


def test_bracketed_requirement_id_is_preserved():
    text = "[REQ-ADB-007] The system shall verify sensor inputs."

    reqs = extract_requirements_from_text(text, system_name="ADB")

    assert len(reqs) == 1
    assert reqs[0].id == "REQ-ADB-007"


def test_bracketed_alphanumeric_requirement_ids_are_preserved_without_collisions():
    text = "\n".join([
        "[REQ-DEMO-A] The brake controller shall cite REQ-DEMO-B before braking.",
        "[REQ-DEMO-B] The wheel subsystem shall provide readings to the brake controller.",
        "[REQ-DEMO-C] The dashboard shall display the selected media title.",
    ])

    reqs = extract_requirements_from_text(text, system_name="DEMO")
    pairs = generate_requirement_pairs(reqs)

    assert [requirement.id for requirement in reqs] == [
        "REQ-DEMO-A",
        "REQ-DEMO-B",
        "REQ-DEMO-C",
    ]
    assert len(pairs) == 3
    assert len({pair.id for pair in pairs}) == 3


def test_extract_requirements_from_file(tmp_path):
    srs_file = str(tmp_path / "srs.txt")
    with open(srs_file, "w", encoding="utf-8") as f:
        f.write("REQ-001: The system shall verify sensor inputs.")
    reqs = extract_requirements_from_file(srs_file, system_name="ADB")
    assert len(reqs) == 1
    assert reqs[0].source == "srs.txt"
    assert reqs[0].source_location == "line 1"


def test_extract_requirements_from_docx_table_rows(tmp_path):
    docx = pytest.importorskip("docx")
    source = tmp_path / "table-srs.docx"
    document = docx.Document()
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "REQ-TABLE-01"
    table.rows[0].cells[1].text = "The system shall record brake status."
    next_row = table.add_row()
    next_row.cells[0].text = "REQ-TABLE-02"
    next_row.cells[1].text = "The system shall display brake status."
    document.save(source)

    requirements = extract_requirements_from_file(str(source), system_name="TABLE-SRS")
    pairs = generate_requirement_pairs(requirements)

    assert [requirement.id for requirement in requirements] == [
        "REQ-TABLE-01",
        "REQ-TABLE-02",
    ]
    assert requirements[0].text.endswith("The system shall record brake status.")
    assert requirements[0].source == "table-srs.docx"
    assert [requirement.source_location for requirement in requirements] == ["line 1", "line 2"]
    assert len(pairs) == 1


def test_extract_requirements_from_pdf_text(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    source = tmp_path / "text-srs.pdf"
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    }))
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    content = DecodedStreamObject()
    content.set_data(
        b"BT /F1 12 Tf 72 720 Td "
        b"(REQ-PDF-01: The system shall record brake status.) Tj "
        b"0 -24 Td (REQ-PDF-02: The system shall display brake status.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(content)
    with source.open("wb") as output:
        writer.write(output)

    requirements = extract_requirements_from_file(str(source), system_name="PDF-SRS")

    assert [requirement.id for requirement in requirements] == [
        "REQ-PDF-01",
        "REQ-PDF-02",
    ]
    assert requirements[0].text.endswith("The system shall record brake status.")
    assert requirements[0].source == "text-srs.pdf"
    assert [requirement.source_location for requirement in requirements] == ["line 1", "line 2"]
    assert len(generate_requirement_pairs(requirements)) == 1


def test_extract_requirements_from_scanned_pdf_fails_with_ocr_guidance(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    source = tmp_path / "scanned-srs.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with source.open("wb") as output:
        writer.write(output)

    with pytest.raises(ValueError, match="no extractable text.*OCR.*not supported"):
        extract_requirements_from_file(str(source))


def test_apply_requirement_metadata_uses_exact_ids_and_preserves_missing_values():
    requirements = [
        Requirement(id="REQ-A", text="Requirement A", system="S"),
        Requirement(id="REQ-B", text="Requirement B", system="S", component="existing"),
    ]

    enriched = apply_requirement_metadata(
        requirements,
        {"REQ-A": {"component": "Brake Control", "subsystem": "Braking"}},
    )

    assert enriched[0].component == "Brake Control"
    assert enriched[0].subsystem == "Braking"
    assert enriched[1].component == "existing"
    assert enriched[1].subsystem is None
    assert requirements[0].component is None


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"UNKNOWN": {"component": "Brakes"}}, "unknown IDs"),
        ({"REQ-A": {"owner": "Team"}}, "Unsupported metadata fields"),
        ({"REQ-A": {"component": 42}}, "must be a string or null"),
        ({"REQ-A": "Brakes"}, "must be an object"),
    ],
)
def test_apply_requirement_metadata_rejects_malformed_or_unknown_records(metadata, message):
    requirements = [Requirement(id="REQ-A", text="Requirement A")]

    with pytest.raises(ValueError, match=message):
        apply_requirement_metadata(requirements, metadata)


def test_legacy_requirement_without_source_location_remains_loadable():
    requirement = Requirement.model_validate({
        "id": "REQ-LEGACY-1",
        "text": "The system shall preserve legacy requirements.",
        "system": "LEGACY",
        "source": "legacy_srs.txt",
    })

    assert requirement.source == "legacy_srs.txt"
    assert requirement.source_location is None


def test_generate_requirement_pairs():
    req1 = Requirement(id="REQ-1", text="Text 1", system="ADB")
    req2 = Requirement(id="REQ-2", text="Text 2", system="ADB")
    req3 = Requirement(id="REQ-3", text="Text 3", system="ADB")

    pairs = generate_requirement_pairs([req1, req2, req3])
    assert len(pairs) == 3
    assert all(pair.dependency_type is None for pair in pairs)
    assert all(pair.provenance.source == AnnotationSource.CANDIDATE_GENERATOR for pair in pairs)
    assert all(pair.provenance.status == VerificationStatus.PROVISIONAL for pair in pairs)


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


def test_unlabeled_pair_csv_roundtrip_remains_unlabeled(tmp_path, sample_pair):
    sample_pair.dependency_type = None
    file_path = str(tmp_path / "unlabeled.csv")
    save_pairs_csv([sample_pair], file_path)
    loaded = load_pairs_csv(file_path)
    assert loaded[0].dependency_type is None
