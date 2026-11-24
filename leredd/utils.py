"""Utility functions for requirement extraction, pair generation, I/O, and Cohen's Kappa calculation."""

import re
import json
from pathlib import Path
import pandas as pd
from typing import Any, List, Mapping
from loguru import logger
from leredd.data_models import (
    AnnotationProvenance, AnnotationSource, DependencyType, Requirement,
    RequirementPair, VerificationStatus,
)
from leredd.synthetic import is_synthetic, synthetic_source_identity


def extract_requirements_from_text(
    text: str,
    system_name: str = "GenericSystem",
    keywords: List[str] | None = None,
    *,
    source_name: str | None = None,
) -> List[Requirement]:
    """Extract requirement statements containing keywords while preserving terminal punctuation."""
    keywords = keywords or ["shall", "should", "must"]
    pattern = re.compile(
        r"(?i)(?:REQ-[A-Z0-9]+[-:]?\s*)?.*?\b(?:" + "|".join(keywords) + r")\b.*?(?:\.|(?=\n|$))"
    )
    requirement_id_pattern = re.compile(
        r"^(?:[\[(]\s*)?(REQ-[A-Z0-9]+(?:[-:][A-Z0-9]+)*)",
        re.IGNORECASE,
    )

    extracted: List[Requirement] = []
    lines = text.split("\n")
    source_name = source_name.strip() if source_name and source_name.strip() else None
    explicit_ids = {
        id_match.group(1).upper()
        for line in lines
        for match in pattern.findall(line.strip())
        if (id_match := requirement_id_pattern.match(match.strip()))
    }

    req_counter = 1
    used_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        line_clean = line.strip()
        if not line_clean:
            continue
        
        matches = pattern.findall(line_clean)
        for match in matches:
            stmt = match.strip()
            if len(stmt) < 15:
                continue

            # Check if explicit ID exists (e.g. REQ-ADB-001)
            id_match = requirement_id_pattern.match(stmt)
            if id_match:
                req_id = id_match.group(1).upper()
                used_ids.add(req_id)
            else:
                while (
                    f"REQ-{system_name.upper()}-{req_counter:03d}" in explicit_ids
                    or f"REQ-{system_name.upper()}-{req_counter:03d}" in used_ids
                ):
                    req_counter += 1
                req_id = f"REQ-{system_name.upper()}-{req_counter:03d}"
                used_ids.add(req_id)
                req_counter += 1

            extracted.append(
                Requirement(
                    id=req_id,
                    text=stmt,
                    system=system_name,
                    source=source_name or "regex_extraction",
                    source_location=f"line {line_number}",
                )
            )

    # Deduplicate by requirement text
    unique_reqs: List[Requirement] = []
    seen = set()
    for req in extracted:
        if req.text not in seen:
            seen.add(req.text)
            unique_reqs.append(req)

    logger.info(f"Extracted {len(unique_reqs)} unique requirements for system '{system_name}'.")
    return unique_reqs


def load_requirement_metadata_json(file_path: str) -> dict[str, dict[str, Any]]:
    """Load an explicit requirement-ID-to-metadata JSON sidecar."""
    try:
        with open(file_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read requirement metadata JSON '{file_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Requirement metadata JSON must be an object keyed by requirement ID.")
    return payload


def apply_requirement_metadata(
    requirements: List[Requirement],
    metadata_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Requirement]:
    """Attach explicit component/subsystem values without inferring metadata.

    The sidecar maps exact requirement IDs to objects containing only `component`
    and/or `subsystem`. Unknown IDs and malformed fields fail rather than silently
    dropping user-provided metadata.
    """
    if not isinstance(metadata_by_id, Mapping):
        raise ValueError("Requirement metadata must be an object keyed by requirement ID.")

    requirement_ids = [requirement.id for requirement in requirements]
    if len(requirement_ids) != len(set(requirement_ids)):
        raise ValueError("Extracted requirements contain duplicate IDs; metadata cannot be joined safely.")

    unknown_ids = set(metadata_by_id) - set(requirement_ids)
    if unknown_ids:
        raise ValueError(
            "Requirement metadata references unknown IDs: "
            + ", ".join(sorted(str(item) for item in unknown_ids))
        )

    allowed_fields = {"component", "subsystem", "module", "metadata"}
    enriched = []
    for requirement in requirements:
        entry = metadata_by_id.get(requirement.id)
        if entry is None:
            enriched.append(requirement)
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(f"Metadata for {requirement.id} must be an object.")
        extra_fields = set(entry) - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Unsupported metadata fields for {requirement.id}: "
                + ", ".join(sorted(str(item) for item in extra_fields))
            )

        updates: dict[str, Any] = {}
        for field in allowed_fields:
            if field not in entry or entry[field] is None:
                continue
            value = entry[field]
            if field == "metadata":
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"Metadata field 'metadata' for {requirement.id} must be an object or null."
                    )
                updates["metadata"] = dict(value)
            else:
                if not isinstance(value, str):
                    raise ValueError(f"Metadata field '{field}' for {requirement.id} must be a string or null.")
                clean_value = value.strip()
                updates[field] = clean_value or None
        enriched.append(requirement.model_copy(update=updates) if updates else requirement)
    return enriched


def extract_requirements_from_file(
    file_path: str,
    system_name: str | None = None
) -> List[Requirement]:
    """Extract requirements from a file (.txt, .pdf, or .docx)."""
    system = system_name or "SRS"

    suffix = Path(file_path).suffix.lower()
    if suffix in {".txt", ".md"}:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif suffix == ".pdf":
        try:
            import pypdf
        except ImportError as exc:
            raise RuntimeError(
                "PDF extraction requires the optional document dependencies; "
                "install the 'documents' extra (pip install 'leredd[documents]')."
            ) from exc
        try:
            reader = pypdf.PdfReader(file_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as exc:
            raise ValueError(f"Could not extract text from PDF file '{file_path}': {exc}") from exc
        if not text.strip():
            raise ValueError(
                f"PDF file '{file_path}' contains no extractable text. "
                "Scanned PDFs require OCR, which is not supported by this extractor."
            )
    elif suffix == ".docx":
        try:
            import docx
        except ImportError as exc:
            raise RuntimeError(
                "DOCX extraction requires the optional document dependencies; "
                "install the 'documents' extra (pip install 'leredd[documents]')."
            ) from exc
        try:
            doc = docx.Document(file_path)
            # Walk body blocks in source order so table-based SRS requirements
            # are preserved alongside ordinary paragraphs.
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import Table
            from docx.text.paragraph import Paragraph

            blocks = []
            for block in doc.element.body.iterchildren():
                if isinstance(block, CT_P):
                    paragraph_text = Paragraph(block, doc).text
                    if paragraph_text.strip():
                        blocks.append(paragraph_text)
                elif isinstance(block, CT_Tbl):
                    table = Table(block, doc)
                    for row in table.rows:
                        cell_text = [
                            " ".join(part.strip() for part in cell.text.splitlines() if part.strip())
                            for cell in row.cells
                        ]
                        row_text = "\t".join(cell_text).strip()
                        if row_text:
                            blocks.append(row_text)
            text = "\n".join(blocks)
        except Exception as exc:
            raise ValueError(f"Could not extract text from DOCX file '{file_path}': {exc}") from exc
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return extract_requirements_from_text(
        text,
        system_name=system,
        source_name=Path(file_path).name,
    )


def generate_requirement_pairs(requirements: List[Requirement]) -> List[RequirementPair]:
    """Generate all n(n-1)/2 unique requirement pairs from a list of n requirements."""
    n = len(requirements)
    pairs: List[RequirementPair] = []

    pair_idx = 1
    for i in range(n):
        for j in range(i + 1, n):
            req_a = requirements[i]
            req_b = requirements[j]
            pair_id = f"PAIR-{req_a.system}-{pair_idx:04d}"
            synthetic = is_synthetic(req_a) or is_synthetic(req_b)
            origin = synthetic_source_identity(req_a, req_b) if synthetic else None
            pairs.append(
                RequirementPair(
                    id=pair_id,
                    req_a=req_a,
                    req_b=req_b,
                    dependency_type=None,
                    provenance=AnnotationProvenance(
                        source=AnnotationSource.CANDIDATE_GENERATOR,
                        status=VerificationStatus.PROVISIONAL,
                        annotator_id="legacy_pairs_generator",
                        synthetic=synthetic,
                        source_identity=origin,
                    ),
                    synthetic=synthetic,
                    source_identity=origin,
                )
            )
            pair_idx += 1

    logger.info(f"Generated {len(pairs)} unique requirement pairs from {n} requirements.")
    return pairs


def calculate_cohens_kappa(
    annotations1: List[DependencyType],
    annotations2: List[DependencyType]
) -> float:
    """Calculate Cohen's kappa coefficient (kappa) for inter-annotator agreement."""
    if len(annotations1) != len(annotations2) or not annotations1:
        return 0.0

    labels1 = [a.value for a in annotations1]
    labels2 = [a.value for a in annotations2]

    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(labels1, labels2))
    except Exception:
        # Manual Cohen's kappa calculation
        n = len(labels1)
        categories = list(set(labels1 + labels2))
        
        # Observed agreement Po
        po = sum(1 for a, b in zip(labels1, labels2) if a == b) / float(n)

        # Expected agreement Pe
        pe = 0.0
        for cat in categories:
            p1 = sum(1 for a in labels1 if a == cat) / float(n)
            p2 = sum(1 for b in labels2 if b == cat) / float(n)
            pe += (p1 * p2)

        if pe == 1.0:
            return 1.0
        return float((po - pe) / (1.0 - pe))


def load_pairs_json(json_path: str) -> List[RequirementPair]:
    """Load requirement pairs from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = [RequirementPair.model_validate(item) for item in data]
    return pairs


def save_pairs_json(pairs: List[RequirementPair], json_path: str) -> None:
    """Save requirement pairs to a JSON file."""
    data = [pair.model_dump(mode="json") for pair in pairs]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(pairs)} requirement pairs to {json_path}")


def load_pairs_csv(csv_path: str) -> List[RequirementPair]:
    """Load requirement pairs from a CSV file."""
    df = pd.read_csv(csv_path)
    pairs = []

    def parse_bool(value: Any) -> bool:
        return False if pd.isna(value) else str(value).strip().casefold() in {
            "true", "1", "yes",
        }

    def parse_optional_text(value: Any) -> str | None:
        return None if pd.isna(value) or not str(value).strip() else str(value)

    for idx, row in df.iterrows():
        req_a = Requirement(
            id=str(row.get("req_a_id", f"REQ-A-{idx}")),
            text=str(row["req_a"]),
            system=str(row.get("system", "Generic")),
            synthetic=parse_bool(row.get("req_a_synthetic", False)),
            source_identity=parse_optional_text(row.get("req_a_source_identity")),
        )
        req_b = Requirement(
            id=str(row.get("req_b_id", f"REQ-B-{idx}")),
            text=str(row["req_b"]),
            system=str(row.get("system", "Generic")),
            synthetic=parse_bool(row.get("req_b_synthetic", False)),
            source_identity=parse_optional_text(row.get("req_b_source_identity")),
        )
        raw_dep_type = row.get("dependency_type")
        dep_type = (
            None
            if pd.isna(raw_dep_type) or not str(raw_dep_type).strip()
            else DependencyType.normalize(str(raw_dep_type))
        )
        pair_id = str(row.get("id", f"PAIR-{idx:04d}"))
        pairs.append(
            RequirementPair(
                id=pair_id,
                req_a=req_a,
                req_b=req_b,
                dependency_type=dep_type,
                synthetic=parse_bool(row.get("synthetic", False)),
                source_identity=parse_optional_text(row.get("source_identity")),
            )
        )
    return pairs


def save_pairs_csv(pairs: List[RequirementPair], csv_path: str) -> None:
    """Save requirement pairs to a CSV file."""
    rows = []
    for pair in pairs:
        rows.append({
            "id": pair.id,
            "synthetic": pair.synthetic,
            "source_identity": pair.source_identity,
            "req_a_id": pair.req_a.id,
            "req_a": pair.req_a.text,
            "req_a_synthetic": pair.req_a.synthetic,
            "req_a_source_identity": pair.req_a.source_identity,
            "req_b_id": pair.req_b.id,
            "req_b": pair.req_b.text,
            "req_b_synthetic": pair.req_b.synthetic,
            "req_b_source_identity": pair.req_b.source_identity,
            "system": pair.req_a.system,
            "dependency_type": pair.dependency_type.value if pair.dependency_type else ""
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(pairs)} requirement pairs to {csv_path}")
