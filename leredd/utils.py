"""Utility functions for requirement extraction, pair generation, I/O, and Cohen's Kappa calculation."""

import re
import json
import pandas as pd
from typing import List, Tuple
from loguru import logger
from leredd.data_models import Requirement, RequirementPair, DependencyType


def extract_requirements_from_text(
    text: str,
    system_name: str = "GenericSystem",
    keywords: List[str] | None = None
) -> List[Requirement]:
    """Extract requirement statements containing keywords ('shall', 'should', 'must') using regex."""
    keywords = keywords or ["shall", "should", "must"]
    pattern = re.compile(
        r"(?i)(?:REQ-[A-Z0-9]+[-:]?\s*)?.*?\b(?:" + "|".join(keywords) + r")\b.*?(?=\.|\n|$)"
    )

    extracted: List[Requirement] = []
    lines = text.split("\n")

    req_counter = 1
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        
        matches = pattern.findall(line_clean)
        for match in matches:
            stmt = match.strip()
            if len(stmt) < 15:
                continue

            # Check if explicit ID exists (e.g. REQ-ADB-001)
            id_match = re.match(r"^(REQ-[A-Z0-9]+[-:]?\d+)", stmt, re.IGNORECASE)
            if id_match:
                req_id = id_match.group(1).upper()
            else:
                req_id = f"REQ-{system_name.upper()}-{req_counter:03d}"
                req_counter += 1

            extracted.append(
                Requirement(
                    id=req_id,
                    text=stmt,
                    system=system_name,
                    source="regex_extraction"
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


def extract_requirements_from_file(
    file_path: str,
    system_name: str | None = None
) -> List[Requirement]:
    """Extract requirements from a file (.txt, .pdf, or .docx)."""
    system = system_name or "SRS"

    if file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif file_path.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            text = ""
    elif file_path.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error reading DOCX file {file_path}: {e}")
            text = ""
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return extract_requirements_from_text(text, system_name=system)


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
            pairs.append(
                RequirementPair(
                    id=pair_id,
                    req_a=req_a,
                    req_b=req_b,
                    dependency_type=DependencyType.NO_DEPENDENCY
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
    for idx, row in df.iterrows():
        req_a = Requirement(
            id=str(row.get("req_a_id", f"REQ-A-{idx}")),
            text=str(row["req_a"]),
            system=str(row.get("system", "Generic"))
        )
        req_b = Requirement(
            id=str(row.get("req_b_id", f"REQ-B-{idx}")),
            text=str(row["req_b"]),
            system=str(row.get("system", "Generic"))
        )
        dep_type = DependencyType.normalize(str(row.get("dependency_type", "No_dependency")))
        pair_id = str(row.get("id", f"PAIR-{idx:04d}"))
        pairs.append(
            RequirementPair(
                id=pair_id,
                req_a=req_a,
                req_b=req_b,
                dependency_type=dep_type
            )
        )
    return pairs


def save_pairs_csv(pairs: List[RequirementPair], csv_path: str) -> None:
    """Save requirement pairs to a CSV file."""
    rows = []
    for pair in pairs:
        rows.append({
            "id": pair.id,
            "req_a_id": pair.req_a.id,
            "req_a": pair.req_a.text,
            "req_b_id": pair.req_b.id,
            "req_b": pair.req_b.text,
            "system": pair.req_a.system,
            "dependency_type": pair.dependency_type.value if pair.dependency_type else "No_dependency"
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(pairs)} requirement pairs to {csv_path}")
