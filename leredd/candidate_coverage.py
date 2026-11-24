"""Design-based analysis of a stratified candidate-coverage pilot.

The analysis estimates positive dependency recall from human-adjudicated labels
for a probability sample of strategy-selected and strategy-omitted pairs. It
does not turn annotations into verified data or accept provisional labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from leredd.data_models import (
    AnnotationSource,
    DependencyType,
    RequirementPair,
    VerificationStatus,
)
from leredd.evaluation import ground_truth_provenance_problem


_STRATA = {"hybrid_selected", "hybrid_omitted"}


def _read_json(path: str | Path, description: str) -> tuple[Path, Any, bytes]:
    source = Path(path).expanduser()
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} '{source}': {exc}") from exc
    return source, payload, raw


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def analyze_candidate_recall(
    pilot_manifest_path: str | Path,
    strata_map_path: str | Path,
    verified_pairs_path: str | Path,
) -> dict[str, Any]:
    """Estimate candidate positive recall from adjudicated, verified pilot labels.

    The sample is a simple random sample without replacement within each
    stratum. The point estimate is a post-stratified ratio estimator. Its
    optional confidence interval uses a finite-population-corrected, linearized
    normal approximation and is omitted when either non-census stratum has an
    all-positive or all-negative sample.
    """
    manifest_path, manifest, _ = _read_json(pilot_manifest_path, "pilot manifest")
    _, map_rows, map_bytes = _read_json(strata_map_path, "analysis-only strata map")
    _, verified_raw, verified_bytes = _read_json(
        verified_pairs_path, "verified adjudication dataset"
    )

    if not isinstance(manifest, dict) or manifest.get("status") != "prepared_not_reviewed":
        raise ValueError("Pilot manifest must have status 'prepared_not_reviewed'.")
    required_manifest = {
        "artifact_type", "reviewer_input", "reviewer_input_sha256",
        "source_srs_path", "source_srs_sha256", "population_pair_count",
        "strategy_candidate_count", "strategy_omitted_count", "sample_total",
        "analysis_only_strata_map_sha256", "candidate_source_pairs_path",
        "candidate_source_pairs_sha256", "full_pair_universe_path",
        "full_pair_universe_sha256",
    }
    missing = sorted(required_manifest - manifest.keys())
    if missing:
        raise ValueError(f"Pilot manifest is missing required fields: {', '.join(missing)}.")
    if manifest["artifact_type"] != "unlabeled_blind_candidate_coverage_pilot":
        raise ValueError("Pilot manifest is not a supported blind coverage-pilot artifact.")
    if manifest.get("strategy") != "hybrid":
        raise ValueError("This pilot analyzer requires the recorded hybrid strategy.")
    count_fields = (
        "population_pair_count", "strategy_candidate_count",
        "strategy_omitted_count", "sample_total",
    )
    if any(
        isinstance(manifest.get(name), bool)
        or not isinstance(manifest.get(name), int)
        or manifest[name] < 1
        for name in count_fields
    ):
        raise ValueError("Pilot manifest population and sample counts must be positive integers.")

    reviewer_path, _, reviewer_bytes = _read_json(
        manifest["reviewer_input"], "pinned blinded reviewer input"
    )
    if _sha256(reviewer_bytes) != manifest["reviewer_input_sha256"]:
        raise ValueError("Blinded reviewer input hash does not match the pilot manifest.")
    if _sha256(map_bytes) != manifest["analysis_only_strata_map_sha256"]:
        raise ValueError("Analysis-only strata map hash does not match the pilot manifest.")
    try:
        source_bytes = Path(manifest["source_srs_path"]).expanduser().read_bytes()
    except OSError as exc:
        raise ValueError(f"Could not read pinned SRS context: {exc}") from exc
    if _sha256(source_bytes) != manifest["source_srs_sha256"]:
        raise ValueError("SRS context hash does not match the pilot manifest.")

    if not isinstance(map_rows, list) or not map_rows:
        raise ValueError("Analysis-only strata map must be a non-empty JSON array.")
    if not isinstance(verified_raw, list) or not verified_raw:
        raise ValueError("Verified adjudication dataset must be a non-empty JSON array.")
    _, candidate_raw, candidate_bytes = _read_json(
        manifest["candidate_source_pairs_path"], "candidate strategy pair output"
    )
    _, universe_raw, universe_bytes = _read_json(
        manifest["full_pair_universe_path"], "complete pair-universe output"
    )
    if _sha256(candidate_bytes) != manifest["candidate_source_pairs_sha256"]:
        raise ValueError("Candidate strategy pair hash does not match the pilot manifest.")
    if _sha256(universe_bytes) != manifest["full_pair_universe_sha256"]:
        raise ValueError("Complete pair-universe hash does not match the pilot manifest.")
    if not isinstance(candidate_raw, list) or not isinstance(universe_raw, list):
        raise ValueError("Candidate and complete pair inputs must be JSON arrays.")
    try:
        reviewer_records = [
            RequirementPair.model_validate(row)
            for row in json.loads(reviewer_bytes.decode("utf-8"))
        ]
        verified_pairs = [RequirementPair.model_validate(row) for row in verified_raw]
        candidate_source_pairs = [RequirementPair.model_validate(row) for row in candidate_raw]
        full_pair_universe = [RequirementPair.model_validate(row) for row in universe_raw]
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Pilot or verified pair input is malformed: {exc}") from exc

    reviewer_by_id = {pair.id: pair for pair in reviewer_records}
    if len(reviewer_by_id) != len(reviewer_records):
        raise ValueError("Blinded reviewer input contains duplicate pair IDs.")
    for pair in reviewer_records:
        if pair.dependency_type is not None or pair.provenance is None:
            raise ValueError(
                "Pinned blinded reviewer input must remain unlabeled and provenance-bearing."
            )
        if (
            pair.provenance.source != AnnotationSource.CANDIDATE_GENERATOR
            or pair.provenance.status != VerificationStatus.PROVISIONAL
        ):
            raise ValueError(
                "Pinned blinded reviewer input must remain candidate-generated and provisional."
            )
    if len(verified_pairs) != len({pair.id for pair in verified_pairs}):
        raise ValueError("Verified adjudication dataset contains duplicate pair IDs.")
    if len(reviewer_records) != manifest["sample_total"]:
        raise ValueError("Blinded reviewer input count does not match the pilot manifest.")

    def canonical_key(pair: RequirementPair) -> tuple[str, str]:
        return tuple(sorted((pair.req_a.id, pair.req_b.id)))

    full_by_id = {pair.id: pair for pair in full_pair_universe}
    full_keys = {canonical_key(pair) for pair in full_pair_universe}
    candidate_keys = {canonical_key(pair) for pair in candidate_source_pairs}
    if len(full_by_id) != len(full_pair_universe) or len(full_keys) != len(full_pair_universe):
        raise ValueError("Complete pair universe contains duplicate IDs or requirement pairs.")
    if len(candidate_keys) != len(candidate_source_pairs) or not candidate_keys <= full_keys:
        raise ValueError(
            "Candidate strategy output is duplicated or outside the complete pair universe."
        )
    if len(full_pair_universe) != manifest["population_pair_count"]:
        raise ValueError("Complete pair file count does not match the pilot manifest.")
    if len(candidate_keys) != manifest["strategy_candidate_count"]:
        raise ValueError("Candidate pair file count does not match the pilot manifest.")
    if len(full_keys - candidate_keys) != manifest["strategy_omitted_count"]:
        raise ValueError("Omitted pair count does not match the pilot manifest.")

    sample_map: dict[str, dict[str, Any]] = {}
    stratum_counts: dict[str, int] = {}
    for row in map_rows:
        if not isinstance(row, dict):
            raise ValueError("Every strata-map row must be a JSON object.")
        pair_id = row.get("blind_pair_id")
        stratum = row.get("stratum")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("Every strata-map row requires a blind_pair_id.")
        if stratum not in _STRATA:
            raise ValueError(f"Unsupported candidate-coverage stratum: {stratum!r}.")
        if pair_id in sample_map:
            raise ValueError(f"Duplicate blind_pair_id in strata map: {pair_id}.")
        original_pair_id = row.get("original_pair_id")
        if original_pair_id not in full_by_id:
            raise ValueError(f"Unknown original pair ID in strata map: {original_pair_id!r}.")
        original_pair = full_by_id[original_pair_id]
        if {row.get("req_a_id"), row.get("req_b_id")} != {
            original_pair.req_a.id, original_pair.req_b.id
        }:
            raise ValueError(f"Requirement IDs do not match original pair {original_pair_id}.")
        actually_selected = canonical_key(original_pair) in candidate_keys
        if actually_selected != (stratum == "hybrid_selected"):
            raise ValueError(f"Stratum assignment does not match strategy output for {pair_id}.")
        sample_pair = reviewer_by_id.get(pair_id)
        if sample_pair is None or canonical_key(sample_pair) != canonical_key(original_pair):
            raise ValueError(f"Blind pair {pair_id} does not match its original universe pair.")
        population_n = row.get("population_stratum_size")
        sample_n = row.get("sample_stratum_size")
        inclusion = row.get("inclusion_probability")
        weight = row.get("analysis_weight")
        if (
            isinstance(population_n, bool) or not isinstance(population_n, int)
            or population_n < 1 or isinstance(sample_n, bool)
            or not isinstance(sample_n, int) or sample_n < 1
            or sample_n > population_n
        ):
            raise ValueError(f"Invalid population/sample sizes for {pair_id}.")
        if not isinstance(inclusion, (int, float)) or not math.isclose(
            float(inclusion), sample_n / population_n, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"Invalid inclusion probability for {pair_id}.")
        if not isinstance(weight, (int, float)) or not math.isclose(
            float(weight), population_n / sample_n, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"Invalid analysis weight for {pair_id}.")
        sample_map[pair_id] = row
        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1

    if set(sample_map) != set(reviewer_by_id):
        raise ValueError("Strata map IDs must exactly match the blinded reviewer input IDs.")
    if set(stratum_counts) != _STRATA:
        raise ValueError("Both selected and omitted strata are required for recall estimation.")

    verified_by_id = {pair.id: pair for pair in verified_pairs}
    if set(verified_by_id) != set(sample_map):
        raise ValueError(
            "Verified adjudication IDs must exactly match the full pilot sample; "
            "unresolved or missing labels cannot be treated as negatives."
        )

    for pair_id, pair in verified_by_id.items():
        sample_pair = reviewer_by_id[pair_id]
        if pair.req_a.model_dump(mode="json") != sample_pair.req_a.model_dump(mode="json") or (
            pair.req_b.model_dump(mode="json") != sample_pair.req_b.model_dump(mode="json")
        ):
            raise ValueError(f"Verified pair {pair_id} does not match the pinned reviewer input.")
        problem = ground_truth_provenance_problem(pair)
        if problem:
            raise ValueError(f"Pair {pair_id} is not verified human ground truth: {problem}.")

    stratum_populations = {
        name: int(
            next(
                row["population_stratum_size"]
                for row in map_rows
                if row["stratum"] == name
            )
        )
        for name in _STRATA
    }
    selected_population = stratum_populations["hybrid_selected"]
    omitted_population = stratum_populations["hybrid_omitted"]
    if selected_population != manifest["strategy_candidate_count"]:
        raise ValueError("Selected-stratum population does not match the pilot manifest.")
    if omitted_population != manifest["strategy_omitted_count"]:
        raise ValueError("Omitted-stratum population does not match the pilot manifest.")
    population_total = selected_population + omitted_population
    if population_total != manifest["population_pair_count"]:
        raise ValueError("Stratum populations do not match the full pair universe.")

    strata: dict[str, dict[str, Any]] = {}
    for name in sorted(_STRATA):
        rows = [row for row in map_rows if row["stratum"] == name]
        first = rows[0]
        population_n = int(first["population_stratum_size"])
        sample_n = len(rows)
        if any(
            row["population_stratum_size"] != population_n
            or row["sample_stratum_size"] != sample_n
            for row in rows
        ):
            raise ValueError(f"Stratum metadata is inconsistent for {name}.")
        positive_n = sum(
            verified_by_id[row["blind_pair_id"]].dependency_type != DependencyType.NO_DEPENDENCY
            for row in rows
        )
        estimated_positive_total = population_n * positive_n / sample_n
        positive_rate = positive_n / sample_n
        strata[name] = {
            "population_size": population_n,
            "sample_size": sample_n,
            "positive_sample_count": positive_n,
            "no_dependency_sample_count": sample_n - positive_n,
            "positive_sample_rate": positive_rate,
            "inclusion_probability": sample_n / population_n,
            "analysis_weight": population_n / sample_n,
            "estimated_positive_population_total": estimated_positive_total,
        }

    selected = strata["hybrid_selected"]
    omitted = strata["hybrid_omitted"]
    estimated_total = (
        selected["estimated_positive_population_total"]
        + omitted["estimated_positive_population_total"]
    )
    if estimated_total <= 0:
        raise ValueError("No positive dependency was observed; positive recall is not estimable.")
    recall = selected["estimated_positive_population_total"] / estimated_total

    boundary_strata = [
        name for name, result in strata.items()
        if result["sample_size"] < result["population_size"]
        and result["positive_sample_count"] in (0, result["sample_size"])
    ]
    standard_error = None
    confidence_interval = None
    uncertainty_note = None
    if boundary_strata:
        uncertainty_note = (
            "Not calculated: at least one sampled stratum had all positive or all "
            "negative labels, making its sample-variance estimate degenerate."
        )
    else:
        variances: dict[str, float] = {}
        for name, result in strata.items():
            n = result["sample_size"]
            population_n = result["population_size"]
            p = result["positive_sample_rate"]
            sample_variance = (
                0.0 if n == population_n else n * p * (1 - p) / (n - 1)
            )
            variances[name] = (
                population_n ** 2
                * (1 - n / population_n)
                * sample_variance / n
            )
        variance_recall = (
            (1 - recall) ** 2 * variances["hybrid_selected"]
            + recall ** 2 * variances["hybrid_omitted"]
        ) / (estimated_total ** 2)
        standard_error = math.sqrt(max(0.0, variance_recall))
        confidence_interval = [
            max(0.0, recall - 1.96 * standard_error),
            min(1.0, recall + 1.96 * standard_error),
        ]
        uncertainty_note = (
            "Approximate 95% normal interval from a finite-population-corrected "
            "linearized stratified ratio estimator; pilot-sized samples may make "
            "this interval unreliable."
        )

    return {
        "analysis_type": "stratified_candidate_positive_recall_estimate",
        "status": "exploratory_estimate_from_verified_adjudications",
        "strategy": manifest.get("strategy", "hybrid"),
        "top_k": manifest.get("top_k", 5),
        "population_pair_count": population_total,
        "verified_sample_count": len(verified_pairs),
        "strata": strata,
        "estimated_positive_dependency_count": estimated_total,
        "estimated_selected_positive_count": selected["estimated_positive_population_total"],
        "estimated_omitted_positive_count": omitted["estimated_positive_population_total"],
        "estimated_positive_recall": recall,
        "approx_standard_error": standard_error,
        "approx_95_percent_confidence_interval": confidence_interval,
        "uncertainty_note": uncertainty_note,
        "method": (
            "Within-stratum simple random sampling without replacement; post-stratified "
            "positive totals and ratio estimate. Any non-No_dependency class counts as positive."
        ),
        "input_sha256": {
            "pilot_manifest": _sha256(manifest_path.read_bytes()),
            "reviewer_input": manifest["reviewer_input_sha256"],
            "analysis_only_strata_map": _sha256(map_bytes),
            "source_srs_context": manifest["source_srs_sha256"],
            "candidate_source_pairs": _sha256(candidate_bytes),
            "full_pair_universe": _sha256(universe_bytes),
            "verified_adjudications": _sha256(verified_bytes),
        },
        "warnings": [
            "This is a pilot estimate, not proof of population-level performance.",
            "Labels must be human-adjudicated and VERIFIED; independent annotations "
            "alone are not ground truth.",
            "The candidate-strategy result does not assess classifier accuracy or generalization.",
        ],
    }
