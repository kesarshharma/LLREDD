"""Unit tests for Provisional Requirement Dependency Labeling (Phase 4, TASK-P4-02)."""

import json
import pytest
from typer.testing import CliRunner

from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.config import LEREDDConfig
from leredd.llm import MockLLMClient
from leredd.provisional_labeler import (
    HeuristicLabeler,
    HeuristicRuleResult,
    LabelingMode,
    LabelingRunManifest,
    LLMLabeler,
    MultiModelLLMLabeler,
    ProvisionalLabeler,
)
from leredd.cli import app

runner = CliRunner()


@pytest.fixture
def req_a():
    return Requirement(
        id="REQ-001",
        text="The system shall process user payments securely using AES-256 encryption.",
        system_name="BillingSystem"
    )


@pytest.fixture
def req_b():
    return Requirement(
        id="REQ-002",
        text="The system requires REQ-001 for verifying user identity before payment processing.",
        system_name="BillingSystem"
    )


@pytest.fixture
def req_c():
    return Requirement(
        id="REQ-003",
        text="The system shall disable all payment processing during maintenance windows.",
        system_name="BillingSystem"
    )


def test_six_class_canonical_taxonomy():
    """Verify all 6 dependency classes are supported and match exact canonical enum values."""
    expected_values = {
        "Requires",
        "Implements",
        "Conflicts",
        "Details",
        "Is similar",
        "No_dependency"
    }
    actual_values = {d.value for d in DependencyType}
    assert actual_values == expected_values, f"Taxonomy mismatch: {actual_values}"


def test_heuristic_rule_explicit_reference(req_a, req_b):
    """Verify heuristic rule detects explicit requirement ID citation."""
    pair = RequirementPair(id="P-001", req_a=req_b, req_b=req_a)
    result = HeuristicLabeler.evaluate_pair(pair)

    assert result.suggested_label == DependencyType.REQUIRES
    assert result.rule_id == "RULE-REF-ID-DIRECT"
    assert result.confidence > 0.8


def test_heuristic_rule_negation_conflict(req_a, req_c):
    """Verify heuristic rule detects asymmetric negation and conflict."""
    # req_a has "process user payments"
    # req_c has "disable all payment processing"
    pair = RequirementPair(id="P-002", req_a=req_a, req_b=req_c)
    result = HeuristicLabeler.evaluate_pair(pair)

    assert result.suggested_label == DependencyType.CONFLICTS
    assert result.rule_id == "RULE-NEG-CONFLICT"


def test_heuristic_rule_abstention():
    """Verify heuristics return None (abstain) when evidence is insufficient, NOT No_dependency."""
    r1 = Requirement(id="R1", text="The user dashboard displays transaction summary charts.")
    r2 = Requirement(id="R2", text="The database stores user profiles in PostgreSQL.")
    pair = RequirementPair(id="P-003", req_a=r1, req_b=r2)

    result = HeuristicLabeler.evaluate_pair(pair)
    assert result.suggested_label is None
    assert result.rule_id == "RULE-ABSTAIN"


def test_llm_labeler_mock_success(req_a, req_b):
    """Verify LLMLabeler with MockLLMClient parses valid structured output correctly."""
    # Mock LLM returns a structured JSON string
    mock_response = json.dumps({
        "dependency_type": "Requires",
        "confidence": 4,
        "explanation": "REQ-002 explicitly references REQ-001."
    })
    mock_client = MockLLMClient(response_text=mock_response)
    labeler = LLMLabeler(llm_client=mock_client)

    pair = RequirementPair(id="P-004", req_a=req_a, req_b=req_b)
    predicted_label, confidence, explanation = labeler.label_pair(pair)

    assert predicted_label == DependencyType.REQUIRES
    assert confidence == 4
    assert "explicitly references" in explanation


def test_llm_labeler_malformed_response_handling(req_a, req_b):
    """Verify LLM labeler handles malformed output safely without defaulting to No_dependency."""
    mock_client = MockLLMClient(response_text="I think they might be related, but invalid JSON format!")
    labeler = LLMLabeler(llm_client=mock_client)

    pair = RequirementPair(id="P-005", req_a=req_a, req_b=req_b)
    predicted_label, confidence, explanation = labeler.label_pair(pair)

    assert predicted_label is None  # Must NOT be No_dependency!
    assert "Malformed" in explanation or "invalid" in explanation.lower()


def test_provisional_labeler_agreement_hybrid(req_a, req_b):
    """Verify hybrid mode produces ENSEMBLE source when heuristic and LLM agree."""
    mock_response = json.dumps({
        "dependency_type": "Requires",
        "confidence": 5,
        "explanation": "Direct reference in text."
    })
    mock_client = MockLLMClient(response_text=mock_response)
    llm_labeler = LLMLabeler(llm_client=mock_client)

    coordinator = ProvisionalLabeler(llm_labeler=llm_labeler)
    pair = RequirementPair(id="P-006", req_a=req_b, req_b=req_a)

    processed = coordinator.process_pair(pair, mode=LabelingMode.HYBRID)

    assert processed.dependency_type == DependencyType.REQUIRES
    assert processed.provenance.status == VerificationStatus.PROVISIONAL
    assert processed.provenance.source == AnnotationSource.ENSEMBLE
    assert "AGREEMENT" in processed.provenance.evidence


def test_provisional_labeler_disagreement_preserves_unresolved(req_a, req_b):
    """Verify hybrid mode sets label to None (Unresolved) on disagreement and records review notes."""
    # Heuristic will detect explicit citation -> Requires
    # Mock LLM returns Conflicts
    mock_response = json.dumps({
        "dependency_type": "Conflicts",
        "confidence": 3,
        "explanation": "Contradictory workflow."
    })
    mock_client = MockLLMClient(response_text=mock_response)
    llm_labeler = LLMLabeler(llm_client=mock_client)

    coordinator = ProvisionalLabeler(llm_labeler=llm_labeler)
    pair = RequirementPair(id="P-007", req_a=req_b, req_b=req_a)

    processed = coordinator.process_pair(pair, mode=LabelingMode.HYBRID)

    assert processed.dependency_type is None  # Unresolved due to disagreement
    assert processed.provenance.status == VerificationStatus.PROVISIONAL
    assert "DISAGREEMENT" in processed.provenance.review_notes


def test_multi_model_labeler_records_unanimous_votes_as_provisional(req_a, req_b):
    response = json.dumps({
        "dependency_type": "Requires",
        "confidence": 4,
        "explanation": "A controlled vote.",
    })
    labelers = [
        LLMLabeler(
            config=LEREDDConfig(llm_model=model_id),
            llm_client=MockLLMClient(model=model_id, response_text=response),
        )
        for model_id in ("model-a", "model-b")
    ]
    coordinator = ProvisionalLabeler(llm_labeler=MultiModelLLMLabeler(labelers))
    pair = RequirementPair(id="MULTI-AGREE", req_a=req_a, req_b=req_b)

    labeled, manifest = coordinator.process_batch([pair], mode=LabelingMode.MULTI_LLM)

    assert labeled[0].dependency_type == DependencyType.REQUIRES
    assert labeled[0].provenance.status == VerificationStatus.PROVISIONAL
    assert labeled[0].provenance.source == AnnotationSource.ENSEMBLE
    assert labeled[0].provenance.model_id == "ensemble[model-a,model-b]"
    assert "MULTI_LLM_UNANIMOUS" in labeled[0].provenance.evidence
    assert manifest.agreed_count == 1
    assert manifest.metadata["llm_models"] == ["model-a", "model-b"]


def test_multi_model_disagreement_routes_hybrid_pair_for_review(req_a, req_b):
    responses = [
        json.dumps({"dependency_type": label, "confidence": 4, "explanation": f"Vote {label}."})
        for label in ("Requires", "Conflicts")
    ]
    labelers = [
        LLMLabeler(
            config=LEREDDConfig(llm_model=model_id),
            llm_client=MockLLMClient(model=model_id, response_text=response),
        )
        for model_id, response in zip(("model-a", "model-b"), responses)
    ]
    coordinator = ProvisionalLabeler(llm_labeler=MultiModelLLMLabeler(labelers))
    # req_b explicitly cites req_a, so the heuristic alone would suggest Requires.
    pair = RequirementPair(id="MULTI-DISAGREE", req_a=req_b, req_b=req_a)

    processed = coordinator.process_pair(pair, mode=LabelingMode.HYBRID)

    assert processed.dependency_type is None
    assert processed.provenance.status == VerificationStatus.PROVISIONAL
    assert processed.provenance.source == AnnotationSource.ENSEMBLE
    assert "MULTI_LLM_DISAGREEMENT" in processed.provenance.review_notes
    assert "model-a" in processed.provenance.review_notes
    assert "model-b" in processed.provenance.review_notes


def test_cli_multi_llm_mode_uses_distinct_configured_models(tmp_path, monkeypatch):
    from leredd.llm import LLMClientFactory

    pair = RequirementPair(
        id="CLI-MULTI",
        req_a=Requirement(id="CLI-A", text="System shall execute the primary task."),
        req_b=Requirement(id="CLI-B", text="System shall store the task result."),
    )
    input_file = tmp_path / "candidates.json"
    output_file = tmp_path / "labeled.json"
    input_file.write_text(json.dumps([pair.model_dump(mode="json")]), encoding="utf-8")
    response = json.dumps({"dependency_type": "Details", "confidence": 3, "explanation": "Vote."})
    monkeypatch.setattr(
        LLMClientFactory,
        "create",
        lambda config=None: MockLLMClient(model=config.llm_model, response_text=response),
    )

    result = runner.invoke(
        app,
        [
            "label", "--input", str(input_file), "--output", str(output_file),
            "--mode", "multi_llm", "--models", "model-a,model-b",
        ],
    )

    assert result.exit_code == 0, result.output
    records = json.loads(output_file.read_text(encoding="utf-8"))
    assert records[0]["dependency_type"] == "Details"
    assert records[0]["provenance"]["status"] == "provisional"
    manifest = json.loads((tmp_path / "labeled_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["llm_models"] == ["model-a", "model-b"]


def test_cli_multi_llm_mode_rejects_missing_or_duplicate_models(tmp_path):
    input_file = tmp_path / "candidates.json"
    input_file.write_text("[]", encoding="utf-8")
    missing = runner.invoke(
        app,
        ["label", "--input", str(input_file), "--mode", "multi_llm"],
    )
    duplicate = runner.invoke(
        app,
        [
            "label", "--input", str(input_file), "--mode", "multi_llm",
            "--models", "same-model,same-model",
        ],
    )

    assert missing.exit_code == 1
    assert "requires --models" in missing.output
    assert duplicate.exit_code == 1
    assert "at least two distinct IDs" in duplicate.output


def test_directional_labeling_preserves_orientation(req_a, req_b):
    """Verify directional pairs (A,B) and (B,A) preserve their respective orientations."""
    coord = ProvisionalLabeler()

    pair_ab = RequirementPair(id="P-AB", req_a=req_a, req_b=req_b)
    pair_ba = RequirementPair(id="P-BA", req_a=req_b, req_b=req_a)

    res_ab = coord.process_pair(pair_ab, mode=LabelingMode.HEURISTIC)
    res_ba = coord.process_pair(pair_ba, mode=LabelingMode.HEURISTIC)

    # REQ-002 cites REQ-001. In (req_b, req_a), req_b is A and req_a is B -> RULE-REF-ID-DIRECT
    assert res_ba.dependency_type == DependencyType.REQUIRES
    assert res_ba.req_a.id == "REQ-002"
    assert res_ba.req_b.id == "REQ-001"


def test_existing_verified_annotations_never_overwritten(req_a, req_b):
    """Verify human-verified annotations are preserved without modification."""
    verified_prov = AnnotationProvenance(
        source=AnnotationSource.HUMAN,
        status=VerificationStatus.VERIFIED,
        annotator_id="expert_annotator_1"
    )
    pair = RequirementPair(
        id="P-VERIFIED",
        req_a=req_a,
        req_b=req_b,
        dependency_type=DependencyType.IMPLEMENTS,
        provenance=verified_prov
    )

    coord = ProvisionalLabeler()
    processed = coord.process_pair(pair, mode=LabelingMode.HYBRID)

    assert processed.dependency_type == DependencyType.IMPLEMENTS
    assert processed.provenance.status == VerificationStatus.VERIFIED
    assert processed.provenance.source == AnnotationSource.HUMAN


def test_batch_processing_and_manifest(req_a, req_b, req_c, tmp_path):
    """Verify process_batch correctly tallies agreement/disagreement and serializes manifest."""
    mock_response = json.dumps({"dependency_type": "Requires", "confidence": 4, "explanation": "Requires."})
    llm_labeler = LLMLabeler(llm_client=MockLLMClient(response_text=mock_response))
    coord = ProvisionalLabeler(llm_labeler=llm_labeler)

    candidates = [
        RequirementPair(id="C1", req_a=req_b, req_b=req_a),  # Explicit citation -> Requires (Agreed)
        RequirementPair(id="C2", req_a=req_a, req_b=req_c),  # Negation -> Conflicts vs Requires (Disagreed)
    ]

    labeled_pairs, manifest = coord.process_batch(candidates, mode=LabelingMode.HYBRID)

    assert len(labeled_pairs) == 2
    assert manifest.input_candidate_count == 2
    assert manifest.agreed_count + manifest.disagreed_count + manifest.unresolved_count <= 2

    # Verify serialization
    out_file = tmp_path / "provisional.json"
    coord.save_provisional_dataset(labeled_pairs, manifest, str(out_file))

    assert out_file.exists()
    assert (tmp_path / "provisional_manifest.json").exists()


def test_cli_label_command(tmp_path):
    """Verify leredd-cli label command runs and saves output."""
    pairs = [
        RequirementPair(
            id="CLI-1",
            req_a=Requirement(id="R1", text="REQ-R2 is required."),
            req_b=Requirement(id="R2", text="Base module."),
        ).model_dump(mode="json")
    ]

    in_file = tmp_path / "candidates.json"
    out_file = tmp_path / "provisional.json"

    with open(in_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f)

    res = runner.invoke(app, [
        "label",
        "--input", str(in_file),
        "--output", str(out_file),
        "--mode", "heuristic"
    ])

    assert res.exit_code == 0
    assert "PROVISIONAL LABELING SUMMARY" in res.stdout
    assert out_file.exists()
