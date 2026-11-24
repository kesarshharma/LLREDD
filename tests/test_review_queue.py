"""Unit tests for Multi-Method Disagreement and Uncertainty Review Queue (Phase 5, TASK-P5-01)."""

import json
import pytest
from typer.testing import CliRunner

from leredd.cli import app
from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.review_queue import (
    QueueStatus,
    ReviewItem,
    ReviewPriority,
    ReviewQueueConfig,
    ReviewQueueManager,
    ReviewReasonCode,
)

runner = CliRunner()


@pytest.fixture
def req_a():
    return Requirement(id="REQ-101", text="The system shall encrypt sensitive data using AES-256.", system_name="SecuritySystem")


@pytest.fixture
def req_b():
    return Requirement(id="REQ-102", text="The system requires REQ-101 for identity validation.", system_name="SecuritySystem")


@pytest.fixture
def req_c():
    return Requirement(id="REQ-103", text="The system shall disable all data encryption during debugging.", system_name="SecuritySystem")


@pytest.fixture
def disagreed_pair(req_a, req_b):
    prov = AnnotationProvenance(
        source=AnnotationSource.ENSEMBLE,
        status=VerificationStatus.PROVISIONAL,
        review_notes="DISAGREEMENT: Heuristic predicted 'Requires', LLM predicted 'Conflicts'.",
        evidence="Heuristic [RULE-REF-ID-DIRECT] | LLM Rationale: Conflicts"
    )
    return RequirementPair(id="P-DIS", req_a=req_b, req_b=req_a, dependency_type=None, provenance=prov)


@pytest.fixture
def abstained_pair(req_a, req_c):
    prov = AnnotationProvenance(
        source=AnnotationSource.HEURISTIC,
        status=VerificationStatus.PROVISIONAL,
        review_notes="UNRESOLVED: Both heuristic and LLM abstained or failed."
    )
    return RequirementPair(id="P-ABS", req_a=req_a, req_b=req_c, dependency_type=None, provenance=prov)


@pytest.fixture
def llm_only_pair(req_a, req_b):
    prov = AnnotationProvenance(
        source=AnnotationSource.LLM,
        status=VerificationStatus.PROVISIONAL,
        confidence_score=2,
        evidence="LLM Rationale: Direct dependency found."
    )
    return RequirementPair(id="P-LLM", req_a=req_a, req_b=req_b, dependency_type=DependencyType.REQUIRES, provenance=prov)


def test_queue_construction_from_disagreement(disagreed_pair):
    """Verify queue constructs ReviewItem from disagreement with CRITICAL priority rank 1."""
    manager = ReviewQueueManager()
    item = manager.create_review_item(disagreed_pair)

    assert item is not None
    assert ReviewReasonCode.DISAGREEMENT in item.review_reasons
    assert item.priority == ReviewPriority.CRITICAL.value
    assert item.status == QueueStatus.PENDING
    assert item.orientation == ("REQ-102", "REQ-101")


def test_queue_construction_from_abstentions_and_failures(abstained_pair):
    """Verify queue constructs ReviewItem from abstentions with MEDIUM priority rank 3."""
    manager = ReviewQueueManager()
    item = manager.create_review_item(abstained_pair)

    assert item is not None
    assert ReviewReasonCode.BOTH_ABSTAINED in item.review_reasons
    assert item.priority == ReviewPriority.MEDIUM.value


def test_single_method_eligibility_policy(llm_only_pair):
    """Verify single-method predictions respect configured eligibility policy."""
    # Included by default when include_single_method = True
    manager_inc = ReviewQueueManager(config=ReviewQueueConfig(include_single_method=True))
    item_inc = manager_inc.create_review_item(llm_only_pair)
    assert item_inc is not None
    assert ReviewReasonCode.LLM_ONLY in item_inc.review_reasons

    # Excluded when include_single_method = False (if no low confidence)
    manager_exc = ReviewQueueManager(config=ReviewQueueConfig(include_single_method=False, include_low_confidence=False))
    item_exc = manager_exc.create_review_item(llm_only_pair)
    assert item_exc is None


def test_short_requirement_text_is_routed_with_configurable_threshold():
    pair = RequirementPair(
        id="P-SHORT",
        req_a=Requirement(id="REQ-SHORT", text="Start."),
        req_b=Requirement(id="REQ-LONG", text="The system shall record and report the requested status."),
        dependency_type=DependencyType.NO_DEPENDENCY,
        provenance=AnnotationProvenance(
            source=AnnotationSource.ENSEMBLE,
            status=VerificationStatus.PROVISIONAL,
            confidence_score=5,
        ),
    )
    config = ReviewQueueConfig(
        include_single_method=False,
        include_low_confidence=False,
        minimum_requirement_characters=20,
    )

    item = ReviewQueueManager(config=config).create_review_item(pair)

    assert item is not None
    assert item.review_reasons == [ReviewReasonCode.SHORT_REQUIREMENT]
    assert item.priority == ReviewPriority.MEDIUM.value
    assert "Requirement A (6 characters)" in item.review_notes
    assert "inspect for incomplete or ambiguous text" in item.review_notes

    at_threshold = pair.model_copy(update={
        "req_a": pair.req_a.model_copy(update={"text": "12345678901234567890"}),
    })
    assert ReviewQueueManager(config=config).create_review_item(at_threshold) is None

    disabled = ReviewQueueConfig(
        include_single_method=False,
        include_low_confidence=False,
        minimum_requirement_characters=0,
    )
    assert ReviewQueueManager(config=disabled).create_review_item(pair) is None


def test_priority_ordering_and_deterministic_tie_breaking(disagreed_pair, abstained_pair, llm_only_pair):
    """Verify deterministic priority ordering: CRITICAL (1) > MEDIUM (3) > LOW (4)."""
    manager = ReviewQueueManager()
    pairs = [llm_only_pair, abstained_pair, disagreed_pair]
    queue = manager.build_queue(pairs)

    assert len(queue) == 3
    assert queue[0].pair_id == "P-DIS"  # Priority 1
    assert queue[1].pair_id == "P-ABS"  # Priority 3
    assert queue[2].pair_id == "P-LLM"  # Priority 4


def test_filtering_queue(disagreed_pair, abstained_pair):
    """Verify queue items can be filtered by status, reason, or priority."""
    manager = ReviewQueueManager()
    queue = manager.build_queue([disagreed_pair, abstained_pair])

    disagreements = manager.filter_queue(queue, reason=ReviewReasonCode.DISAGREEMENT)
    assert len(disagreements) == 1
    assert disagreements[0].pair_id == "P-DIS"

    pending_items = manager.filter_queue(queue, status=QueueStatus.PENDING)
    assert len(pending_items) == 2


def test_recording_human_decision_accept(disagreed_pair):
    """Verify ACCEPT decision promotes annotation to VERIFIED with AnnotationSource.HUMAN."""
    manager = ReviewQueueManager()
    item = manager.create_review_item(disagreed_pair)

    updated_item, verified_pair = manager.record_decision(
        item=item,
        decision_type=QueueStatus.ACCEPTED,
        final_label=DependencyType.REQUIRES,
        reviewer_id="expert_1",
        reviewer_notes="Confirmed explicit citation."
    )

    assert updated_item.status == QueueStatus.ACCEPTED
    assert updated_item.human_decision.final_label == DependencyType.REQUIRES
    assert verified_pair.provenance.status == VerificationStatus.VERIFIED
    assert verified_pair.provenance.source == AnnotationSource.HUMAN
    assert verified_pair.dependency_type == DependencyType.REQUIRES


def test_recording_human_decision_correct(disagreed_pair):
    """Verify CORRECTED decision updates final label and creates verified pair."""
    manager = ReviewQueueManager()
    item = manager.create_review_item(disagreed_pair)

    updated_item, verified_pair = manager.record_decision(
        item=item,
        decision_type=QueueStatus.CORRECTED,
        final_label=DependencyType.IMPLEMENTS,
        reviewer_id="expert_2",
        reviewer_notes="Corrected to Implements based on architecture doc."
    )

    assert updated_item.status == QueueStatus.CORRECTED
    assert verified_pair.dependency_type == DependencyType.IMPLEMENTS
    assert verified_pair.provenance.status == VerificationStatus.VERIFIED


def test_ground_truth_protection_and_raw_prediction_preservation(disagreed_pair):
    """Verify original predictions remain untouched in item audit trail after human decision."""
    manager = ReviewQueueManager()
    item = manager.create_review_item(disagreed_pair)
    original_h_pred = item.heuristic_prediction
    original_l_pred = item.llm_prediction

    updated_item, _ = manager.record_decision(
        item=item,
        decision_type=QueueStatus.ACCEPTED,
        final_label=DependencyType.REQUIRES,
        reviewer_id="fixture-reviewer",
    )

    # Raw predictions must remain untouched!
    assert updated_item.heuristic_prediction == original_h_pred
    assert updated_item.llm_prediction == original_l_pred


def test_idempotent_queue_refresh_preserves_decisions(disagreed_pair, req_a, req_b):
    """Verify refreshing queue preserves completed human decisions and does not reset status to PENDING."""
    manager = ReviewQueueManager()
    queue = manager.build_queue([disagreed_pair])

    # Record decision on item
    manager.record_decision(queue[0], decision_type=QueueStatus.ACCEPTED, final_label=DependencyType.REQUIRES, reviewer_id="fixture-reviewer")
    assert queue[0].status == QueueStatus.ACCEPTED

    # Refresh queue with same input pairs
    refreshed_queue = manager.refresh_queue(queue, [disagreed_pair])

    assert len(refreshed_queue) == 1
    assert refreshed_queue[0].status == QueueStatus.ACCEPTED  # Preserved!
    assert refreshed_queue[0].human_decision is not None


def test_record_decision_rejects_missing_and_placeholder_reviewer(disagreed_pair):
    manager = ReviewQueueManager()
    item = manager.create_review_item(disagreed_pair)
    with pytest.raises(ValueError, match="nonblank reviewer_id"):
        manager.record_decision(item, QueueStatus.ACCEPTED, final_label=DependencyType.REQUIRES)
    with pytest.raises(ValueError, match="nonblank reviewer_id"):
        manager.record_decision(item, QueueStatus.ACCEPTED, final_label=DependencyType.REQUIRES, reviewer_id="   ")
    with pytest.raises(ValueError, match="generic placeholder"):
        manager.record_decision(item, QueueStatus.ACCEPTED, final_label=DependencyType.REQUIRES, reviewer_id="human_reviewer_1")
    assert item.status == QueueStatus.PENDING


def test_directionality_preservation_and_orientation_display(req_a, req_b):
    """Verify directional orientation (A, B) vs (B, A) yields distinct stable queue IDs."""
    manager = ReviewQueueManager()

    prov = AnnotationProvenance(source=AnnotationSource.LLM, status=VerificationStatus.PROVISIONAL)
    pair_ab = RequirementPair(id="P-001", req_a=req_a, req_b=req_b, dependency_type=DependencyType.REQUIRES, provenance=prov)
    pair_ba = RequirementPair(id="P-001", req_a=req_b, req_b=req_a, dependency_type=DependencyType.REQUIRES, provenance=prov)

    item_ab = manager.create_review_item(pair_ab)
    item_ba = manager.create_review_item(pair_ba)

    assert item_ab.orientation == ("REQ-101", "REQ-102")
    assert item_ba.orientation == ("REQ-102", "REQ-101")
    assert item_ab.queue_id != item_ba.queue_id  # Unique stable IDs for different orientations!


def test_queue_persistence_json_roundtrip(disagreed_pair, tmp_path):
    """Verify queue items and manifest serialize and deserialized without data loss."""
    manager = ReviewQueueManager()
    queue = manager.build_queue([disagreed_pair])
    manifest = manager.create_manifest(queue)

    out_queue = tmp_path / "queue.json"
    out_manifest = tmp_path / "manifest.json"

    manager.save_queue(queue, manifest, str(out_queue), str(out_manifest))

    loaded_queue = manager.load_queue(str(out_queue))
    assert len(loaded_queue) == 1
    assert loaded_queue[0].queue_id == queue[0].queue_id
    assert loaded_queue[0].pair_id == queue[0].pair_id
    assert loaded_queue[0].review_reasons == queue[0].review_reasons


def test_requirement_metadata_survives_queue_persistence_and_verified_decision(tmp_path):
    manager = ReviewQueueManager()
    pair = RequirementPair(
        id="P-METADATA",
        req_a=Requirement(
            id="REQ-A",
            text="Requirement A.",
            system="System A",
            component="Authentication",
            subsystem="Identity",
            source="auth-srs.md",
            source_location="line 12",
        ),
        req_b=Requirement(
            id="REQ-B",
            text="Requirement B.",
            system="System B",
            component="Token Service",
            subsystem="Security",
            source="security-srs.md",
            source_location="line 27",
        ),
        dependency_type=DependencyType.REQUIRES,
        provenance=AnnotationProvenance(
            source=AnnotationSource.LLM,
            status=VerificationStatus.PROVISIONAL,
        ),
    )
    queue_path = tmp_path / "queue.json"
    items = manager.build_queue([pair])
    manager.save_queue(items, manager.create_manifest(items), str(queue_path))

    loaded_item = manager.load_queue(str(queue_path))[0]
    _, verified_pair = manager.record_decision(
        loaded_item,
        QueueStatus.ACCEPTED,
        final_label=DependencyType.REQUIRES,
        reviewer_id="metadata-reviewer",
    )

    assert verified_pair.req_a.model_dump() == pair.req_a.model_dump()
    assert verified_pair.req_b.model_dump() == pair.req_b.model_dump()
    assert verified_pair.provenance.status == VerificationStatus.VERIFIED


def test_legacy_queue_item_without_requirement_metadata_remains_decidable(llm_only_pair):
    item = ReviewQueueManager().create_review_item(llm_only_pair)
    legacy_data = item.model_dump(mode="json")
    for field in (
        "req_a_system", "req_b_system", "req_a_component", "req_b_component",
        "req_a_subsystem", "req_b_subsystem", "req_a_source", "req_b_source",
        "req_a_source_location", "req_b_source_location",
    ):
        legacy_data.pop(field)
    legacy_item = ReviewItem.model_validate(legacy_data)

    _, verified_pair = ReviewQueueManager.record_decision(
        legacy_item,
        QueueStatus.ACCEPTED,
        final_label=DependencyType.REQUIRES,
        reviewer_id="legacy-reviewer",
    )

    assert verified_pair.req_a.system == legacy_item.system_name
    assert verified_pair.req_a.component is None


def test_queue_save_replace_failure_preserves_existing_outputs(disagreed_pair, tmp_path, monkeypatch):
    """A failed atomic replacement must not truncate the prior queue or manifest."""
    import leredd.review_queue as review_queue_module

    manager = ReviewQueueManager()
    queue = manager.build_queue([disagreed_pair])
    manifest = manager.create_manifest(queue)
    out_queue = tmp_path / "queue.json"
    out_manifest = tmp_path / "manifest.json"
    original_queue = '[{"existing": "queue"}]'
    original_manifest = '{"existing": "manifest"}'
    out_queue.write_text(original_queue, encoding="utf-8")
    out_manifest.write_text(original_manifest, encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(review_queue_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        manager.save_queue(queue, manifest, str(out_queue), str(out_manifest))

    assert out_queue.read_text(encoding="utf-8") == original_queue
    assert out_manifest.read_text(encoding="utf-8") == original_manifest
    assert list(tmp_path.glob(".*.tmp")) == []


def test_cli_queue_commands(tmp_path):
    """Verify CLI commands queue-create, queue-list, and queue-decide run cleanly."""
    from leredd.review_queue import ReviewQueueManager, ReviewReasonCode

    prov = AnnotationProvenance(
        source=AnnotationSource.ENSEMBLE,
        status=VerificationStatus.PROVISIONAL,
        review_notes="DISAGREEMENT: Heuristic predicted 'Requires', LLM predicted 'Conflicts'."
    )
    pair = RequirementPair(
        id="CLI-P1",
        req_a=Requirement(id="R1", text="REQ-R2 is needed."),
        req_b=Requirement(id="R2", text="Base component."),
        dependency_type=None,
        provenance=prov
    )

    in_file = tmp_path / "provisional.json"
    queue_file = tmp_path / "review_queue.json"
    default_threshold_queue = tmp_path / "default_threshold_queue.json"

    with open(in_file, "w", encoding="utf-8") as f:
        json.dump([pair.model_dump(mode="json")], f)

    # 1. Test queue-create
    res_create = runner.invoke(app, [
        "queue-create",
        "--input", str(in_file),
        "--output", str(queue_file)
    ])
    assert res_create.exit_code == 0
    assert "REVIEW QUEUE SUMMARY" in res_create.stdout
    assert queue_file.exists()

    default_threshold = runner.invoke(app, [
        "queue-create",
        "--input", str(in_file),
        "--output", str(default_threshold_queue),
    ])
    assert default_threshold.exit_code == 0, default_threshold.stdout
    default_items = ReviewQueueManager.load_queue(str(default_threshold_queue))
    assert ReviewReasonCode.SHORT_REQUIREMENT in default_items[0].review_reasons
    default_manifest = json.loads(
        (tmp_path / "default_threshold_queue_manifest.json").read_text(encoding="utf-8")
    )
    assert default_manifest["eligibility_config"]["minimum_requirement_characters"] == 20

    disabled_threshold = runner.invoke(app, [
        "queue-create",
        "--input", str(in_file),
        "--output", str(queue_file),
        "--minimum-requirement-characters", "0",
    ])
    assert disabled_threshold.exit_code == 0, disabled_threshold.stdout
    disabled_items = ReviewQueueManager.load_queue(str(queue_file))
    assert ReviewReasonCode.SHORT_REQUIREMENT not in disabled_items[0].review_reasons
    disabled_manifest = json.loads(
        (tmp_path / "review_queue_manifest.json").read_text(encoding="utf-8")
    )
    assert disabled_manifest["eligibility_config"]["minimum_requirement_characters"] == 0

    # 2. Test queue-list
    res_list = runner.invoke(app, [
        "queue-list",
        "--queue", str(queue_file)
    ])
    assert res_list.exit_code == 0
    assert "Review Queue Items" in res_list.stdout

    # 3. Test queue-decide
    res_decide = runner.invoke(app, [
        "queue-decide",
        "--queue", str(queue_file),
        "--item-id", "CLI-P1",
        "--decision", "accepted",
        "--label", "Requires",
        "--reviewer", "tester_1",
        "--notes", "CLI decision test"
    ])
    assert res_decide.exit_code == 0
    assert "Successfully recorded decision" in res_decide.stdout
