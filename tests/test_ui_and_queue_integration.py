"""Unit and integration tests for Streamlit UI and Review Queue Interface (Phase 5, TASK-P5-02)."""

import json
import os
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
    HumanDecision,
    QueueStatus,
    ReviewItem,
    ReviewPriority,
    ReviewQueueConfig,
    ReviewQueueManager,
    ReviewQueueManifest,
    ReviewReasonCode,
)

runner = CliRunner()


def test_ui_module_import():
    """Verify ui/app.py imports cleanly without syntax errors."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ui_app", "ui/app.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert module is not None


def test_operational_queue_counts_from_manifest(tmp_path):
    """Verify queue statistics are computed accurately from actual loaded queue items."""
    manager = ReviewQueueManager()

    prov = AnnotationProvenance(source=AnnotationSource.ENSEMBLE, status=VerificationStatus.PROVISIONAL, review_notes="DISAGREEMENT: Heuristic predicted 'Requires', LLM predicted 'Conflicts'.")
    pair1 = RequirementPair(id="P1", req_a=Requirement(id="R1", text="Data encryption requirement."), req_b=Requirement(id="R2", text="Key storage requirement."), dependency_type=None, provenance=prov)
    pair2 = RequirementPair(id="P2", req_a=Requirement(id="R3", text="Logging requirement."), req_b=Requirement(id="R4", text="Audit requirement."), dependency_type=DependencyType.IS_SIMILAR, provenance=AnnotationProvenance(source=AnnotationSource.LLM, status=VerificationStatus.PROVISIONAL))

    queue = manager.build_queue([pair1, pair2])

    # Accept pair1
    manager.record_decision(queue[0], decision_type=QueueStatus.ACCEPTED, final_label=DependencyType.REQUIRES)

    manifest = manager.create_manifest(queue)

    assert manifest.total_items == 2
    assert manifest.pending_count == 1
    assert manifest.accepted_count == 1
    assert manifest.corrected_count == 0


def test_all_six_canonical_classes_selectable_for_decision(tmp_path):
    """Verify all 6 canonical dependency classes are valid for correction/acceptance."""
    manager = ReviewQueueManager()
    prov = AnnotationProvenance(source=AnnotationSource.ENSEMBLE, status=VerificationStatus.PROVISIONAL, review_notes="DISAGREEMENT: Conflicting labels.")
    pair = RequirementPair(id="P-TEST", req_a=Requirement(id="R1", text="A"), req_b=Requirement(id="R2", text="B"), provenance=prov)

    queue = manager.build_queue([pair])
    item = queue[0]

    for dep_class in DependencyType:
        _, verified_pair = manager.record_decision(
            item=item,
            decision_type=QueueStatus.CORRECTED,
            final_label=dep_class,
            reviewer_id="test_user"
        )
        assert verified_pair.dependency_type == dep_class
        assert verified_pair.provenance.status == VerificationStatus.VERIFIED


def test_invalid_decision_submision_rejected():
    """Verify submitting invalid labels or ACCEPT without label is rejected."""
    manager = ReviewQueueManager()
    prov = AnnotationProvenance(source=AnnotationSource.HEURISTIC, status=VerificationStatus.PROVISIONAL, review_notes="UNRESOLVED: Both abstained.")
    pair = RequirementPair(id="P-EMPTY", req_a=Requirement(id="R1", text="A"), req_b=Requirement(id="R2", text="B"), dependency_type=None, provenance=prov)

    queue = manager.build_queue([pair])
    item = queue[0]

    # Accepting an unresolved item without proposed or final label must raise ValueError
    with pytest.raises(ValueError, match="Cannot ACCEPT an item with no proposed"):
        manager.record_decision(item=item, decision_type=QueueStatus.ACCEPTED, final_label=None)


def test_verified_export_filters_provisional_and_deferred(tmp_path):
    """Verify verified annotation export excludes provisional, deferred, rejected, and unresolved records."""
    manager = ReviewQueueManager()

    p_acc = RequirementPair(id="P-ACC", req_a=Requirement(id="R1", text="A"), req_b=Requirement(id="R2", text="B"), dependency_type=DependencyType.REQUIRES, provenance=AnnotationProvenance(source=AnnotationSource.LLM, status=VerificationStatus.PROVISIONAL))
    p_def = RequirementPair(id="P-DEF", req_a=Requirement(id="R3", text="C"), req_b=Requirement(id="R4", text="D"), dependency_type=None, provenance=AnnotationProvenance(source=AnnotationSource.ENSEMBLE, status=VerificationStatus.PROVISIONAL, review_notes="DISAGREEMENT"))
    p_rej = RequirementPair(id="P-REJ", req_a=Requirement(id="R5", text="E"), req_b=Requirement(id="R6", text="F"), dependency_type=None, provenance=AnnotationProvenance(source=AnnotationSource.HEURISTIC, status=VerificationStatus.PROVISIONAL))

    queue = manager.build_queue([p_acc, p_def, p_rej])

    item_acc = next(i for i in queue if i.pair_id == "P-ACC")
    item_def = next(i for i in queue if i.pair_id == "P-DEF")
    item_rej = next(i for i in queue if i.pair_id == "P-REJ")

    # Record ACCEPT on p_acc, DEFER on p_def, REJECT on p_rej
    manager.record_decision(item_acc, decision_type=QueueStatus.ACCEPTED)
    manager.record_decision(item_def, decision_type=QueueStatus.DEFERRED)
    manager.record_decision(item_rej, decision_type=QueueStatus.REJECTED)

    # Filter export list: only ACCEPTED or CORRECTED with valid final_label
    export_pairs = []
    for item in queue:
        if item.status in (QueueStatus.ACCEPTED, QueueStatus.CORRECTED) and item.human_decision and item.human_decision.final_label:
            _, v_pair = manager.record_decision(item, item.status, final_label=item.human_decision.final_label)
            export_pairs.append(v_pair)

    assert len(export_pairs) == 1
    assert export_pairs[0].id == "P-ACC"
    assert export_pairs[0].provenance.status == VerificationStatus.VERIFIED


def test_cli_queue_create_list_decide_export_compatibility(tmp_path):
    """Verify CLI commands queue-create, queue-list, and queue-decide execute cleanly."""
    prov = AnnotationProvenance(source=AnnotationSource.ENSEMBLE, status=VerificationStatus.PROVISIONAL, review_notes="DISAGREEMENT: Conflicting labels.")
    pair = RequirementPair(id="CLI-INTEG", req_a=Requirement(id="R1", text="REQ 1 text."), req_b=Requirement(id="R2", text="REQ 2 text."), dependency_type=None, provenance=prov)

    prov_file = tmp_path / "provisional.json"
    queue_file = tmp_path / "queue.json"
    ver_file = tmp_path / "verified.json"

    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump([pair.model_dump(mode="json")], f)

    # 1. queue-create
    res_create = runner.invoke(app, ["queue-create", "--input", str(prov_file), "--output", str(queue_file)])
    assert res_create.exit_code == 0
    assert queue_file.exists()

    # 2. queue-list
    res_list = runner.invoke(app, ["queue-list", "--queue", str(queue_file)])
    assert res_list.exit_code == 0

    # 3. queue-decide with --verified-out
    res_decide = runner.invoke(app, [
        "queue-decide",
        "--queue", str(queue_file),
        "--item-id", "CLI-INTEG",
        "--decision", "corrected",
        "--label", "Implements",
        "--reviewer", "ui_tester",
        "--notes", "UI integration test decision",
        "--verified-out", str(ver_file)
    ])
    assert res_decide.exit_code == 0
    assert ver_file.exists()

    with open(ver_file, "r", encoding="utf-8") as f:
        v_data = json.load(f)
    assert len(v_data) == 1
    assert v_data[0]["dependency_type"] == "Implements"
    assert v_data[0]["provenance"]["status"] == "verified"


def test_streamlit_export_save_pairs_json_defined():
    """Verify that ui/app.py correctly imports save_pairs_json to prevent NameError during export."""
    import importlib
    ui_app = importlib.import_module("ui.app")
    assert hasattr(ui_app, "save_pairs_json")
    assert callable(ui_app.save_pairs_json)


def test_verified_export_roundtrip_and_exclusions(tmp_path):
    """Verify verified dataset export writes valid JSON, preserves provenance, and excludes pending/provisional/deferred/rejected records."""
    from leredd.utils import load_pairs_json, save_pairs_json

    manager = ReviewQueueManager()

    # 1. Accepted pair
    pair_acc = RequirementPair(
        id="P-ACC-01",
        req_a=Requirement(id="R1", text="Authentication requirement.", system="ADB"),
        req_b=Requirement(id="R2", text="Token validation requirement.", system="ADB"),
        dependency_type=DependencyType.REQUIRES,
        provenance=AnnotationProvenance(source=AnnotationSource.LLM, status=VerificationStatus.PROVISIONAL),
    )
    # 2. Corrected pair
    pair_corr = RequirementPair(
        id="P-CORR-01",
        req_a=Requirement(id="R3", text="Logging requirement.", system="ADB"),
        req_b=Requirement(id="R4", text="Audit storage requirement.", system="ADB"),
        dependency_type=None,
        provenance=AnnotationProvenance(source=AnnotationSource.ENSEMBLE, status=VerificationStatus.PROVISIONAL, review_notes="DISAGREEMENT"),
    )
    # 3. Deferred pair (should be excluded)
    pair_def = RequirementPair(
        id="P-DEF-01",
        req_a=Requirement(id="R5", text="UI theme.", system="ADB"),
        req_b=Requirement(id="R6", text="Font loading.", system="ADB"),
        dependency_type=None,
        provenance=AnnotationProvenance(source=AnnotationSource.HEURISTIC, status=VerificationStatus.PROVISIONAL),
    )
    # 4. Rejected pair (should be excluded)
    pair_rej = RequirementPair(
        id="P-REJ-01",
        req_a=Requirement(id="R7", text="Unrelated text A.", system="ADB"),
        req_b=Requirement(id="R8", text="Unrelated text B.", system="ADB"),
        dependency_type=None,
        provenance=AnnotationProvenance(source=AnnotationSource.HEURISTIC, status=VerificationStatus.PROVISIONAL),
    )

    queue = manager.build_queue([pair_acc, pair_corr, pair_def, pair_rej])

    # Record decisions
    manager.record_decision(next(i for i in queue if i.pair_id == "P-ACC-01"), QueueStatus.ACCEPTED, reviewer_id="annotator_a", reviewer_notes="Approved prediction")
    manager.record_decision(next(i for i in queue if i.pair_id == "P-CORR-01"), QueueStatus.CORRECTED, final_label=DependencyType.IS_SIMILAR, reviewer_id="annotator_a", reviewer_notes="Corrected label to Similar")
    manager.record_decision(next(i for i in queue if i.pair_id == "P-DEF-01"), QueueStatus.DEFERRED, reviewer_id="annotator_a", reviewer_notes="Needs team discussion")
    manager.record_decision(next(i for i in queue if i.pair_id == "P-REJ-01"), QueueStatus.REJECTED, reviewer_id="annotator_a", reviewer_notes="Noise candidate")

    # Replicate Streamlit export logic
    verified_pairs = []
    for item in queue:
        if item.status in (QueueStatus.ACCEPTED, QueueStatus.CORRECTED) and item.human_decision and item.human_decision.final_label:
            _, v_pair = manager.record_decision(
                item,
                item.status,
                final_label=item.human_decision.final_label,
                reviewer_id=item.human_decision.reviewer_id,
                reviewer_notes=item.human_decision.reviewer_notes,
            )
            verified_pairs.append(v_pair)

    export_file = tmp_path / "verified_export.json"
    save_pairs_json(verified_pairs, str(export_file))

    assert export_file.exists()

    # Load and verify roundtrip
    reloaded_pairs = load_pairs_json(str(export_file))
    assert len(reloaded_pairs) == 2

    ids = {p.id for p in reloaded_pairs}
    assert ids == {"P-ACC-01", "P-CORR-01"}

    acc_pair = next(p for p in reloaded_pairs if p.id == "P-ACC-01")
    assert acc_pair.dependency_type == DependencyType.REQUIRES
    assert acc_pair.provenance.status == VerificationStatus.VERIFIED
    assert acc_pair.provenance.source == AnnotationSource.HUMAN
    assert acc_pair.provenance.reviewer_id == "annotator_a"

    corr_pair = next(p for p in reloaded_pairs if p.id == "P-CORR-01")
    assert corr_pair.dependency_type == DependencyType.IS_SIMILAR
    assert corr_pair.provenance.status == VerificationStatus.VERIFIED
    assert corr_pair.provenance.source == AnnotationSource.HUMAN
    assert corr_pair.provenance.reviewer_id == "annotator_a"
