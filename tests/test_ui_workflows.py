"""Streamlit page journeys with temporary outputs and controlled model services."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT_ROOT / "ui" / "app.py"


def _run_page(page: str) -> AppTest:
    app = AppTest.from_file(str(APP_FILE))
    app.run(timeout=30)
    app.sidebar.radio[0].set_value(page)
    app.run(timeout=30)
    assert not app.exception, [str(error) for error in app.exception]
    return app


def test_all_dashboard_pages_render_without_uncaught_exceptions():
    pages = [
        "1. Home Overview",
        "2. Requirement Annotation",
        "3. LEREDD Detection",
        "4. Model Evaluation",
        "5. Graph Visualization",
        "6. Human Review Queue",
    ]
    app = AppTest.from_file(str(APP_FILE))
    app.run(timeout=30)
    for page in pages:
        app.sidebar.radio[0].set_value(page)
        app.run(timeout=30)
        assert not app.exception, f"{page}: {[str(error) for error in app.exception]}"


def test_annotation_page_extracts_and_persists_verified_pair(tmp_path):
    app = _run_page("2. Requirement Annotation")
    candidates_file = tmp_path / "candidates.json"
    annotations_file = tmp_path / "verified.json"

    app.text_input[0].set_value("E2E")
    app.text_area[0].set_value(
        "[REQ-1] The system shall provide braking.\n"
        "[REQ-2] The system shall report brake status."
    )
    app.text_input[1].set_value(str(candidates_file))
    app.button[0].click()
    app.run(timeout=30)

    assert not app.exception, [str(error) for error in app.exception]
    candidates = json.loads(candidates_file.read_text(encoding="utf-8"))
    assert len(candidates) == 1
    assert candidates[0]["id"] == "PAIR-E2E-0001"
    assert candidates[0]["dependency_type"] is None
    assert candidates[0]["provenance"]["status"] == "provisional"

    app.selectbox[0].set_value("Requires")
    app.text_input[2].set_value("ui-test-reviewer")
    app.text_input[3].set_value(str(annotations_file))
    app.button[1].click()
    app.run(timeout=30)

    assert not app.exception, [str(error) for error in app.exception]
    annotated = json.loads(annotations_file.read_text(encoding="utf-8"))
    assert annotated[0]["dependency_type"] == "Requires"
    assert annotated[0]["provenance"]["status"] == "verified"
    assert annotated[0]["provenance"]["annotator_id"] == "ui-test-reviewer"
    assert json.loads(candidates_file.read_text(encoding="utf-8"))[0]["dependency_type"] is None
    assert any("second independently collected" in item.value for item in app.info)
    assert [button.label for button in app.button] == [
        "Extract Requirements & Generate Pairs",
        "Save Annotation for Pair",
    ]


def test_annotation_page_accepts_uploaded_markdown(tmp_path):
    app = _run_page("2. Requirement Annotation")
    output = tmp_path / "uploaded_candidates.json"
    app.file_uploader[0].set_value((
        "requirements.md",
        b"[REQ-UP-1] The system shall encrypt records.\n"
        b"[REQ-UP-2] The system shall audit access.",
        "text/markdown",
    ))
    app.text_input[1].set_value(str(output))
    app.button[0].click()
    app.run(timeout=30)

    assert not app.exception, [str(error) for error in app.exception]
    assert output.exists()
    assert len(json.loads(output.read_text(encoding="utf-8"))) == 1


def test_detection_page_runs_controlled_detector(monkeypatch):
    from leredd.data_models import DependencyType, Prediction
    from leredd import detector as detector_module

    received_configs = []

    class ControlledDetector:
        def __init__(self, config=None):
            self.config = config
            received_configs.append(config)

        def detect_pair(self, target_pair, annotated_pool, srs_text):
            assert target_pair.id == "PAIR-DEMO-001"
            return Prediction(
                pair_id=target_pair.id,
                dependency_type=DependencyType.REQUIRES,
                rationale="controlled UI test response",
                confidence=4,
                raw_response="controlled response",
            )

    monkeypatch.setattr(detector_module, "LEREDDDetector", ControlledDetector)
    app = _run_page("3. LEREDD Detection")
    next(text for text in app.text_input if text.label == "Model Name").set_value("mock-ui-test")
    next(slider for slider in app.slider if slider.label == "Dynamic Examples (k)").set_value(3)
    next(box for box in app.checkbox if box.label == "Enable RAG Context").check()
    next(slider for slider in app.slider if slider.label == "Confidence Threshold").set_value(2)
    app.button[0].click()
    app.run(timeout=30)

    assert not app.exception, [str(error) for error in app.exception]
    assert app.session_state["detection_results"][0].dependency_type.value == "Requires"
    assert any("controlled UI test response" in item.value for item in app.markdown)
    assert received_configs[0].llm_provider == "mock"
    assert received_configs[0].llm_model == "mock-ui-test"
    assert received_configs[0].k_examples == 3
    assert received_configs[0].use_rag is True
    assert received_configs[0].confidence_threshold == 2


def test_detection_page_gates_external_provider_until_user_confirms(monkeypatch):
    from leredd.data_models import DependencyType, Prediction
    from leredd import detector as detector_module

    calls = []

    class ControlledDetector:
        def __init__(self, config=None):
            calls.append(config.llm_provider)

        def detect_pair(self, target_pair, annotated_pool, srs_text):
            return Prediction(
                pair_id=target_pair.id,
                dependency_type=DependencyType.REQUIRES,
                rationale="controlled external-request gate test",
                confidence=4,
            )

    monkeypatch.setattr(detector_module, "LEREDDDetector", ControlledDetector)
    app = _run_page("3. LEREDD Detection")
    next(box for box in app.selectbox if box.label == "LLM Provider").set_value("openai")
    app.run(timeout=30)
    app.button[0].click()
    app.run(timeout=30)

    assert calls == []
    assert any("Confirm the external OpenAI request" in warning.value for warning in app.warning)
    next(box for box in app.checkbox if "authorize sending requirement text" in box.label).check()
    app.button[0].click()
    app.run(timeout=30)

    assert not app.exception, [str(error) for error in app.exception]
    assert calls == ["openai"]
    assert app.session_state["detection_results"][-1].rationale == "controlled external-request gate test"


def test_evaluation_page_uses_system_disjoint_report_contract(tmp_path, monkeypatch):
    import scripts.run_full_evaluation as evaluation_module

    calls = []

    def controlled_evaluation(**kwargs):
        calls.append(kwargs)
        report_file = Path(kwargs["output_path"])
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text("{}", encoding="utf-8")
        result = {
            "model": "Controlled model",
            "overall_accuracy": 0.6,
            "macro_f1": 0.5,
            "weighted_f1": 0.55,
            "confusion_matrix": {},
        }
        return {
            "model_results": {
                "controlled": result,
                "LEREDD_Hybrid": {**result, "model": "LEREDD Hybrid"},
            },
            "evaluation_limitations": ["MockLLM; demonstrative sample only."],
        }

    monkeypatch.setattr(evaluation_module, "run_evaluation", controlled_evaluation)
    monkeypatch.chdir(tmp_path)
    app = _run_page("4. Model Evaluation")
    assert [button.label for button in app.button] == ["Run System-Disjoint Evaluation"]
    app.button[0].click()
    app.run(timeout=30)

    assert not app.exception, [str(error) for error in app.exception]
    assert calls[0]["split_dir"] == "data/splits/system_disjoint"
    assert calls[0]["strict_contamination"] is True
    assert Path(calls[0]["output_path"]).is_file()
    assert any("MockLLM" in warning.value for warning in app.warning)


def test_review_queue_page_records_and_exports_human_decision(tmp_path, monkeypatch):
    from leredd.data_models import (
        AnnotationProvenance,
        AnnotationSource,
        DependencyType,
        Requirement,
        RequirementPair,
        VerificationStatus,
    )
    from leredd.review_queue import ReviewQueueManager

    monkeypatch.chdir(tmp_path)
    queue_path = tmp_path / "review_queue.json"
    export_path = tmp_path / "verified.json"
    pair = RequirementPair(
        id="UI-QUEUE-1",
        req_a=Requirement(id="UI-Q1", text="First requirement.", system="ADB"),
        req_b=Requirement(id="UI-Q2", text="Second requirement.", system="ADB"),
        dependency_type=DependencyType.REQUIRES,
        provenance=AnnotationProvenance(
            source=AnnotationSource.LLM,
            status=VerificationStatus.PROVISIONAL,
        ),
    )
    manager = ReviewQueueManager()
    items = manager.build_queue([pair])
    manifest = manager.create_manifest(items, source_identity="ui-test")
    manager.save_queue(items, manifest, str(queue_path))

    app = _run_page("6. Human Review Queue")
    app.text_input[0].set_value(str(queue_path))
    app.run(timeout=30)
    next(button for button in app.button if button.label == "🔄 Reload Queue File").click()
    app.run(timeout=30)
    next(select for select in app.selectbox if select.label == "Filter Queue Status").set_value("ALL")
    next(select for select in app.selectbox if select.label == "Filter Review Reason Code").set_value("ALL")
    next(select for select in app.selectbox if select.label == "Filter Priority Rank").set_value("ALL")
    next(text for text in app.text_input if text.label == "Search Pair / Requirement Text").set_value("UI-QUEUE-1")
    next(select for select in app.selectbox if select.label == "Select Queue Item to Inspect & Review").set_value(0)
    app.run(timeout=30)
    assert any("Showing **1** of **1**" in element.value for element in app.markdown)
    next(radio for radio in app.radio if radio.label == "Decision Action").set_value(
        "CORRECT (Assign Different Label)"
    )
    next(
        select for select in app.selectbox
        if select.label == "Target Dependency Class (for ACCEPT or CORRECT)"
    ).set_value("Details")
    next(text for text in app.text_input if text.label == "Reviewer Identifier").set_value("reviewer-ui")
    next(check for check in app.checkbox if "I confirm this human decision" in check.label).check()
    next(button for button in app.button if button.label == "💾 Submit & Persist Human Decision").click()
    app.run(timeout=30)

    assert not app.exception, [str(error) for error in app.exception]
    saved_queue = json.loads(queue_path.read_text(encoding="utf-8"))
    assert saved_queue[0]["status"] == "corrected"
    assert saved_queue[0]["human_decision"]["final_label"] == "Details"

    next(text for text in app.text_input if text.label == "Verified Dataset Output File Path").set_value(str(export_path))
    next(button for button in app.button if button.label == "🚀 Export Verified Dataset").click()
    app.run(timeout=30)

    assert not app.exception, [str(error) for error in app.exception]
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert len(exported) == 1
    assert exported[0]["dependency_type"] == "Details"
    assert exported[0]["provenance"]["status"] == "verified"
