"""Tests for FastAPI REST API endpoints."""

import asyncio

import httpx

from api.main import app
import pytest



class _InProcessAPIClient:
    """Synchronous test helper backed by HTTPX's in-process ASGI transport.

    Avoid Starlette TestClient's worker-thread portal, which can hang on
    Windows when its asyncio Proactor loop creates a socket-pair fallback.
    The app has no lifespan hooks, so ASGITransport covers the full route path.
    """

    def request(self, method, url, **kwargs):
        async def send_request():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as async_client:
                return await async_client.request(method, url, **kwargs)

        return asyncio.run(send_request())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


client = _InProcessAPIClient()


@pytest.fixture
def controlled_api_detector(monkeypatch):
    """Exercise real API/detector routing with an offline deterministic client."""
    import numpy as np
    from api import routes
    from leredd.config import LEREDDConfig
    from leredd.detector import LEREDDDetector
    from leredd.llm import MockLLMClient

    class OfflineEmbedder:
        def encode(self, texts):
            return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)

    detector = LEREDDDetector(
        config=LEREDDConfig(llm_provider="mock", k_examples=0),
        embedder=OfflineEmbedder(),
        llm=MockLLMClient(response_text=(
            "Dependency_Type: Requires\nRationale: controlled API test.\nConfidence Score: 5"
        )),
    )
    monkeypatch.setattr(routes, "get_detector", lambda: detector)
    return detector


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to LEREDD API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_api_config():
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "embedding_model" in data
    assert "openai_api_key" not in data


def test_api_config_never_returns_openai_secret(monkeypatch):
    from leredd.config import LEREDDConfig
    from api import routes

    monkeypatch.setattr(
        routes, "get_config", lambda: LEREDDConfig(openai_api_key="test-secret")
    )

    response = client.get("/api/v1/config")

    assert response.status_code == 200
    assert "test-secret" not in response.text
    assert response.json()["openai_api_key_configured"] is True


def test_api_single_detect(controlled_api_detector):
    payload = {
        "req_a": "The system shall include the Brake Control Subsystem (BCS).",
        "req_b": "The BCS shall accept camera input.",
        "system": "ADB"
    }
    response = client.post("/api/v1/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "dependency_type" in data
    assert "rationale" in data
    assert "confidence" in data
    assert data["validation_status"] == "passed_threshold"
    assert data["stability_status"] == "not_checked"


def test_api_single_detect_accepts_per_request_retrieval_mode(
    controlled_api_detector, monkeypatch
):
    calls = []
    original_detect_pair = controlled_api_detector.detect_pair

    def record_mode(*args, **kwargs):
        calls.append(kwargs.get("retrieval_mode"))
        return original_detect_pair(*args, **kwargs)

    monkeypatch.setattr(controlled_api_detector, "detect_pair", record_mode)
    response = client.post(
        "/api/v1/detect",
        json={
            "req_a": "The system shall control braking.",
            "req_b": "The controller shall read wheel speed.",
            "retrieval_mode": "hybrid",
        },
    )

    assert response.status_code == 200
    assert calls == ["hybrid"]


def test_api_single_detect_can_enable_rag_per_request(controlled_api_detector, monkeypatch):
    calls = []
    prompts = []

    class RecordingRAG:
        def retrieve(self, **kwargs):
            calls.append(kwargs["srs_text"])
            return ["SRS_CONTEXT_FOR_API_TEST"]

    controlled_api_detector._rag_retriever = RecordingRAG()
    original_build = controlled_api_detector.prompt_builder.build

    def record_prompt(*args, **kwargs):
        prompt = original_build(*args, **kwargs)
        prompts.append(prompt)
        return prompt

    monkeypatch.setattr(controlled_api_detector.prompt_builder, "build", record_prompt)
    base_payload = {
        "req_a": "The system shall control braking.",
        "req_b": "The controller shall read wheel speed.",
        "srs_text": "Controlled API SRS context.",
    }

    default_request = client.post("/api/v1/detect", json=base_payload)
    assert default_request.status_code == 200
    assert calls == []  # Config is disabled and the request omitted use_rag.

    enabled_request = client.post(
        "/api/v1/detect", json={**base_payload, "use_rag": True}
    )
    assert enabled_request.status_code == 200
    assert calls == ["Controlled API SRS context."]
    assert "SRS_CONTEXT_FOR_API_TEST" in prompts[-1]

    controlled_api_detector.config.use_rag = True
    disabled_request = client.post(
        "/api/v1/detect", json={**base_payload, "use_rag": False}
    )
    assert disabled_request.status_code == 200
    assert calls == ["Controlled API SRS context."]

    configured_default_request = client.post("/api/v1/detect", json=base_payload)
    assert configured_default_request.status_code == 200
    assert calls == ["Controlled API SRS context.", "Controlled API SRS context."]


def test_api_batch_detect_can_enable_rag_per_request(controlled_api_detector):
    calls = []

    class RecordingRAG:
        def retrieve(self, **kwargs):
            calls.append(kwargs["srs_text"])
            return ["SRS_CONTEXT_FOR_BATCH_TEST"]

    controlled_api_detector._rag_retriever = RecordingRAG()
    response = client.post(
        "/api/v1/batch_detect",
        json={
            "pairs": [{
                "id": "PAIR-API-RAG",
                "req_a": {"id": "REQ-A", "text": "The system shall brake.", "system": "ADB"},
                "req_b": {"id": "REQ-B", "text": "The subsystem shall read speed.", "system": "ADB"},
            }],
            "srs_text": "Batch SRS context.",
            "use_rag": True,
        },
    )

    assert response.status_code == 200
    assert calls == ["Batch SRS context."]


def test_api_repeated_inference_reports_stability_and_validates_run_count(controlled_api_detector):
    response = client.post(
        "/api/v1/detect",
        json={
            "req_a": "The system shall control braking.",
            "req_b": "The controller shall read wheel speed.",
            "stability_runs": 2,
        },
    )
    assert response.status_code == 200
    assert response.json()["stability_status"] == "consistent"
    assert response.json()["stability_run_count"] == 2
    assert response.json()["stability_labels"] == ["Requires", "Requires"]

    invalid = client.post(
        "/api/v1/detect",
        json={
            "req_a": "The system shall control braking.",
            "req_b": "The controller shall read wheel speed.",
            "stability_runs": 6,
        },
    )
    assert invalid.status_code == 422


def test_api_batch_detect_accepts_per_request_retrieval_mode(
    controlled_api_detector, monkeypatch
):
    calls = []
    original_detect_batch = controlled_api_detector.detect_batch

    def record_mode(*args, **kwargs):
        calls.append((kwargs.get("retrieval_mode"), kwargs.get("stability_runs")))
        return original_detect_batch(*args, **kwargs)

    monkeypatch.setattr(controlled_api_detector, "detect_batch", record_mode)
    response = client.post(
        "/api/v1/batch_detect",
        json={
            "pairs": [
                {
                    "id": "PAIR-API-MODE",
                    "req_a": {"id": "REQ-A", "text": "The system shall brake.", "system": "ADB"},
                    "req_b": {"id": "REQ-B", "text": "The subsystem shall read speed.", "system": "ADB"},
                }
            ],
            "retrieval_mode": "bm25",
            "stability_runs": 2,
        },
    )

    assert response.status_code == 200
    assert calls == [("bm25", 2)]
    assert response.json()[0]["stability_status"] == "consistent"


def test_api_batch_detect(controlled_api_detector):
    payload = {
        "pairs": [
            {
                "id": "PAIR-API-1",
                "req_a": {"id": "REQ-1", "text": "The system shall control speed.", "system": "ADB"},
                "req_b": {"id": "REQ-2", "text": "The BCS subsystem shall brake.", "system": "ADB"},
                "dependency_type": None
            }
        ]
    }
    response = client.post("/api/v1/batch_detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["pair_id"] == "PAIR-API-1"


def test_api_extract():
    payload = {
        "text": "[REQ-ADB-001] The system shall accelerate safely.\n[REQ-ADB-002] The system shall decelerate.",
        "system_name": "ADB",
        "source_name": "vehicle_srs.md",
    }
    response = client.post("/api/v1/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["system"] == "ADB"
    assert data[0]["source"] == "vehicle_srs.md"
    assert data[0]["source_location"] == "line 1"


def test_api_extract_attaches_explicit_component_metadata():
    response = client.post(
        "/api/v1/extract",
        json={
            "text": "[REQ-API-A] The brake controller shall calculate distance.",
            "system_name": "API-DEMO",
            "metadata_by_id": {
                "REQ-API-A": {"component": "Brake Controller", "subsystem": "Braking"}
            },
        },
    )

    assert response.status_code == 200
    requirement = response.json()[0]
    assert requirement["component"] == "Brake Controller"
    assert requirement["subsystem"] == "Braking"


def test_api_extract_rejects_metadata_for_unknown_requirement():
    response = client.post(
        "/api/v1/extract",
        json={
            "text": "[REQ-API-A] The brake controller shall calculate distance.",
            "metadata_by_id": {"REQ-NOT-IN-SRS": {"component": "Brakes"}},
        },
    )

    assert response.status_code == 422
    assert "unknown IDs" in response.json()["detail"]


def test_api_single_detect_passes_component_metadata_to_target(controlled_api_detector, monkeypatch):
    received_pairs = []
    original_detect = controlled_api_detector.detect_pair

    def capture_pair(target_pair, **kwargs):
        received_pairs.append(target_pair)
        return original_detect(target_pair, **kwargs)

    monkeypatch.setattr(controlled_api_detector, "detect_pair", capture_pair)
    response = client.post(
        "/api/v1/detect",
        json={
            "req_a": "The brake controller shall calculate distance.",
            "req_b": "The display shall report distance.",
            "req_a_component": "Brake Controller",
            "req_a_subsystem": "Braking",
            "req_b_component": "Display",
        },
    )

    assert response.status_code == 200
    assert received_pairs[0].req_a.component == "Brake Controller"
    assert received_pairs[0].req_a.subsystem == "Braking"
    assert received_pairs[0].req_b.component == "Display"


def test_api_invalid_payload_422():
    # Missing required field req_b
    payload = {
        "req_a": "The system shall control speed."
    }
    response = client.post("/api/v1/detect", json=payload)
    assert response.status_code == 422


def test_api_rejects_unsupported_retrieval_mode():
    response = client.post(
        "/api/v1/detect",
        json={
            "req_a": "The system shall control speed.",
            "req_b": "The brake controller shall apply braking.",
            "retrieval_mode": "unsupported",
        },
    )

    assert response.status_code == 422


def test_api_main_starts_uvicorn_with_requested_host_and_port(monkeypatch):
    import sys
    import uvicorn
    from api import main as api_main

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, "argv", ["api/main.py", "--host", "127.0.0.1", "--port", "8123"])

    assert api_main.main() == 0
    assert calls == [(("api.main:app",), {"host": "127.0.0.1", "port": 8123, "reload": False})]
