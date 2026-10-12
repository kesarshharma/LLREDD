"""Tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["health"] == "/api/v1/health"


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


def test_api_single_detect():
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


def test_api_batch_detect():
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
        "system_name": "ADB"
    }
    response = client.post("/api/v1/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["system"] == "ADB"


def test_api_invalid_payload_422():
    # Missing required field req_b
    payload = {
        "req_a": "The system shall control speed."
    }
    response = client.post("/api/v1/detect", json=payload)
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
