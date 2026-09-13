"""Tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


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
