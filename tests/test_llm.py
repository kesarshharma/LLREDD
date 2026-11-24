"""Tests for LLM Client abstraction and Mock LLM."""

import json
import httpx
import pytest
from unittest.mock import patch
from leredd.llm import (
    DiskCachedLLMClient,
    MockLLMClient,
    LLMClientFactory,
    OllamaLLMClient,
)
from leredd.config import LEREDDConfig


def test_mock_llm_client():
    client = MockLLMClient()
    prompt = "Requirement A: The system shall include BCS.\nRequirement B: The BCS requires camera input.\n"
    text, usage = client.generate(prompt)

    assert "Dependency_Type:" in text
    assert "Rationale:" in text
    assert "Confidence Score:" in text
    assert usage["provider"] == "mock"


def test_llm_factory():
    cfg = LEREDDConfig(llm_provider="mock")
    client = LLMClientFactory.create(cfg)
    assert client is not None


def test_disk_cache_separates_model_ids_for_same_prompt(tmp_path):
    class NamedClient:
        provider = "test-provider"

        def __init__(self, model):
            self.model = model
            self.calls = 0

        def generate(self, prompt, temperature=0.2):
            self.calls += 1
            return self.model, {"provider": self.provider, "model": self.model}

    model_a = NamedClient("model-a")
    model_b = NamedClient("model-b")
    cached_a = DiskCachedLLMClient(model_a, cache_dir=str(tmp_path))
    cached_b = DiskCachedLLMClient(model_b, cache_dir=str(tmp_path))

    assert cached_a.generate("same prompt")[0] == "model-a"
    assert cached_b.generate("same prompt")[0] == "model-b"
    assert cached_a.generate("same prompt")[0] == "model-a"
    assert model_a.calls == 1
    assert model_b.calls == 1


def test_disk_cached_client_uncached_generation_bypasses_cache(tmp_path):
    class CountingClient:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, temperature=0.2):
            self.calls += 1
            return f"fresh-{self.calls}", {"cost_usd": 0.0}

    inner = CountingClient()
    client = DiskCachedLLMClient(inner, cache_dir=str(tmp_path))

    assert client.generate("same prompt")[0] == "fresh-1"
    assert client.generate_uncached("same prompt")[0] == "fresh-2"
    assert client.generate("same prompt")[0] == "fresh-1"
    assert inner.calls == 2


class MockResponse:
    def __init__(self, status_code=200, payload=None, bad_json=False):
        self.status_code = status_code
        self.payload = payload
        self.bad_json = bad_json
        self.request = object()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=self.request, response=self)

    def json(self):
        if self.bad_json:
            raise json.JSONDecodeError("bad json", "bad", 0)
        return self.payload if self.payload is not None else {}


class MockHttpxClient:
    def __init__(self, responses, *args, **kwargs):
        self.responses = responses
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return False

    def post(self, url, json=None):
        resp = self.responses[self.index]
        self.index += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


def test_ollama_retry_connection_then_success():
    responses = [
        httpx.ConnectError("connection refused"),
        MockResponse(200, payload={"response": "hello"}),
    ]
    inst = MockHttpxClient(responses)
    client = OllamaLLMClient(model="test-model")
    with patch("time.sleep"), patch("httpx.Client", lambda *a, **k: inst):
        text, usage = client.generate("prompt")
    assert inst.index == 2
    assert text == "hello"
    assert usage["provider"] == "ollama"
    assert usage["model"] == "test-model"


def test_ollama_retry_503_then_success():
    responses = [
        MockResponse(503),
        MockResponse(200, payload={"response": "world"}),
    ]
    inst = MockHttpxClient(responses)
    client = OllamaLLMClient(model="test-model")
    with patch("time.sleep"), patch("httpx.Client", lambda *a, **k: inst):
        text, usage = client.generate("prompt")
    assert inst.index == 2
    assert text == "world"
    assert usage["provider"] == "ollama"


def test_ollama_retry_404_no_retry():
    responses = [MockResponse(404)]
    inst = MockHttpxClient(responses)
    client = OllamaLLMClient(model="test-model")
    with patch("time.sleep"), patch("httpx.Client", lambda *a, **k: inst):
        try:
            client.generate("prompt")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "Ollama LLM call failed" in str(e)
    assert inst.index == 1


def test_ollama_retry_bad_json_no_retry():
    responses = [MockResponse(200, bad_json=True)]
    inst = MockHttpxClient(responses)
    client = OllamaLLMClient(model="test-model")
    with patch("time.sleep"), patch("httpx.Client", lambda *a, **k: inst):
        try:
            client.generate("prompt")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "Ollama LLM call failed" in str(e)
    assert inst.index == 1


def test_ollama_retry_repeated_transient_failures():
    responses = [
        MockResponse(503),
        MockResponse(503),
        MockResponse(503),
    ]
    inst = MockHttpxClient(responses)
    client = OllamaLLMClient(model="test-model")
    with patch("time.sleep"), patch("httpx.Client", lambda *a, **k: inst):
        try:
            client.generate("prompt")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "Ollama LLM call failed" in str(e)
    assert inst.index == 3


def test_ollama_retry_400_no_retry():
    responses = [MockResponse(400)]
    inst = MockHttpxClient(responses)
    client = OllamaLLMClient(model="test-model")
    with patch("time.sleep"), patch("httpx.Client", lambda *a, **k: inst):
        try:
            client.generate("prompt")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "Ollama LLM call failed" in str(e)
    assert inst.index == 1


def test_gemini_llm_client_mocked():
    from leredd.llm import GeminiLLMClient
    gemini_resp = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "**Dependency_Type: Requires**\n**Rationale: Necessary.**\n**Confidence Score: 5**"}]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 10
        }
    }
    client = GeminiLLMClient(api_key="test_key", model="gemini-2.5-flash")
    with patch("httpx.Client.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = gemini_resp
        mock_post.return_value.raise_for_status = lambda: None

        text, usage = client.generate("test prompt")
        assert "Dependency_Type: Requires" in text
        assert usage["provider"] == "gemini"
        assert usage["model"] == "gemini-2.5-flash"


def test_gemini_missing_api_key_raises():
    import pytest
    from leredd.llm import GeminiLLMClient
    client = GeminiLLMClient(api_key=None)
    client.api_key = None
    with pytest.raises(ValueError, match="Gemini API key not configured"):
        client.generate("test prompt")


def test_llm_factory_gemini():
    cfg = LEREDDConfig(llm_provider="gemini", gemini_api_key="test-key")
    client = LLMClientFactory.create(cfg)
    assert client is not None


def test_dependency_type_normalize_word_boundary():
    from leredd.data_models import DependencyType
    # Substring 'No' in 'Notification' or 'Notable' should NOT normalize to NO_DEPENDENCY
    with pytest.raises(ValueError, match="Unsupported dependency type"):
        DependencyType.normalize("Notification requirement")

    # Explicit 'No dependency', 'No', or 'Independent' SHOULD normalize to NO_DEPENDENCY
    assert DependencyType.normalize("No dependency") == DependencyType.NO_DEPENDENCY
    assert DependencyType.normalize("Independent") == DependencyType.NO_DEPENDENCY
    assert DependencyType.normalize("No") == DependencyType.NO_DEPENDENCY
