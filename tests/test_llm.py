"""Tests for LLM Client abstraction and Mock LLM."""

from leredd.llm import MockLLMClient, LLMClientFactory
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
