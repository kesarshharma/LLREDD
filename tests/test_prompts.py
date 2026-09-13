"""Tests for PromptBuilder."""

from leredd.prompts import PromptBuilder
from leredd.data_models import DependencyType


def test_prompt_builder_format(mock_config, sample_pair):
    builder = PromptBuilder(config=mock_config)
    prompt = builder.build(target_pair=sample_pair)

    assert "Requirement A:" in prompt
    assert "Requirement B:" in prompt
    assert "#Dependency Definitions:" in prompt
    assert "**Dependency_Type: [TYPE]**" in prompt
    assert "**Rationale: [EXPLANATION]**" in prompt
    assert "**Confidence Score: [SCORE]**" in prompt
