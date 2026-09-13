"""Prompt construction module implementing Paper Figure 2 and Table I definitions."""

from typing import List, Dict
from leredd.config import LEREDDConfig, get_config
from leredd.data_models import RequirementPair, DependencyType

# Formal definitions strictly matching Table I of the paper
DEPENDENCY_DEFINITIONS = {
    DependencyType.REQUIRES: (
        "Requires: if the fulfillment of one requirement is a prerequisite to the fulfillment of the other requirement."
    ),
    DependencyType.IMPLEMENTS: (
        "Implements: if one is a higher-level requirement (e.g., a system or subsystem level requirement) "
        "that is fulfilled by the other lower-level requirement (e.g., a subsystem or component level requirement)."
    ),
    DependencyType.CONFLICTS: (
        "Conflicts: if the fulfillment of one requirement restricts the fulfillment of the other requirement."
    ),
    DependencyType.DETAILS: (
        "Details: if both requirements describe the same action under the same condition, "
        "and one requirement provides additional details specifically regarding the shared action."
    ),
    DependencyType.IS_SIMILAR: (
        "Is similar: if one requirement replicates partially or totally the content of the other requirement, "
        "resulting in redundancy."
    ),
    DependencyType.NO_DEPENDENCY: (
        "No_dependency: if no direct or indirect dependency of any above type exists between Requirement A and Requirement B."
    )
}


class PromptBuilder:
    """Constructs structured prompts for LEREDD dependency inference."""

    def __init__(self, config: LEREDDConfig | None = None):
        self.config = config or get_config()

    def build_definitions_block(self) -> str:
        """Format formal definitions block."""
        lines = []
        for _, def_str in DEPENDENCY_DEFINITIONS.items():
            lines.append(f"- {def_str}")
        return "\n".join(lines)

    def build_examples_block(self, retrieved_examples: Dict[DependencyType, List[RequirementPair]] | None) -> str:
        """Format retrieved in-context examples block."""
        if not retrieved_examples:
            return "No in-context examples provided."

        lines = []
        for dep_type, pairs in retrieved_examples.items():
            if not pairs:
                continue
            lines.append(f"[{dep_type.value} Examples]")
            for idx, pair in enumerate(pairs, 1):
                lines.append(f"  Example {idx}:")
                lines.append(f"    Requirement A: \"{pair.req_a.text}\"")
                lines.append(f"    Requirement B: \"{pair.req_b.text}\"")
                lines.append(f"    Dependency_Type: {dep_type.value}")
        
        return "\n".join(lines) if lines else "No in-context examples provided."

    def build_context_block(self, context_chunks: List[str] | None) -> str:
        """Format RAG context chunks block."""
        if not context_chunks:
            return "No additional domain-specific context provided."

        lines = []
        for idx, chunk in enumerate(context_chunks, 1):
            lines.append(f"--- Context Chunk {idx} ---")
            lines.append(chunk)
        return "\n".join(lines)

    def build(
        self,
        target_pair: RequirementPair,
        retrieved_examples: Dict[DependencyType, List[RequirementPair]] | None = None,
        context_chunks: List[str] | None = None,
        domain: str = "automotive software engineering",
        system_name: str | None = None
    ) -> str:
        """Build full paper Figure 2 prompt string."""
        system_str = system_name or target_pair.req_a.system or "the system"

        req_a_text = target_pair.req_a.text
        req_b_text = target_pair.req_b.text
        definitions_text = self.build_definitions_block()
        examples_text = self.build_examples_block(retrieved_examples)
        context_text = self.build_context_block(context_chunks)

        prompt = f"""You are an expert requirements engineer from the {domain}. You will be provided with a pair of requirements extracted from the software requirements specification for {system_str}.
Given the following requirement dependency type definitions, examples, and context, your task is to analyze the requirement pair and determine whether a direct or indirect dependency exists between them.

#Requirements to analyze:
Requirement A: {req_a_text}
Requirement B: {req_b_text}

#Dependency Definitions:
{definitions_text}

#Examples:
{examples_text}

#Context:
{context_text}

#Instructions
- You should analyze the requirements pair according to all dependency types.
- If a dependency exists between a pair, you should annotate it with the type of dependency.
- If it does not fall into one of the above types of dependency, annotate it with "No_dependency".
- Explain the rationale behind your annotation.
- Provide a confidence score for the annotation. The score should range from 0 to 5, with 0 indicating no confidence and 5 indicating the highest confidence.
- **The output MUST be structured using these exact labels, each on a new line:**
**Dependency_Type: [TYPE]**
**Rationale: [EXPLANATION]**
**Confidence Score: [SCORE]**
"""
        return prompt
