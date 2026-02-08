"""
V2 dataset pipeline for agentic LLM pretraining.

Outputs structured parts format to enable loss masking support:
{"parts": [{"type": "system", "text": "..."}, {"type": "prompt", "text": "..."}, ...]}

Part types and loss mask semantics:
- text: mask=0 (train) - generic text samples
- system: mask=1 (excluded) - system prompts
- prompt: mask=1 (excluded) - user prompts
- response: mask=0 (train) - model responses
- thinking: mask=0 (train) - reasoning traces

Template types:
- TextTemplate: All tokens trained on (type="text")
- SplitTemplate: Prompt masked, response trained
- DialogueTemplate: Role-based masking via format_chat_parts()
"""

from examples.agentic_llm_pretraining.datasets.v2.parts_templates import (
    TemplateBase,
    TextTemplate,
    SplitTemplate,
    DialogueTemplate,
    TEMPLATES,
)
from examples.agentic_llm_pretraining.datasets.v2.transform_functions import Part
from examples.agentic_llm_pretraining.datasets.v2.preprocessing import DialogueData

__all__ = [
    "TemplateBase",
    "TextTemplate",
    "SplitTemplate",
    "DialogueTemplate",
    "TEMPLATES",
    "Part",
    "DialogueData",
]
