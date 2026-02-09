"""
V2 transform functions for converting raw dataset samples to structured parts format.

Provides transform functions that output parts for loss masking during tokenization.

Output format: list of Part objects with type and text fields.

Three dataset types are supported:
1. Text datasets: All tokens trained on (type="text")
2. Split template datasets: Prompt masked, response trained
3. Dialogue datasets: Role-based masking (system/user/tool masked, assistant trained)
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from mlpug.mlpug_logging import get_logger, use_fancy_colors

from examples.agentic_llm_pretraining.datasets.common import (
    TransformStats,
    VALID_CHAT_ROLES,
)

from examples.agentic_llm_pretraining.datasets.v2.parts_templates import (
    TemplateBase,
    TextTemplate,
    SplitTemplate,
    DialogueTemplate,
)
from examples.agentic_llm_pretraining.datasets.v2.preprocessing import DialogueData


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


@dataclass
class Part:
    """A typed part of a sample for loss masking."""
    type: str  # "text", "system", "prompt", "response", "thinking"
    text: str

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {"type": self.type, "text": self.text}


# =============================================================================
# Chat formatting
# =============================================================================


def format_chat_parts(
    messages: list[dict],
    system: Optional[str] = None,
) -> list[Part]:
    """
    Format messages into Qwen3 chat format as typed parts.

    Each message dict should have exactly one key indicating the role:
    - {"user": "message text"}
    - {"assistant": "response text"}
    - {"tool": "tool result"}

    Parts are formatted such that when concatenated, they match the output
    of v1's format_chat() which uses "\\n".join(parts).

    :param messages: List of message dicts with role as key.
    :param system: Optional system prompt.

    :return: List of Part objects with appropriate types.

    :raises ValueError: If a message has no valid role key.
    """
    parts = []

    if system:
        # System prompt with Qwen3 markers
        parts.append(Part(
            type="system",
            text=f"<|im_start|>system\n{system}<|im_end|>"
        ))

    for msg in messages:
        role = next(iter(msg.keys()), None)
        if role not in VALID_CHAT_ROLES:
            raise ValueError(
                f"Message has no valid role key. Expected one of {VALID_CHAT_ROLES}, "
                f"got keys: {list(msg.keys())}"
            )

        content = msg[role]
        # Add \n prefix if not first part (matches v1's "\n".join behavior)
        prefix = "\n" if parts else ""
        formatted_text = f"{prefix}<|im_start|>{role}\n{content}<|im_end|>"

        # Map role to part type for loss masking
        if role == "user":
            part_type = "prompt"
        elif role == "assistant":
            part_type = "response"
        elif role == "tool":
            part_type = "prompt"  # Tool results are like user input
        else:
            part_type = "text"

        parts.append(Part(type=part_type, text=formatted_text))

    return parts


# =============================================================================
# Template application
# =============================================================================


def apply_text_template(fields: dict, template: TextTemplate) -> list[Part]:
    """
    Apply a text template to create a single text part.

    Used for datasets where all tokens are trained on (no loss masking).

    :param fields: Dict of field values from preprocess function.
    :param template: TextTemplate with text_template field.

    :return: List with single Part of type "text".
    """
    text = template.text_template.format(**fields)
    return [Part(type="text", text=text)]


def apply_split_template(fields: dict, template: SplitTemplate) -> list[Part]:
    """
    Apply a split template to create prompt + response parts.

    Prompt part is masked (excluded from loss), response part is trained on.

    :param fields: Dict of field values from preprocess function.
    :param template: SplitTemplate with prompt_template and response_template.

    :return: List with Part(type="prompt") and Part(type="response").
    """
    prompt_text = template.prompt_template.format(**fields)
    response_text = template.response_template.format(**fields)

    return [
        Part(type="prompt", text=prompt_text),
        Part(type="response", text=response_text),
    ]


def apply_dialogue_data(dialogue_data: DialogueData) -> list[Part]:
    """
    Convert DialogueData to parts using chat formatting.

    Handles optional suffix task (prompt + response appended after dialogue).

    :param dialogue_data: DialogueData with messages, system, and optional suffix.

    :return: List of Part objects.
    """
    parts = format_chat_parts(
        messages=dialogue_data.messages,
        system=dialogue_data.system,
    )

    # Add suffix task if present
    if dialogue_data.suffix_prompt and dialogue_data.suffix_response:
        parts.append(Part(type="prompt", text=dialogue_data.suffix_prompt))
        parts.append(Part(type="response", text=dialogue_data.suffix_response))

    return parts


# =============================================================================
# Batch transformation
# =============================================================================


def transform_samples_to_parts(
    samples: list[dict],
    dataset_name: str,
    preprocess_fn,
    template: TemplateBase,
    logger: Optional[logging.Logger] = None,
) -> tuple[list[list[Part]], TransformStats]:
    """
    Transform raw samples to structured parts format.

    Supports three template types:
    1. TextTemplate: preprocess_fn returns dict, all tokens trained on
    2. SplitTemplate: preprocess_fn returns dict, prompt masked, response trained
    3. DialogueTemplate: preprocess_fn returns DialogueData, role-based masking

    :param samples: List of raw sample dicts from dataset.
    :param dataset_name: Name of the dataset (used in log messages).
    :param preprocess_fn: Function to extract data from sample.
        For TextTemplate/SplitTemplate: (sample, index, dataset_name, logger) -> dict | None
        For DialogueTemplate: (sample, index, dataset_name, logger) -> DialogueData | None
    :param template: TemplateBase instance (TextTemplate, SplitTemplate, or DialogueTemplate).
    :param logger: Optional logger. Falls back to module logger if None.

    :return: Tuple of (list of parts lists, TransformStats).
    """
    if logger is None:
        logger = module_logger

    results = []
    failed = 0

    for index, sample in enumerate(samples):
        try:
            preprocessed = preprocess_fn(sample, index, dataset_name, logger)
            if preprocessed is None:
                failed += 1
                continue

            # Convert to parts based on template type
            if isinstance(template, DialogueTemplate):
                parts = apply_dialogue_data(preprocessed)
            elif isinstance(template, TextTemplate):
                parts = apply_text_template(preprocessed, template)
            elif isinstance(template, SplitTemplate):
                parts = apply_split_template(preprocessed, template)
            else:
                raise ValueError(f"Unknown template type: {type(template)}")

            results.append(parts)

        except Exception as e:
            logger.warning(
                f"{dataset_name}[{index}]: transform failed with "
                f"{type(e).__name__}: {e}"
            )
            failed += 1

    stats = TransformStats(total=len(samples), success=len(results), failed=failed)
    return results, stats
