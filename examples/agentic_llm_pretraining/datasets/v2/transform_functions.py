"""
V2 transform functions for converting raw dataset samples to structured parts format.

Provides transform functions that output parts for loss masking during tokenization.

Output format: list of Part objects with type and text fields.

Three dataset types are supported:
1. Text datasets: All tokens trained on (type="text")
2. Split template datasets: Prompt masked, response trained
3. Dialogue datasets: Role-based masking with Qwen3-compatible thinking control.
   System/user/tool messages are masked. Assistant messages are split into a masked
   prompt cue (including empty or filled <think> block) and trained response/thinking.
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
    expects_thinking: bool = False,
    add_empty_think_block: bool = True,
) -> list[Part]:
    """
    Format messages into Qwen3 chat format as typed parts.

    Each message dict should have exactly one key indicating the role:
    - {"user": "message text"}
    - {"assistant": "response text"}
    - {"tool": "tool result"}

    Assistant messages are split into prompt cue + response for proper masking:
    - The ``<|im_start|>assistant\\n`` prefix is masked (prompt)
    - For thinking datasets: ``<think>\\n`` is added to the prompt cue, thinking
      content becomes ``Part(type="thinking")``, answer becomes ``Part(type="response")``
    - For non-thinking datasets: an empty ``<think>\\n\\n</think>\\n\\n`` block is
      appended to the prompt cue (masked), unless add_empty_think_block is False.

    This follows the Qwen3 thinking control convention where ``<think>`` blocks
    are always present (empty or filled) and controlled via the prompt.

    :param messages: List of message dicts with role as key.
    :param system: Optional system prompt.
    :param expects_thinking: If True, extract ``<think>`` content from
        assistant messages into separate thinking parts. If False, add
        empty ``<think>`` blocks to the masked prompt.
    :param add_empty_think_block: If False, skip empty ``<think>`` blocks
        for non-thinking assistant messages. Only relevant when
        expects_thinking is False.

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

        if role == "assistant":
            _format_assistant_parts(
                content, prefix, expects_thinking, parts,
                add_empty_think_block=add_empty_think_block,
            )
        else:
            # User and tool messages are masked prompts
            formatted_text = f"{prefix}<|im_start|>{role}\n{content}<|im_end|>"
            part_type = "prompt" if role in ("user", "tool") else "text"
            parts.append(Part(type=part_type, text=formatted_text))

    return parts


def _format_assistant_parts(
    content: str,
    prefix: str,
    expects_thinking: bool,
    parts: list[Part],
    add_empty_think_block: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Format an assistant message into prompt cue + thinking/response parts.

    The ``<|im_start|>assistant\\n`` prefix is always masked as part of the
    prompt. Thinking control follows the Qwen3 convention:

    - Thinking dataset with ``<think>`` content: prompt cue includes
      ``<think>\\n``, thinking content is trained, answer is trained.
    - Non-thinking dataset (or no ``<think>`` in content): prompt cue
      includes empty ``<think>\\n\\n</think>\\n\\n`` block, response is trained.

    Normalized output format for thinking samples::

        <|im_start|>assistant\\n<think>\\n
        {thinking_content}\\n</think>\\n
        \\n{answer}<|im_end|>

    :param content: The assistant message content.
    :param prefix: Newline prefix for non-first parts.
    :param expects_thinking: Whether this dataset has thinking traces.
    :param parts: List to append parts to (modified in place).
    :param logger: Optional logger. Falls back to module logger if None.
    """
    if logger is None:
        logger = module_logger

    cue = f"{prefix}<|im_start|>assistant\n"
    content = content.strip()

    if expects_thinking and content.startswith("<think>"):
        # Extract thinking content and answer
        think_close_tag = "</think>"
        think_end = content.find(think_close_tag)
        if think_end == -1:
            logger.warning(
                "No closing </think> tag found in thinking content — "
                "treating entire content as thinking"
            )
            thinking_text = content[len("<think>"):].strip("\n ")
            parts.append(Part(type="prompt", text=cue + "<think>\n"))
            parts.append(Part(type="thinking", text=thinking_text + "\n</think>\n"))
            parts.append(Part(type="response", text="<|im_end|>"))
            return

        # Normalize: strip whitespace between <think> and content,
        # and between </think> and answer. Handles both "<think>\n..."
        # (openthoughts3) and "<think> ..." (nemotron) variants.
        thinking_text = content[len("<think>"):think_end].strip("\n ")
        answer_text = content[think_end + len(think_close_tag):].strip("\n ")

        parts.append(Part(type="prompt", text=cue + "<think>\n"))
        parts.append(Part(type="thinking", text=thinking_text + "\n</think>\n"))
        parts.append(Part(type="response", text="\n" + answer_text + "<|im_end|>"))
    else:
        # Non-thinking: optionally add empty think block to masked prompt
        if add_empty_think_block:
            cue += "<think>\n\n</think>\n\n"
        parts.append(Part(type="prompt", text=cue))
        parts.append(Part(type="response", text=content + "<|im_end|>"))


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


def apply_dialogue_data(
    dialogue_data: DialogueData,
    expects_thinking: bool = False,
    add_empty_think_block: bool = True,
) -> list[Part]:
    """
    Convert DialogueData to parts using chat formatting.

    Handles optional suffix task (prompt + response appended after dialogue).

    :param dialogue_data: DialogueData with messages, system, and optional suffix.
    :param expects_thinking: Whether this dataset has thinking traces.
    :param add_empty_think_block: If False, skip empty ``<think>`` blocks.

    :return: List of Part objects.
    """
    parts = format_chat_parts(
        messages=dialogue_data.messages,
        system=dialogue_data.system,
        expects_thinking=expects_thinking,
        add_empty_think_block=add_empty_think_block,
    )

    # Add suffix task if present
    if dialogue_data.suffix_prompt and dialogue_data.suffix_response:
        parts.append(Part(type="prompt", text=dialogue_data.suffix_prompt))
        parts.append(Part(type="response", text=dialogue_data.suffix_response))

    return parts


# =============================================================================
# Batch transformation
# =============================================================================


MASKED_PART_TYPES = {"system", "prompt"}


def exceeds_max_masked_chars(
    parts: list[Part],
    max_masked_chars: int,
) -> tuple[bool, Optional[Part]]:
    """
    Check if any single masked part exceeds the character limit.

    :param parts: List of Part objects.
    :param max_masked_chars: Maximum character count for any single masked part.

    :return: Tuple of (exceeds, offending_part). offending_part is None
        if no part exceeds the limit.
    """
    for part in parts:
        if part.type in MASKED_PART_TYPES and len(part.text) > max_masked_chars:
            return True, part
    return False, None


def transform_samples_to_parts(
    samples: list[dict],
    dataset_name: str,
    preprocess_fn,
    template: TemplateBase,
    max_masked_chars: int = 0,
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
    :param max_masked_chars: Skip samples where any single masked part
        exceeds this character count. 0 disables filtering.
    :param logger: Optional logger. Falls back to module logger if None.

    :return: Tuple of (list of parts lists, TransformStats).
    """
    if logger is None:
        logger = module_logger

    results = []
    failed = 0
    filtered = 0

    for index, sample in enumerate(samples):
        try:
            preprocessed = preprocess_fn(sample, index, dataset_name, logger)
            if preprocessed is None:
                failed += 1
                continue

            # Convert to parts based on template type
            if isinstance(template, DialogueTemplate):
                parts = apply_dialogue_data(
                    preprocessed,
                    expects_thinking=template.expects_thinking,
                    add_empty_think_block=template.add_empty_think_block,
                )
            elif isinstance(template, TextTemplate):
                parts = apply_text_template(preprocessed, template)
            elif isinstance(template, SplitTemplate):
                parts = apply_split_template(preprocessed, template)
            else:
                raise ValueError(f"Unknown template type: {type(template)}")

            # Filter by max masked part length
            if max_masked_chars > 0:
                exceeds, offending_part = exceeds_max_masked_chars(
                    parts, max_masked_chars,
                )
                if exceeds:
                    logger.debug(
                        f"{dataset_name}[{index}]: filtered — "
                        f"{offending_part.type} part has {len(offending_part.text)} chars "
                        f"(max: {max_masked_chars})"
                    )
                    filtered += 1
                    continue

            results.append(parts)

        except Exception as e:
            logger.warning(
                f"{dataset_name}[{index}]: transform failed with "
                f"{type(e).__name__}: {e}"
            )
            failed += 1

    stats = TransformStats(
        total=len(samples), success=len(results),
        failed=failed, filtered=filtered,
    )
    return results, stats
