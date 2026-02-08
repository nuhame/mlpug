"""
V1 transform functions for converting raw dataset samples to training text.

Provides a generic transform function that applies optional preprocessing
and template formatting, plus helpers for Qwen3 chat format.

Output format: single text string per sample.
"""

import logging
import os
from typing import Optional

from mlpug.mlpug_logging import get_logger, use_fancy_colors

from examples.agentic_llm_pretraining.datasets.common import (
    TransformStats,
    PreprocessFunc,
    VALID_CHAT_ROLES,
)


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


def transform(
    samples: list[dict],
    dataset_name: str,
    preprocess_fn: Optional[PreprocessFunc] = None,
    template: str = "{text}",
    logger: Optional[logging.Logger] = None,
) -> tuple[list[str], TransformStats]:
    """
    Transform raw samples to formatted text strings.

    :param samples: List of raw sample dicts from dataset.
    :param dataset_name: Name of the dataset (used in log messages).
    :param preprocess_fn: Optional function to extract/validate fields.
        Signature: (sample, index, dataset_name, logger) -> dict | None
        Returns dict of template fields, or None to skip invalid samples.
    :param template: Format string using {field} placeholders.
        Applied as template.strip().format(**fields).
    :param logger: Optional logger. Falls back to module logger if None.

    :return: Tuple of (transformed texts, TransformStats).
    """
    if logger is None:
        logger = module_logger

    results = []
    failed = 0

    for index, sample in enumerate(samples):
        try:
            if preprocess_fn:
                fields = preprocess_fn(sample, index, dataset_name, logger)
                if fields is None:
                    failed += 1
                    continue
            else:
                fields = sample

            text = template.strip().format(**fields)
            results.append(text)

        except Exception as e:
            logger.warning(f"{dataset_name}[{index}]: preprocess failed with {type(e).__name__}: {e}")
            failed += 1

    stats = TransformStats(total=len(samples), success=len(results), failed=failed)
    return results, stats


def format_chat(
    messages: list[dict],
    system: Optional[str] = None,
) -> str:
    """
    Format messages into Qwen3 chat format.

    Each message dict should have exactly one key indicating the role:
    - {"user": "message text"}
    - {"assistant": "response text"}
    - {"tool": "tool result"}

    :param messages: List of message dicts with role as key.
    :param system: Optional system prompt.

    :return: Formatted chat string with <|im_start|>/<|im_end|> tokens.

    :raises ValueError: If a message has no valid role key.
    """
    parts = []

    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")

    for msg in messages:
        role = next(iter(msg.keys()), None)
        if role in VALID_CHAT_ROLES:
            parts.append(f"<|im_start|>{role}\n{msg[role]}<|im_end|>")
        else:
            raise ValueError(
                f"Message has no valid role key. Expected one of {VALID_CHAT_ROLES}, "
                f"got keys: {list(msg.keys())}"
            )

    return "\n".join(parts)
