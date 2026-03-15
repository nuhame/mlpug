"""
Reward functions for RLVR training.

This module provides reward functions compatible with TRL's GRPOTrainer interface.
Reward functions are also reusable with the future MLPug GRPO implementation (Phase 2).

TRL reward function interface:
    def reward_func(completions, **kwargs) -> list[float]
    - completions: list[str] or list[list[dict]] (conversational format)
    - **kwargs: any extra dataset columns (e.g., ground_truth)
    - returns: list of float rewards, one per completion
"""

import re

import os

from basics.logging import get_logger

import mlpug.pytorch as mlp
mlp.logging.use_fancy_colors()

from examples.agentic_llm_pretraining.evaluation.lm_eval.tasks.gsm8k.utils import (
    _normalize_answer,
    _answers_match,
)

module_logger = get_logger(os.path.basename(__file__))


def extract_gsm8k_answer(text: str) -> str:
    """Extract final numeric answer from a model completion.

    Tries multiple extraction patterns in priority order:
    1. #### <answer>  (GSM8K format, what our NTP training data uses)
    2. \\boxed{<answer>}  (LaTeX format, from OpenThoughts3/math exposure)
    3. Last number in text  (fallback)

    :param text: The model's generated text.

    :return: Extracted answer string, or empty string if none found.
    """
    # Pattern 1: GSM8K "#### <answer>"
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()

    # Pattern 2: LaTeX \boxed{<answer>}
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()

    # Pattern 3: Last number in text (fallback)
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].strip()

    return ""


def gsm8k_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """Binary reward for GSM8K math problems.

    Compatible with TRL GRPOTrainer reward function interface.
    Handles both standard format (completions are strings) and
    conversational format (completions are list of message dicts).

    :param completions: Generated completions. Either list[str] (standard)
        or list[list[dict]] (conversational, e.g. [{"role": "assistant",
        "content": "..."}]).
    :param ground_truth: Ground truth answers from dataset (list[str]).

    :return: List of 1.0 (correct) or 0.0 (incorrect), one per completion.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        # Handle conversational format from TRL
        if isinstance(completion, list):
            # TRL conversational: [{"role": "assistant", "content": "..."}]
            text = completion[-1]["content"] if completion else ""
        else:
            text = completion

        predicted = extract_gsm8k_answer(text)
        reward = 1.0 if _answers_match(predicted, gt) else 0.0
        rewards.append(reward)

    return rewards
