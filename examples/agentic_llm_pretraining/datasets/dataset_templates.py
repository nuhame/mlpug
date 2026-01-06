"""
Dataset templates and preprocess functions for agentic LLM pretraining.

Each dataset has:
- A template string with {field} placeholders
- A preprocess function that validates and extracts fields

See transform_analysis.md for detailed template specifications.
"""

import json
import logging
import os
import random
from typing import Optional

from mlpug.mlpug_logging import get_logger, use_fancy_colors

from .transform_functions import format_chat


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


# =============================================================================
# Templates
# =============================================================================

FINEWEB_EDU_TEMPLATE = "{text}"

GSM8K_TEMPLATE = """
Problem: {question}

Solution: {answer}
"""

# Chat-formatted datasets use format_chat() in preprocess, output via passthrough
PASSTHROUGH_TEMPLATE = "{text}"


# =============================================================================
# Preprocess Functions
# =============================================================================

def preprocess_fineweb_edu(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess fineweb-edu sample.

    Validates that text field exists and is non-empty.
    """
    if logger is None:
        logger = module_logger

    text = sample.get("text", "")

    valid = True
    if not text:
        logger.warning(f"{dataset_name}[{index}]: empty text field")
        valid = False

    if not valid:
        return None

    return {"text": text}


def preprocess_gsm8k(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess gsm8k sample.

    Validates that question and answer fields exist and are non-empty.
    """
    if logger is None:
        logger = module_logger

    question = sample.get("question", "")
    answer = sample.get("answer", "")

    valid = True
    if not question:
        logger.warning(f"{dataset_name}[{index}]: empty question field")
        valid = False
    if not answer:
        logger.warning(f"{dataset_name}[{index}]: empty answer field")
        valid = False

    if not valid:
        return None

    return {"question": question, "answer": answer}


def preprocess_soda(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess soda (DialogStudio) sample.

    Extracts narrative from original dialog info, randomly selects a prompt,
    and formats conversation turns using Qwen3 chat format.
    """
    if logger is None:
        logger = module_logger

    log = sample.get("log", [])
    original_dialog_info_str = sample.get("original dialog info", "{}")
    prompts = sample.get("prompt", [])

    valid = True
    if not log:
        logger.warning(f"{dataset_name}[{index}]: empty log field")
        valid = False
    if not prompts:
        logger.warning(f"{dataset_name}[{index}]: empty prompt field")
        valid = False

    # Parse original dialog info JSON
    try:
        original_dialog_info = json.loads(original_dialog_info_str)
    except json.JSONDecodeError as e:
        logger.warning(f"{dataset_name}[{index}]: failed to parse original dialog info: {e}")
        valid = False
        original_dialog_info = {}

    narrative = original_dialog_info.get("narrative", "")
    if not narrative:
        logger.warning(f"{dataset_name}[{index}]: empty narrative in original dialog info")
        valid = False

    if not valid:
        return None

    # Build system message from narrative + random prompt
    selected_prompt = random.choice(prompts)
    system_message = f"{narrative}\n\n{selected_prompt}"

    # Build conversation messages
    messages = []
    for turn in log:
        user_utterance = turn.get("user utterance", "")
        system_response = turn.get("system response", "")

        if user_utterance:
            messages.append({"user": user_utterance})
        if system_response:
            messages.append({"assistant": system_response})

    if not messages:
        logger.warning(f"{dataset_name}[{index}]: no valid turns in log")
        return None

    # Format as Qwen3 chat
    text = format_chat(messages, system=system_message)

    return {"text": text}


def preprocess_toolace(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess toolace sample.

    Extracts system prompt and conversations, maps 'from' values to Qwen3 roles,
    and formats using Qwen3 chat format.
    """
    if logger is None:
        logger = module_logger

    system = sample.get("system", "")
    conversations = sample.get("conversations", [])

    valid = True
    if not system:
        logger.warning(f"{dataset_name}[{index}]: empty system field")
        valid = False
    if not conversations:
        logger.warning(f"{dataset_name}[{index}]: empty conversations field")
        valid = False

    if not valid:
        return None

    # Map toolace roles to Qwen3 roles
    role_mapping = {
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }

    # Build conversation messages
    messages = []
    for conv in conversations:
        from_role = conv.get("from", "")
        value = conv.get("value", "")

        if not value:
            continue

        qwen_role = role_mapping.get(from_role)
        if qwen_role is None:
            logger.warning(
                f"{dataset_name}[{index}]: unknown role '{from_role}', skipping turn"
            )
            continue

        messages.append({qwen_role: value})

    if not messages:
        logger.warning(f"{dataset_name}[{index}]: no valid turns in conversations")
        return None

    # Format as Qwen3 chat
    text = format_chat(messages, system=system)

    return {"text": text}


# =============================================================================
# Registry
# =============================================================================

TEMPLATES = {
    "fineweb-edu": FINEWEB_EDU_TEMPLATE,
    "gsm8k": GSM8K_TEMPLATE,
    "soda": PASSTHROUGH_TEMPLATE,
    "toolace": PASSTHROUGH_TEMPLATE,
}

PREPROCESS_FUNCTIONS = {
    "fineweb-edu": preprocess_fineweb_edu,
    "gsm8k": preprocess_gsm8k,
    "soda": preprocess_soda,
    "toolace": preprocess_toolace,
}
