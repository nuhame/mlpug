"""
V2 preprocess functions for dialogue datasets.

These functions return structured dialogue data instead of formatted text,
enabling proper loss masking via format_chat_parts().

Return format:
- messages: list of message dicts [{"user": "..."}, {"assistant": "..."}, ...]
- system: optional system message
- suffix_prompt: optional prompt text appended after dialogue (for tasks)
- suffix_response: optional response text for the suffix task

Standard dialogue preprocess functions (v1) return {"text": formatted_text}.
V2 dialogue preprocess functions return the raw components for format_chat_parts().
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

from mlpug.mlpug_logging import get_logger, use_fancy_colors

from examples.agentic_llm_pretraining.datasets.common import extract_dialogstudio_messages


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


@dataclass
class DialogueData:
    """
    Structured dialogue data for v2 transform pipeline.

    Fields:
    - messages: Conversation turns [{"user": "..."}, {"assistant": "..."}, ...]
    - system: System message (prefix instruction, masked)
    - suffix_prompt: Task prompt appended AFTER dialogue (e.g., "Summarize...")
    - suffix_response: Response to suffix task (trained on)
    """
    messages: list[dict] = field(default_factory=list)
    system: Optional[str] = None
    suffix_prompt: Optional[str] = None
    suffix_response: Optional[str] = None


# =============================================================================
# Helper: Standard Chat Format
# =============================================================================


def _extract_standard_chat_messages(
    messages_raw: list[dict],
    index: int,
    dataset_name: str,
    logger: logging.Logger,
) -> Optional[list[dict]]:
    """
    Extract messages from standard chat format (role/content dicts).

    Handles datasets like Crab-SFT and tulu3-if that use
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    :param messages_raw: Raw messages list from sample.
    :param index: Sample index for logging.
    :param dataset_name: Dataset name for logging.
    :param logger: Logger instance.

    :return: List of message dicts for DialogueData, or None if invalid.
    """
    if not messages_raw:
        logger.warning(f"{dataset_name}[{index}]: empty messages field")
        return None

    messages = []
    for msg in messages_raw:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if not content:
            continue

        if role in ("user", "assistant"):
            messages.append({role: content})
        else:
            logger.warning(
                f"{dataset_name}[{index}]: unknown role '{role}', skipping turn"
            )

    if not messages:
        logger.warning(f"{dataset_name}[{index}]: no valid turns in messages")
        return None

    return messages


# =============================================================================
# Reasoning Trace Preprocessing
# =============================================================================


def preprocess_openthoughts3_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess OpenThoughts3-1.2M sample for v2 parts format.

    Maps human→user, gpt→assistant. GPT responses contain <think>...</think>
    reasoning traces followed by the final answer.
    """
    if logger is None:
        logger = module_logger

    conversations = sample.get("conversations", [])

    if not conversations:
        logger.warning(f"{dataset_name}[{index}]: empty conversations field")
        return None

    role_mapping = {
        "human": "user",
        "gpt": "assistant",
    }

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

    return DialogueData(messages=messages)


# =============================================================================
# Instruction Following Preprocessing
# =============================================================================


def preprocess_crab_sft_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess Crab-SFT sample for v2 parts format.

    Multi-turn instruction-following with format/length/style constraints.
    Standard role/content chat format.
    """
    if logger is None:
        logger = module_logger

    messages = _extract_standard_chat_messages(
        sample.get("messages", []), index, dataset_name, logger,
    )
    if messages is None:
        return None

    return DialogueData(messages=messages)


def preprocess_tulu3_if_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess tulu-3-sft-personas-instruction-following sample for v2 parts format.

    Single-turn IFEval-style verifiable constraint satisfaction.
    Standard content/role chat format. The 'prompt' and 'constraints' fields
    are redundant metadata — the actual data is in 'messages'.
    """
    if logger is None:
        logger = module_logger

    messages = _extract_standard_chat_messages(
        sample.get("messages", []), index, dataset_name, logger,
    )
    if messages is None:
        return None

    return DialogueData(messages=messages)


# =============================================================================
# Agentic Dialogue Preprocessing
# =============================================================================


def preprocess_toolace_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess toolace sample for v2 parts format.

    Returns structured dialogue data with system prompt and conversation messages.
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

    return DialogueData(messages=messages, system=system)


def preprocess_hermes_function_calling_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess hermes-function-calling sample for v2 parts format.

    Maps 'from' values to Qwen3 roles: system→system, human→user, gpt→assistant.
    """
    if logger is None:
        logger = module_logger

    conversations = sample.get("conversations", [])

    if not conversations:
        logger.warning(f"{dataset_name}[{index}]: empty conversations field")
        return None

    # Map hermes roles to Qwen3 roles
    role_mapping = {
        "system": "system",
        "human": "user",
        "gpt": "assistant",
    }

    system_message = None
    messages = []

    for conv in conversations:
        from_role = conv.get("from", "")
        value = conv.get("value", "")

        if not value:
            continue

        if from_role == "system":
            system_message = value
        else:
            qwen_role = role_mapping.get(from_role)
            if qwen_role is None:
                logger.warning(
                    f"{dataset_name}[{index}]: unknown role '{from_role}', skipping"
                )
                continue
            messages.append({qwen_role: value})

    if not messages:
        logger.warning(f"{dataset_name}[{index}]: no valid turns in conversations")
        return None

    return DialogueData(messages=messages, system=system_message)


def preprocess_glaive_function_calling_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess glaive-function-calling sample for v2 parts format.

    Parses system and chat fields. Chat format uses USER:/ASSISTANT:/FUNCTION RESPONSE:
    prefixes. Removes <|endoftext|> tokens.
    """
    if logger is None:
        logger = module_logger

    system = sample.get("system", "")
    chat = sample.get("chat", "")

    if not chat:
        logger.warning(f"{dataset_name}[{index}]: empty chat field")
        return None

    # Strip "SYSTEM: " prefix from system
    if system.startswith("SYSTEM: "):
        system = system[8:]

    # Remove any <|endoftext|> tokens
    chat = chat.replace("<|endoftext|>", "")

    # Parse chat by splitting on role markers
    markers = ["USER:", "ASSISTANT:", "FUNCTION RESPONSE:"]
    messages = []

    # Find all marker positions
    positions = []
    for marker in markers:
        start = 0
        while True:
            pos = chat.find(marker, start)
            if pos == -1:
                break
            positions.append((pos, marker))
            start = pos + len(marker)

    # Sort by position
    positions.sort(key=lambda x: x[0])

    # Extract content between markers
    for i, (pos, marker) in enumerate(positions):
        start = pos + len(marker)
        if i + 1 < len(positions):
            end = positions[i + 1][0]
        else:
            end = len(chat)

        content = chat[start:end].strip()
        if not content:
            continue

        if marker == "USER:":
            messages.append({"user": content})
        elif marker == "ASSISTANT:":
            messages.append({"assistant": content})
        elif marker == "FUNCTION RESPONSE:":
            messages.append({"tool": content})

    if not messages:
        logger.warning(f"{dataset_name}[{index}]: no valid messages parsed from chat")
        return None

    return DialogueData(
        messages=messages,
        system=system if system else None,
    )


# =============================================================================
# Social Dialogue Preprocessing (DialogStudio)
# =============================================================================


def preprocess_soda_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess soda (DialogStudio) sample for v2 parts format.

    Extracts narrative from original dialog info, randomly selects a prompt,
    and returns structured dialogue data.
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
        logger.warning(
            f"{dataset_name}[{index}]: failed to parse original dialog info: {e}"
        )
        valid = False
        original_dialog_info = {}

    narrative = original_dialog_info.get("narrative", "")
    if not narrative:
        logger.warning(
            f"{dataset_name}[{index}]: empty narrative in original dialog info"
        )
        valid = False

    if not valid:
        return None

    messages = extract_dialogstudio_messages(log, index, dataset_name, logger)
    if messages is None:
        return None

    # Build system message from narrative + random prompt
    selected_prompt = random.choice(prompts)
    system_message = f"{narrative}\n\n{selected_prompt}"

    return DialogueData(messages=messages, system=system_message)


def preprocess_multiwoz_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess multiwoz (DialogStudio) sample for v2 parts format.

    Task-oriented dialogues. Randomly selects one of 5 prompts as system message.
    """
    if logger is None:
        logger = module_logger

    log = sample.get("log", [])
    prompts = sample.get("prompt", [])

    if not prompts:
        logger.warning(f"{dataset_name}[{index}]: empty prompt field")
        return None

    messages = extract_dialogstudio_messages(log, index, dataset_name, logger)
    if messages is None:
        return None

    selected_prompt = random.choice(prompts)

    return DialogueData(messages=messages, system=selected_prompt)


def preprocess_wizard_of_wikipedia_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess wizard-of-wikipedia (DialogStudio) sample for v2 parts format.

    Knowledge-grounded dialogues. Extracts persona, topic, and background passages.
    """
    if logger is None:
        logger = module_logger

    log = sample.get("log", [])
    original_dialog_info_str = sample.get("original dialog info", "{}")

    # Parse original dialog info
    try:
        original_dialog_info = json.loads(original_dialog_info_str)
    except json.JSONDecodeError as e:
        logger.warning(
            f"{dataset_name}[{index}]: failed to parse original dialog info: {e}"
        )
        return None

    chosen_topic = original_dialog_info.get("chosen_topic", "")
    chosen_topic_passage = original_dialog_info.get("chosen_topic_passage", [])
    persona = original_dialog_info.get("persona", "")

    if not chosen_topic:
        logger.warning(f"{dataset_name}[{index}]: empty chosen_topic")
        return None

    messages = extract_dialogstudio_messages(log, index, dataset_name, logger)
    if messages is None:
        return None

    # Build system message
    system_parts = []
    if persona:
        system_parts.append(
            f"You act as a person with the following persona: {persona}"
        )
    system_parts.append(f"Together with a user you will talk about: {chosen_topic}")

    if chosen_topic_passage:
        system_parts.append("\nBackground info:")
        for passage in chosen_topic_passage:
            system_parts.append(passage)

    system_message = "\n".join(system_parts)

    return DialogueData(messages=messages, system=system_message)


def preprocess_sharegpt_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess sharegpt (DialogStudio) sample for v2 parts format.

    General user-assistant conversations. No system prompt.
    """
    if logger is None:
        logger = module_logger

    log = sample.get("log", [])

    messages = extract_dialogstudio_messages(log, index, dataset_name, logger)
    if messages is None:
        return None

    return DialogueData(messages=messages, system=None)


def preprocess_empathetic_dialogues_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess empathetic-dialogues (DialogStudio) sample for v2 parts format.

    Emotionally aware peer conversations. Extracts emotion from context.
    Appends emotion recognition Q&A as suffix task.
    """
    if logger is None:
        logger = module_logger

    log = sample.get("log", [])
    original_dialog_info_str = sample.get("original dialog info", "{}")

    # Parse original dialog info
    try:
        original_dialog_info = json.loads(original_dialog_info_str)
    except json.JSONDecodeError as e:
        logger.warning(
            f"{dataset_name}[{index}]: failed to parse original dialog info: {e}"
        )
        return None

    emotion = original_dialog_info.get("context", "")

    messages = extract_dialogstudio_messages(log, index, dataset_name, logger)
    if messages is None:
        return None

    # Suffix task for emotion recognition
    suffix_prompt = None
    suffix_response = None
    if emotion:
        suffix_prompt = (
            "\n\nTask: Identify how the user feels in the above conversation.\n"
            "Emotion: "
        )
        suffix_response = emotion

    return DialogueData(
        messages=messages,
        system=None,
        suffix_prompt=suffix_prompt,
        suffix_response=suffix_response,
    )


def preprocess_samsum_parts(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[DialogueData]:
    """
    Preprocess samsum (DialogStudio) sample for v2 parts format.

    Dialogue summarization. Appends summarization Q&A as suffix task.
    """
    if logger is None:
        logger = module_logger

    log = sample.get("log", [])
    original_dialog_info_str = sample.get("original dialog info", "{}")

    # Parse original dialog info
    try:
        original_dialog_info = json.loads(original_dialog_info_str)
    except json.JSONDecodeError as e:
        logger.warning(
            f"{dataset_name}[{index}]: failed to parse original dialog info: {e}"
        )
        return None

    summary = original_dialog_info.get("summary", "")

    messages = extract_dialogstudio_messages(log, index, dataset_name, logger)
    if messages is None:
        return None

    # Suffix task for summarization
    suffix_prompt = None
    suffix_response = None
    if summary:
        suffix_prompt = "\n\nTask: Summarize the above conversation.\nSummary: "
        suffix_response = summary
    else:
        logger.warning(f"{dataset_name}[{index}]: empty summary")

    return DialogueData(
        messages=messages,
        system=None,
        suffix_prompt=suffix_prompt,
        suffix_response=suffix_response,
    )


# =============================================================================
# Registry
# =============================================================================

# Map dataset names to v2 preprocessing functions
DIALOGUE_PREPROCESS_FUNCS = {
    "openthoughts3": preprocess_openthoughts3_parts,
    "crab-sft": preprocess_crab_sft_parts,
    "tulu3-if": preprocess_tulu3_if_parts,
    "toolace": preprocess_toolace_parts,
    "hermes-function-calling": preprocess_hermes_function_calling_parts,
    "glaive-function-calling": preprocess_glaive_function_calling_parts,
    "soda": preprocess_soda_parts,
    "multiwoz": preprocess_multiwoz_parts,
    "wizard-of-wikipedia": preprocess_wizard_of_wikipedia_parts,
    "sharegpt": preprocess_sharegpt_parts,
    "empathetic-dialogues": preprocess_empathetic_dialogues_parts,
    "samsum": preprocess_samsum_parts,
}
