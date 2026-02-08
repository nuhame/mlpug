"""
Preprocess functions for dataset transformation.

Each function validates and extracts fields from raw samples,
returning a dict of template fields or None if invalid.

See transform_analysis.md for detailed specifications.
"""

import json
import logging
import os
import random
from typing import Optional

from mlpug.mlpug_logging import get_logger, use_fancy_colors

from examples.agentic_llm_pretraining.datasets.common import extract_dialogstudio_messages
from examples.agentic_llm_pretraining.datasets.v1.transform_functions import format_chat


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


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
# Grammar/Language Preprocessing
# =============================================================================


def preprocess_wikipedia(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess wikipedia/simple-wikipedia sample.

    Extracts title and text, validates both are non-empty.
    """
    if logger is None:
        logger = module_logger

    title = sample.get("title", "")
    text = sample.get("text", "")

    valid = True
    if not title:
        logger.warning(f"{dataset_name}[{index}]: empty title field")
        valid = False
    if not text:
        logger.warning(f"{dataset_name}[{index}]: empty text field")
        valid = False

    if not valid:
        return None

    return {"title": title, "text": text}


# =============================================================================
# Reasoning/Math Preprocessing
# =============================================================================


def preprocess_math(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess competition_math sample.

    Extracts problem, solution, type, and level. Keeps LaTeX notation.
    """
    if logger is None:
        logger = module_logger

    problem = sample.get("problem", "")
    solution = sample.get("solution", "")
    math_type = sample.get("type", "")
    level = sample.get("level", "")

    valid = True
    if not problem:
        logger.warning(f"{dataset_name}[{index}]: empty problem field")
        valid = False
    if not solution:
        logger.warning(f"{dataset_name}[{index}]: empty solution field")
        valid = False

    if not valid:
        return None

    # Default values for missing metadata
    if not math_type:
        math_type = "Math"
    if not level:
        level = "Unknown"

    return {
        "problem": problem,
        "solution": solution,
        "type": math_type,
        "level": level,
    }


def preprocess_openmath_instruct_1(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess openmath-instruct-1 sample.

    Extracts question and generated_solution.
    Keeps <llm-code> tags as-is for HTML/XML syntax exposure.

    Note: Filtering for is_correct=True should be done via filter_func during download.
    """
    if logger is None:
        logger = module_logger

    question = sample.get("question", "")
    generated_solution = sample.get("generated_solution", "")

    valid = True
    if not question:
        logger.warning(f"{dataset_name}[{index}]: empty question field")
        valid = False
    if not generated_solution:
        logger.warning(f"{dataset_name}[{index}]: empty generated_solution field")
        valid = False

    if not valid:
        return None

    return {"question": question, "generated_solution": generated_solution}


def preprocess_openmath_instruct_2(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess openmath-instruct-2 sample.

    Extracts problem and generated_solution. All solutions are correct.
    """
    if logger is None:
        logger = module_logger

    problem = sample.get("problem", "")
    generated_solution = sample.get("generated_solution", "")

    valid = True
    if not problem:
        logger.warning(f"{dataset_name}[{index}]: empty problem field")
        valid = False
    if not generated_solution:
        logger.warning(f"{dataset_name}[{index}]: empty generated_solution field")
        valid = False

    if not valid:
        return None

    return {"problem": problem, "generated_solution": generated_solution}


def preprocess_ecqa(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess ecqa sample.

    Extracts question, options, reasoning (positive/negative), conclusion, answer.
    Teaches comparative reasoning.
    """
    if logger is None:
        logger = module_logger

    q_text = sample.get("q_text", "")
    q_op1 = sample.get("q_op1", "")
    q_op2 = sample.get("q_op2", "")
    q_op3 = sample.get("q_op3", "")
    q_op4 = sample.get("q_op4", "")
    q_op5 = sample.get("q_op5", "")
    q_ans = sample.get("q_ans", "")
    taskA_pos = sample.get("taskA_pos", "")
    taskA_neg = sample.get("taskA_neg", "")
    taskB = sample.get("taskB", "")

    valid = True
    if not q_text:
        logger.warning(f"{dataset_name}[{index}]: empty q_text field")
        valid = False
    if not q_ans:
        logger.warning(f"{dataset_name}[{index}]: empty q_ans field")
        valid = False
    if not taskB:
        logger.warning(f"{dataset_name}[{index}]: empty taskB (conclusion) field")
        valid = False

    if not valid:
        return None

    return {
        "q_text": q_text,
        "q_op1": q_op1 or "(no option)",
        "q_op2": q_op2 or "(no option)",
        "q_op3": q_op3 or "(no option)",
        "q_op4": q_op4 or "(no option)",
        "q_op5": q_op5 or "(no option)",
        "q_ans": q_ans,
        "taskA_pos": taskA_pos or "(no positive reasoning)",
        "taskA_neg": taskA_neg or "(no negative reasoning)",
        "taskB": taskB,
    }


# =============================================================================
# Procedural Preprocessing
# =============================================================================


def preprocess_cosmopedia_wikihow(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess cosmopedia-wikihow sample.

    Extracts text field (already well-formatted with steps).
    """
    if logger is None:
        logger = module_logger

    text = sample.get("text", "")

    if not text:
        logger.warning(f"{dataset_name}[{index}]: empty text field")
        return None

    return {"text": text}


def preprocess_stackexchange(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess stackexchange sample.

    Extracts question and best answer(s) by pm_score. Keeps HTML as-is.
    If multiple answers have the same highest score, include all.
    """
    if logger is None:
        logger = module_logger

    question = sample.get("question", "")
    answers = sample.get("answers", [])

    valid = True
    if not question:
        logger.warning(f"{dataset_name}[{index}]: empty question field")
        valid = False
    if not answers:
        logger.warning(f"{dataset_name}[{index}]: empty answers field")
        valid = False

    if not valid:
        return None

    # Find highest pm_score
    max_score = float("-inf")
    for ans in answers:
        score = ans.get("pm_score", 0)
        if score > max_score:
            max_score = score

    # Collect all answers with the highest score
    best_answers = []
    for ans in answers:
        if ans.get("pm_score", 0) == max_score:
            answer_text = ans.get("text", "")
            if answer_text:
                best_answers.append(answer_text)

    if not best_answers:
        logger.warning(f"{dataset_name}[{index}]: no valid answer text found")
        return None

    # Format answers
    if len(best_answers) == 1:
        answers_text = f"Answer: {best_answers[0]}"
    else:
        answers_parts = []
        for i, ans_text in enumerate(best_answers, 1):
            answers_parts.append(f"Answer {i}: {ans_text}")
        answers_text = "\n\n".join(answers_parts)

    return {"question": question, "answers": answers_text}


# =============================================================================
# Code Preprocessing
# =============================================================================


def preprocess_swe_bench(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess swe-bench sample.

    Extracts repository name, problem statement (issue), and patch (solution).
    """
    if logger is None:
        logger = module_logger

    repo = sample.get("repo", "")
    problem_statement = sample.get("problem_statement", "")
    patch = sample.get("patch", "")

    valid = True
    if not repo:
        logger.warning(f"{dataset_name}[{index}]: empty repo field")
        valid = False
    if not problem_statement:
        logger.warning(f"{dataset_name}[{index}]: empty problem_statement field")
        valid = False
    if not patch:
        logger.warning(f"{dataset_name}[{index}]: empty patch field")
        valid = False

    if not valid:
        return None

    return {
        "repo": repo,
        "problem_statement": problem_statement,
        "patch": patch,
    }


# Language ID mapping for code-contests
# Full mapping for logging purposes
CODE_CONTESTS_ALL_LANGUAGES = {
    0: "Unknown",
    1: "Python2",
    2: "C++",
    3: "Python3",
    4: "Java",
}
# Target languages with output format
CODE_CONTESTS_LANGUAGE_IDS = {
    2: ("C++", "cpp"),
    3: ("Python3", "python"),
    4: ("Java", "java"),
}
CODE_CONTESTS_WEIGHTS = {3: 0.5, 2: 0.25, 4: 0.25}


def preprocess_code_contests(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess code-contests sample.

    Selects a language with weighted probability (Python3: 0.5, C++: 0.25, Java: 0.25).
    Picks shortest solution in chosen language. Includes all public test cases.
    """
    if logger is None:
        logger = module_logger

    name = sample.get("name", "")
    description = sample.get("description", "")
    public_tests = sample.get("public_tests", {})
    solutions = sample.get("solutions", {})

    valid = True
    if not name:
        logger.warning(f"{dataset_name}[{index}]: empty name field")
        valid = False
    if not description:
        logger.warning(f"{dataset_name}[{index}]: empty description field")
        valid = False

    if not valid:
        return None

    # Get solution languages and code
    solution_languages = solutions.get("language", [])
    solution_codes = solutions.get("solution", [])

    if not solution_languages or not solution_codes:
        logger.warning(f"{dataset_name}[{index}]: no solutions available")
        return None

    # Find available languages from our target set
    available_langs = set(solution_languages)
    candidates = [lid for lid in CODE_CONTESTS_WEIGHTS if lid in available_langs]

    if not candidates:
        # Log what languages are actually available
        lang_names = [
            CODE_CONTESTS_ALL_LANGUAGES.get(lid, f"Unknown language ID:{lid}")
            for lid in available_langs
        ]
        logger.warning(
            f"{dataset_name}[{index}]: no Python3/C++/Java solutions, "
            f"available: {', '.join(lang_names)}"
        )
        return None

    # Weighted random selection
    weights = [CODE_CONTESTS_WEIGHTS[lid] for lid in candidates]
    chosen_id = random.choices(candidates, weights=weights)[0]

    # Get indices for chosen language, find shortest solution
    indices = [
        i for i, lid in enumerate(solution_languages) if lid == chosen_id
    ]
    if not indices:
        logger.warning(f"{dataset_name}[{index}]: no solution for chosen language")
        return None

    shortest_idx = min(indices, key=lambda i: len(solution_codes[i]))
    solution_code = solution_codes[shortest_idx]
    language_name, language_ext = CODE_CONTESTS_LANGUAGE_IDS[chosen_id]

    # Format public test cases
    test_inputs = public_tests.get("input", [])
    test_outputs = public_tests.get("output", [])
    examples_parts = []
    for inp, out in zip(test_inputs, test_outputs):
        examples_parts.append(f"Input: {inp}\nOutput: {out}")
    examples_text = "\n\n".join(examples_parts) if examples_parts else "(no examples)"

    return {
        "name": name,
        "description": description,
        "examples": examples_text,
        "language": language_name,
        "language_ext": language_ext,
        "solution": solution_code,
    }


def preprocess_codesearchnet(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess codesearchnet sample.

    Extracts whole_func_string (complete function with docstring).
    """
    if logger is None:
        logger = module_logger

    func_string = sample.get("whole_func_string", "")

    if not func_string:
        logger.warning(f"{dataset_name}[{index}]: empty whole_func_string field")
        return None

    return {"text": func_string}


# =============================================================================
# Agentic Preprocessing
# =============================================================================


def preprocess_hermes_function_calling(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess hermes-function-calling sample.

    Maps 'from' values to Qwen3 roles: system→system, human→user, gpt→assistant.
    Formats as Qwen3 chat.
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

    # First message is typically system prompt
    system_message = None
    messages = []

    for conv in conversations:
        from_role = conv.get("from", "")
        value = conv.get("value", "")

        if not value:
            continue

        if from_role == "system":
            # Use as system message
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

    # Format as Qwen3 chat
    text = format_chat(messages, system=system_message)

    return {"text": text}


def preprocess_glaive_function_calling(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess glaive-function-calling sample.

    Parses system and chat fields. Chat format uses USER:/ASSISTANT:/FUNCTION RESPONSE:
    prefixes. Maps to Qwen3 format. Removes <|endoftext|> tokens.
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
    # Order matters: check FUNCTION RESPONSE before just RESPONSE patterns
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

    # Format as Qwen3 chat
    text = format_chat(messages, system=system if system else None)

    return {"text": text}


# =============================================================================
# Knowledge Preprocessing
# =============================================================================


def preprocess_generics_kb(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess generics-kb sample.

    Extracts generic_sentence. Simple passthrough.
    """
    if logger is None:
        logger = module_logger

    sentence = sample.get("generic_sentence", "")

    if not sentence:
        logger.warning(f"{dataset_name}[{index}]: empty generic_sentence field")
        return None

    return {"text": sentence}


def preprocess_openbookqa(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess openbookqa sample.

    Extracts question, choices (A/B/C/D), and answer key with text.
    """
    if logger is None:
        logger = module_logger

    question_stem = sample.get("question_stem", "")
    choices = sample.get("choices", {})
    answer_key = sample.get("answerKey", "")

    valid = True
    if not question_stem:
        logger.warning(f"{dataset_name}[{index}]: empty question_stem field")
        valid = False
    if not answer_key:
        logger.warning(f"{dataset_name}[{index}]: empty answerKey field")
        valid = False

    if not valid:
        return None

    # Extract choice texts
    choice_texts = choices.get("text", [])
    choice_labels = choices.get("label", [])

    # Build label -> text mapping
    label_to_text = {}
    for label, text in zip(choice_labels, choice_texts):
        label_to_text[label] = text

    # Get individual choices
    choice_a = label_to_text.get("A", "")
    choice_b = label_to_text.get("B", "")
    choice_c = label_to_text.get("C", "")
    choice_d = label_to_text.get("D", "")

    if not all([choice_a, choice_b, choice_c, choice_d]):
        logger.warning(f"{dataset_name}[{index}]: missing choice text")
        return None

    # Get answer text
    answer_text = label_to_text.get(answer_key, "")
    if not answer_text:
        logger.warning(f"{dataset_name}[{index}]: answer key not in choices")
        return None

    return {
        "question_stem": question_stem,
        "choice_a": choice_a,
        "choice_b": choice_b,
        "choice_c": choice_c,
        "choice_d": choice_d,
        "answer_key": answer_key,
        "answer_text": answer_text,
    }


# =============================================================================
# Dialogue Preprocessing (DialogStudio format)
# =============================================================================


def preprocess_multiwoz(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess multiwoz (DialogStudio) sample.

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

    # Randomly select a prompt as system message
    selected_prompt = random.choice(prompts)

    text = format_chat(messages, system=selected_prompt)

    return {"text": text}


def preprocess_wizard_of_wikipedia(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess wizard-of-wikipedia (DialogStudio) sample.

    Knowledge-grounded dialogues. Extracts persona, topic, and background passages.
    Handles both assistant-initiated (~80%) and user-initiated (~20%) conversations.
    """
    if logger is None:
        logger = module_logger

    log = sample.get("log", [])
    original_dialog_info_str = sample.get("original dialog info", "{}")

    # Parse original dialog info
    try:
        original_dialog_info = json.loads(original_dialog_info_str)
    except json.JSONDecodeError as e:
        logger.warning(f"{dataset_name}[{index}]: failed to parse original dialog info: {e}")
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
        system_parts.append(f"You act as a person with the following persona: {persona}")
    system_parts.append(f"Together with a user you will talk about: {chosen_topic}")

    if chosen_topic_passage:
        system_parts.append("\nBackground info:")
        for passage in chosen_topic_passage:
            system_parts.append(passage)

    system_message = "\n".join(system_parts)

    text = format_chat(messages, system=system_message)

    return {"text": text}


def preprocess_sharegpt(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess sharegpt (DialogStudio) sample.

    General user-assistant conversations. No system prompt.
    """
    if logger is None:
        logger = module_logger

    log = sample.get("log", [])

    messages = extract_dialogstudio_messages(log, index, dataset_name, logger)
    if messages is None:
        return None

    # No system message for ShareGPT
    text = format_chat(messages, system=None)

    return {"text": text}


def preprocess_empathetic_dialogues(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess empathetic-dialogues (DialogStudio) sample.

    Emotionally aware peer conversations. Extracts emotion from context.
    Appends emotion recognition Q&A after conversation.
    """
    if logger is None:
        logger = module_logger

    log = sample.get("log", [])
    original_dialog_info_str = sample.get("original dialog info", "{}")

    # Parse original dialog info
    try:
        original_dialog_info = json.loads(original_dialog_info_str)
    except json.JSONDecodeError as e:
        logger.warning(f"{dataset_name}[{index}]: failed to parse original dialog info: {e}")
        return None

    emotion = original_dialog_info.get("context", "")

    messages = extract_dialogstudio_messages(log, index, dataset_name, logger)
    if messages is None:
        return None

    # Format conversation
    chat_text = format_chat(messages, system=None)

    # Append emotion recognition task
    if emotion:
        text = f"{chat_text}\n\nTask: Identify how the user feels in the above conversation.\nEmotion: {emotion}"
    else:
        text = chat_text

    return {"text": text}


def preprocess_samsum(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess samsum (DialogStudio) sample.

    Dialogue summarization. Appends summarization Q&A after conversation.
    """
    if logger is None:
        logger = module_logger

    log = sample.get("log", [])
    original_dialog_info_str = sample.get("original dialog info", "{}")

    # Parse original dialog info
    try:
        original_dialog_info = json.loads(original_dialog_info_str)
    except json.JSONDecodeError as e:
        logger.warning(f"{dataset_name}[{index}]: failed to parse original dialog info: {e}")
        return None

    summary = original_dialog_info.get("summary", "")

    messages = extract_dialogstudio_messages(log, index, dataset_name, logger)
    if messages is None:
        return None

    # Format conversation
    chat_text = format_chat(messages, system=None)

    # Append summarization task
    if summary:
        text = f"{chat_text}\n\nTask: Summarize the above conversation.\nSummary: {summary}"
    else:
        logger.warning(f"{dataset_name}[{index}]: empty summary")
        text = chat_text

    return {"text": text}


# =============================================================================
# RAG Preprocessing
# =============================================================================


def preprocess_rag_dataset_12000(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess rag-dataset-12000 sample.

    Reading comprehension format: context is given, then question and answer.
    """
    if logger is None:
        logger = module_logger

    context = sample.get("context", "")
    question = sample.get("question", "")
    answer = sample.get("answer", "")

    valid = True
    if not context:
        logger.warning(f"{dataset_name}[{index}]: empty context field")
        valid = False
    if not question:
        logger.warning(f"{dataset_name}[{index}]: empty question field")
        valid = False
    if not answer:
        logger.warning(f"{dataset_name}[{index}]: empty answer field")
        valid = False

    if not valid:
        return None

    return {"context": context, "question": question, "answer": answer}


def preprocess_ragbench_hotpotqa(
    sample: dict,
    index: int,
    dataset_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Preprocess ragbench-hotpotqa sample.

    RAG workflow: question triggers retrieval, then answer.
    Documents formatted as numbered list.
    """
    if logger is None:
        logger = module_logger

    question = sample.get("question", "")
    documents = sample.get("documents", [])
    response = sample.get("response", "")

    valid = True
    if not question:
        logger.warning(f"{dataset_name}[{index}]: empty question field")
        valid = False
    if not documents:
        logger.warning(f"{dataset_name}[{index}]: empty documents field")
        valid = False
    if not response:
        logger.warning(f"{dataset_name}[{index}]: empty response field")
        valid = False

    if not valid:
        return None

    # Format documents as numbered list
    doc_parts = []
    for i, doc in enumerate(documents, 1):
        doc_parts.append(f"[{i}] {doc}")
    documents_text = "\n".join(doc_parts)

    return {"question": question, "documents": documents_text, "response": response}
