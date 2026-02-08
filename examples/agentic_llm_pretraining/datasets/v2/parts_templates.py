"""
V2 parts templates for agentic LLM pretraining.

Templates define prompt/response split for loss masking during tokenization.

Template hierarchy:
- TemplateBase: Common base class for all templates
  - TextTemplate: Single text field (all tokens trained on)
  - SplitTemplate: prompt_template + response_template (prompt masked, response trained)
  - DialogueTemplate: Marker for dialogue datasets (uses format_chat_parts)

Design decisions:
- All formatting (spaces, newlines) goes in prompt_template
- response_template starts with actual content (no leading space/newline)
- Use explicit \\n instead of triple quotes to make whitespace unambiguous
"""

from dataclasses import dataclass


@dataclass
class TemplateBase:
    """Base class for all template types."""
    pass


@dataclass
class TextTemplate(TemplateBase):
    """Template for text-only datasets (no loss masking)."""
    text_template: str


@dataclass
class SplitTemplate(TemplateBase):
    """Template with prompt/response split for loss masking."""
    prompt_template: str
    response_template: str


@dataclass
class DialogueTemplate(TemplateBase):
    """Marker for dialogue datasets (uses format_chat_parts instead of templates)."""
    pass


# =============================================================================
# Text templates (train on all tokens)
# =============================================================================

FINEWEB_EDU_TEMPLATE = TextTemplate(
    text_template="{text}",
)

WIKIPEDIA_TEMPLATE = TextTemplate(
    text_template="# {title}\n\n{text}",
)

COSMOPEDIA_WIKIHOW_TEMPLATE = TextTemplate(
    text_template="{text}",
)

GENERICS_KB_TEMPLATE = TextTemplate(
    text_template="{text}",
)

CODESEARCHNET_TEMPLATE = TextTemplate(
    text_template="{text}",
)

# =============================================================================
# Q&A / Reasoning templates (prompt masked, response trained)
# =============================================================================

GSM8K_TEMPLATE = SplitTemplate(
    prompt_template="Problem: {question}\n\nSolution: ",
    response_template="{answer}",
)

MATH_TEMPLATE = SplitTemplate(
    prompt_template="Problem ({type}, {level}): {problem}\n\nSolution: ",
    response_template="{solution}",
)

OPENMATH_1_TEMPLATE = SplitTemplate(
    prompt_template="Problem: {question}\n\nCorrect generated solution: ",
    response_template="{generated_solution}",
)

OPENMATH_2_TEMPLATE = SplitTemplate(
    prompt_template="Problem: {problem}\n\nSolution: ",
    response_template="{generated_solution}",
)

ECQA_TEMPLATE = SplitTemplate(
    prompt_template=(
        "Question: {q_text}\n\n"
        "Options:\n"
        "A) {q_op1}\n"
        "B) {q_op2}\n"
        "C) {q_op3}\n"
        "D) {q_op4}\n"
        "E) {q_op5}\n\n"
        "Reasoning:\n"
    ),
    response_template=(
        "{taskA_pos}\n"
        "{taskA_neg}\n\n"
        "Conclusion: {taskB}\n\n"
        "Answer: {q_ans}"
    ),
)

STACKEXCHANGE_TEMPLATE = SplitTemplate(
    prompt_template="Question: {question}\n\n",
    response_template="{answers}",
)

OPENBOOKQA_TEMPLATE = SplitTemplate(
    prompt_template=(
        "Question: {question_stem}\n\n"
        "A) {choice_a}\n"
        "B) {choice_b}\n"
        "C) {choice_c}\n"
        "D) {choice_d}\n\n"
        "Answer: "
    ),
    response_template="{answer_key}) {answer_text}",
)

# =============================================================================
# Code templates (prompt masked, response trained)
# =============================================================================

SWE_BENCH_TEMPLATE = SplitTemplate(
    prompt_template=(
        "Repository: {repo}\n\n"
        "Issue: {problem_statement}\n\n"
        "Solution:\n"
    ),
    response_template="{patch}",
)

CODE_CONTESTS_TEMPLATE = SplitTemplate(
    prompt_template=(
        "Problem: {name}\n\n"
        "{description}\n\n"
        "Solution ({language}):\n"
    ),
    response_template="```{language_ext}\n{solution}\n```",
)

# =============================================================================
# RAG templates (prompt masked, response trained)
# =============================================================================

RAG_DATASET_TEMPLATE = SplitTemplate(
    prompt_template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer: ",
    response_template="{answer}",
)

RAGBENCH_TEMPLATE = SplitTemplate(
    prompt_template=(
        "Question: {question}\n\n"
        "Retrieved documents:\n"
        "{documents}\n\n"
        "Answer: "
    ),
    response_template="{response}",
)

# =============================================================================
# Dialogue template (marker for dialogue datasets)
# =============================================================================

# Shared instance for all dialogue datasets
# Note: empathetic-dialogues and samsum append tasks after dialogue,
# handled via suffix_prompt/suffix_response in their preprocessing functions.
DIALOGUE_TEMPLATE = DialogueTemplate()

# =============================================================================
# Registry
# =============================================================================

# Unified template registry mapping dataset names to their templates
TEMPLATES: dict[str, TemplateBase] = {
    # Text templates (no loss masking)
    "fineweb-edu": FINEWEB_EDU_TEMPLATE,
    "fineweb-edu-long": FINEWEB_EDU_TEMPLATE,
    "wikipedia": WIKIPEDIA_TEMPLATE,
    "simple-wikipedia": WIKIPEDIA_TEMPLATE,
    "cosmopedia-wikihow": COSMOPEDIA_WIKIHOW_TEMPLATE,
    "generics-kb": GENERICS_KB_TEMPLATE,
    "codesearchnet": CODESEARCHNET_TEMPLATE,
    # Split templates (prompt masked, response trained)
    # Reasoning/Math
    "gsm8k": GSM8K_TEMPLATE,
    "math": MATH_TEMPLATE,
    "openmath-instruct-1": OPENMATH_1_TEMPLATE,
    "openmath-instruct-2": OPENMATH_2_TEMPLATE,
    "ecqa": ECQA_TEMPLATE,
    # Procedural
    "stackexchange": STACKEXCHANGE_TEMPLATE,
    # Code
    "swe-bench": SWE_BENCH_TEMPLATE,
    "code-contests": CODE_CONTESTS_TEMPLATE,
    # Knowledge
    "openbookqa": OPENBOOKQA_TEMPLATE,
    # RAG
    "rag-dataset-12000": RAG_DATASET_TEMPLATE,
    "ragbench-hotpotqa": RAGBENCH_TEMPLATE,
    # Dialogue templates (role-based masking via format_chat_parts)
    "toolace": DIALOGUE_TEMPLATE,
    "hermes-function-calling": DIALOGUE_TEMPLATE,
    "glaive-function-calling": DIALOGUE_TEMPLATE,
    "soda": DIALOGUE_TEMPLATE,
    "multiwoz": DIALOGUE_TEMPLATE,
    "wizard-of-wikipedia": DIALOGUE_TEMPLATE,
    "sharegpt": DIALOGUE_TEMPLATE,
    "empathetic-dialogues": DIALOGUE_TEMPLATE,
    "samsum": DIALOGUE_TEMPLATE,
}
