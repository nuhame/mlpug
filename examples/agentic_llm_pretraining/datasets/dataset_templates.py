"""
Dataset templates for agentic LLM pretraining.

Templates define the output format structure using {field} placeholders.
Preprocess functions (in preprocessing.py) extract the fields from raw data.

See transform_analysis.md for detailed template specifications.
"""

# =============================================================================
# Templates
# =============================================================================

# --- Passthrough templates ---
# For raw text or chat-formatted datasets (format_chat() in preprocess)
PASSTHROUGH_TEMPLATE = "{text}"

# --- Grammar/Language ---
FINEWEB_EDU_TEMPLATE = "{text}"

WIKIPEDIA_TEMPLATE = """
# {title}

{text}
"""

# --- Reasoning/Math ---
GSM8K_TEMPLATE = """
Problem: {question}

Solution: {answer}
"""

MATH_TEMPLATE = """
Problem ({type}, {level}): {problem}

Solution: {solution}
"""

OPENMATH_1_TEMPLATE = """
Problem: {question}

Correct generated solution: {generated_solution}
"""

OPENMATH_2_TEMPLATE = """
Problem: {problem}

Solution: {generated_solution}
"""

ECQA_TEMPLATE = """
Question: {q_text}

Options:
A) {q_op1}
B) {q_op2}
C) {q_op3}
D) {q_op4}
E) {q_op5}

Reasoning:
{taskA_pos}
{taskA_neg}

Conclusion: {taskB}

Answer: {q_ans}
"""

# --- Procedural ---
# cosmopedia-wikihow uses PASSTHROUGH_TEMPLATE

STACKEXCHANGE_TEMPLATE = """
Question: {question}

{answers}
"""

# --- Code ---
SWE_BENCH_TEMPLATE = """
Repository: {repo}

Issue: {problem_statement}

Solution:
{patch}
"""

CODE_CONTESTS_TEMPLATE = """
Problem: {name}

{description}

Examples:
{examples}

Solution ({language}):
```{language_ext}
{solution}
```
"""

# codesearchnet uses PASSTHROUGH_TEMPLATE

# --- Knowledge ---
# generics-kb uses PASSTHROUGH_TEMPLATE

OPENBOOKQA_TEMPLATE = """
Question: {question_stem}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Answer: {answer_key}) {answer_text}
"""

# --- RAG ---
RAG_DATASET_TEMPLATE = """
Context:
{context}

Question: {question}

Answer: {answer}
"""

RAGBENCH_TEMPLATE = """
Question: {question}

Retrieved documents:
{documents}

Answer: {response}
"""


# =============================================================================
# Registry
# =============================================================================

TEMPLATES = {
    # Grammar/Language
    "fineweb-edu": FINEWEB_EDU_TEMPLATE,
    "wikipedia": WIKIPEDIA_TEMPLATE,
    "simple-wikipedia": WIKIPEDIA_TEMPLATE,
    # Reasoning/Math
    "gsm8k": GSM8K_TEMPLATE,
    "math": MATH_TEMPLATE,
    "openmath-instruct-1": OPENMATH_1_TEMPLATE,
    "openmath-instruct-2": OPENMATH_2_TEMPLATE,
    "ecqa": ECQA_TEMPLATE,
    # Procedural
    "cosmopedia-wikihow": PASSTHROUGH_TEMPLATE,
    "stackexchange": STACKEXCHANGE_TEMPLATE,
    # Code
    "swe-bench": SWE_BENCH_TEMPLATE,
    "code-contests": CODE_CONTESTS_TEMPLATE,
    "codesearchnet": PASSTHROUGH_TEMPLATE,
    # Agentic (chat-formatted via preprocess)
    "toolace": PASSTHROUGH_TEMPLATE,
    "hermes-function-calling": PASSTHROUGH_TEMPLATE,
    "glaive-function-calling": PASSTHROUGH_TEMPLATE,
    # Knowledge
    "generics-kb": PASSTHROUGH_TEMPLATE,
    "openbookqa": OPENBOOKQA_TEMPLATE,
    # Dialogue (chat-formatted via preprocess)
    "soda": PASSTHROUGH_TEMPLATE,
    "multiwoz": PASSTHROUGH_TEMPLATE,
    "wizard-of-wikipedia": PASSTHROUGH_TEMPLATE,
    "sharegpt": PASSTHROUGH_TEMPLATE,
    "empathetic-dialogues": PASSTHROUGH_TEMPLATE,
    "samsum": PASSTHROUGH_TEMPLATE,
    # RAG
    "rag-dataset-12000": RAG_DATASET_TEMPLATE,
    "ragbench-hotpotqa": RAGBENCH_TEMPLATE,
}
