"""
Dataset templates for agentic LLM pretraining.

Templates define the output format structure using {field} placeholders.
Preprocess functions (in preprocessing.py) extract the fields from raw data.

See transform_analysis.md for detailed template specifications.
"""

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
# Registry
# =============================================================================

TEMPLATES = {
    "fineweb-edu": FINEWEB_EDU_TEMPLATE,
    "gsm8k": GSM8K_TEMPLATE,
    "soda": PASSTHROUGH_TEMPLATE,
    "toolace": PASSTHROUGH_TEMPLATE,
}
