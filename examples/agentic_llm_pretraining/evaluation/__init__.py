"""
LLM evaluation module for agentic LLM pretraining.

This module provides functions to evaluate model checkpoints using various
evaluation frameworks:

- lm_eval: lm-evaluation-harness for general benchmarks (hellaswag, arc, etc.)
- bfcl: Berkeley Function Calling Leaderboard for tool/function calling

Subpackages:
    lm_eval: Benchmarks using lm-evaluation-harness
    bfcl: Benchmarks using BFCL

Shared utilities:
    checkpoint: Functions for loading/converting MLPug checkpoints

Usage (lm-evaluation-harness):
    from examples.agentic_llm_pretraining.evaluation.lm_eval import (
        evaluate_checkpoint,
        DEFAULT_BENCHMARKS,
    )

    results = evaluate_checkpoint(
        checkpoint_path="/path/to/checkpoint.pt",
        tasks=DEFAULT_BENCHMARKS,
    )

Usage (BFCL):
    from examples.agentic_llm_pretraining.evaluation.bfcl import (
        evaluate_checkpoint,
        DEFAULT_TEST_CATEGORIES,
    )

    results = evaluate_checkpoint(
        checkpoint_path="/path/to/checkpoint.pt",
        test_categories=DEFAULT_TEST_CATEGORIES,
    )

CLI:
    # lm-evaluation-harness
    python -m examples.agentic_llm_pretraining.evaluation.lm_eval.evaluate_checkpoint \
        --checkpoint /path/to/checkpoint.pt

    # BFCL
    python -m examples.agentic_llm_pretraining.evaluation.bfcl.evaluate_checkpoint \
        --checkpoint /path/to/checkpoint.pt
"""
# Re-export checkpoint utilities for convenience
from examples.agentic_llm_pretraining.evaluation.checkpoint import (
    load_model_from_checkpoint,
    save_model_as_hf,
    convert_checkpoint_to_hf,
)

__all__ = [
    "load_model_from_checkpoint",
    "save_model_as_hf",
    "convert_checkpoint_to_hf",
]
