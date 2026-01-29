"""
BFCL (Berkeley Function Calling Leaderboard) based benchmarking.

This subpackage provides evaluation functions using the BFCL framework
for measuring function/tool calling capabilities.

Requirements:
    pip install bfcl-eval[oss_eval_vllm]  # For vLLM backend (recommended)
    # or
    pip install bfcl-eval[oss_eval_sglang]  # For SGLang backend (faster multi-turn)
"""
from examples.agentic_llm_pretraining.evaluation.bfcl.benchmarks import (
    DEFAULT_TEST_CATEGORIES,
    BASIC_TEST_CATEGORIES,
    ALL_SCORING_CATEGORIES,
    bfcl_generate,
    bfcl_evaluate,
    evaluate_checkpoint,
    get_results_summary,
)

__all__ = [
    "DEFAULT_TEST_CATEGORIES",
    "BASIC_TEST_CATEGORIES",
    "ALL_SCORING_CATEGORIES",
    "bfcl_generate",
    "bfcl_evaluate",
    "evaluate_checkpoint",
    "get_results_summary",
]
