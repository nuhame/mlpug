"""
lm-evaluation-harness based benchmarking.

This subpackage provides evaluation functions using lm-evaluation-harness.
"""
from examples.agentic_llm_pretraining.evaluation.lm_eval.benchmarks import (
    DEFAULT_BENCHMARKS,
    EXTENDED_BENCHMARKS,
    EXTENDED_BENCHMARKS_INCLUDING_MMLU,
    evaluate_checkpoint,
    evaluate_hf_model,
    evaluate_hf_model_distributed,
    evaluate_model,
    get_results_summary,
)

__all__ = [
    "DEFAULT_BENCHMARKS",
    "EXTENDED_BENCHMARKS",
    "EXTENDED_BENCHMARKS_INCLUDING_MMLU",
    "evaluate_checkpoint",
    "evaluate_hf_model",
    "evaluate_hf_model_distributed",
    "evaluate_model",
    "get_results_summary",
]
