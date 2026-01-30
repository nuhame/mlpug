"""
Core evaluation functions using BFCL (Berkeley Function Calling Leaderboard).

This module provides functions for evaluating language model checkpoints
on function/tool calling benchmarks using the BFCL framework.

WARNING: This module has not been tested yet. The implementation is based on
BFCL documentation and may require adjustments. Test with a small category
(e.g., simple_python) before running comprehensive evaluations.

Requirements:
    pip install bfcl-eval[oss_eval_vllm]  # For vLLM backend
    # or
    pip install bfcl-eval[oss_eval_sglang]  # For SGLang backend

Usage (standalone):
    from examples.agentic_llm_pretraining.evaluation.bfcl.benchmarks import (
        evaluate_checkpoint,
        DEFAULT_TEST_CATEGORIES,
    )

    results = evaluate_checkpoint(
        checkpoint_path="/path/to/checkpoint.pt",
        model_name="Qwen/Qwen3-1.7B-Base",
        test_categories=DEFAULT_TEST_CATEGORIES,
    )

Note:
    BFCL uses vLLM or SGLang as the inference backend. The model must be
    converted to HuggingFace format before evaluation.
"""
from typing import Optional

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from basics.logging import get_logger

from examples.agentic_llm_pretraining.evaluation.checkpoint import (
    convert_checkpoint_to_hf,
)

module_logger = get_logger(os.path.basename(__file__))


# Basic function calling test categories (recommended for initial testing)
BASIC_TEST_CATEGORIES = [
    "simple_python",      # Basic function calls (Python)
    "simple_java",        # Basic function calls (Java)
    "simple_javascript",  # Basic function calls (JavaScript)
]

# Default test categories for comprehensive function calling evaluation
DEFAULT_TEST_CATEGORIES = BASIC_TEST_CATEGORIES + [
    "parallel",   # Concurrent function execution
    "multiple",   # Sequential function calls
]

# All scoring categories (comprehensive evaluation)
ALL_SCORING_CATEGORIES = "all_scoring"


def bfcl_generate(
    model_path: str,
    test_category: str | list[str],
    templates_model_name: str = "Qwen/Qwen3-1.7B",
    backend: str = "vllm",
    num_gpus: int = 1,
    gpu_memory_utilization: float = 0.9,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Generate responses for BFCL evaluation using a local model.

    This runs `bfcl generate` to create model responses for evaluation.

    :param model_path: Path to HuggingFace model directory containing weights.
    :param test_category: Test category or list of categories to evaluate.
        Use "all_scoring" for all scoring categories.
    :param templates_model_name: Model name whose prompt templates to use.
        Must be recognized by BFCL (e.g., "Qwen/Qwen3-1.7B").
        The actual weights come from model_path via --local-model-path.
    :param backend: Inference backend ("vllm" or "sglang").
    :param num_gpus: Number of GPUs to use.
    :param gpu_memory_utilization: GPU memory utilization (0.0-1.0).
    :param output_dir: Directory for BFCL output (default: temp directory).
    :param logger: Optional logger for status messages.

    :return: Path to the output directory containing results.
    """
    if logger is None:
        logger = module_logger

    # Handle test_category as string or list
    if isinstance(test_category, list):
        categories = test_category
    else:
        categories = [test_category]

    # Set up output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="bfcl_output_")

    logger.info(f"Running BFCL generate")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Templates model: {templates_model_name}")
    logger.info(f"Test categories: {categories}")
    logger.info(f"Backend: {backend}, GPUs: {num_gpus}")

    # Build command for each category
    for category in categories:
        cmd = [
            "bfcl", "generate",
            "--model", templates_model_name,  # Model whose templates to use
            "--test-category", category,
            "--backend", backend,
            "--num-gpus", str(num_gpus),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--local-model-path", model_path,  # Actual path to model weights
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        # Set BFCL_PROJECT_ROOT to our output directory
        env = os.environ.copy()
        env["BFCL_PROJECT_ROOT"] = output_dir

        try:
            subprocess.run(
                cmd,
                check=True,
                env=env,
            )
            logger.info(f"Generate completed for category: {category}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Generate failed for category {category}: {e}")
            raise

    return output_dir


def bfcl_evaluate(
    test_category: str | list[str],
    output_dir: str,
    templates_model_name: str = "Qwen/Qwen3-1.7B",
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Evaluate generated BFCL responses.

    This runs `bfcl evaluate` on previously generated responses.

    :param test_category: Test category or list of categories to evaluate.
    :param output_dir: Directory containing BFCL output from bfcl_generate().
    :param templates_model_name: Model name (must match bfcl_generate).
    :param logger: Optional logger for status messages.

    :return: Dictionary with evaluation results.
    """
    if logger is None:
        logger = module_logger

    # Handle test_category as string or list
    if isinstance(test_category, list):
        categories = test_category
    else:
        categories = [test_category]

    logger.info(f"Running BFCL evaluate")
    logger.info(f"Templates model: {templates_model_name}")
    logger.info(f"Test categories: {categories}")

    results = {}

    for category in categories:
        cmd = [
            "bfcl", "evaluate",
            "--model", templates_model_name,  # Must match the model used in generate
            "--test-category", category,
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        # Set BFCL_PROJECT_ROOT to our output directory
        env = os.environ.copy()
        env["BFCL_PROJECT_ROOT"] = output_dir

        try:
            subprocess.run(
                cmd,
                check=True,
                env=env,
            )
            logger.info(f"Evaluate completed for category: {category}")

            # Try to read the score file
            # BFCL creates: score/MODEL_NAME/BFCL_v3_TEST_CATEGORY_score.json
            # Note: templates_model_name may contain "/" which becomes directory separator
            score_file = Path(output_dir) / "score" / templates_model_name / f"BFCL_v3_{category}_score.json"

            if score_file.exists():
                with open(score_file) as f:
                    results[category] = json.load(f)
            else:
                logger.warning(f"Score file not found: {score_file}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluate failed for category {category}: {e}")
            raise

    return results


def evaluate_checkpoint(
    checkpoint_path: Optional[str] = None,
    hf_model: Optional[str] = None,
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    templates_model_name: str = "Qwen/Qwen3-1.7B",
    test_categories: str | list[str] | None = None,
    backend: str = "vllm",
    num_gpus: int = 1,
    gpu_memory_utilization: float = 0.9,
    output_path: Optional[str] = None,
    keep_temp_model: bool = False,
    temp_model_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Evaluate an MLPug checkpoint or HuggingFace model using BFCL.

    This is the main entry point for BFCL evaluation. Provide either:
    - checkpoint_path: Loads the checkpoint, converts to HF format, then evaluates
    - hf_model: Evaluates the HuggingFace model directly (no conversion needed)

    :param checkpoint_path: Path to the .pt checkpoint file (mutually exclusive with hf_model).
    :param hf_model: HuggingFace model name/path to evaluate directly (mutually exclusive with checkpoint_path).
    :param model_name: HuggingFace model name for architecture (only used with checkpoint_path).
    :param templates_model_name: Model name whose prompt templates to use.
        Must be recognized by BFCL (e.g., "Qwen/Qwen3-1.7B"). The actual weights
        come from the checkpoint or hf_model.
    :param test_categories: Test category or list of categories (default: DEFAULT_TEST_CATEGORIES).
    :param backend: Inference backend ("vllm" or "sglang").
    :param num_gpus: Number of GPUs to use.
    :param gpu_memory_utilization: GPU memory utilization (0.0-1.0).
    :param output_path: Path to save results JSON file.
    :param keep_temp_model: If True, don't delete temporary model directory (only used with checkpoint_path).
    :param temp_model_dir: Custom directory for temporary model (only used with checkpoint_path).
    :param logger: Optional logger.

    :return: Dictionary with evaluation results.
    """
    if logger is None:
        logger = module_logger

    if checkpoint_path is None and hf_model is None:
        raise ValueError("Must provide either checkpoint_path or hf_model")
    if checkpoint_path is not None and hf_model is not None:
        raise ValueError("Cannot provide both checkpoint_path and hf_model")

    if test_categories is None:
        test_categories = DEFAULT_TEST_CATEGORIES

    # Determine model path
    temp_dir = None
    cleanup_temp = False

    if hf_model is not None:
        # HuggingFace model: use directly
        model_path = hf_model
    else:
        # Checkpoint: convert to HF format
        if temp_model_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="bfcl_model_")
            cleanup_temp = not keep_temp_model
        else:
            temp_dir = temp_model_dir
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

        model_path = convert_checkpoint_to_hf(
            checkpoint_path=checkpoint_path,
            output_dir=temp_dir,
            model_name=model_name,
            device="cpu",  # Use CPU to minimize memory
            logger=logger,
        )

    # Create temp directory for BFCL output
    bfcl_output_dir = tempfile.mkdtemp(prefix="bfcl_output_")

    try:
        # Generate responses
        bfcl_generate(
            model_path=model_path,
            test_category=test_categories,
            templates_model_name=templates_model_name,
            backend=backend,
            num_gpus=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            output_dir=bfcl_output_dir,
            logger=logger,
        )

        # Evaluate responses
        results = bfcl_evaluate(
            test_category=test_categories,
            output_dir=bfcl_output_dir,
            templates_model_name=templates_model_name,
            logger=logger,
        )

        # Log summary
        _log_results_summary(results, logger)

        # Save results if output path specified
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results

    finally:
        # Cleanup temporary directories
        if cleanup_temp and temp_dir:
            logger.info(f"Cleaning up temporary model directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Always cleanup BFCL output directory
        shutil.rmtree(bfcl_output_dir, ignore_errors=True)


def _log_results_summary(results: dict, logger: logging.Logger) -> None:
    """Log a summary of BFCL evaluation results."""
    if not results:
        return

    logger.info("=" * 60)
    logger.info("BFCL Evaluation Results Summary")
    logger.info("=" * 60)

    for category, category_results in results.items():
        if isinstance(category_results, dict):
            # Try to find accuracy metric
            accuracy = category_results.get('accuracy', category_results.get('ast_accuracy'))
            if accuracy is not None:
                logger.info(f"  {category}: {accuracy:.4f}")
            else:
                logger.info(f"  {category}: {category_results}")
        else:
            logger.info(f"  {category}: {category_results}")

    logger.info("=" * 60)


def get_results_summary(results: dict) -> dict:
    """
    Extract a summary of key metrics from BFCL evaluation results.

    :param results: Full results dictionary from BFCL evaluation.

    :return: Dictionary mapping categories to their primary accuracy metric.
    """
    summary = {}

    for category, category_results in results.items():
        if isinstance(category_results, dict):
            accuracy = category_results.get('accuracy', category_results.get('ast_accuracy'))
            if accuracy is not None:
                summary[category] = {
                    'metric': 'accuracy',
                    'value': accuracy,
                }

    return summary
