"""
Core evaluation functions using lm-evaluation-harness.

This module provides reusable functions for evaluating language model checkpoints.
The functions are designed to be used both standalone and in training callbacks.
Supports both single-GPU and multi-GPU evaluation (via accelerate).

Requirements:
    pip install lm-eval accelerate

Usage (standalone, single GPU):
    from examples.agentic_llm_pretraining.evaluation.lm_eval.benchmarks import (
        evaluate_checkpoint,
        DEFAULT_BENCHMARKS,
    )

    results = evaluate_checkpoint(
        checkpoint_path="/path/to/checkpoint.pt",
        model_name="Qwen/Qwen3-1.7B-Base",
        tasks=DEFAULT_BENCHMARKS,
    )

Usage (standalone, multi-GPU):
    results = evaluate_checkpoint(
        checkpoint_path="/path/to/checkpoint.pt",
        model_name="Qwen/Qwen3-1.7B-Base",
        tasks=DEFAULT_BENCHMARKS,
        num_gpus=6,  # Uses accelerate for data-parallel evaluation
    )

Usage (in callback):
    from examples.agentic_llm_pretraining.evaluation.lm_eval.benchmarks import (
        evaluate_model,
        DEFAULT_BENCHMARKS,
    )

    # model is already loaded and on device
    results = evaluate_model(model, tokenizer, tasks=DEFAULT_BENCHMARKS)
"""
from typing import Optional

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from basics.logging import get_logger

from examples.agentic_llm_pretraining.evaluation.checkpoint import (
    load_model_from_checkpoint,
    save_model_as_hf,
)

module_logger = get_logger(os.path.basename(__file__))


# Default benchmarks for quick evaluation
# These are common benchmarks available in lm-evaluation-harness
DEFAULT_BENCHMARKS = [
    "hellaswag",      # Commonsense reasoning
    "arc_easy",       # Science questions (easy)
    "arc_challenge",  # Science questions (hard)
    "winogrande",     # Coreference resolution
    "boolq",          # Boolean questions
    "truthfulqa_mc2", # Truthfulness
]

# Extended benchmarks including math reasoning
EXTENDED_BENCHMARKS = DEFAULT_BENCHMARKS + [
    "gsm8k",          # Grade school math
]

# Benchmarks including MMLU (slow but comprehensive)
EXTENDED_BENCHMARKS_INCLUDING_MMLU = EXTENDED_BENCHMARKS + [
    "mmlu",           # Multi-task language understanding
]


def evaluate_hf_model(
    model_path: str,
    tasks: list[str],
    batch_size: int = 8,
    device: str = "cuda",
    dtype: str = "bfloat16",
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Evaluate a HuggingFace model using lm-evaluation-harness.

    This is the core evaluation function that calls lm-evaluation-harness.

    :param model_path: Path to HuggingFace model directory or model name.
    :param tasks: List of benchmark task names.
    :param batch_size: Batch size for evaluation.
    :param device: Device to run evaluation on.
    :param dtype: Model dtype for loading (default: bfloat16).
    :param num_fewshot: Number of few-shot examples (None = use task default).
    :param limit: Limit number of samples per task (for quick testing).
    :param output_path: Optional path to save results JSON file.
    :param logger: Optional logger for status messages.

    :return: Dictionary with evaluation results.
    """
    if logger is None:
        logger = module_logger

    try:
        import lm_eval
        from lm_eval import simple_evaluate
    except ImportError:
        raise ImportError(
            "lm-evaluation-harness not installed. Install with: pip install lm-eval"
        )

    logger.info(f"Running lm-evaluation-harness on tasks: {tasks}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Batch size: {batch_size}, Device: {device}, dtype: {dtype}")
    if limit:
        logger.info(f"Sample limit per task: {limit}")

    # Run evaluation
    # confirm_run_unsafe_code=True needed for code evaluation tasks (humaneval, mbpp)
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},device={device},dtype={dtype}",
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=False,
        confirm_run_unsafe_code=True,
    )

    logger.info("Evaluation complete")
    _log_results_summary(results, logger)

    # Save results if output path specified
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    return results


def evaluate_hf_model_distributed(
    model_path: str,
    tasks: list[str],
    batch_size: int = 8,
    num_gpus: int = 1,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    dtype: str = "bfloat16",
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Evaluate a HuggingFace model using lm-evaluation-harness with multi-GPU support.

    For single GPU (num_gpus=1), uses the in-process evaluate_hf_model().
    For multiple GPUs, launches accelerate as a subprocess to distribute
    evaluation across all GPUs using data parallelism.

    :param model_path: Path to HuggingFace model directory or model name.
    :param tasks: List of benchmark task names.
    :param batch_size: Batch size for evaluation (per GPU).
    :param num_gpus: Number of GPUs to use (1 = single GPU, >1 = multi-GPU).
    :param num_fewshot: Number of few-shot examples (None = use task default).
    :param limit: Limit number of samples per task (for quick testing).
    :param output_path: Optional path to save results JSON file.
    :param dtype: Model dtype for loading (default: bfloat16).
    :param logger: Optional logger for status messages.

    :return: Dictionary with evaluation results.
    """
    if logger is None:
        logger = module_logger

    if num_gpus == 1:
        # Single GPU: use in-process evaluation
        return evaluate_hf_model(
            model_path=model_path,
            tasks=tasks,
            batch_size=batch_size,
            device="cuda",
            dtype=dtype,
            num_fewshot=num_fewshot,
            limit=limit,
            output_path=output_path,
            logger=logger,
        )

    # Multi-GPU: launch accelerate as subprocess
    # lm-eval CLI creates: output_path/MODEL_NAME/results_TIMESTAMP.json
    # We use a unique temp directory to guarantee we find only our results,
    # then copy to user's output_path if specified.
    temp_output_dir = tempfile.mkdtemp(prefix="lm_eval_output_")

    logger.info(f"Launching multi-GPU evaluation with {num_gpus} GPUs")
    logger.info(f"Model: {model_path}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Batch size: {batch_size}, dtype: {dtype}")

    # Build command
    # Note: --log_samples and --confirm_run_unsafe_code are boolean flags (no value)
    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes", str(num_gpus),
        "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype={dtype}",
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
        "--output_path", temp_output_dir,
        "--confirm_run_unsafe_code",
    ]

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    logger.info(f"Command: {' '.join(cmd)}")

    try:
        # Run subprocess, streaming output to main process stdout/stderr
        subprocess.run(
            cmd,
            check=True,  # Raise exception on non-zero exit
        )

        # Find results file in temp directory
        # lm-eval creates: temp_output_dir/MODEL_NAME/results_TIMESTAMP.json
        # Since temp_output_dir is unique, any results file must be ours
        results_files = list(Path(temp_output_dir).rglob("results_*.json"))

        if not results_files:
            raise FileNotFoundError(
                f"No results file found in {temp_output_dir}"
            )

        if len(results_files) > 1:
            # This should never happen with a unique temp directory
            logger.error(
                f"UNEXPECTED: Multiple results files found in temp directory: "
                f"{results_files}. This indicates a bug or race condition. "
                f"Using most recent file."
            )
            results_file = max(results_files, key=lambda p: p.stat().st_mtime)
        else:
            results_file = results_files[0]

        # Read results
        with open(results_file) as f:
            results = json.load(f)

        # Copy to user's output_path if specified
        if output_path is not None:
            shutil.copy(results_file, output_path)
            logger.info(f"Results saved to {output_path}")

        logger.info("Evaluation complete")
        _log_results_summary(results, logger)

        return results

    except subprocess.CalledProcessError as e:
        error_msg = f"Evaluation subprocess failed with exit code {e.returncode}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    except FileNotFoundError as e:
        error_msg = f"Results file not found: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_output_dir, ignore_errors=True)


def evaluate_checkpoint(
    checkpoint_path: Optional[str] = None,
    hf_model: Optional[str] = None,
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    tasks: Optional[list[str]] = None,
    batch_size: int = 8,
    num_gpus: int = 1,
    dtype: str = "bfloat16",
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    keep_temp_model: bool = False,
    temp_model_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Evaluate an MLPug checkpoint or HuggingFace model using lm-evaluation-harness.

    This is the main entry point for model evaluation. Provide either:
    - checkpoint_path: Loads the checkpoint, converts to HF format, then evaluates
    - hf_model: Evaluates the HuggingFace model directly (no conversion needed)

    :param checkpoint_path: Path to the .pt checkpoint file (mutually exclusive with hf_model).
    :param hf_model: HuggingFace model name to evaluate directly (mutually exclusive with checkpoint_path).
    :param model_name: HuggingFace model name for architecture (only used with checkpoint_path).
    :param tasks: List of benchmark tasks (default: DEFAULT_BENCHMARKS).
    :param batch_size: Batch size for evaluation.
    :param num_gpus: Number of GPUs to use (1 = single GPU, >1 = multi-GPU).
    :param dtype: Model dtype for loading (default: bfloat16).
    :param num_fewshot: Number of few-shot examples.
    :param limit: Sample limit per task (for quick testing).
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

    if tasks is None:
        tasks = DEFAULT_BENCHMARKS

    # HuggingFace model mode: evaluate directly without conversion
    if hf_model is not None:
        return evaluate_hf_model_distributed(
            model_path=hf_model,
            tasks=tasks,
            batch_size=batch_size,
            num_gpus=num_gpus,
            dtype=dtype,
            num_fewshot=num_fewshot,
            limit=limit,
            output_path=output_path,
            logger=logger,
        )

    # Checkpoint mode: load, convert to HF format, evaluate
    # For multi-GPU, load on CPU to avoid CUDA init before subprocess spawn
    # For single GPU, load directly on CUDA
    load_device = "cpu" if num_gpus > 1 else "cuda"

    # Load checkpoint
    model, tokenizer = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=load_device,
        logger=logger,
    )

    # Save to temporary HuggingFace format
    if temp_model_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="lm_eval_model_")
    else:
        temp_dir = temp_model_dir
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

    try:
        model_path = save_model_as_hf(
            model=model,
            tokenizer=tokenizer,
            output_dir=temp_dir,
            logger=logger,
        )

        # Free memory before evaluation
        del model
        if load_device == "cuda":
            torch.cuda.empty_cache()

        # Run evaluation (single or multi-GPU)
        results = evaluate_hf_model_distributed(
            model_path=model_path,
            tasks=tasks,
            batch_size=batch_size,
            num_gpus=num_gpus,
            dtype=dtype,
            num_fewshot=num_fewshot,
            limit=limit,
            output_path=output_path,
            logger=logger,
        )

        return results

    finally:
        # Cleanup temporary model directory
        if not keep_temp_model and temp_model_dir is None:
            logger.info(f"Cleaning up temporary model directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


def evaluate_model(
    model,
    tokenizer,
    tasks: Optional[list[str]] = None,
    batch_size: int = 8,
    num_gpus: int = 1,
    dtype: str = "bfloat16",
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Evaluate an already-loaded model using lm-evaluation-harness.

    This function is designed for use in training callbacks where the model
    is already loaded in memory. It temporarily saves the model to disk
    for lm-evaluation-harness.

    :param model: PyTorch model (already loaded).
    :param tokenizer: Tokenizer for the model.
    :param tasks: List of benchmark tasks.
    :param batch_size: Batch size for evaluation.
    :param num_gpus: Number of GPUs to use (1 = single GPU, >1 = multi-GPU).
    :param dtype: Model dtype for loading (default: bfloat16).
    :param num_fewshot: Number of few-shot examples.
    :param limit: Sample limit per task.
    :param output_path: Path to save results JSON file.
    :param logger: Optional logger.

    :return: Dictionary with evaluation results.
    """
    if logger is None:
        logger = module_logger

    if tasks is None:
        tasks = DEFAULT_BENCHMARKS

    # Save to temporary directory
    with tempfile.TemporaryDirectory(prefix="lm_eval_model_") as temp_dir:
        model_path = save_model_as_hf(
            model=model,
            tokenizer=tokenizer,
            output_dir=temp_dir,
            logger=logger,
        )

        # Run evaluation (single or multi-GPU)
        results = evaluate_hf_model_distributed(
            model_path=model_path,
            tasks=tasks,
            batch_size=batch_size,
            num_gpus=num_gpus,
            dtype=dtype,
            num_fewshot=num_fewshot,
            limit=limit,
            output_path=output_path,
            logger=logger,
        )

        return results


def _find_metric_key(task_results: dict, metric: str) -> tuple:
    """
    Find the best matching key for a metric in task results.

    lm-evaluation-harness uses various formats:
    - "metric,none" (e.g., "acc_norm,none")
    - "metric,filter" (e.g., "exact_match,flexible-extract")
    - "metric" (legacy format)

    :param task_results: Dictionary of results for a task.
    :param metric: The metric name to search for (e.g., "acc_norm", "exact_match").

    :return: Tuple of (full_key, value) or (None, None) if not found.
    """
    # Preferred suffixes in order (more lenient/informative first)
    preferred_suffixes = ['flexible-extract', 'none', 'strict-match']

    # First, try preferred formats in order
    for suffix in preferred_suffixes:
        key = f'{metric},{suffix}'
        if key in task_results:
            return key, task_results[key]

    # Then, search for any key starting with the metric (excluding stderr)
    for key, value in task_results.items():
        if key.startswith(f'{metric},') and 'stderr' not in key:
            return key, value

    # Finally, try bare metric name (legacy format)
    if metric in task_results:
        return metric, task_results[metric]

    return None, None


def _log_results_summary(results: dict, logger: logging.Logger) -> None:
    """Log a summary of evaluation results."""
    if 'results' not in results:
        return

    logger.info("=" * 60)
    logger.info("Evaluation Results Summary")
    logger.info("=" * 60)

    for task_name, task_results in results['results'].items():
        # Get the main accuracy metric
        # Try metrics in order of preference
        found_key = None
        found_value = None
        for metric in ['acc_norm', 'acc', 'exact_match', 'mc2']:
            found_key, found_value = _find_metric_key(task_results, metric)
            if found_key is not None:
                break

        if found_key is not None and found_value is not None:
            if isinstance(found_value, float):
                logger.info(f"  {task_name}: {found_value:.4f} ({found_key})")
            else:
                logger.info(f"  {task_name}: {found_value} ({found_key})")

    logger.info("=" * 60)


def get_results_summary(results: dict) -> dict:
    """
    Extract a summary of key metrics from evaluation results.

    :param results: Full results dictionary from lm-evaluation-harness.

    :return: Dictionary mapping task names to their primary accuracy metric.
    """
    summary = {}

    if 'results' not in results:
        return summary

    for task_name, task_results in results['results'].items():
        # Get the main accuracy metric
        for metric in ['acc_norm', 'acc', 'exact_match', 'mc2']:
            found_key, found_value = _find_metric_key(task_results, metric)
            if found_key is not None:
                summary[task_name] = {
                    'metric': found_key,
                    'value': found_value,
                }
                break

    return summary
