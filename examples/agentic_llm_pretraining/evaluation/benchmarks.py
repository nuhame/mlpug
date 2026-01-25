"""
Core evaluation functions using lm-evaluation-harness.

This module provides reusable functions for evaluating language model checkpoints.
The functions are designed to be used both standalone and in training callbacks.
Supports both single-GPU and multi-GPU evaluation (via accelerate).

Requirements:
    pip install lm-eval accelerate

Usage (standalone, single GPU):
    from examples.agentic_llm_pretraining.evaluation.benchmarks import (
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
    from examples.agentic_llm_pretraining.evaluation.benchmarks import (
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


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


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    device: str = "cuda",
    logger: Optional[logging.Logger] = None,
) -> tuple:
    """
    Load a model from an MLPug checkpoint file.

    MLPug saves model checkpoints as state_dict files using torch.save().
    This function loads the state_dict into a fresh model instance.

    :param checkpoint_path: Path to the .pt checkpoint file.
    :param model_name: HuggingFace model name to initialize the architecture.
    :param device: Device to load the model to.
    :param logger: Optional logger for status messages.

    :return: Tuple of (model, tokenizer).
    """
    if logger:
        logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load the checkpoint
    # weights_only=False needed because MLPug checkpoints contain pickled objects
    # (e.g., MicroBatchResults in manager_state). This is safe for our own checkpoints.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # The checkpoint contains model state under 'model' key
    # (from MLPug's trainer.get_model_components() with convert_to_dict())
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # Assume it's a direct state_dict
        state_dict = checkpoint

    # Initialize model architecture
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)

    # Load the trained weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if logger:
        logger.info(f"Model loaded successfully on {device}")

    return model, tokenizer


def save_model_for_lm_eval(
    model,
    tokenizer,
    output_dir: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Save model in HuggingFace format for lm-evaluation-harness.

    lm-evaluation-harness works best with HuggingFace model paths.
    This function saves the model and tokenizer in the expected format.

    :param model: The PyTorch model to save.
    :param tokenizer: The tokenizer to save.
    :param output_dir: Directory to save the model.
    :param logger: Optional logger for status messages.

    :return: Path to the saved model directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(f"Saving model to {output_path}")

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    if logger:
        logger.info(f"Model saved successfully")

    return str(output_path)


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
    try:
        import lm_eval
        from lm_eval import simple_evaluate
    except ImportError:
        raise ImportError(
            "lm-evaluation-harness not installed. Install with: pip install lm-eval"
        )

    if logger:
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

    if logger:
        logger.info("Evaluation complete")
        _log_results_summary(results, logger)

    # Save results if output path specified
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if logger:
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
    # lm-eval CLI creates a directory at output_path and writes results.json inside
    # We use a temp directory, then copy results.json to user-specified output_path
    temp_output_dir = tempfile.mkdtemp(prefix="lm_eval_output_")

    if logger:
        logger.info(f"Launching multi-GPU evaluation with {num_gpus} GPUs")
        logger.info(f"Model: {model_path}")
        logger.info(f"Tasks: {tasks}")
        logger.info(f"Batch size: {batch_size}, dtype: {dtype}")

    # Build command
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
        "--log_samples", "False",
        "--confirm_run_unsafe_code",
    ]

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if logger:
        logger.info(f"Command: {' '.join(cmd)}")

    try:
        # Run subprocess, streaming output to main process stdout/stderr
        subprocess.run(
            cmd,
            check=True,  # Raise exception on non-zero exit
        )

        # Read results from temp directory
        # lm-eval writes results to temp_output_dir/results.json
        results_file = Path(temp_output_dir) / "results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found at {results_file}")

        with open(results_file) as f:
            results = json.load(f)

        # Copy results to user-specified output_path if provided
        if output_path is not None:
            shutil.copy(results_file, output_path)
            if logger:
                logger.info(f"Results saved to {output_path}")

        if logger:
            logger.info("Evaluation complete")
            _log_results_summary(results, logger)

        return results

    except subprocess.CalledProcessError as e:
        error_msg = f"Evaluation subprocess failed with exit code {e.returncode}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    except FileNotFoundError as e:
        error_msg = f"Results file not found: {e}"
        if logger:
            logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_output_dir, ignore_errors=True)


def evaluate_checkpoint(
    checkpoint_path: str,
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
    Evaluate an MLPug checkpoint using lm-evaluation-harness.

    This is the main entry point for checkpoint evaluation. It:
    1. Loads the checkpoint into a model
    2. Saves the model in HuggingFace format (temporary)
    3. Runs lm-evaluation-harness (single or multi-GPU)
    4. Cleans up temporary files (unless keep_temp_model=True)

    :param checkpoint_path: Path to the .pt checkpoint file.
    :param model_name: HuggingFace model name for architecture.
    :param tasks: List of benchmark tasks (default: DEFAULT_BENCHMARKS).
    :param batch_size: Batch size for evaluation.
    :param num_gpus: Number of GPUs to use (1 = single GPU, >1 = multi-GPU).
    :param dtype: Model dtype for loading (default: bfloat16).
    :param num_fewshot: Number of few-shot examples.
    :param limit: Sample limit per task (for quick testing).
    :param output_path: Path to save results JSON file.
    :param keep_temp_model: If True, don't delete temporary model directory.
    :param temp_model_dir: Custom directory for temporary model.
    :param logger: Optional logger.

    :return: Dictionary with evaluation results.
    """
    if tasks is None:
        tasks = DEFAULT_BENCHMARKS

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
        model_path = save_model_for_lm_eval(
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
            if logger:
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
    if tasks is None:
        tasks = DEFAULT_BENCHMARKS

    # Save to temporary directory
    with tempfile.TemporaryDirectory(prefix="lm_eval_model_") as temp_dir:
        model_path = save_model_for_lm_eval(
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
