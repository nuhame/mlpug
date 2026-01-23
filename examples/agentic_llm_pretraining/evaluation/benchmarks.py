"""
Core evaluation functions using lm-evaluation-harness.

This module provides reusable functions for evaluating language model checkpoints.
The functions are designed to be used both standalone and in training callbacks.

Requirements:
    pip install lm-eval

Usage (standalone):
    from examples.agentic_llm_pretraining.evaluation.benchmarks import (
        evaluate_checkpoint,
        DEFAULT_BENCHMARKS,
    )

    results = evaluate_checkpoint(
        checkpoint_path="/path/to/checkpoint.pt",
        model_name="Qwen/Qwen3-1.7B-Base",
        tasks=DEFAULT_BENCHMARKS,
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

import os
import logging
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
    :param num_fewshot: Number of few-shot examples (None = use task default).
    :param limit: Limit number of samples per task (for quick testing).
    :param output_path: Optional path to save results JSON.
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
        logger.info(f"Batch size: {batch_size}, Device: {device}")
        if limit:
            logger.info(f"Sample limit per task: {limit}")

    # Run evaluation
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},device={device}",
        tasks=tasks,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=False,
    )

    if logger:
        logger.info("Evaluation complete")
        _log_results_summary(results, logger)

    # Save results if output path specified
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        if logger:
            logger.info(f"Results saved to {output_path}")

    return results


def evaluate_checkpoint(
    checkpoint_path: str,
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    tasks: Optional[list[str]] = None,
    batch_size: int = 8,
    device: str = "cuda",
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
    3. Runs lm-evaluation-harness
    4. Cleans up temporary files (unless keep_temp_model=True)

    :param checkpoint_path: Path to the .pt checkpoint file.
    :param model_name: HuggingFace model name for architecture.
    :param tasks: List of benchmark tasks (default: DEFAULT_BENCHMARKS).
    :param batch_size: Batch size for evaluation.
    :param device: Device to run on.
    :param num_fewshot: Number of few-shot examples.
    :param limit: Sample limit per task (for quick testing).
    :param output_path: Path to save results JSON.
    :param keep_temp_model: If True, don't delete temporary model directory.
    :param temp_model_dir: Custom directory for temporary model.
    :param logger: Optional logger.

    :return: Dictionary with evaluation results.
    """
    if tasks is None:
        tasks = DEFAULT_BENCHMARKS

    # Load checkpoint
    model, tokenizer = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
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

        # Free GPU memory before evaluation
        del model
        torch.cuda.empty_cache()

        # Run evaluation
        results = evaluate_hf_model(
            model_path=model_path,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            num_fewshot=num_fewshot,
            limit=limit,
            output_path=output_path,
            logger=logger,
        )

        return results

    finally:
        # Cleanup temporary model directory
        if not keep_temp_model and temp_model_dir is None:
            import shutil
            if logger:
                logger.info(f"Cleaning up temporary model directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


def evaluate_model(
    model,
    tokenizer,
    tasks: Optional[list[str]] = None,
    batch_size: int = 8,
    device: str = "cuda",
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
    :param device: Device to run on.
    :param num_fewshot: Number of few-shot examples.
    :param limit: Sample limit per task.
    :param output_path: Path to save results JSON.
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

        # Run evaluation
        results = evaluate_hf_model(
            model_path=model_path,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            num_fewshot=num_fewshot,
            limit=limit,
            output_path=output_path,
            logger=logger,
        )

        return results


def _log_results_summary(results: dict, logger: logging.Logger) -> None:
    """Log a summary of evaluation results."""
    if 'results' not in results:
        return

    logger.info("=" * 60)
    logger.info("Evaluation Results Summary")
    logger.info("=" * 60)

    for task_name, task_results in results['results'].items():
        # Get the main accuracy metric
        # lm-evaluation-harness uses "metric,aggregation" format (e.g., "acc_norm,none")
        acc_key = None
        acc_value = None
        for metric in ['acc_norm', 'acc', 'exact_match', 'mc2']:
            # Try both formats: "metric,none" and "metric"
            for key_format in [f'{metric},none', metric]:
                if key_format in task_results:
                    acc_key = metric
                    acc_value = task_results[key_format]
                    break
            if acc_key:
                break

        if acc_key and acc_value is not None:
            if isinstance(acc_value, float):
                logger.info(f"  {task_name}: {acc_value:.4f} ({acc_key})")
            else:
                logger.info(f"  {task_name}: {acc_value} ({acc_key})")

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
        # lm-evaluation-harness uses "metric,aggregation" format (e.g., "acc_norm,none")
        for metric in ['acc_norm', 'acc', 'exact_match', 'mc2']:
            # Try both formats: "metric,none" and "metric"
            for key_format in [f'{metric},none', metric]:
                if key_format in task_results:
                    summary[task_name] = {
                        'metric': metric,
                        'value': task_results[key_format],
                    }
                    break
            if task_name in summary:
                break

    return summary
