"""
CLI script to evaluate model checkpoints or HuggingFace models using lm-evaluation-harness.

Usage (checkpoint, single GPU):
    python -m examples.agentic_llm_pretraining.evaluation.evaluate_checkpoint \
        --checkpoint /path/to/checkpoint.pt \
        --output-dir /path/to/results

Usage (HuggingFace model, single GPU):
    python -m examples.agentic_llm_pretraining.evaluation.evaluate_checkpoint \
        --hf-model Qwen/Qwen3-1.7B-Base \
        --output-dir /path/to/results

Multi-GPU evaluation (uses accelerate for data parallelism):
    python -m examples.agentic_llm_pretraining.evaluation.evaluate_checkpoint \
        --checkpoint /path/to/checkpoint.pt \
        --num-gpus 6 \
        --batch-size 24

Quick test (limited samples):
    python -m examples.agentic_llm_pretraining.evaluation.evaluate_checkpoint \
        --checkpoint /path/to/checkpoint.pt \
        --limit 10

Extended benchmarks including GSM8K:
    python -m examples.agentic_llm_pretraining.evaluation.evaluate_checkpoint \
        --checkpoint /path/to/checkpoint.pt \
        --extended
"""
import argparse
import os
import sys
from pathlib import Path

from basics.logging import get_logger

import mlpug.pytorch as mlp
from mlpug.utils.git_logging import log_git_state

from examples.agentic_llm_pretraining.evaluation.benchmarks import (
    evaluate_checkpoint,
    DEFAULT_BENCHMARKS,
    EXTENDED_BENCHMARKS,
    EXTENDED_BENCHMARKS_INCLUDING_MMLU,
)

module_logger = get_logger(os.path.basename(__file__))


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate model checkpoint or HuggingFace model using lm-evaluation-harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the model checkpoint (.pt file)",
    )
    model_group.add_argument(
        "--hf-model",
        type=str,
        help="HuggingFace model name to evaluate directly (e.g., Qwen/Qwen3-1.7B-Base)",
    )

    # Model configuration (only used with --checkpoint)
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="HuggingFace model name for architecture (only used with --checkpoint)",
    )

    # Benchmark selection
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific benchmark tasks to run (overrides --extended/--mmlu)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run extended benchmarks including GSM8K",
    )
    parser.add_argument(
        "--mmlu",
        action="store_true",
        help="Run benchmarks including MMLU (slow)",
    )

    # Evaluation settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (per GPU)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (1 = single GPU, >1 = multi-GPU with accelerate)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype for loading",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (None = task default)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples per task (for quick testing)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: same as checkpoint)",
    )

    return parser


def describe_config(
    checkpoint: str | None,
    hf_model: str | None,
    model_name: str,
    tasks: list[str],
    batch_size: int,
    num_gpus: int,
    dtype: str,
    num_fewshot: int | None,
    limit: int | None,
    output_dir: str | None,
    logger=None,
) -> None:
    """Log script configuration."""
    if logger is None:
        logger = module_logger

    logger.info("Configuration:")
    logger.info(f"  checkpoint: {checkpoint}")
    logger.info(f"  hf_model: {hf_model}")
    logger.info(f"  model_name: {model_name}")
    logger.info(f"  tasks: {tasks}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  num_gpus: {num_gpus}")
    logger.info(f"  dtype: {dtype}")
    logger.info(f"  num_fewshot: {num_fewshot}")
    logger.info(f"  limit: {limit}")
    logger.info(f"  output_dir: {output_dir}")


def main() -> None:
    """Main entry point for checkpoint or HuggingFace model evaluation."""
    mlp.logging.use_fancy_colors()
    log_git_state()

    parser = create_arg_parser()
    args = parser.parse_args()

    # Determine which tasks to run
    if args.tasks:
        tasks = args.tasks
    elif args.mmlu:
        tasks = EXTENDED_BENCHMARKS_INCLUDING_MMLU
    elif args.extended:
        tasks = EXTENDED_BENCHMARKS
    else:
        tasks = DEFAULT_BENCHMARKS

    # Determine output directory and filename
    if args.checkpoint:
        # Checkpoint mode
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = str(Path(args.checkpoint).parent)
        output_name = Path(args.checkpoint).stem
    else:
        # HuggingFace model mode
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = "."
        # Convert model name to safe filename (e.g., "Qwen/Qwen3-1.7B-Base" -> "Qwen--Qwen3-1.7B-Base")
        output_name = args.hf_model.replace("/", "--")

    output_path = Path(output_dir) / f"{output_name}-eval-results.json"

    describe_config(
        checkpoint=args.checkpoint,
        hf_model=args.hf_model,
        model_name=args.model_name,
        tasks=tasks,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        dtype=args.dtype,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        output_dir=output_dir,
        logger=module_logger,
    )

    # Run evaluation
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        hf_model=args.hf_model,
        model_name=args.model_name,
        tasks=tasks,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        dtype=args.dtype,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        output_path=str(output_path),
        logger=module_logger,
    )

    module_logger.info(f"Evaluation complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
