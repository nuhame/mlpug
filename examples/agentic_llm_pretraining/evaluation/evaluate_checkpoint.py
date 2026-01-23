"""
CLI script to evaluate model checkpoints using lm-evaluation-harness.

Usage:
    python -m examples.agentic_llm_pretraining.evaluation.evaluate_checkpoint \
        --checkpoint /path/to/checkpoint.pt \
        --output-dir /path/to/results

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
        description="Evaluate model checkpoint using lm-evaluation-harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file)",
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="HuggingFace model name for architecture",
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
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda, cpu)",
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
    checkpoint: str,
    model_name: str,
    tasks: list[str],
    batch_size: int,
    device: str,
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
    logger.info(f"  model_name: {model_name}")
    logger.info(f"  tasks: {tasks}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  device: {device}")
    logger.info(f"  num_fewshot: {num_fewshot}")
    logger.info(f"  limit: {limit}")
    logger.info(f"  output_dir: {output_dir}")


def main() -> None:
    """Main entry point for checkpoint evaluation."""
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

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(args.checkpoint).parent)

    # Create output filename based on checkpoint name
    checkpoint_name = Path(args.checkpoint).stem
    output_path = Path(output_dir) / f"{checkpoint_name}-eval-results.json"

    describe_config(
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        output_dir=output_dir,
        logger=module_logger,
    )

    # Run evaluation
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        output_path=str(output_path),
        logger=module_logger,
    )

    module_logger.info(f"Evaluation complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
