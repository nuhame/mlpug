"""
CLI script to evaluate model checkpoints or HuggingFace models using BFCL.

Usage (checkpoint):
    python -m examples.agentic_llm_pretraining.evaluation.bfcl.evaluate_checkpoint \
        --checkpoint /path/to/checkpoint.pt \
        --output-dir /path/to/results

Usage (HuggingFace model):
    python -m examples.agentic_llm_pretraining.evaluation.bfcl.evaluate_checkpoint \
        --hf-model Qwen/Qwen3-1.7B-Base \
        --output-dir /path/to/results

Basic test (simple function calls only):
    python -m examples.agentic_llm_pretraining.evaluation.bfcl.evaluate_checkpoint \
        --checkpoint /path/to/checkpoint.pt \
        --basic

All scoring categories:
    python -m examples.agentic_llm_pretraining.evaluation.bfcl.evaluate_checkpoint \
        --checkpoint /path/to/checkpoint.pt \
        --all-scoring

Requirements:
    pip install bfcl-eval[oss_eval_vllm]  # For vLLM backend
"""
import argparse
import os
from pathlib import Path

from basics.logging import get_logger

import mlpug.pytorch as mlp
from mlpug.utils.git_logging import log_git_state

from examples.agentic_llm_pretraining.evaluation.bfcl.benchmarks import (
    evaluate_checkpoint,
    DEFAULT_TEST_CATEGORIES,
    BASIC_TEST_CATEGORIES,
    ALL_SCORING_CATEGORIES,
)

module_logger = get_logger(os.path.basename(__file__))


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for BFCL evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate model checkpoint or HuggingFace model using BFCL",
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
        help="HuggingFace model name/path to evaluate directly",
    )

    # Model configuration (only used with --checkpoint)
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="HuggingFace model name for architecture (only used with --checkpoint)",
    )
    parser.add_argument(
        "--templates-model-name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name whose prompt templates to use (must be BFCL-supported)",
    )

    # Test category selection
    parser.add_argument(
        "--test-categories",
        type=str,
        nargs="+",
        default=None,
        help="Specific test categories to run (overrides --basic/--all-scoring)",
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Run basic test categories only (simple_python, simple_java, simple_javascript)",
    )
    parser.add_argument(
        "--all-scoring",
        action="store_true",
        help="Run all scoring categories (comprehensive)",
    )

    # Evaluation settings
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "sglang"],
        help="Inference backend",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: same as checkpoint)",
    )

    # Sample capture
    parser.add_argument(
        "--capture-samples",
        action="store_true",
        help="Capture sample prompts and model responses for analysis",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to capture per category (only used with --capture-samples)",
    )

    return parser


def describe_config(
    checkpoint: str | None,
    hf_model: str | None,
    model_name: str,
    templates_model_name: str,
    test_categories: str | list[str],
    backend: str,
    num_gpus: int,
    gpu_memory_utilization: float,
    output_dir: str | None,
    capture_samples: bool,
    num_samples: int,
    logger=None,
) -> None:
    """Log script configuration."""
    if logger is None:
        logger = module_logger

    logger.info("Configuration:")
    logger.info(f"  checkpoint: {checkpoint}")
    logger.info(f"  hf_model: {hf_model}")
    logger.info(f"  model_name: {model_name}")
    logger.info(f"  templates_model_name: {templates_model_name}")
    logger.info(f"  test_categories: {test_categories}")
    logger.info(f"  backend: {backend}")
    logger.info(f"  num_gpus: {num_gpus}")
    logger.info(f"  gpu_memory_utilization: {gpu_memory_utilization}")
    logger.info(f"  output_dir: {output_dir}")
    logger.info(f"  capture_samples: {capture_samples}")
    logger.info(f"  num_samples: {num_samples}")


def main() -> None:
    """Main entry point for BFCL evaluation."""
    mlp.logging.use_fancy_colors()
    log_git_state()

    parser = create_arg_parser()
    args = parser.parse_args()

    # Determine which test categories to run
    if args.test_categories:
        test_categories = args.test_categories
    elif args.all_scoring:
        test_categories = ALL_SCORING_CATEGORIES
    elif args.basic:
        test_categories = BASIC_TEST_CATEGORIES
    else:
        test_categories = DEFAULT_TEST_CATEGORIES

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
        # Convert model name to safe filename
        output_name = args.hf_model.replace("/", "--")

    output_path = Path(output_dir) / f"{output_name}-bfcl-results.json"

    # Determine sample capture path if requested
    capture_samples_path = None
    if args.capture_samples:
        capture_samples_path = str(Path(output_dir) / f"{output_name}-bfcl-samples.txt")

    describe_config(
        checkpoint=args.checkpoint,
        hf_model=args.hf_model,
        model_name=args.model_name,
        templates_model_name=args.templates_model_name,
        test_categories=test_categories,
        backend=args.backend,
        num_gpus=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_dir=output_dir,
        capture_samples=args.capture_samples,
        num_samples=args.num_samples,
        logger=module_logger,
    )

    # Run evaluation
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        hf_model=args.hf_model,
        model_name=args.model_name,
        templates_model_name=args.templates_model_name,
        test_categories=test_categories,
        backend=args.backend,
        num_gpus=args.num_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_path=str(output_path),
        capture_samples_path=capture_samples_path,
        num_samples_to_capture=args.num_samples,
        logger=module_logger,
    )

    module_logger.info(f"Evaluation complete. Results saved to: {output_path}")
    if capture_samples_path:
        module_logger.info(f"Sample responses saved to: {capture_samples_path}")


if __name__ == "__main__":
    main()
