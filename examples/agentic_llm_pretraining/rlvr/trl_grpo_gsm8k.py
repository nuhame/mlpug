"""
TRL GRPO training script for GSM8K math RLVR.

Phase 1: Use TRL's GRPOTrainer as a baseline before building custom MLPug
GRPO implementation (Phase 2). Trains the model with binary rewards on GSM8K
math problems using GRPO.

Usage:
    # Smoke test (no vLLM, small config)
    python -m examples.agentic_llm_pretraining.rlvr.trl_grpo_gsm8k \
        --hf-model /path/to/hf-model \
        --output-dir /path/to/output \
        --micro-batch-size 2 --batch-size 2 \
        --num-generations 4 --max-completion-length 512

    # Full run on 1xA100 with vLLM
    python -m examples.agentic_llm_pretraining.rlvr.trl_grpo_gsm8k \
        --checkpoint /path/to/checkpoint.pt \
        --output-dir /path/to/output \
        --use-vllm
"""

import argparse
import logging
import os
import re

from basics.logging import get_logger

import mlpug.pytorch as mlp
mlp.logging.use_fancy_colors()

from mlpug.utils.git_logging import log_git_state

from examples.agentic_llm_pretraining.checkpoint import convert_checkpoint_to_hf
from examples.agentic_llm_pretraining.rlvr.rewards import gsm8k_reward_func

module_logger = get_logger(os.path.basename(__file__))


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves math problems. "
    "Solve the problem step by step, showing your reasoning. "
    "At the end, provide the final numeric answer on a new line after ####."
)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for TRL GRPO GSM8K training."""
    parser = argparse.ArgumentParser(
        description="TRL GRPO training on GSM8K math problems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to MLPug .pt checkpoint file (will be converted to HF format).",
    )
    model_group.add_argument(
        "--hf-model",
        type=str,
        help="Path to HuggingFace model directory (already in HF format).",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-1.7B-Base",
        help="HuggingFace model name for architecture (used with --checkpoint).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save trained model and checkpoints.",
    )

    # GRPO hyperparameters
    parser.add_argument(
        "--num-generations",
        type=int,
        default=16,
        help="Number of completions per prompt (G in GRPO).",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=4096,
        help="Maximum tokens for each generated completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability threshold.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Peak learning rate.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL divergence coefficient. 0.0 disables KL penalty "
             "(PPO clipping still constrains policy drift).",
    )

    # Batch size (MLPug convention)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Total prompts per optimizer step (effective batch size).",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=4,
        help="Prompts per forward pass. "
             "gradient_accumulation_steps = batch_size / micro_batch_size.",
    )

    # Training schedule
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of steps for linear warmup.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer.",
    )

    # vLLM
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for generation (colocate mode, sleep/wake cycle).",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.3,
        help="Fraction of GPU memory for vLLM (colocate mode).",
    )

    # Logging and saving
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Log metrics every N steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Save checkpoint every N steps. 0 = per-epoch only.",
    )
    parser.add_argument(
        "--log-completions",
        action="store_true",
        help="Log sample completions during training.",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the model.",
    )

    return parser


def describe_config(
    checkpoint: str | None,
    hf_model: str | None,
    model_name: str,
    output_dir: str,
    num_generations: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    learning_rate: float,
    beta: float,
    batch_size: int,
    micro_batch_size: int,
    num_epochs: int,
    warmup_ratio: float,
    max_grad_norm: float,
    weight_decay: float,
    use_vllm: bool,
    vllm_gpu_memory_utilization: float,
    logging_steps: int,
    save_steps: int,
    log_completions: bool,
    seed: int,
    system_prompt: str,
    logger: logging.Logger | None = None,
) -> None:
    """Log script configuration."""
    if logger is None:
        logger = module_logger

    gradient_accumulation_steps = batch_size // micro_batch_size

    logger.info("Configuration:")
    logger.info(f"  checkpoint: {checkpoint}")
    logger.info(f"  hf_model: {hf_model}")
    logger.info(f"  model_name: {model_name}")
    logger.info(f"  output_dir: {output_dir}")
    logger.info(f"  num_generations: {num_generations}")
    logger.info(f"  max_completion_length: {max_completion_length}")
    logger.info(f"  temperature: {temperature}")
    logger.info(f"  top_p: {top_p}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  beta: {beta}")
    logger.info(f"  batch_size: {batch_size} (total prompts per optimizer step)")
    logger.info(f"  micro_batch_size: {micro_batch_size} (per_device_train_batch_size)")
    logger.info(f"  gradient_accumulation_steps: {gradient_accumulation_steps} (derived)")
    logger.info(f"  num_epochs: {num_epochs}")
    logger.info(f"  warmup_ratio: {warmup_ratio}")
    logger.info(f"  max_grad_norm: {max_grad_norm}")
    logger.info(f"  weight_decay: {weight_decay}")
    logger.info(f"  use_vllm: {use_vllm}")
    logger.info(f"  vllm_gpu_memory_utilization: {vllm_gpu_memory_utilization}")
    logger.info(f"  logging_steps: {logging_steps}")
    logger.info(f"  save_steps: {save_steps}")
    logger.info(f"  log_completions: {log_completions}")
    logger.info(f"  seed: {seed}")
    logger.info(f"  system_prompt: {system_prompt[:80]}...")


def get_model_path(args) -> str:
    """Resolve model path: convert MLPug checkpoint to HF format if needed.

    :param args: Parsed CLI arguments.

    :return: Path to HuggingFace model directory.
    """
    if args.hf_model:
        module_logger.info(f"Using HF model: {args.hf_model}")
        return args.hf_model

    # Convert MLPug checkpoint to HF format in output directory
    hf_dir = os.path.join(args.output_dir, "hf-converted-model")
    module_logger.info(
        f"Converting checkpoint {args.checkpoint} to HF format at {hf_dir}"
    )
    convert_checkpoint_to_hf(
        checkpoint_path=args.checkpoint,
        output_dir=hf_dir,
        model_name=args.model_name,
        device="cpu",
    )
    return hf_dir


def prepare_gsm8k_dataset(system_prompt: str):
    """Load and format GSM8K training set for TRL GRPOTrainer.

    Formats each sample as a conversational prompt with system instructions
    and the math question. Extracts ground truth numeric answer from the
    GSM8K answer column (which contains ``#### <number>``).

    :param system_prompt: System prompt to include in each conversation.

    :return: HuggingFace Dataset with 'prompt' and 'ground_truth' columns.
    """
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="train")

    # TODO: failed_extractions tracking relies on single-process dataset.map().
    #  If num_proc is added, the closure won't propagate mutations back to the
    #  main process. Move to a separate pass if parallelism is needed.
    failed_extractions = []

    def format_sample(sample: dict) -> dict:
        # Extract numeric answer from "#### <number>" in answer column
        match = re.search(r"####\s*(.+?)(?:\n|$)", sample["answer"])
        if match:
            ground_truth = match.group(1).strip()
        else:
            ground_truth = ""
            failed_extractions.append(sample["answer"])

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["question"]},
        ]

        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
        }

    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    # Filter out samples where ground truth extraction failed
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: x["ground_truth"] != "")
    if len(dataset) < original_len:
        module_logger.warning(
            f"Filtered {original_len - len(dataset)} samples "
            f"with missing ground truth"
        )
        for i, answer_text in enumerate(failed_extractions):
            module_logger.warning(
                f"  Failed extraction [{i}]: ...{answer_text[-200:]}"
            )

    module_logger.info(f"GSM8K dataset prepared: {len(dataset)} samples")

    return dataset


def main():
    """Main entry point for TRL GRPO GSM8K training."""
    # Delayed imports — TRL and transformers are heavy
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError(
            "TRL is required for GRPO training. Install it with: "
            "pip install trl"
        )

    # Log git state for reproducibility
    log_git_state()

    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()
    config = vars(args)

    # Log configuration
    describe_config(**config)

    # Validate batch size divisibility
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError(
            f"batch_size ({args.batch_size}) must be divisible by "
            f"micro_batch_size ({args.micro_batch_size})"
        )

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    # Resolve model path
    model_path = get_model_path(args)

    # Prepare dataset
    dataset = prepare_gsm8k_dataset(args.system_prompt)

    # Configure save strategy
    if args.save_steps > 0:
        save_strategy = "steps"
        save_steps = args.save_steps
    else:
        save_strategy = "epoch"
        save_steps = None

    # Create GRPOConfig
    grpo_config_kwargs = dict(
        output_dir=args.output_dir,
        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        # Training
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        # Precision and memory
        bf16=True,
        gradient_checkpointing=True,
        # vLLM
        use_vllm=args.use_vllm,
        # Logging
        logging_steps=args.logging_steps,
        log_completions=args.log_completions,
        report_to="tensorboard",
        # Saving
        save_strategy=save_strategy,
        # Misc
        seed=args.seed,
        # Chat template
        chat_template_kwargs={"enable_thinking": False},
    )

    if args.use_vllm:
        grpo_config_kwargs["vllm_gpu_memory_utilization"] = (
            args.vllm_gpu_memory_utilization
        )

    if save_steps is not None:
        grpo_config_kwargs["save_steps"] = save_steps

    training_config = GRPOConfig(**grpo_config_kwargs)

    # Create trainer
    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=gsm8k_reward_func,
        args=training_config,
        train_dataset=dataset,
    )

    module_logger.info("Starting GRPO training...")

    # Train
    trainer.train()

    # Save final model
    module_logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model(args.output_dir)

    module_logger.info("Training complete.")


if __name__ == "__main__":
    main()
