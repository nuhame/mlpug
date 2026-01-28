"""
Shared CLI arguments for NTP training.

This module provides argument parsing for NTP training that is backend-agnostic.
Backend-specific arguments (e.g., num_dataloader_workers for PyTorch) are added
in the respective backend modules.
"""
import argparse
import logging
import os

from basics.logging import get_logger

import mlpug.pytorch as mlp

from examples.shared_args import create_arg_parser as create_base_arg_parser
from examples.shared_args import describe_config as describe_base_config

mlp.logging.use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


def create_arg_parser(
    description: str = "Train a language model from scratch using Next-Token Prediction",
) -> argparse.ArgumentParser:
    """
    Create argument parser for NTP training.

    Extends the base argument parser with NTP-specific arguments.
    Backend-specific arguments should be added by the backend module.

    :param description: Parser description.

    :return: Configured argument parser.
    """
    parser = create_base_arg_parser(description=description)

    # -------------------------------------------------------------------------
    # Data paths
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to tokenized training data directory (must contain Data Forager index)",
    )

    parser.add_argument(
        "--val-data-path",
        type=str,
        required=False,
        default=None,
        help="Path to tokenized validation data directory (optional)",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        required=False,
        default=None,
        help="Fraction of training data to use (0-1, default: None = all data)",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        required=False,
        default=None,
        help="Fraction of validation data to use (0-1, default: None = all data)",
    )

    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--model-name",
        type=str,
        required=False,
        default="Qwen/Qwen3-1.7B-Base",
        help="HuggingFace model name for architecture config (default: Qwen/Qwen3-1.7B-Base)",
    )

    parser.add_argument(
        "--attn-dropout",
        type=float,
        required=False,
        default=0.0,
        help=(
            "Attention dropout rate on attention weights (Qwen3 built-in). "
            "Default: 0.0, recommended: 0.1 for multi-epoch training."
        ),
    )

    parser.add_argument(
        "--mlp-dropout",
        type=float,
        required=False,
        default=0.0,
        help=(
            "MLP dropout rate after MLP layers, before residual connection. "
            "Default: 0.0, recommended: 0.1 for multi-epoch training."
        ),
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=False,
        default="../trained-models",
        help="Directory for saving model checkpoints",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        required=False,
        default="../training-logs",
        help="Directory for training logs (TensorBoard)",
    )

    parser.add_argument(
        "--archive-epoch-checkpoints",
        action="store_true",
        help="Archive model checkpoint at end of each epoch (for evaluation at different stages)",
    )

    # -------------------------------------------------------------------------
    # LR scheduler configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        required=False,
        default=0.1,
        help="Fraction of total steps for LR warmup (default: 0.1 = 10%%)",
    )

    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        required=False,
        default=0.01,
        help="Minimum LR as fraction of peak LR at end of training (default: 0.01 = 1%%)",
    )

    # -------------------------------------------------------------------------
    # Optimizer configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--weight-decay",
        type=float,
        required=False,
        default=0.1,
        help="Weight decay for AdamW (default: 0.1, typical for LLM pretraining)",
    )

    parser.add_argument(
        "--beta1",
        type=float,
        required=False,
        default=0.9,
        help="AdamW beta1 parameter (default: 0.9)",
    )

    parser.add_argument(
        "--beta2",
        type=float,
        required=False,
        default=0.95,
        help="AdamW beta2 parameter (default: 0.95, lower than default for LLM stability)",
    )

    # -------------------------------------------------------------------------
    # Performance options
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory at cost of speed",
    )

    parser.add_argument(
        "--no-liger-kernel",
        action="store_true",
        help="Disable Liger Kernel (enabled by default for memory-efficient cross-entropy)",
    )

    # Override defaults from base parser for LLM training
    parser.set_defaults(
        batch_size=32,
        learning_rate=3e-4,
        num_epochs=1,
        seed=42,
        progress_log_period=100,
    )

    return parser


def describe_config(
    train_data_path: str,
    val_data_path: str | None,
    train_fraction: float | None,
    val_fraction: float | None,
    model_name: str,
    attn_dropout: float,
    mlp_dropout: float,
    checkpoint_dir: str,
    log_dir: str,
    archive_epoch_checkpoints: bool,
    warmup_ratio: float,
    min_lr_ratio: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    activation_checkpointing: bool,
    no_liger_kernel: bool,
    logger: logging.Logger | None = None,
    **kwargs,
) -> None:
    """
    Log NTP training configuration.

    Calls the base describe_config and adds NTP-specific arguments.

    :param train_data_path: Path to tokenized training data.
    :param val_data_path: Path to tokenized validation data.
    :param train_fraction: Fraction of training data to use.
    :param val_fraction: Fraction of validation data to use.
    :param model_name: HuggingFace model name.
    :param attn_dropout: Attention dropout rate (Qwen3 built-in).
    :param mlp_dropout: MLP dropout rate (wrapper).
    :param checkpoint_dir: Directory for checkpoints.
    :param log_dir: Directory for logs.
    :param warmup_ratio: LR warmup ratio.
    :param min_lr_ratio: Minimum LR ratio.
    :param weight_decay: Weight decay.
    :param beta1: AdamW beta1.
    :param beta2: AdamW beta2.
    :param activation_checkpointing: Whether to use gradient checkpointing.
    :param no_liger_kernel: Whether Liger Kernel is disabled.
    :param logger: Logger to use. If None, uses module logger.
    :param kwargs: Additional arguments passed to base describe_config.
    """
    if logger is None:
        logger = module_logger

    # Log base config (includes "Configuration:" header)
    describe_base_config(logger=logger, **kwargs)

    # Log NTP-specific config
    logger.info(f"  train_data_path: {train_data_path}")
    logger.info(f"  val_data_path: {val_data_path}")
    logger.info(f"  train_fraction: {train_fraction}")
    logger.info(f"  val_fraction: {val_fraction}")
    logger.info(f"  model_name: {model_name}")
    logger.info(f"  attn_dropout: {attn_dropout}")
    logger.info(f"  mlp_dropout: {mlp_dropout}")
    logger.info(f"  checkpoint_dir: {checkpoint_dir}")
    logger.info(f"  log_dir: {log_dir}")
    logger.info(f"  archive_epoch_checkpoints: {archive_epoch_checkpoints}")
    logger.info(f"  warmup_ratio: {warmup_ratio}")
    logger.info(f"  min_lr_ratio: {min_lr_ratio}")
    logger.info(f"  weight_decay: {weight_decay}")
    logger.info(f"  beta1: {beta1}")
    logger.info(f"  beta2: {beta2}")
    logger.info(f"  activation_checkpointing: {activation_checkpointing}")
    logger.info(f"  use_liger_kernel: {not no_liger_kernel}")
