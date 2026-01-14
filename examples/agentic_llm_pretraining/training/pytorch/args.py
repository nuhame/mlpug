"""
PyTorch-specific CLI arguments for NTP training.

Extends the NTP shared arguments with PyTorch-specific options.
"""
import argparse
import logging
import os

from basics.logging import get_logger

import mlpug.pytorch as mlp

from examples.agentic_llm_pretraining.training.shared_args import (
    create_arg_parser as create_base_arg_parser,
    describe_config as describe_base_config,
)

mlp.logging.use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


def create_arg_parser(
    description: str = "Train a language model from scratch using Next-Token Prediction (PyTorch)",
) -> argparse.ArgumentParser:
    """
    Create argument parser for PyTorch NTP training.

    Extends the NTP argument parser with PyTorch-specific arguments.

    :param description: Parser description.

    :return: Configured argument parser.
    """
    parser = create_base_arg_parser(description=description)

    # -------------------------------------------------------------------------
    # PyTorch-specific options
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--num-dataloader-workers",
        type=int,
        required=False,
        default=2,
        help="Number of DataLoader worker processes",
    )

    return parser


def describe_config(
    num_dataloader_workers: int,
    logger: logging.Logger | None = None,
    **kwargs,
) -> None:
    """
    Log PyTorch NTP training configuration.

    Calls the NTP describe_config and adds PyTorch-specific arguments.

    :param num_dataloader_workers: Number of DataLoader workers.
    :param logger: Logger to use. If None, uses module logger.
    :param kwargs: Additional arguments passed to NTP describe_config.
    """
    if logger is None:
        logger = module_logger

    # Log NTP config (which calls base config)
    describe_base_config(logger=logger, **kwargs)

    # Log PyTorch-specific config
    logger.info(f"  num_dataloader_workers: {num_dataloader_workers}")
