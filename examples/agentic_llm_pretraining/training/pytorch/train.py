"""
NTP (Next-Token Prediction) training entry point for PyTorch.

This script trains a language model from scratch using Next-Token Prediction.
It is designed to be run with torchrun for distributed training:

    # Single GPU
    python -m examples.agentic_llm_pretraining.training.pytorch.train \\
        --train-data-path /path/to/tokenized/train

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 \\
        -m examples.agentic_llm_pretraining.training.pytorch.train \\
        --train-data-path /path/to/tokenized/train

Environment variables (set by torchrun):
    - RANK or LOCAL_RANK: Device rank
    - WORLD_SIZE: Total number of devices
"""
import os

from basics.logging import get_logger

import mlpug.pytorch as mlp
from mlpug.lr_scheduler_configs import CosineDecayConfig
from mlpug.utils.git_logging import log_git_state

from examples.agentic_llm_pretraining.training.pytorch.args import (
    create_arg_parser,
    describe_config,
)
from examples.agentic_llm_pretraining.training.pytorch.training_process import (
    NTPTrainingProcess,
)

module_logger = get_logger(os.path.basename(__file__))


def get_rank_and_world_size() -> tuple[int, int]:
    """
    Get rank and world_size from environment (torchrun) or default to single device.

    torchrun sets RANK, LOCAL_RANK, and WORLD_SIZE environment variables.
    For single GPU runs without torchrun, defaults to rank=0, world_size=1.

    :return: (rank, world_size) tuple.
    """
    # Try RANK first (global rank), then LOCAL_RANK (per-node rank)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def main() -> None:
    """Main entry point for NTP training."""
    # Setup logging
    mlp.logging.use_fancy_colors()

    # Log git state for reproducibility
    log_git_state()

    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()
    config = vars(args)

    # Get rank and world_size from environment (torchrun) or defaults
    rank, world_size = get_rank_and_world_size()

    # Only log config on primary device
    if rank == 0:
        describe_config(**config, logger=module_logger)

    # use_loss_scaling is derived from use_mixed_precision
    # --use-mixed-precision → float16 + loss scaling
    # --autocast-dtype X (without --use-mixed-precision) → autocast only, no scaling
    config["use_loss_scaling"] = config["use_mixed_precision"]

    # Create LR scheduler config from CLI args
    lr_scheduler_config = CosineDecayConfig(
        warmup_ratio=config.pop("warmup_ratio"),
        min_lr_ratio=config.pop("min_lr_ratio"),
    )

    # Create optimizer config from CLI args
    optimizer_config = {
        "betas": (config.pop("beta1"), config.pop("beta2")),
    }

    # Map progress_log_period to log_frequency (TrainingProcess parameter name)
    config["log_frequency"] = config.pop("progress_log_period")

    # Remove args that aren't TrainingProcess parameters
    config.pop("remote_debug_ip", None)
    config.pop("distributed", None)
    config.pop("num_devices", None)

    # Create and setup training process
    training_process = NTPTrainingProcess(
        rank=rank,
        num_devices=world_size,
        lr_scheduler_config=lr_scheduler_config,
        optimizer_config=optimizer_config,
        distributed=(world_size > 1),
        **config,
    )

    training_process.setup()
    training_process.start()


if __name__ == "__main__":
    main()
