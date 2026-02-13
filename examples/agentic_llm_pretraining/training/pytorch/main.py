"""
Shared NTP training entry point logic for PyTorch.

Contains the main training function parameterized by the training process class.
Version-specific entry points (v1/train.py, v2/train.py) inject the appropriate
NTPTrainingProcess subclass.

Designed to be run with torchrun for distributed training:

    # Single GPU (v1)
    python -m examples.agentic_llm_pretraining.training.pytorch.v1.train \\
        --train-data-path /path/to/tokenized/train

    # Multi-GPU with torchrun (v2)
    torchrun --nproc_per_node=4 \\
        -m examples.agentic_llm_pretraining.training.pytorch.v2.train \\
        --train-data-path /path/to/tokenized/train

Environment variables (set by torchrun):
    - RANK or LOCAL_RANK: Device rank
    - WORLD_SIZE: Total number of devices
"""
from typing import Type

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


def main(process_class: Type[NTPTrainingProcess] = NTPTrainingProcess) -> None:
    """
    Main entry point for NTP training.

    :param process_class: Training process class to instantiate.
        Defaults to NTPTrainingProcess (v1). V2 passes NTPTrainingProcessV2.
    """
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
        module_logger.info(f"Training process: {process_class.__name__}")
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

    # Convert no_liger_kernel to use_liger_kernel (enabled by default)
    config["use_liger_kernel"] = not config.pop("no_liger_kernel")

    # Configure torch.compile to capture scalar outputs (e.g., .item() calls)
    # This prevents graph breaks from Liger Kernel's fused cross-entropy.
    # On ROCm, capture_scalar_outputs=True causes inductor C++ code generation bugs,
    # so --allow-liger-kernel-graph-breaks disables this setting.
    allow_graph_breaks = config.pop("allow_liger_kernel_graph_breaks")
    if not allow_graph_breaks:
        import torch._dynamo.config
        torch._dynamo.config.capture_scalar_outputs = True
        if rank == 0:
            module_logger.info("torch._dynamo.config.capture_scalar_outputs = True")
    else:
        if rank == 0:
            module_logger.info(
                "Allowing Liger Kernel graph breaks (capture_scalar_outputs=False)"
            )

    # Remove args that aren't TrainingProcess parameters
    config.pop("remote_debug_ip", None)
    config.pop("distributed", None)
    config.pop("num_devices", None)

    # Create and setup training process
    training_process = process_class(
        rank=rank,
        num_devices=world_size,
        lr_scheduler_config=lr_scheduler_config,
        optimizer_config=optimizer_config,
        distributed=(world_size > 1),
        **config,
    )

    training_process.setup()
    training_process.start()
