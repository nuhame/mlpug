"""
V2 NTP training entry point (next-token prediction with loss masking).

Loss masking excludes system/user prompts from loss computation while training
on responses and reasoning traces. Requires v2 tokenized data with auxiliary
loss mask arrays.

Usage:
    # Single GPU
    python -m examples.agentic_llm_pretraining.training.pytorch.v2.train \\
        --train-data-path /path/to/v2/tokenized/train

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 \\
        -m examples.agentic_llm_pretraining.training.pytorch.v2.train \\
        --train-data-path /path/to/v2/tokenized/train
"""
from examples.agentic_llm_pretraining.training.pytorch.main import main
from examples.agentic_llm_pretraining.training.pytorch.v2.training_process import (
    NTPTrainingProcessV2,
)


if __name__ == "__main__":
    main(NTPTrainingProcessV2)
