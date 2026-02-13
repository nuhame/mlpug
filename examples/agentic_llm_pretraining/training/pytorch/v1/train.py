"""
V1 NTP training entry point (standard next-token prediction without loss masking).

Usage:
    # Single GPU
    python -m examples.agentic_llm_pretraining.training.pytorch.v1.train \\
        --train-data-path /path/to/tokenized/train

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 \\
        -m examples.agentic_llm_pretraining.training.pytorch.v1.train \\
        --train-data-path /path/to/tokenized/train
"""
from examples.agentic_llm_pretraining.training.pytorch.main import main
from examples.agentic_llm_pretraining.training.pytorch.training_process import (
    NTPTrainingProcess,
)


if __name__ == "__main__":
    main(NTPTrainingProcess)
