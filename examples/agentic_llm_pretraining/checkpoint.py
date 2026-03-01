"""
Reusable checkpoint handling utilities.

This module provides functions for loading MLPug checkpoints and converting them
to HuggingFace format for use with various evaluation frameworks.
"""
from typing import Optional

import logging
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from basics.logging import get_logger

module_logger = get_logger(os.path.basename(__file__))


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    device: str = "cuda",
    logger: Optional[logging.Logger] = None,
) -> tuple:
    """
    Load a model from an MLPug checkpoint file.

    MLPug saves model checkpoints as state_dict files using torch.save().
    This function loads the state_dict into a fresh model instance.

    :param checkpoint_path: Path to the .pt checkpoint file.
    :param model_name: HuggingFace model name to initialize the architecture.
    :param device: Device to load the model to.
    :param logger: Optional logger for status messages.

    :return: Tuple of (model, tokenizer).
    """
    if logger is None:
        logger = module_logger

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load the checkpoint
    # weights_only=False needed because MLPug checkpoints contain pickled objects
    # (e.g., MicroBatchResults in manager_state). This is safe for our own checkpoints.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # The checkpoint contains model state under 'model' key
    # (from MLPug's trainer.get_model_components() with convert_to_dict())
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # Assume it's a direct state_dict
        state_dict = checkpoint

    # Initialize model architecture
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)

    # Load the trained weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Model loaded successfully on {device}")

    return model, tokenizer


def save_model_as_hf(
    model,
    tokenizer,
    output_dir: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Save model in HuggingFace format.

    This saves the model and tokenizer in the standard HuggingFace format,
    which is required by most evaluation frameworks (lm-eval, BFCL, etc.).

    :param model: The PyTorch model to save.
    :param tokenizer: The tokenizer to save.
    :param output_dir: Directory to save the model.
    :param logger: Optional logger for status messages.

    :return: Path to the saved model directory.
    """
    if logger is None:
        logger = module_logger

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_path}")

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    logger.info(f"Model saved successfully")

    return str(output_path)


def convert_checkpoint_to_hf(
    checkpoint_path: str,
    output_dir: str,
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    device: str = "cpu",
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Convert an MLPug checkpoint to HuggingFace format.

    This is a convenience function that combines load_model_from_checkpoint()
    and save_model_as_hf(). Useful for preparing checkpoints for evaluation
    frameworks that require HuggingFace format.

    :param checkpoint_path: Path to the .pt checkpoint file.
    :param output_dir: Directory to save the HuggingFace model.
    :param model_name: HuggingFace model name for architecture.
    :param device: Device to use for loading (default: cpu to minimize memory).
    :param logger: Optional logger for status messages.

    :return: Path to the saved HuggingFace model directory.
    """
    model, tokenizer = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
        logger=logger,
    )

    output_path = save_model_as_hf(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        logger=logger,
    )

    # Free memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return output_path
