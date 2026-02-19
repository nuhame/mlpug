"""
V2 NTP training process with loss masking.

Extends NTPTrainingProcess to load tokenized data with auxiliary loss masks
and optionally apply them during training. When applied, masked positions
(system prompts, user prompts) are excluded from loss by setting their labels
to -100.

This module expects tokenized data created with the v2 tokenization pipeline
(Data Forager with auxiliary data support).
See examples/agentic_llm_pretraining/datasets/v2/tokenize_dataset.py.
"""
from typing import Callable, Dict

from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from examples.agentic_llm_pretraining.datasets.loading import load_tokens_with_aux_dataset
from examples.agentic_llm_pretraining.training.pytorch.datasets import PyTorchForagerDataset
from examples.agentic_llm_pretraining.training.pytorch.training_process import (
    NTPTrainingProcess,
)


def loss_mask_collate_fn(
    samples: list[Dict[str, np.ndarray]],
    apply_loss_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Collate samples into (input_ids, loss_mask) tensors for NTP training.

    Always returns a tuple. When apply_loss_mask is True, loss_mask is included
    so the model can derive labels inside the compiled graph (clone input_ids,
    then mask). When False, loss_mask is None and the model uses
    labels = input_ids directly.

    This design avoids passing pre-constructed labels as an independent tensor
    input to the compiled model, which causes torch.compile + Liger FLCE
    compatibility issues with AOTAutograd.

    Loss mask convention: 1 = masked (excluded from loss),
    0 = trained (included in loss).

    :param samples: List of dicts with "tokens" and "loss_mask" numpy arrays.
    :param apply_loss_mask: Whether to include the loss mask. When False,
        loss_mask is None.

    :return: Tuple of (input_ids, loss_mask). loss_mask is None when
        apply_loss_mask is False.
    """
    tokens_list = []

    if apply_loss_mask:
        mask_list = []
        for sample in samples:
            tokens_list.append(torch.from_numpy(sample["tokens"]).long())
            mask_list.append(torch.from_numpy(sample["loss_mask"]))

        return torch.stack(tokens_list), torch.stack(mask_list)

    # No masking: return (input_ids, None) — model uses labels = input_ids
    for sample in samples:
        tokens_list.append(torch.from_numpy(sample["tokens"]).long())

    return torch.stack(tokens_list), None


class NTPTrainingProcessV2(NTPTrainingProcess):
    """
    V2 training process with loss masking for selective training.

    Loads tokenized data with auxiliary loss masks and optionally applies them
    by setting masked positions to -100 in labels. This allows training on
    responses and reasoning traces while excluding system/user prompts from loss.

    When apply_loss_mask is False, all tokens are trained on (useful for
    curriculum training phase 1).

    :param apply_loss_mask: Whether to apply loss masking. Default: True.
    :param kwargs: All other parameters passed to NTPTrainingProcess.
    """

    def __init__(self, *args, apply_loss_mask: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_loss_mask = apply_loss_mask

    def _load_dataset(self, path: str, name: str) -> TorchDataset:
        """
        Load a TokensWithAuxDataset and wrap as PyTorch dataset.

        :param path: Path to tokenized data directory with aux data.
        :param name: Dataset name for logging.

        :return: PyTorchForagerDataset instance.
        """
        fraction = self._train_fraction if name == "training" else self._val_fraction

        tokens_with_aux_dataset = load_tokens_with_aux_dataset(path)
        tokens_with_aux_dataset = self._apply_subsampling(
            tokens_with_aux_dataset, fraction, name,
        )

        return PyTorchForagerDataset[Dict[str, np.ndarray]](
            tokens_with_aux_dataset,
            copy_sample_func=lambda s: {k: v.copy() for k, v in s.items()},
        )

    def _get_sample_properties(self, dataset: TorchDataset) -> tuple[int, np.dtype]:
        """
        Extract context length and token dtype from a dict-returning dataset.

        :param dataset: PyTorchForagerDataset instance.

        :return: Tuple of (context_length, token_dtype).
        """
        sample = dataset[0]
        tokens = sample["tokens"]
        return len(tokens), tokens.dtype

    def _get_collate_fn(self) -> Callable:
        """
        Return the collation function with loss mask setting applied.

        :return: Collate function that produces (input_ids, loss_mask) tuples.
        """
        return partial(loss_mask_collate_fn, apply_loss_mask=self._apply_loss_mask)
