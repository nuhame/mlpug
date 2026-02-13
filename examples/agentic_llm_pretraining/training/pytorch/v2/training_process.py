"""
V2 NTP training process with loss masking.

Extends NTPTrainingProcess to load tokenized data with auxiliary loss masks
and apply them during training. Masked positions (system prompts, user prompts)
are excluded from loss by setting their labels to -100.

This module expects tokenized data created with the v2 tokenization pipeline
(Data Forager with auxiliary data support).
See examples/agentic_llm_pretraining/datasets/v2/tokenize_dataset.py.
"""
from typing import Callable, Dict

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from examples.agentic_llm_pretraining.datasets.loading import load_tokens_with_aux_dataset
from examples.agentic_llm_pretraining.training.pytorch.datasets import PyTorchForagerDataset
from examples.agentic_llm_pretraining.training.pytorch.training_process import (
    NTPTrainingProcess,
)


# Label value that HuggingFace CrossEntropyLoss (and Liger Kernel) ignores
IGNORE_INDEX = -100


def loss_mask_collate_fn(
    samples: list[Dict[str, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate samples with loss masks into (input_ids, labels) tensors.

    Stacks token arrays into input_ids, creates labels with -100 at masked
    positions. Loss mask convention: 1 = masked (excluded from loss),
    0 = trained (included in loss).

    :param samples: List of dicts with "tokens" and "loss_mask" numpy arrays.

    :return: Tuple of (input_ids, labels) tensors, both shape (batch_size, context_length).
    """
    tokens_list = []
    labels_list = []

    for sample in samples:
        # Convert to long (int64) upfront — required for embedding layer and
        # for masked fill (uint32 doesn't support it). This moves the dtype
        # conversion from the model to the DataLoader workers.
        tokens = torch.from_numpy(sample["tokens"]).long()
        loss_mask = sample["loss_mask"]

        # Labels start as a copy of tokens; masked positions become IGNORE_INDEX
        labels = tokens.clone()
        labels[loss_mask == 1] = IGNORE_INDEX

        tokens_list.append(tokens)
        labels_list.append(labels)

    input_ids = torch.stack(tokens_list)
    labels = torch.stack(labels_list)

    return input_ids, labels


class NTPTrainingProcessV2(NTPTrainingProcess):
    """
    V2 training process with loss masking for selective training.

    Loads tokenized data with auxiliary loss masks and applies them by setting
    masked positions to -100 in labels. This allows training on responses and
    reasoning traces while excluding system/user prompts from loss.

    All parameters are inherited from NTPTrainingProcess.
    """

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
        Return the loss-mask-aware collation function.

        :return: Collate function that produces (input_ids, labels) tuples.
        """
        return loss_mask_collate_fn
