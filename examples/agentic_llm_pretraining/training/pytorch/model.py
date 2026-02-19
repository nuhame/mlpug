"""
NTP (Next-Token Prediction) training model wrapper.

Wraps a HuggingFace causal LM model and computes the NTP loss for MLPug training.
"""
from typing import Any

import torch
import torch.nn as nn
from torch.nn import Module

# Label value that HuggingFace CrossEntropyLoss (and Liger Kernel) ignores
IGNORE_INDEX = -100


class NTPTrainModel(nn.Module):
    """
    Training model wrapper for Next-Token Prediction.

    Wraps a causal language model and computes cross-entropy loss for
    next-token prediction. The model handles the label shifting internally.

    Supports two input modes:

    1. Plain tensor (v1): input_ids used as both input and labels.
    2. Tuple (v2): (input_ids, loss_mask) where loss_mask is optional.
       When loss_mask is provided, labels are derived from input_ids inside
       the compiled graph by cloning and masking. When loss_mask is None,
       labels = input_ids (same as v1). This ensures torch.compile + Liger
       FLCE compatibility by keeping labels derived from input_ids rather
       than as an independent tensor input.

    :param model: Causal language model (e.g., Qwen3ForCausalLM). Must support
        forward(input_ids, labels, use_cache) and return an object with .loss attribute.
    :param device: Device to run the model on (e.g., torch.device("cuda")).
    :param activation_checkpointing: Enable gradient checkpointing to reduce
        memory usage at the cost of recomputing activations during backward pass.
    """

    def __init__(
        self,
        model: Module,
        device: torch.device,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.model = model
        self.device = device
        self._activation_checkpointing = activation_checkpointing

        if activation_checkpointing:
            # HuggingFace models have gradient_checkpointing_enable()
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

    def forward(
        self,
        batch_data: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
        evaluate_settings: dict[str, Any] | None = None,
        inference_mode: bool | None = None,
    ) -> dict[str, Any]:
        """
        Forward pass computing NTP loss.

        :param batch_data: Either a token IDs tensor of shape
            (batch_size, context_length), or a tuple of (input_ids, loss_mask).
            When a single tensor, it is used as both input_ids and labels.
            When a tuple with loss_mask, labels are derived by cloning
            input_ids and setting masked positions to -100.
            When a tuple with loss_mask=None, labels = input_ids.
        :param evaluate_settings: Optional evaluation settings
            (unused, for MLPug compatibility).
        :param inference_mode: Optional inference mode flag
            (unused, for MLPug compatibility).

        :return: Dict with 'loss', 'num_samples', and 'auxiliary_results' (None).
        """
        if isinstance(batch_data, tuple):
            input_ids, loss_mask = batch_data
            input_ids = input_ids.to(self.device)

            if loss_mask is not None:
                # Derive labels inside the compiled graph: clone input_ids
                # and mask positions. This keeps labels traceable from
                # input_ids for AOTAutograd compatibility with Liger FLCE.
                loss_mask = loss_mask.to(self.device)
                labels = input_ids.clone()
                labels.masked_fill_(loss_mask == 1, IGNORE_INDEX)
            else:
                labels = input_ids
        else:
            # V1 path: plain tensor, convert to long for embedding layer
            # (PyTorch embeddings require Long/Int, not unsigned types like uint32)
            input_ids = batch_data.to(self.device).long()
            labels = input_ids

        # NTP: predict next token, labels are shifted input_ids
        # HuggingFace CausalLM handles the shift internally:
        # - logits are computed for positions 0..N-1
        # - loss is computed against labels at positions 1..N
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            # use_cache must be False when using gradient checkpointing
            # (cached KV values interfere with recomputation)
            use_cache=not self._activation_checkpointing,
        )

        loss = outputs.loss
        batch_size = input_ids.shape[0]

        return {
            "loss": loss,
            "num_samples": batch_size,
            "auxiliary_results": None,
        }
