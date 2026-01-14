"""
NTP (Next-Token Prediction) training model wrapper.

Wraps a HuggingFace causal LM model and computes the NTP loss for MLPug training.
"""
import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import Module


class NTPTrainModel(nn.Module):
    """
    Training model wrapper for Next-Token Prediction.

    Wraps a causal language model and computes cross-entropy loss for
    next-token prediction. The model handles the label shifting internally.

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
        batch_data: torch.Tensor,
        evaluate_settings: dict[str, Any] | None = None,
        inference_mode: bool | None = None,
    ) -> dict[str, Any]:
        """
        Forward pass computing NTP loss.

        :param batch_data: Token IDs tensor of shape (batch_size, context_length).
        :param evaluate_settings: Optional evaluation settings (unused, for MLPug compatibility).
        :param inference_mode: Optional inference mode flag (unused, for MLPug compatibility).

        :return: Dict with 'loss', 'num_samples', and 'auxiliary_results'.
            auxiliary_results contains 'perplexity' computed from the loss.
        """
        # Move to device and convert to long for embedding layer compatibility
        # (PyTorch embeddings require Long/Int, not unsigned types like uint32)
        input_ids = batch_data.to(self.device).long()

        # NTP: predict next token, labels are shifted input_ids
        # HuggingFace CausalLM handles the shift internally:
        # - logits are computed for positions 0..N-1
        # - loss is computed against labels at positions 1..N
        outputs = self.model(
            input_ids=input_ids,
            labels=input_ids,
            # use_cache must be False when using gradient checkpointing
            # (cached KV values interfere with recomputation)
            use_cache=not self._activation_checkpointing,
        )

        loss = outputs.loss
        batch_size = input_ids.shape[0]

        # Compute perplexity from loss
        # perplexity = exp(cross_entropy_loss)
        with torch.no_grad():
            perplexity = math.exp(loss.item()) if loss.item() < 100 else float('inf')

        return {
            "loss": loss,
            "num_samples": batch_size,
            "auxiliary_results": {
                "perplexity": perplexity,
            },
        }
