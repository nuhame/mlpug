"""
Dropout wrapper for Qwen3 models.

Qwen3 models have built-in attention dropout (on attention weights), but no MLP dropout.
This module provides a wrapper that adds dropout after MLP layers (before residual),
which is important for multi-epoch training on small datasets (~1B tokens).

For attention dropout, use Qwen3's built-in config.attention_dropout parameter.

Usage:
    from transformers import AutoConfig, AutoModelForCausalLM
    from examples.agentic_llm_pretraining.models.qwen3 import add_mlp_dropout_to_qwen3

    # Set attention dropout via config (Qwen3 built-in)
    config = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B-Base")
    config.attention_dropout = 0.1
    model = AutoModelForCausalLM.from_config(config)

    # Add MLP dropout via wrapper
    model = add_mlp_dropout_to_qwen3(model, dropout_rate=0.1)

References:
    - Google AI Mode recommendation: 0.1 for both attention and MLP dropout
    - "Repetition In Repetition Out" (NeurIPS 2023): dropout helps with multi-epoch training
    - "Drop Dropout" paper: single-epoch doesn't need dropout, but multi-epoch does
"""

import logging
import os

import torch
import torch.nn as nn

from basics.logging import get_logger

module_logger = get_logger(os.path.basename(__file__))


class Qwen3DecoderLayerWithMLPDropout(nn.Module):
    """
    Wrapper that adds MLP dropout to Qwen3DecoderLayer.

    Adds dropout after MLP, before residual connection. For attention dropout,
    use Qwen3's built-in config.attention_dropout parameter instead.

    This is recommended for multi-epoch training on small datasets.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize the MLP dropout wrapper.

        :param original_layer: The original Qwen3DecoderLayer to wrap.
        :param dropout_rate: Dropout probability after MLP (default: 0.1).
        """
        super().__init__()
        self.layer = original_layer
        self.mlp_dropout = nn.Dropout(dropout_rate)

    def __getattr__(self, name: str):
        """Forward attribute access to the wrapped layer."""
        if name in ("layer", "mlp_dropout", "training"):
            return super().__getattr__(name)
        try:
            return getattr(self.layer, name)
        except AttributeError:
            return super().__getattr__(name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with dropout added after MLP.

        Follows the same structure as Qwen3DecoderLayer.forward() but adds
        dropout before the MLP residual connection.

        For attention dropout, use Qwen3's built-in config.attention_dropout.
        """
        # Input layernorm + self-attention (using original layer's attention dropout)
        residual = hidden_states
        hidden_states = self.layer.input_layernorm(hidden_states)

        hidden_states, _ = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Post-attention layernorm + MLP
        residual = hidden_states
        hidden_states = self.layer.post_attention_layernorm(hidden_states)
        hidden_states = self.layer.mlp(hidden_states)

        # Dropout BEFORE residual connection (post-MLP)
        hidden_states = self.mlp_dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def add_mlp_dropout_to_qwen3(
    model: nn.Module,
    dropout_rate: float = 0.1,
    logger: logging.Logger | None = None,
) -> nn.Module:
    """
    Replace all decoder layers with MLP dropout-enabled versions.

    This modifies the model in-place by wrapping each Qwen3DecoderLayer
    with Qwen3DecoderLayerWithMLPDropout.

    For attention dropout, set config.attention_dropout before model creation.

    :param model: The Qwen3 model (AutoModelForCausalLM).
    :param dropout_rate: Dropout probability after MLP
        (default: 0.1, recommended by research for multi-epoch training).
    :param logger: Optional logger instance.

    :return: The modified model with MLP dropout added.
    """
    if logger is None:
        logger = module_logger

    num_layers = len(model.model.layers)

    for i in range(num_layers):
        original_layer = model.model.layers[i]
        model.model.layers[i] = Qwen3DecoderLayerWithMLPDropout(
            original_layer,
            dropout_rate=dropout_rate,
        )

    logger.info(
        f"Added MLP dropout to {num_layers} decoder layers (rate={dropout_rate})"
    )

    return model
