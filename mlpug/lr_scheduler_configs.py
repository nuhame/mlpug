"""
LR scheduler configuration classes.

These dataclasses define the configuration for different learning rate schedules.
The actual schedule implementations are in scheduler_funcs.py.
"""
from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto


class DecayType(Enum):
    """Decay function types for WSD schedule."""
    LINEAR = auto()
    EXPONENTIAL = auto()
    COSINE = auto()


@dataclass
class LRSchedulerConfig(ABC):
    """Base class for all LR scheduler configurations."""
    pass


@dataclass
class LinearDecayConfig(LRSchedulerConfig):
    """
    Warmup + linear decay to min LR.

    Linear warmup from 0 to peak LR, then linear decay to min LR.
    """
    warmup_ratio: float = 0.1
    """Fraction of total steps for warmup phase."""

    min_lr_ratio: float = 0.0
    """Final LR as fraction of peak LR (final_lr = peak_lr * min_lr_ratio)."""


@dataclass
class CosineDecayConfig(LRSchedulerConfig):
    """
    Warmup + cosine decay to min LR.

    Standard approach used by most LLM pretraining.
    """
    warmup_ratio: float = 0.1
    """Fraction of total steps for warmup phase."""

    min_lr_ratio: float = 0.01
    """Final LR as fraction of peak LR (final_lr = peak_lr * min_lr_ratio)."""


@dataclass
class WSDConfig(LRSchedulerConfig):
    """
    Warmup-Stable-Decay (WSD) schedule.

    Three phases:
    1. Warmup: Linear increase from 0 to peak LR
    2. Stable: Constant at peak LR (can be extended arbitrarily)
    3. Decay: Decrease from peak to min LR using specified decay function

    Adopted by DeepSeek-V3, Kimi-K2, InternLM for training flexibility.
    """
    warmup_ratio: float = 0.1
    """Fraction of total steps for warmup phase."""

    stable_ratio: float = 0.7
    """Fraction of total steps for stable phase (constant LR)."""

    decay_type: DecayType = DecayType.LINEAR
    """Type of decay function for the decay phase."""

    min_lr_ratio: float = 0.0
    """Final LR as fraction of peak LR (final_lr = peak_lr * min_lr_ratio)."""


@dataclass
class ConstantLRConfig(LRSchedulerConfig):
    """
    Constant learning rate (no scheduling).

    Use when you want a fixed LR throughout training.
    """
    pass
