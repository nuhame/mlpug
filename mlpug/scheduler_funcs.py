"""
Learning rate schedule implementations.

These schedules return a scale factor (0 to 1) that is multiplied with the base LR.
Use with PyTorch's LambdaLR: `LambdaLR(optimizer, schedule)`

The schedule classes can be instantiated directly or via the factory function
`create_lr_schedule(config, total_steps)`.
"""
import abc
import math
from typing import Callable

from mlpug.base import Base
from mlpug.lr_scheduler_configs import (
    LRSchedulerConfig,
    LinearDecayConfig,
    CosineDecayConfig,
    WSDConfig,
    ConstantLRConfig,
    DecayType,
)


def _cosine_decay(progress: float, min_lr_ratio: float) -> float:
    """
    Compute cosine decay scale factor.

    :param progress: Decay progress from 0.0 (start) to 1.0 (end).
    :param min_lr_ratio: Minimum LR as fraction of peak LR.

    :return: Scale factor between 1.0 and min_lr_ratio.
    """
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor


class LRScheduleBase(Base, metaclass=abc.ABCMeta):
    """
    Base class for LR schedules.

    Subclasses must implement `__call__(step)` to return a scale factor.
    """

    def __init__(
        self,
        total_steps: int,
        name: str = "LRScheduleBase",
        **kwargs,
    ):
        """
        :param total_steps: Total number of training steps.
        :param name: Logger name.
        """
        super().__init__(pybase_logger_name=name, **kwargs)

        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")

        self._total_steps = total_steps

    @abc.abstractmethod
    def __call__(self, step: int) -> float:
        """
        Compute LR scale factor for the given step.

        :param step: Current training step.

        :return: Scale factor to multiply with base LR (typically 0.0 to 1.0).
        """
        raise NotImplementedError


class LinearDecaySchedule(LRScheduleBase):
    """
    Linear warmup followed by linear decay to min_lr_ratio.

    Scale factor:
    - Warmup phase: linear increase from 0 to 1
    - Decay phase: linear decrease from 1 to min_lr_ratio
    """

    def __init__(
        self,
        total_steps: int,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.0,
        name: str = "LinearDecaySchedule",
        **kwargs,
    ):
        """
        :param total_steps: Total number of training steps.
        :param warmup_ratio: Fraction of total steps for warmup phase.
        :param min_lr_ratio: Final LR as fraction of peak LR.
        :param name: Logger name.
        """
        super().__init__(total_steps=total_steps, name=name, **kwargs)

        self._warmup_steps = int(warmup_ratio * total_steps)
        self._min_lr_ratio = min_lr_ratio

        # Ensure at least one decay step
        if self._warmup_steps >= total_steps:
            self._log.warning(
                f"warmup_steps ({self._warmup_steps}) >= total_steps ({total_steps}), "
                f"setting warmup_steps to {total_steps - 1}"
            )
            self._warmup_steps = total_steps - 1

    def __call__(self, step: int) -> float:
        step = float(step)

        if step <= self._warmup_steps:
            # Linear warmup: 0 -> 1
            if self._warmup_steps == 0:
                return 1.0
            return step / self._warmup_steps
        else:
            # Linear decay: 1 -> min_lr_ratio
            decay_steps = self._total_steps - self._warmup_steps
            progress = (step - self._warmup_steps) / decay_steps
            return max(1.0 - progress * (1.0 - self._min_lr_ratio), self._min_lr_ratio)


class CosineDecaySchedule(LRScheduleBase):
    """
    Linear warmup followed by cosine decay to min_lr_ratio.

    Scale factor:
    - Warmup phase: linear increase from 0 to 1
    - Decay phase: cosine decrease from 1 to min_lr_ratio
    """

    def __init__(
        self,
        total_steps: int,
        warmup_ratio: float = 0.1,
        min_lr_ratio: float = 0.01,
        name: str = "CosineDecaySchedule",
        **kwargs,
    ):
        """
        :param total_steps: Total number of training steps.
        :param warmup_ratio: Fraction of total steps for warmup phase.
        :param min_lr_ratio: Final LR as fraction of peak LR.
        :param name: Logger name.
        """
        super().__init__(total_steps=total_steps, name=name, **kwargs)

        self._warmup_steps = int(warmup_ratio * total_steps)
        self._min_lr_ratio = min_lr_ratio

        if self._warmup_steps >= total_steps:
            self._log.warning(
                f"warmup_steps ({self._warmup_steps}) >= total_steps ({total_steps}), "
                f"setting warmup_steps to {total_steps - 1}"
            )
            self._warmup_steps = total_steps - 1

    def __call__(self, step: int) -> float:
        step = float(step)

        if step <= self._warmup_steps:
            # Linear warmup: 0 -> 1
            if self._warmup_steps == 0:
                return 1.0
            return step / self._warmup_steps
        else:
            # Cosine decay: 1 -> min_lr_ratio
            decay_steps = self._total_steps - self._warmup_steps
            progress = (step - self._warmup_steps) / decay_steps
            return _cosine_decay(progress, self._min_lr_ratio)


class WSDSchedule(LRScheduleBase):
    """
    Warmup-Stable-Decay (WSD) schedule.

    Three phases:
    1. Warmup: Linear increase from 0 to 1
    2. Stable: Constant at 1 (peak LR)
    3. Decay: Decrease from 1 to min_lr_ratio using specified decay function
    """

    # Minimum value for exponential decay (avoids log(0))
    _EXPONENTIAL_MIN_LR_RATIO = 1e-8

    def __init__(
        self,
        total_steps: int,
        warmup_ratio: float = 0.1,
        stable_ratio: float = 0.7,
        decay_type: DecayType = DecayType.LINEAR,
        min_lr_ratio: float = 0.0,
        name: str = "WSDSchedule",
        **kwargs,
    ):
        """
        :param total_steps: Total number of training steps.
        :param warmup_ratio: Fraction of total steps for warmup phase.
        :param stable_ratio: Fraction of total steps for stable phase.
        :param decay_type: Type of decay function (LINEAR, COSINE, EXPONENTIAL).
        :param min_lr_ratio: Final LR as fraction of peak LR.
        :param name: Logger name.
        """
        super().__init__(total_steps=total_steps, name=name, **kwargs)

        if warmup_ratio + stable_ratio >= 1.0:
            raise ValueError(
                f"warmup_ratio ({warmup_ratio}) + stable_ratio ({stable_ratio}) must be < 1.0"
            )

        self._warmup_steps = int(warmup_ratio * total_steps)
        self._stable_steps = int(stable_ratio * total_steps)
        self._decay_type = decay_type

        # Handle min_lr_ratio for exponential decay
        if decay_type == DecayType.EXPONENTIAL and min_lr_ratio <= 0:
            self._log.warning(
                f"Exponential decay requires min_lr_ratio > 0, "
                f"setting to {self._EXPONENTIAL_MIN_LR_RATIO}"
            )
            self._min_lr_ratio = self._EXPONENTIAL_MIN_LR_RATIO
        else:
            self._min_lr_ratio = min_lr_ratio

        # Decay starts after warmup + stable
        self._decay_start = self._warmup_steps + self._stable_steps
        self._decay_steps = total_steps - self._decay_start

    def __call__(self, step: int) -> float:
        step = float(step)

        if step <= self._warmup_steps:
            # Warmup phase: 0 -> 1
            if self._warmup_steps == 0:
                return 1.0
            return step / self._warmup_steps

        elif step <= self._decay_start:
            # Stable phase: constant at 1
            return 1.0

        else:
            # Decay phase: 1 -> min_lr_ratio
            if self._decay_steps == 0:
                return self._min_lr_ratio

            progress = (step - self._decay_start) / self._decay_steps
            progress = min(progress, 1.0)  # Clamp to [0, 1]

            if self._decay_type == DecayType.LINEAR:
                scale = 1.0 - progress * (1.0 - self._min_lr_ratio)
            elif self._decay_type == DecayType.COSINE:
                scale = _cosine_decay(progress, self._min_lr_ratio)
            elif self._decay_type == DecayType.EXPONENTIAL:
                # Exponential decay: scale = exp(progress * log(min_lr_ratio))
                scale = math.exp(progress * math.log(self._min_lr_ratio))
            else:
                raise ValueError(f"Unknown decay type: {self._decay_type}")

            return max(scale, self._min_lr_ratio)


class ConstantLRSchedule(LRScheduleBase):
    """
    Constant learning rate (no scheduling).

    Always returns 1.0 (full base LR).
    """

    def __init__(self, name: str = "ConstantLRSchedule", **kwargs):
        # Use total_steps=1 as a dummy value (not used)
        super().__init__(total_steps=1, name=name, **kwargs)

    def __call__(self, step: int) -> float:
        return 1.0


def create_lr_schedule(
    config: LRSchedulerConfig,
    total_steps: int,
    **kwargs,
) -> Callable[[int], float]:
    """
    Factory function to create an LR schedule from a config.

    :param config: LR scheduler configuration.
    :param total_steps: Total number of training steps.
    :param kwargs: Additional arguments passed to schedule constructor.

    :return: Callable that takes step number and returns scale factor.
    """
    if isinstance(config, LinearDecayConfig):
        return LinearDecaySchedule(
            total_steps=total_steps,
            warmup_ratio=config.warmup_ratio,
            min_lr_ratio=config.min_lr_ratio,
            **kwargs,
        )
    elif isinstance(config, CosineDecayConfig):
        return CosineDecaySchedule(
            total_steps=total_steps,
            warmup_ratio=config.warmup_ratio,
            min_lr_ratio=config.min_lr_ratio,
            **kwargs,
        )
    elif isinstance(config, WSDConfig):
        return WSDSchedule(
            total_steps=total_steps,
            warmup_ratio=config.warmup_ratio,
            stable_ratio=config.stable_ratio,
            decay_type=config.decay_type,
            min_lr_ratio=config.min_lr_ratio,
            **kwargs,
        )
    elif isinstance(config, ConstantLRConfig):
        return ConstantLRSchedule(**kwargs)
    else:
        raise ValueError(f"Unknown LR scheduler config type: {type(config)}")
