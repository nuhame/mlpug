"""
PyTorch XLA-specific LR schedule implementations.

These add MultiProcessingMixin to ensure only device 0 shows logs in distributed training.
"""
from mlpug.scheduler_funcs import (
    LinearDecaySchedule as LinearDecayScheduleBase,
    CosineDecaySchedule as CosineDecayScheduleBase,
    WSDSchedule as WSDScheduleBase,
    ConstantLRSchedule as ConstantLRScheduleBase,
)

from mlpug.pytorch.xla.multi_processing import MultiProcessingMixin


class LinearDecaySchedule(MultiProcessingMixin, LinearDecayScheduleBase):
    pass


class CosineDecaySchedule(MultiProcessingMixin, CosineDecayScheduleBase):
    pass


class WSDSchedule(MultiProcessingMixin, WSDScheduleBase):
    pass


class ConstantLRSchedule(MultiProcessingMixin, ConstantLRScheduleBase):
    pass
