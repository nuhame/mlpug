import torch_xla.core.xla_model as xm

from mlpug.mlpug_exceptions import TrainerInvalidException

from mlpug.pytorch.utils import SlidingWindow

from mlpug.trainers.training import TrainingManager as TrainingManagerBase
from mlpug.trainers.training import Trainer as TrainerBase

from mlpug.pytorch.trainers.training import (
    PTTrainerMixin,
    DefaultTrainerMixin
)

from mlpug.pytorch.xla.multi_processing import MultiProcessingMixin


class TrainingManager(MultiProcessingMixin, TrainingManagerBase):
    def __init__(self, *args, sliding_window_factory=SlidingWindow, **kwargs):
        super().__init__(*args, sliding_window_factory=sliding_window_factory, **kwargs)

    def _training_ended(self):
        if self.is_distributed:
            # Wait for all processes to finish
            xm.rendezvous("mlpug-training-ended")


class Trainer(MultiProcessingMixin, PTTrainerMixin, TrainerBase):
    pass


class DefaultTrainer(MultiProcessingMixin, DefaultTrainerMixin):

    def __init__(self, *args, use_mixed_precision=False, name="DefaultTrainer", **kwargs):
        if use_mixed_precision:
            raise TrainerInvalidException("Mixed precision training not supported with XLA devices")

        super().__init__(
            *args,
            use_mixed_precision=use_mixed_precision,
            name=name,
            **kwargs
        )

    def _execute_optimizer(self, optimizer) -> bool:
        did_update = super()._execute_optimizer(optimizer)
        if not self.is_distributed:
            xm.mark_step()

        return did_update


