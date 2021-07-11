from mlpug.pytorch.multi_processing import MultiProcessingMixin

from mlpug.mlpug_exceptions import CallbackBadUseException

from mlpug.trainers.callbacks.tensorboard import *
from mlpug.trainers.callbacks.tensorboard import \
    Tensorboard as TensorboardBase, \
    AutoTensorboard as AutoTensorboardBase


class Tensorboard(MultiProcessingMixin, TensorboardBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.is_primary:
            raise CallbackBadUseException(self.name, "Only add Tensorboard logging to the primary worker")


class AutoTensorboard(MultiProcessingMixin, AutoTensorboardBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.is_primary:
            raise CallbackBadUseException(self.name, "Only add AutoTensorboard logging to the primary worker")
