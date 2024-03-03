from mlpug.trainers.callbacks.callback import *
from mlpug.trainers.callbacks.callback import Callback as CallbackBase

from mlpug.pytorch.multi_processing import MultiProcessingMixin


class Callback(MultiProcessingMixin, CallbackBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
