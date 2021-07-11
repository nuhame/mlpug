from mlpug.pytorch.multi_processing import MultiProcessingMixin

from mlpug.trainers.callbacks.callback import *
from mlpug.trainers.callbacks.callback import Callback as CallbackBase


class Callback(MultiProcessingMixin, CallbackBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
