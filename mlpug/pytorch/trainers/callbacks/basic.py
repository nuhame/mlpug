from mlpug.pytorch.multi_processing import MultiProcessingMixin

from mlpug.trainers.callbacks.basic import *
from mlpug.trainers.callbacks.basic import LogProgress as LogProgressBase


class LogProgress(MultiProcessingMixin, LogProgressBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.is_primary:
            self._log.warning("LogProgress is intended to be used only by the primary worker")
