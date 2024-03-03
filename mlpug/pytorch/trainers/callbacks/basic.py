from mlpug.trainers.callbacks.basic import LogProgress as LogProgressBase
from mlpug.trainers.callbacks.basic import BatchSizeLogger as BatchSizeLoggerBase
from mlpug.trainers.callbacks.basic import DescribeLogsObject as DescribeLogsObjectBase

from mlpug.pytorch.multi_processing import MultiProcessingMixin


class LogProgressMixin(LogProgressBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self.is_primary:
            self._log.warning("LogProgress is intended to be used only by the primary worker")

class LogProgress(MultiProcessingMixin, LogProgressMixin):
    pass


class BatchSizeLogger(MultiProcessingMixin, BatchSizeLoggerBase):
    pass


class DescribeLogsObject(MultiProcessingMixin, DescribeLogsObjectBase):
    pass
