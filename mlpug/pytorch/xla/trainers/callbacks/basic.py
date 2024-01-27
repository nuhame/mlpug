from mlpug.trainers.callbacks.basic import BatchSizeLogger as BatchSizeLoggerBase
from mlpug.trainers.callbacks.basic import DescribeLogsObject as DescribeLogsObjectBase

from mlpug.pytorch.trainers.callbacks.basic import LogProgressMixin

from mlpug.pytorch.xla.multi_processing import MultiProcessingMixin


class LogProgress(MultiProcessingMixin, LogProgressMixin):
    pass


class BatchSizeLogger(MultiProcessingMixin, BatchSizeLoggerBase):
    pass


class DescribeLogsObject(MultiProcessingMixin, DescribeLogsObjectBase):
    pass
