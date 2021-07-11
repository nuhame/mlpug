from mlpug.trainers.callbacks.metrics_logger import *

from mlpug.trainers.callbacks.metrics_logger import \
    TrainingMetricsLogger as TrainingMetricsLoggerBase, \
    TestMetricsLogger as TestMetricsLoggerBase

from mlpug.pytorch.multi_processing import MultiProcessingMixin


class TrainingMetricsLogger(MultiProcessingMixin, TrainingMetricsLoggerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TestMetricsLogger(MultiProcessingMixin, TestMetricsLoggerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
