from mlpug.trainers.callbacks.metrics_logger import *

from mlpug.trainers.callbacks.metrics_logger import \
    TrainingMetricsLogger as TrainingMetricsLoggerBase, \
    TestMetricsLogger as TestMetricsLoggerBase

from mlpug.pytorch.multi_processing import MultiProcessingMixin

from mlpug.pytorch.utils import SlidingWindow


class TrainingMetricsLogger(MultiProcessingMixin, TrainingMetricsLoggerBase):

    def __init__(self, *args, sliding_window_factory=SlidingWindow, **kwargs):
        super().__init__(*args, sliding_window_factory=sliding_window_factory, **kwargs)


class TestMetricsLogger(MultiProcessingMixin, TestMetricsLoggerBase):

    def __init__(self, *args, sliding_window_factory=SlidingWindow, **kwargs):
        super().__init__(*args, sliding_window_factory=sliding_window_factory, **kwargs)
