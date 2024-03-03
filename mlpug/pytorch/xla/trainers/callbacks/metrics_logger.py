from mlpug.pytorch.utils import SlidingWindow

from mlpug.trainers.callbacks.metrics_logger import \
    TrainingMetricsLogger as TrainingMetricsLoggerBase, \
    DatasetMetricsLogger as DatasetMetricsLoggerBase

from mlpug.pytorch.xla.multi_processing import MultiProcessingMixin


class TrainingMetricsLogger(MultiProcessingMixin, TrainingMetricsLoggerBase):

    def __init__(self, *args, sliding_window_factory=SlidingWindow, **kwargs):
        super().__init__(*args, sliding_window_factory=sliding_window_factory, **kwargs)


class DatasetMetricsLogger(MultiProcessingMixin, DatasetMetricsLoggerBase):

    def __init__(self, *args, sliding_window_factory=SlidingWindow, **kwargs):
        super().__init__(*args, sliding_window_factory=sliding_window_factory, **kwargs)
