from mlpug.pytorch.trainers.callbacks.lr_scheduler_wrapper import LRSchedulerWrapperMixin

from mlpug.pytorch.xla.multi_processing import MultiProcessingMixin


class LRSchedulerWrapper(MultiProcessingMixin, LRSchedulerWrapperMixin):
    pass
