from mlpug.scheduler_funcs import LRWarmupSchedule as LRWarmupScheduleBase

from mlpug.pytorch.multi_processing import MultiProcessingMixin


class LRWarmupSchedule(MultiProcessingMixin, LRWarmupScheduleBase):
    pass
