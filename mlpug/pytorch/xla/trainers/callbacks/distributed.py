from mlpug.pytorch.trainers.callbacks.distributed import DistributedSamplerManagerMixin

from mlpug.pytorch.xla.trainers.callbacks.callback import Callback


class DistributedSamplerManager(DistributedSamplerManagerMixin, Callback):
    pass
