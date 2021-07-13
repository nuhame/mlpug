import torch.distributed as dist

from mlpug.multi_processing import \
    MultiProcessingContextBase, \
    MultiProcessingMixin


class PyTorchDistributedContext(MultiProcessingContextBase):

    def __init__(self, name="PyTorchDistributedContext"):
        super().__init__(name=name)

    def is_distributed(self):
        return dist.is_initialized()

    def is_primary(self):
        return dist.get_rank() == 0

    def device_rank(self):
        return dist.get_rank()

    def world_size(self):
        return dist.get_world_size()


