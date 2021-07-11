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

    def process_index(self):
        return dist.get_rank()

