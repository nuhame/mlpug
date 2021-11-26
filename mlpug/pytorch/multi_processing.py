import torch.distributed as dist

from mlpug.multi_processing import \
    MultiProcessingContextBase, \
    MultiProcessingMixin


# TODO : All the mlpug components that derive from MultiProcessingMixin, here in the pytorch package
#        should be in a seperate package. This creates an intermediate level between the mlpug base classes
#        and the Pytorch specializations. In this way the MultiProcessing versions of the components can easily be
#        reused for another specialization.


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
