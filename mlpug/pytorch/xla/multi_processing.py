import torch_xla.core.xla_model as xm

from mlpug.multi_processing import \
    MultiProcessingContextBase, \
    MultiProcessingMixin


class XLADistributedContext(MultiProcessingContextBase):

    def __init__(self, name="XLADistributedContext"):
        super().__init__(name=name)

    def is_distributed(self):
        return xm.xrt_world_size() > 1

    def is_primary(self):
        return xm.is_master_ordinal()

    def device_rank(self):
        return xm.get_ordinal()

    def world_size(self):
        return xm.xrt_world_size()
