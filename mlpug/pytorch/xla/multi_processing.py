import torch_xla.core.xla_model as xm

from mlpug.multi_processing import \
    MultiProcessingContextBase, \
    MultiProcessingMixin


class XLADistributedContext(MultiProcessingContextBase):

    def __init__(self, name="PyTorchDistributedContext"):
        super().__init__(name=name)

    def is_distributed(self):
        return xm.xrt_world_size() > 1

    def is_primary(self):
        return xm.is_master_ordinal()

    def process_index(self):
        return xm.get_ordinal()

