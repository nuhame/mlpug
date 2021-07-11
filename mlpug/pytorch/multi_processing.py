import torch.distributed as dist

from mlpug.multi_processing import MultiProcessingMixin as MPMixinBase


class MultiProcessingMixin(MPMixinBase):

    def __init__(self, *args,
                 is_distributed=None,
                 is_primary=None,
                 process_index=None,
                 **kwargs):

        if is_distributed is None:
            is_distributed = dist.is_initialized()

        if process_index is None:
            process_index = dist.get_rank() if is_distributed else 0

        if is_primary is None:
            is_primary = (not is_distributed) or process_index == 0

        super().__init__(*args,
                         is_distributed=is_distributed,
                         is_primary=is_primary,
                         process_index=process_index,
                         **kwargs)
