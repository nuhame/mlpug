import torch

from mlpug.utils.mlpug_data import MLPugDataCleaner as MLPugDataCleanerBase


class MLPugDataCleaner(MLPugDataCleanerBase):

    def _handle_other_data_type(self, data):
        if isinstance(data, torch.Tensor):
            return self(data.cpu().numpy())
        else:
            return data
