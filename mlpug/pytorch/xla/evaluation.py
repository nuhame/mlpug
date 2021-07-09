import torch_xla.core.xla_model as xm

import numpy as np

from mlpug.trainers.training import BatchChunkingResults
from mlpug.evaluation import default_metric_reducer_func

from mlpug.pytorch.evaluation import MetricEvaluator as MetricEvaluatorPyTorch

from mlpug.utils import is_chunkable

from mlpug.pytorch.evaluation import \
    GatherLossSimple as GatherLossSimplePytorch, \
    GatherMaskedLoss as GatherMaskedLossPyTorch


class GatherLossSimple(GatherLossSimplePytorch):

    def _gather_loss_distributed(self, loss, **kwargs):
        loss_sum = xm.mesh_reduce('gather_loss', loss, np.sum)
        num_devices = xm.xrt_world_size()
        loss = loss_sum / num_devices

        return loss, loss_sum, num_devices


class GatherMaskedLoss(GatherMaskedLossPyTorch):

    def _gather_loss_distributed(self, loss_sum, num_samples, **kwargs):
        loss_sum = xm.mesh_reduce('gather_loss', loss_sum, np.sum)
        num_samples = xm.mesh_reduce('gather_num_samples', num_samples, np.sum)

        loss = loss_sum / num_samples

        return loss, loss_sum, num_samples


class MetricEvaluator(MetricEvaluatorPyTorch):

    def __init__(self, *args, batch_metric_funcs=None, name="MetricEvaluator", **kwargs):

        if batch_metric_funcs is None:
            batch_metric_funcs = {
                "loss": GatherLossSimple(requester=name)
            }

        super().__init__(*args, batch_metric_funcs=batch_metric_funcs, name=name, **kwargs)
