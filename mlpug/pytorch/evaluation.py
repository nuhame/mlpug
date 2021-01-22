import os

from functools import reduce

import torch
import torch.distributed as dist

from mlpug.trainers.training import BatchChunkingResults
from mlpug.evaluation import default_metric_reducer_func, MetricEvaluatorBase

from mlpug.utils import is_chunkable

from basics.base import Base
from basics.logging import get_logger

logger = get_logger(os.path.basename(__file__))


# ####### DEFAULT GATHER LOSS METHODS ########
class GatherLossSimple(Base):

    def __init__(self, requester=None):
        name = "GatherLossSimple"
        if requester is not None:
            name += f'[{requester}]'

        super(GatherLossSimple, self).__init__(pybase_logger_name=name)

        self.requester = requester

        self._gather_loss_func = None
        if dist.is_initialized():
            self._log.info(f"Using simple distributed gather loss function")
            self._gather_loss_func = self._gather_loss_distributed
        else:
            self._log.info(f"Using simple gather loss function")
            self._gather_loss_func = self._gather_loss

    def __call__(self, *args, **kwargs):
        return self._gather_loss_func(*args, **kwargs)

    def _gather_loss(self, loss, **kwargs):
        loss = loss.item()
        return loss, loss, 1

    def _gather_loss_distributed(self, loss, **kwargs):
        loss_sum = loss
        dist.reduce(loss_sum, 0)
        num_devices = dist.get_world_size()
        loss = loss_sum / num_devices

        return loss.item(), loss_sum.item(), num_devices


class GatherMaskedLoss(Base):

    def __init__(self, requester=None):
        name = "GatherMaskedLoss"
        if requester is not None:
            name += f'[{requester}]'

        super(GatherMaskedLoss, self).__init__(pybase_logger_name=name)

        self.requester = requester

        self._gather_loss_func = None
        if dist.is_initialized():
            self._log.info(f"Using distributed gather masked loss function")
            self._gather_loss_func = self._gather_loss_distributed
        else:
            self._log.info(f"Using gather masked loss function")
            self._gather_loss_func = self._gather_loss

    def __call__(self, *args, **kwargs):
        return self._gather_loss_func(*args, **kwargs)

    def _gather_loss(self, auxiliary_results, **kwargs):
        # When auxiliary_results is a BatchChunkingResults list, it was created by batch chunking
        if type(auxiliary_results) is BatchChunkingResults:
            loss_sum = sum([aux[0] for aux in auxiliary_results])
            num_samples = sum([aux[1] for aux in auxiliary_results])
        else:
            loss_sum = auxiliary_results[0]
            num_samples = auxiliary_results[1]

        loss_sum = loss_sum.item()
        num_samples = num_samples.item()

        loss = loss_sum/num_samples

        return loss, loss_sum, num_samples

    def _gather_loss_distributed(self, auxiliary_results, **kwargs):
        # When auxiliary_results is a BatchChunkingResults list, it was created by batch chunking
        if type(auxiliary_results) is BatchChunkingResults:
            loss_sum = sum([aux[0] for aux in auxiliary_results])
            num_samples = sum([aux[1] for aux in auxiliary_results])
        else:
            loss_sum = auxiliary_results[0]
            num_samples = auxiliary_results[1]

        dist.reduce(loss_sum, 0)
        dist.reduce(num_samples, 0)

        loss_sum = loss_sum.item()
        num_samples = num_samples.item()

        loss = loss_sum / num_samples

        return loss, loss_sum, num_samples


# ############################################


class MetricEvaluator(MetricEvaluatorBase):

    def __init__(self, *args, batch_metric_funcs=None, name="MetricEvaluator", **kwargs):

        if batch_metric_funcs is None:
            batch_metric_funcs = {
                "loss": GatherLossSimple(requester=name)
            }

        super().__init__(batch_metric_funcs, *args, name=name, **kwargs)

    def _create_default_model_evaluate_func(self):

        def evaluate_loss(batch, evaluate_settings=None):
            if is_chunkable(batch):
                # Get raw batch
                batch = batch[:]

            with torch.no_grad():
                results = self._trainer.evaluate_loss(
                    batch,
                    inference_mode=True,
                    evaluate_settings=evaluate_settings)

            return results

        return evaluate_loss

