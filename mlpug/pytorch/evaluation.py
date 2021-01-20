import os

import torch
import torch.distributed as dist

from mlpug.evaluation import MetricEvaluatorBase

from mlpug.utils import is_chunkable

from basics.logging import get_logger

logger = get_logger(os.path.basename(__file__))


# ####### DEFAULT GATHER LOSS METHODS ########
def gather_loss(loss, **kwargs):
    loss = loss.item()
    return loss, loss, 1


def create_gather_distributed_loss_func():

    # Pytorch Distributed Data Parallel averages gradients, so to reflect this in the loss, it needs to be averaged
    def gather_distributed_loss(loss, **kwargs):
        loss_sum = loss
        dist.reduce(loss_sum, 0)
        num_devices = dist.get_world_size()
        loss = loss_sum/num_devices

        return loss.item(), loss_sum.item(), num_devices

    return gather_distributed_loss


def create_default_gather_loss_func(requester=None):
    if requester is None:
        requester = ''
    else:
        requester += ' : '

    if dist.is_initialized():
        logger.info(f"{requester}Using default distributed gather loss function")
        gather_loss_func = create_gather_distributed_loss_func()
    else:
        logger.info(f"{requester}Using default gather loss function")
        gather_loss_func = gather_loss

    return gather_loss_func
# ############################################


class MetricEvaluator(MetricEvaluatorBase):

    def __init__(self, *args, batch_metric_funcs=None, name="MetricEvaluator", **kwargs):

        if batch_metric_funcs is None:
            batch_metric_funcs = {
                "loss": create_default_gather_loss_func(requester=name)
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

