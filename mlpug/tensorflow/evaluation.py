import os

import tensorflow as tf

from mlpug.evaluation import MetricEvaluatorBase

from mlpug.utils import is_chunkable

from basics.logging import get_logger

logger = get_logger(os.path.basename(__file__))


# ####### DEFAULT GATHER LOSS METHODS ########
def gather_loss(loss, **kwargs):
    return loss.numpy(), 1


def create_gather_distributed_loss_func(distribution_strategy):

    # Tensorflow distribution strategies sum gradients, so to reflect this in the loss, it needs to be summed
    def gather_distributed_loss(loss, **kwargs):
        loss = distribution_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

        return loss.numpy(), 1

    return gather_distributed_loss


def create_default_gather_loss_func(distribution_strategy, requester=None):
    if requester is None:
        requester = ''
    else:
        requester += ' : '

    if distribution_strategy is not None:
        logger.info(f"{requester}Using default distributed gather loss function")
        gather_loss_func = create_gather_distributed_loss_func(distribution_strategy)
    else:
        logger.info(f"{requester}Using default gather loss function")
        gather_loss_func = gather_loss

    return gather_loss_func
# ############################################


class MetricEvaluator(MetricEvaluatorBase):

    def __init__(self, *args, distribution_strategy=None, batch_metric_funcs=None, name="MetricEvaluator", **kwargs):

        self.distribution_strategy = distribution_strategy

        if batch_metric_funcs is None:
            batch_metric_funcs = {
                "loss": create_default_gather_loss_func(distribution_strategy, requester=name)
            }

        super().__init__(batch_metric_funcs, *args, name=name, **kwargs)

    def _create_default_model_evaluate_func(self):

        def evaluate_loss(batch, evaluate_settings=None):
            if is_chunkable(batch):
                # Get raw batch
                batch = batch[:]

            results = self._trainer.evaluate_loss(
                batch,
                inference_mode=True,
                evaluate_settings=evaluate_settings)

            return results

        return evaluate_loss
