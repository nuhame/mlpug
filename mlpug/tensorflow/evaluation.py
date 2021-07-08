import os

import tensorflow as tf

from mlpug.evaluation import default_metric_reducer_func, MetricEvaluatorBase

from mlpug.utils import is_chunkable

from mlpug.base import Base
from basics.logging import get_logger

logger = get_logger(os.path.basename(__file__))


# ####### DEFAULT GATHER LOSS METHODS ########
class GatherLossSimple(Base):

    def __init__(self, distribution_strategy=None, requester=None):
        name = "GatherLossSimple"
        if requester is not None:
            name += f'[{requester}]'

        super(GatherLossSimple, self).__init__(pybase_logger_name=name)

        self._distribution_strategy = distribution_strategy
        self.requester = requester

        self._gather_loss_func = None
        if self._distribution_strategy is not None:
            self._log.info(f"Using simple distributed gather loss function")
            self._gather_loss_func = self._gather_loss_distributed
        else:
            self._log.info(f"Using simple gather loss function")
            self._gather_loss_func = self._gather_loss

    def __call__(self, *args, **kwargs):
        return self._gather_loss_func(*args, **kwargs)

    def _gather_loss(self, loss, **kwargs):
        loss = loss.numpy()
        return loss, loss, 1

    def _gather_loss_distributed(self, loss, **kwargs):
        loss = self._distribution_strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        loss = loss.numpy()

        return loss, loss, 1


class GatherMaskedLoss(Base):

    def __init__(self, distribution_strategy=None, requester=None):
        name = "GatherMaskedLoss"
        if requester is not None:
            name += f'[{requester}]'

        super(GatherMaskedLoss, self).__init__(pybase_logger_name=name)

        self._distribution_strategy = distribution_strategy
        self.requester = requester

        self._gather_loss_func = None
        if self._distribution_strategy is not None:
            self._log.info(f"Using distributed gather masked loss function")
            self._gather_loss_func = self._gather_loss_distributed
        else:
            self._log.info(f"Using gather masked loss function")
            self._gather_loss_func = self._gather_loss

    def __call__(self, *args, **kwargs):
        return self._gather_loss_func(*args, **kwargs)

    def _gather_loss(self, auxiliary_results, **kwargs):
        loss_sum = auxiliary_results[0].item()
        num_samples = auxiliary_results[1].item()

        loss = loss_sum/num_samples

        return loss, loss_sum, num_samples

    def _gather_loss_distributed(self, auxiliary_results, **kwargs):
        loss_sum_per_replica = auxiliary_results[0]
        num_samples_per_replica = auxiliary_results[1]

        loss_sum = self._distribution_strategy.reduce(tf.distribute.ReduceOp.SUM, loss_sum_per_replica, axis=None)
        num_samples = self._distribution_strategy.reduce(tf.distribute.ReduceOp.SUM, num_samples_per_replica, axis=None)

        loss = loss_sum/num_samples

        return loss.numpy(), loss_sum.numpy(), num_samples.numpy()

# ############################################


class MetricEvaluator(MetricEvaluatorBase):

    def __init__(self, *args, distribution_strategy=None, batch_metric_funcs=None, name="MetricEvaluator", **kwargs):

        self.distribution_strategy = distribution_strategy

        if batch_metric_funcs is None:
            batch_metric_funcs = {
                "loss": GatherLossSimple(distribution_strategy, requester=name)
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
