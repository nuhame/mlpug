import os

import abc

import torch
import torch.distributed as dist

from mlpug.evaluation import CombineBatchTuples as CombineBatchTuplesBase
from mlpug.evaluation import CombineBatchDicts as CombineBatchDictsBase
from mlpug.evaluation import average_loss
from mlpug.evaluation import MetricEvaluator as MetricEvaluatorBase

from mlpug.base import Base

from mlpug.pytorch.multi_processing import MultiProcessingMixin


# ####### DEFAULT GATHER LOSS METHODS ########
class GatherLoss(Base):
    """
    To calculate the average loss, this class GatherLoss that all the batches have equal size.
    """

    def __init__(self, requester=None, name="GatherLoss", **kwargs):
        if requester is not None:
            name += f'[{requester}]'

        super().__init__(pybase_logger_name=name, **kwargs)

        self.requester = requester

    def __call__(self, loss, **kwargs):
        loss = loss

        return loss, 1


class GatherMaskedLoss(Base):
    """
    Useful when using masking. In your TrainingModel, return the summed loss and number of unmasked samples as a part
    of the auxiliary result. E.g.:

    return loss, loss_sum, num_samples
    """

    def __init__(self, requester=None, name="GatherMaskedLoss", **kwargs):
        if requester is not None:
            name += f'[{requester}]'

        super().__init__(name, requester, **kwargs)

    def __call__(self, loss, auxiliary_results, **kwargs):
        loss_sum = auxiliary_results[0]
        num_samples = auxiliary_results[1]

        return loss_sum, num_samples
# ############################################


# DEFAULT FUNCTION TO GATHER LOSS IN DISTRIBUTED COMPUTING CONTEXT
def gather_loss_distributed(loss_sum, tot_num_samples):
    dist.all_reduce(loss_sum)
    dist.all_reduce(tot_num_samples)

    return loss_sum.item(), tot_num_samples.item()


# DEFAULT FUNCTION TO COMBINE GATHERED METRIC INPUTS
class ConcatTensorsMixin:

    def _handle_other_type(self, list_of_items, first_item=None):
        if isinstance(first_item, torch.Tensor):
            return torch.concat(list_of_items, dim=self.dim)
        else:
            return list_of_items


class CombineBatchTuples(ConcatTensorsMixin, CombineBatchTuplesBase):
    pass


class CombineBatchDicts(ConcatTensorsMixin, CombineBatchDictsBase):
    pass


class DefaultLossEvaluator:

    def __init__(self, trainer):
        self._trainer = trainer

    def __call__(self, batch, evaluate_settings=None):
        with torch.no_grad():
            results = self._trainer.evaluate_loss(
                batch,
                inference_mode=True,
                evaluate_settings=evaluate_settings)

        return results


class MetricEvaluator(MultiProcessingMixin, MetricEvaluatorBase):

    def __init__(self,
                 gather_metric_inputs_funcs=None,
                 gather_distributed_inputs_funcs=None,
                 combine_metric_inputs_funcs=None,
                 combine_metric_inputs_func=None,
                 metric_funcs=None,
                 batch_dim=0,
                 name="MetricEvaluator",
                 **kwargs):
        """

        See MetricEvaluatorBase for information on all parameters

        :param batch_dim: Default is 0, used by default combine_metric_inputs_funcs:
            CombineBatchTuples(dim=batch_dim)

        :param name:
        :param kwargs:
        """

        if gather_metric_inputs_funcs is None:
            gather_metric_inputs_funcs = {}

        if "loss" not in gather_metric_inputs_funcs:
            gather_metric_inputs_funcs["loss"] = GatherLoss(requester=name)

        metric_names = gather_metric_inputs_funcs.keys()

        if gather_distributed_inputs_funcs is None:
            gather_distributed_inputs_funcs = {}

        if "loss" not in gather_distributed_inputs_funcs:
            gather_distributed_inputs_funcs["loss"] = gather_loss_distributed

        if combine_metric_inputs_funcs is None:
            if not callable(combine_metric_inputs_func):
                combine_metric_inputs_funcs = {}

        if combine_metric_inputs_funcs is not None:
            for metric_name in metric_names:
                if metric_name not in combine_metric_inputs_funcs:
                    combine_metric_inputs_funcs[metric_name] = CombineBatchTuples(dim=batch_dim)

        if metric_funcs is None:
            metric_funcs = {}

        if "loss" not in metric_funcs:
            metric_funcs["loss"] = average_loss

        super().__init__(gather_metric_inputs_funcs=gather_metric_inputs_funcs,
                         gather_distributed_inputs_funcs=gather_distributed_inputs_funcs,
                         combine_metric_inputs_funcs=combine_metric_inputs_funcs,
                         combine_metric_inputs_func=combine_metric_inputs_func,
                         name=name,
                         **kwargs)

    def _create_default_model_evaluate_func(self):
        return DefaultLossEvaluator(self._trainer)

