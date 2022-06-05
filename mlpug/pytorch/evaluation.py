import abc
from typing import Collection, Tuple, Dict

import torch
import torch.distributed as dist

from evaluation import GatherLoss
from mlpug.evaluation import CombineBatchTuples as CombineBatchTuplesBase
from mlpug.evaluation import CombineBatchDicts as CombineBatchDictsBase
from mlpug.evaluation import average_loss
from mlpug.evaluation import MetricEvaluator as MetricEvaluatorBase

from mlpug.pytorch.multi_processing import MultiProcessingMixin


# DEFAULT FUNCTION TO GATHER LOSS IN DISTRIBUTED COMPUTING CONTEXT
def gather_loss_distributed(loss_data: Tuple):
    loss_sum, tot_num_samples = loss_data

    dist.all_reduce(loss_sum)
    dist.all_reduce(tot_num_samples)

    return loss_sum.item(), tot_num_samples.item()


# DEFAULT FUNCTION TO GATHER METRIC INPUT TENSORS IN DISTRIBUTED COMPUTING CONTEXT
class GatherTensorData(MultiProcessingMixin, metaclass=abc.ABCMeta):
    def __init__(self, device, batch_dim=0, **kwargs):
        """
        If applicable, concatenate the tensors from different devices in a distributed computing setting

        :param device:
        """
        super().__init__(**kwargs)
        self.device = device
        self.batch_dim = batch_dim

    @abc.abstractmethod
    def __call__(self, gathered_input_data: Collection):
        raise NotImplementedError()

    def _gather(self, tensor):
        gathered_tensors = None

        if self.is_primary:
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]

        torch.distributed.gather(tensor, gather_list=gathered_tensors)

        if self.is_primary:
            gathered_tensors = torch.concat(gathered_tensors, dim=self.batch_dim)

        return gathered_tensors


class GatherTensorTuple(GatherTensorData):

    def __call__(self, gathered_input_data: Tuple):
        return (self._gather(tensor) for tensor in gathered_input_data)


class GatherTensorDict(GatherTensorData):

    def __call__(self, gathered_input_data: Dict):
        return {metric_name: self._gather(tensor)
                for metric_name, tensor in gathered_input_data.items()}


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

