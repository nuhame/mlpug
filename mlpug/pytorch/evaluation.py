import abc
from typing import Collection, Tuple, Dict

import torch
import torch.distributed as dist

from mlpug.base import Base

from mlpug.evaluation import GatherLoss
from mlpug.evaluation import CombineBatchTuples as CombineBatchTuplesBase
from mlpug.evaluation import CombineBatchDicts as CombineBatchDictsBase
from mlpug.evaluation import average_loss
from mlpug.evaluation import MetricEvaluator as MetricEvaluatorBase

from mlpug.pytorch.multi_processing import MultiProcessingMixin


# DEFAULT FUNCTION TO GATHER LOSS IN DISTRIBUTED COMPUTING CONTEXT
# Using mixins for easy code reuse with MLPug for pytorch/xla
class GatherLossDistributedMixin(Base):

    def __init__(self, requester=None, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__

        if requester is not None:
            name += f'[{requester}]'

        super().__init__(pybase_logger_name=name, **kwargs)

    def __call__(self, loss_data: Tuple[torch.Tensor, int]):
        loss_sum, tot_num_samples = loss_data

        tot_num_samples = torch.tensor(tot_num_samples).to(loss_sum.device)

        if self.is_distributed:
            # Reduce to primary device
            loss_sum = self._reduce(loss_sum)
            tot_num_samples = self._reduce(tot_num_samples)

            if self.is_primary:
                return loss_sum.item(), tot_num_samples.item()
            else:
                return None, None
        else:
            return loss_sum.item(), tot_num_samples.item()

    def _reduce(self, tensor):
        dist.reduce(tensor, 0)
        return tensor


class GatherLossDistributed(MultiProcessingMixin, GatherLossDistributedMixin):
    pass


# DEFAULT FUNCTION TO GATHER METRIC INPUT TENSORS IN DISTRIBUTED COMPUTING CONTEXT
class GatherDistributedTensorDataMixin(Base, metaclass=abc.ABCMeta):
    def __init__(self,
                 device,
                 batch_dim=0,
                 convert_to_numpy=True,
                 requester=None,
                 name=None,
                 **kwargs):
        """
        If applicable, concatenate the tensors from different devices in a distributed computing setting

        :param device:
        """
        if name is None:
            name = self.__class__.__name__

        if requester is not None:
            name += f'[{requester}]'

        super().__init__(pybase_logger_name=name, **kwargs)
        self.device = device
        self.batch_dim = batch_dim
        self.convert_to_numpy = convert_to_numpy

    @abc.abstractmethod
    def __call__(self, gathered_input_data: Collection) -> Collection:
        raise NotImplementedError()

    def _gather(self, tensor):
        gathered_tensors = self._gather_distributed(tensor) if self.is_distributed else tensor

        if gathered_tensors is not None and self.convert_to_numpy:
            gathered_tensors = gathered_tensors.cpu().numpy()

        return gathered_tensors

    @abc.abstractmethod
    def _gather_distributed(self, tensor):
        raise NotImplementedError()


class GatherDistributedTensorTupleMixin:

    def __call__(self, gathered_input_data: Tuple) -> Collection:
        return tuple(self._gather(tensor.to(self.device))
                     for tensor in gathered_input_data)


class GatherDistributedTensorDictMixin:

    def __call__(self, gathered_input_data: Dict) -> Collection:
        return {metric_name: self._gather(tensor.to(self.device))
                for metric_name, tensor in gathered_input_data.items()}

class GatherDistributedTensorData(MultiProcessingMixin, GatherDistributedTensorDataMixin, metaclass=abc.ABCMeta):
    def _gather_distributed(self, tensor):
        gathered_tensors = None

        if self.is_primary:
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]

        torch.distributed.gather(tensor, gather_list=gathered_tensors)

        if self.is_primary:
            gathered_tensors = torch.concat(gathered_tensors, dim=self.batch_dim)

        return gathered_tensors


class GatherDistributedTensorTuple(GatherDistributedTensorTupleMixin, GatherDistributedTensorData):
    pass


class GatherDistributedTensorDict(GatherDistributedTensorDictMixin, GatherDistributedTensorData):
    pass


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
                 eager_mode: bool = False,
                 batch_dim=0,
                 show_warning_distributed_metric_evaluation=True,
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
            gather_distributed_inputs_funcs["loss"] = GatherLossDistributed(requester=name)

        if combine_metric_inputs_funcs is None:
            if not callable(combine_metric_inputs_func):
                combine_metric_inputs_funcs = {}

        if type(combine_metric_inputs_funcs) is dict:
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
                         metric_funcs=metric_funcs,
                         eager_mode=eager_mode,
                         name=name,
                         **kwargs)

        if not eager_mode:
            self._model_evaluate_func = torch.compile(self._model_evaluate_func)

            for metric_name in metric_names:
                gather_metric_inputs_func = self._gather_metric_inputs_funcs[metric_name]
                self._gather_metric_inputs_funcs[metric_name] = torch.compile(gather_metric_inputs_func)

        self._did_show_warning_distributed_metric_evaluation = not show_warning_distributed_metric_evaluation

    def _eval_metric_func(self, metric_func, metric_inputs, metric_name=None):
        if self.is_distributed and not self.is_primary:
            if not self._did_show_warning_distributed_metric_evaluation:
                self._log.warning(
                    f"Not evaluating metric functions on non-primary device. "
                    f"It is assumed that all metric inputs are gathered on primary device. "
                    f"This is default behaviour, override _eval_metric_func to change this behaviour.")
                self._did_show_warning_distributed_metric_evaluation = True

            return None

        return metric_func(metric_inputs)

    def _create_default_model_evaluate_func(self):
        return DefaultLossEvaluator(self._trainer)

    def _write_to_stdout(self, text):
        if self.is_primary:
            super()._write_to_stdout(text)