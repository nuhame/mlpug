import abc

import numpy as np

import torch_xla.core.xla_model as xm

from mlpug.evaluation import GatherLoss
from mlpug.evaluation import average_loss
from mlpug.evaluation import MetricEvaluator as MetricEvaluatorBase

from mlpug.pytorch.evaluation import (
    GatherLossDistributedMixin,
    GatherDistributedTensorDataMixin,
    GatherDistributedTensorTupleMixin,
    GatherDistributedTensorDictMixin
)

from mlpug.pytorch.evaluation import (
    CombineBatchTuples,
    CombineBatchDicts,
)

from mlpug.pytorch.evaluation import (
    DefaultLossEvaluator,
)

from mlpug.pytorch.xla.multi_processing import MultiProcessingMixin


class GatherLossDistributed(MultiProcessingMixin, GatherLossDistributedMixin):
    def _reduce(self, tensor):
        return xm.mesh_reduce("GatherLossDistributed_reduce", tensor, np.sum)


# DEFAULT FUNCTION TO GATHER METRIC INPUT TENSORS IN DISTRIBUTED COMPUTING CONTEXT
class GatherDistributedTensorData(MultiProcessingMixin, GatherDistributedTensorDataMixin, metaclass=abc.ABCMeta):
    def _gather_distributed(self, tensor):
        return xm.mesh_reduce("GatherDistributedTensorData_gather_distributed", tensor, np.concatenate)


class GatherDistributedTensorTuple(MultiProcessingMixin, GatherDistributedTensorTupleMixin):
    pass


class GatherDistributedTensorDict(MultiProcessingMixin, GatherDistributedTensorDictMixin):
    pass


class MetricEvaluator(MultiProcessingMixin, MetricEvaluatorBase):
    def __init__(
            self,
            gather_metric_inputs_funcs=None,
            gather_distributed_inputs_funcs=None,
            combine_metric_inputs_funcs=None,
            combine_metric_inputs_func=None,
            metric_funcs=None,
            batch_dim=0,
            show_warning_distributed_metric_evaluation=True,
            name="MetricEvaluator",
            **kwargs
    ):
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

        if combine_metric_inputs_funcs is not None:
            for metric_name in metric_names:
                if metric_name not in combine_metric_inputs_funcs:
                    combine_metric_inputs_funcs[metric_name] = CombineBatchTuples(dim=batch_dim)

        if metric_funcs is None:
            metric_funcs = {}

        if "loss" not in metric_funcs:
            metric_funcs["loss"] = average_loss

        super().__init__(
            gather_metric_inputs_funcs=gather_metric_inputs_funcs,
            gather_distributed_inputs_funcs=gather_distributed_inputs_funcs,
            combine_metric_inputs_funcs=combine_metric_inputs_funcs,
            combine_metric_inputs_func=combine_metric_inputs_func,
            metric_funcs=metric_funcs,
            name=name,
            **kwargs
        )

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

