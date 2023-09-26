import abc
from functools import partial
from typing import Collection, Tuple, Dict

import os

import tensorflow as tf

from basics.logging import get_logger

from mlpug.base import Base

from mlpug.utils.utils import has_method

from mlpug.batch_chunking import ChunkableBatchDatasetInterface

from mlpug.evaluation import GatherLoss as GatherLossBase
from mlpug.evaluation import CombineBatchTuples as CombineBatchTuplesBase
from mlpug.evaluation import CombineBatchDicts as CombineBatchDictsBase
from mlpug.evaluation import average_loss
from mlpug.evaluation import MetricEvaluator as MetricEvaluatorBase

from mlpug.tensorflow.distributed_utils import create_distributed_func, contains_per_replica_data

from mlpug.tensorflow.batch_chunking import DistributedChunkableBatchDataset


logger = get_logger(os.path.basename(__file__))


class GatherLoss(GatherLossBase):

    def __call__(self, loss, num_samples, **kwargs):
        """
        Gathers the loss sum over the samples and the number of samples

        :param loss:
        :param num_samples:
        :param kwargs:
        :return:
        """
        # Convert back from average to loss sum
        return tf.cast(num_samples, loss.dtype)*loss, num_samples


# DEFAULT FUNCTION TO GATHER LOSS IN DISTRIBUTED COMPUTING CONTEXT
class GatherLossDistributed(Base):

    def __init__(self, distribution_strategy=None, requester=None, name=None, **kwargs):
        if name is None:
            name = self.__class__.__name__

        if requester is not None:
            name += f'[{requester}]'

        super().__init__(pybase_logger_name=name, **kwargs)

        self.distribution_strategy = distribution_strategy

    def __call__(self, loss_data: Tuple[tf.Tensor, tf.Tensor]):
        loss_sum, tot_num_samples = loss_data

        if self.distribution_strategy is not None:
            loss_sum = self.distribution_strategy.reduce(
                tf.distribute.ReduceOp.SUM,
                loss_sum,
                axis=None)

            tot_num_samples = self.distribution_strategy.reduce(
                tf.distribute.ReduceOp.SUM,
                tot_num_samples,
                axis=None)

        if has_method(loss_sum, "numpy"):
            loss_sum = loss_sum.numpy()

        if has_method(tot_num_samples, "numpy"):
            tot_num_samples = tot_num_samples.numpy()

        return loss_sum, tot_num_samples


# DEFAULT FUNCTION TO GATHER METRIC INPUT TENSORS IN DISTRIBUTED COMPUTING CONTEXT
class GatherDistributedTensorData(Base, metaclass=abc.ABCMeta):
    def __init__(self,
                 distribution_strategy=None,
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

        self.distribution_strategy = distribution_strategy

        self.batch_dim = batch_dim
        self.convert_to_numpy = convert_to_numpy

    @abc.abstractmethod
    def __call__(self, gathered_input_data: Collection) -> Collection:
        raise NotImplementedError()

    def _gather(self, tensor):

        if self.distribution_strategy is not None:
            tensor = self.distribution_strategy.gather(tensor, axis=self.batch_dim)

        if self.convert_to_numpy:
            tensor = tensor.numpy()

        return tensor


class GatherDistributedTensorTuple(GatherDistributedTensorData):

    def __call__(self, gathered_input_data: Tuple) -> Collection:
        return tuple(self._gather(tensor) for tensor in gathered_input_data)


class GatherDistributedTensorDict(GatherDistributedTensorData):

    def __call__(self, gathered_input_data: Dict) -> Collection:
        return {metric_name: self._gather(tensor)
                for metric_name, tensor in gathered_input_data.items()}


# DEFAULT FUNCTION TO COMBINE GATHERED METRIC INPUTS
class ConcatTensorsMixin:

    def _handle_other_type(self, list_of_items, first_item=None):
        if isinstance(first_item, tf.Tensor):
            return tf.concat(list_of_items, axis=self.dim)
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
        results = self._trainer.evaluate_loss(
            batch,
            inference_mode=True,
            evaluate_settings=evaluate_settings)

        return results


class MetricEvaluator(MetricEvaluatorBase):

    def __init__(self,
                 distribution_strategy=None,
                 gather_metric_inputs_funcs=None,
                 gather_distributed_inputs_funcs=None,
                 clean_up_batch_data_func=None,
                 combine_metric_inputs_funcs=None,
                 combine_metric_inputs_func=None,
                 metric_funcs=None,
                 batch_dim=0,
                 monitor_tracing=False,
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
            gather_distributed_inputs_funcs["loss"] = GatherLossDistributed(
                distribution_strategy=distribution_strategy,
                requester=name)

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
                         clean_up_batch_data_func=clean_up_batch_data_func,
                         combine_metric_inputs_funcs=combine_metric_inputs_funcs,
                         combine_metric_inputs_func=combine_metric_inputs_func,
                         metric_funcs=metric_funcs,
                         name=name,
                         **kwargs)

        # Wrap in as distributed and tf.function
        if distribution_strategy is not None:
            distributed_func_factory = partial(
                create_distributed_func,
                logger=self._log,
                monitor_tracing=monitor_tracing
            )

            for metric_name in metric_names:
                gather_metric_inputs_func = self._gather_metric_inputs_funcs[metric_name]
                gather_metric_inputs_func = tf.function(gather_metric_inputs_func)
                self._gather_metric_inputs_funcs[metric_name] = distributed_func_factory(
                    gather_metric_inputs_func,
                    distribution_strategy
                )

            if self._clean_up_batch_data_func is not None:

                clean_up_batch_data_func = self._clean_up_batch_data_func

                def distributed_clean_up_batch_data_func(batch, evaluate_settings, model_output):
                    unpacked_batch = distribution_strategy.experimental_local_results(batch)
                    unpacked_model_output = distribution_strategy.experimental_local_results(model_output)

                    for replica_batch, replica_model_output in zip(unpacked_batch, unpacked_model_output):
                        clean_up_batch_data_func(
                            batch=replica_batch,
                            evaluate_settings=evaluate_settings,
                            model_output=replica_model_output
                        )

                self._clean_up_batch_data_func = distributed_clean_up_batch_data_func

        self.distribution_strategy = distribution_strategy

    def _create_default_model_evaluate_func(self):
        return DefaultLossEvaluator(self._trainer)

    def _create_chunkable_batch_dataset(self, batch_data) -> ChunkableBatchDatasetInterface:
        if self.distribution_strategy is None or not contains_per_replica_data(batch_data):
            return super()._create_chunkable_batch_dataset(batch_data)
        else:
            return DistributedChunkableBatchDataset(
                batch_data,
                self._chunkable_batch_wrapper,
                self._batch_chunk_size,
                self.distribution_strategy
            )