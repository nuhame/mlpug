import sys
import abc

import math

import numpy as np
from mlpug.base import Base

import basics.base_utils as _

from mlpug.mlpug_exceptions import BatchNotChunkableException, InvalidChunkableBatch, InvalidParametersException
from mlpug.utils import *


def forward_loss(loss, **kwargs):
    return loss, 1


def default_metric_reducer_func(batch_metrics_list):
    # unzip
    _, metric_sum_list, num_samples_list = list(zip(*batch_metrics_list))

    metric_sum = sum(metric_sum_list)
    num_samples = sum(num_samples_list)

    metric = metric_sum/num_samples

    return metric, metric_sum, num_samples


class ChunkableTupleBatchBase(Base, metaclass=abc.ABCMeta):

    def __init__(self, *batch, dim=0):
        super().__init__()

        self._batch = batch
        self._dim = dim

    def __len__(self):
        # get batch size
        for v in self._batch:
            if not hasattr(v, 'shape'):
                continue

            if self._dim < len(v.shape):
                return v.shape[self._dim]
            else:
                raise InvalidChunkableBatch(
                    f"Unable to assess length of chunkable batch. "
                    f"The tensors in the batch should at least have {self._dim+1} dimension(s)."
                )

        raise InvalidChunkableBatch(
            "Unable to assess length of chunkable batch. "
            "A chunkable batch should contain at least one tensors that has a shape attribute."
        )

    @abc.abstractmethod
    def __getitem__(self, sample_slice):
        raise NotImplementedError("Please implement this in your child class")


class ChunkableTupleBatchDim0(ChunkableTupleBatchBase):

    def __init__(self, *batch):
        super().__init__(*batch, dim=0)

    def __getitem__(self, sample_slice):
        return tuple(v[sample_slice, ...] if v is not None else None
                     for v in self._batch)


class ChunkableTupleBatchDim1(ChunkableTupleBatchBase):

    def __init__(self, *batch):
        super().__init__(*batch, dim=1)

    def __getitem__(self, sample_slice):
        return tuple(v[:, sample_slice, ...] if v is not None else None
                     for v in self._batch)


class ChunkableBatchDataset(Base):

    def __init__(self, batch, batch_chunk_size):
        """
        Turns a chunkable batch in to an iterable dataset
        
        :param batch: A chunkable batch must implement the `__len__` and `__getitem__` methods.
                      len(batch) must return the number of batch samples
                      Here the `__getitem__` method must be able to deal with slices.

        :param batch_chunk_size:
                      The sample size of each batch chunk
        """
        super().__init__()

        self._batch = batch
        self._batch_chunk_size = batch_chunk_size

        if not is_chunkable(batch):
            raise BatchNotChunkableException()

        self._batch_size = len(batch)
        self._num_chunks = math.ceil(self._batch_size / self._batch_chunk_size)
        self._chunk_idx = -1

    def __iter__(self):
        self._chunk_idx = -1
        return self

    def __next__(self):
        self._chunk_idx += 1

        if self._chunk_idx >= self._num_chunks:
            raise StopIteration()

        chunk_start = self._chunk_idx * self._batch_chunk_size
        chunk_end = min((self._chunk_idx + 1) * self._batch_chunk_size, self._batch_size)

        return self._batch[chunk_start:chunk_end]


class MetricEvaluatorBase(Base, metaclass=abc.ABCMeta):

    def __init__(self,
                 batch_metric_funcs,
                 model_evaluate_func=None,
                 trainer=None,
                 batch_metric_reducer_funcs=None,
                 batch_chunk_size=None,
                 batch_chunk_metric_reducer_funcs=None,
                 show_dataset_evaluation_progress=False,
                 name="MetricEvaluatorBase",
                 **kwargs):
        """

        TODO : Add more documentation

        :param batch_metric_funcs: A dict with keys representing the metric names
             (e.g. "loss", "recall", etc.) and the corresponding values are functions to calculate a
             metric value, or to gather information to calculate or reduce an overall metric value, also see
             batch_metric_reducer_funcs.

             The functions will be called as follows:

             metric_func(**kwargs)

             Where kwargs will contain the following keys:
             'batch', 'evaluate_settings' and the keys of the model evaluation results, see model_evaluate_func
             Usually that is 'loss' and 'auxiliary_results'.

             Example batch_metric_funcs dict:

                 def gather_loss_data(batch, loss, **kwargs):
                    inputs = batch[0]
                    num_samples = inputs.shape[1]

                    return loss, num_samples

                 batch_metric_funcs = {
                    'loss': gather_loss_data
                 }

             Using the above function, loss data can be gathered over multiple batches and reduced to, e.g., an
             average loss using a reducer function, see batch_metric_reducer_funcs constructor parameter and the
             default_metric_reducer_func in this package.

             NOTE : Although the above gather_loss_data function returns a tuple of loss, num_samples, the LogProgress
             logger will always only print the first value in the tuple, that is, the loss

             The function can also return a dict with metrics. For instance:

                 def calc_metrics(loss, **kwargs):
                    return {
                        "loss": loss,
                        "perplexity": math.exp(loss),
                    }

                 batch_metric_funcs = {
                    'metrics': calc_metrics
                 }

             In the example below, per batch, target and predicted values are gathered and subsequently reduced
             to a recall score using a reducer func, also see batch_metric_reducer_funcs:

                 def get_target_and_predicted(batch, auxiliary_results, **kwargs):
                        target = batch[1]
                        predicted = auxiliary_results[0]

                        return target, predicted

                 batch_metric_funcs = {
                    'recall': get_target_and_predicted
                 }

                 The corresponding reducer_func could be:

                def calc_recall(batch_metrics_list):
                    target = []
                    predicted = []
                    # append all batch-level data
                    for t, p in batch_metrics_list:
                        target.append(t)
                        predicted.append(p)

                    return recall_score(t, p)

                 batch_metric_reducer_funcs = {
                    'recall': calc_recall
                 }

        :type batch_metric_funcs: dict

        :param model_evaluate_func: Optional. f(batch_data, evaluate_settings) -> model_output
                                    Instead of providing a model_evaluate_func, you can provide a trainer instance.
                                    Also see below.

                                    IMPORTANT: it is assumed that the `model_evaluate_func` takes all
                                    appropriate measures to disable training specific layers such as
                                    dropout and gradient calculations.

                                    Eg. for Pytorch:

                                    def eval_model(batch_data, evaluate_settings):

                                        my_training_model.eval()

                                        with torch.no_grad():
                                            return my_training_model(batch_data, evaluate_settings)

        :type model_evaluate_func: callable

        :param trainer: An optional trainer instance to evaluate a model, you can provide this instead of a
                        custom model_evaluate_func
        :type trainer: TrainerBase child instance

        :param batch_metric_reducer_funcs: Optional.
                 A dict with keys representing the metric names (e.g. "loss", "recall", etc.) and
                 the corresponding values are functions to calculate the overall or reduced metric value,
                 based on metrics, or other data, gathered per batch, also see batch_metric_funcs

                 The functions will be called as follows:

                 reducer_func(batch_metrics_list), where batch_metrics_list is a list with metrics or
                 other data gathered per batch.

                 See `batch_metric_funcs` for example.

                 By default, for each metric name, available as keys of the batch_metric_funcs, an averaging function
                 is provided. This averaging function assumes a list of tuples:
                 (summed_metric, num_samples)
                 ...
                 (summed_metric, num_samples)

                 See default_metric_reducer_func in this package.

        :type batch_metric_reducer_funcs: dict

        :param batch_chunk_size: If given, batches will be evaluated by chunking it to
                 smaller batches of size batch_chunk_size

                 When specifying this option, you can specify batch_chunk_metric_reducer_funcs.
                 The way the batch is evaluated works as follows:

                 for all chunks in batch:
                    eval chunk
                    use eval result to calculate batch metrics with batch_metric_funcs

                 Reduce all batch metric results of the chunks using batch_chunk_metric_reducer_funcs
        :type batch_chunk_size: int

        :param batch_chunk_metric_reducer_funcs: Optional, when providing batch_chunk_size
            Similar to batch_metric_reducer_funcs, reduces the batch metric results for all batch chunk to a final
            metric

            If batch_chunk_size is provided, but no batch_chunk_metric_reducer_funcs are provided, the
            batch_metric_reducer_funcs are used instead by default.

        :type batch_chunk_metric_reducer_funcs: dict

        :param show_dataset_evaluation_progress If True, the progress of dataset evaluation will be logged
        :type show_dataset_evaluation_progress
        """

        super().__init__(**kwargs)

        if trainer is not None:
            if model_evaluate_func is not None:
                raise InvalidParametersException("Either provide a trainer instance or a model_evaluate_func, not both")

            self._trainer = trainer

            self._model_evaluate_func = self._create_default_model_evaluate_func()
        else:
            if not callable(model_evaluate_func):
                raise InvalidParametersException("If no trainer object is provided, "
                                                 "you should provide a callable model_evaluate_func")

            self._model_evaluate_func = model_evaluate_func

        try:
            if batch_metric_funcs is None:
                batch_metric_funcs = {
                    "loss": forward_loss
                }

            self.check_funcs(batch_metric_funcs)
        except InvalidParametersException as e:
            raise InvalidParametersException("The batch metrics funcs are invalid") from e

        if batch_metric_reducer_funcs is None:
            batch_metric_reducer_funcs = {}

        # Add default metric averaging funcs for metrics that don't have a metric averaging func provided:
        for metric_name in batch_metric_funcs.keys():
            if metric_name not in batch_metric_reducer_funcs:
                self._log.debug(f'batch_metric_reducer_funcs: '
                                f'Adding default_metric_reducer_func for metric {metric_name}')
                batch_metric_reducer_funcs[metric_name] = default_metric_reducer_func

        try:
            self.check_funcs(batch_metric_reducer_funcs, func_names=batch_metric_funcs.keys())
        except InvalidParametersException as e:
            raise InvalidParametersException("The batch metric reducer funcs are invalid") from e

        if batch_chunk_size is not None:
            if batch_chunk_metric_reducer_funcs is None:
                batch_chunk_metric_reducer_funcs = batch_metric_reducer_funcs.copy()

            # Add default metric averaging funcs for metrics that don't have a metric averaging func provided:
            for metric_name in batch_metric_funcs.keys():
                if metric_name not in batch_chunk_metric_reducer_funcs:
                    self._log.debug(f'batch_chunk_metric_reducer_funcs: '
                                    f'Adding default_metric_reducer_func for metric {metric_name}')
                    batch_chunk_metric_reducer_funcs[metric_name] = default_metric_reducer_func

            try:
                self.check_funcs(batch_chunk_metric_reducer_funcs, func_names=batch_metric_funcs.keys())
            except InvalidParametersException as e:
                raise InvalidParametersException("The batch CHUNK metric reducer funcs are invalid") from e

        self._batch_metric_funcs = batch_metric_funcs
        self._batch_metric_reducer_funcs = batch_metric_reducer_funcs

        self._batch_chunk_size = batch_chunk_size
        self._batch_chunk_metric_reducer_funcs = batch_chunk_metric_reducer_funcs

        self._show_dataset_evaluation_progress = show_dataset_evaluation_progress

        self._name = name

    def get_name(self):
        return self._name

    def get_metric_names(self):
        """
        Get the names of the metrics that are calculated by this `MetricEvaluator`.

        :return:
        :rtype:
        """
        return list(self._batch_metric_funcs.keys())

    def calc_batch_metrics_for(self,
                               batch_data,
                               metrics_output,
                               evaluate_settings=None,
                               model_output=None,
                               force_no_chunking=False):
        """

        Calculate metrics of given batch, optionally applying evaluate_settings

        :param batch_data:
        :type batch_data:

        :param metrics_output: Dict to which the calculated metrics will be written
        :type metrics_output: Dict

        :param evaluate_settings:
        :type evaluate_settings:

        :param model_output: Optional, externally calculated model output for given batch_data
        :type model_output: dict

        :param force_no_chunking: Optional, will force not to chunk batch when set to true
        :param force_no_chunking: bool

        :return: True on success, else False
        :rtype: Bool

        """
        if not force_no_chunking and self._batch_chunk_size is not None and model_output is None:
            chunk_dataset = ChunkableBatchDataset(batch_data, self._batch_chunk_size)

            # Set force_no_chunking = True, because the batches will now already be chunks of a batch
            return self.calc_dataset_metrics_for(
                chunk_dataset,
                metrics_output,
                evaluate_settings=evaluate_settings,
                batch_metric_reducer_funcs=self._batch_chunk_metric_reducer_funcs,
                show_dataset_evaluation_progress=False,
                force_no_chunking=True,
                dataset_name="batch chunks")

        if not can_get_and_set_items(metrics_output):
            self._log.error(f"The given metrics output variable is not valid ({metrics_output}), "
                            f"unable to calculate metrics on batch")
            return False

        if model_output is None:
            try:
                model_output = self._model_evaluate_func(batch_data, evaluate_settings)
            except Exception as e:
                _.log_exception(self._log, "Evaluating model on batch input data failed", e)
                return None, False

        metric_func_args = {**model_output, **{
            'batch': batch_data,
            'evaluate_settings': evaluate_settings
        }}

        success = True
        for metric_name, batch_metric_func in self._batch_metric_funcs.items():
            try:
                metrics_output[metric_name] = batch_metric_func(**metric_func_args)
            except Exception as e:
                _.log_exception(self._log, f"Evaluating metric {metric_name} using model batch output failed", e)
                success = False

        return success

    def calc_dataset_metrics_for(self,
                                 dataset,
                                 metrics_output,
                                 evaluate_settings=None,
                                 dataset_name=None,
                                 batch_metric_reducer_funcs=None,
                                 show_dataset_evaluation_progress=None,
                                 force_no_chunking=False):
        """
        Calculate metrics over whole dataset, by looping over the batches in the given dataset, optionally
        applying evaluate_settings

        :param dataset:
        :type dataset:

        :param metrics_output: Dict to which the calculated metrics will be written
        :type metrics_output: Dict

        :param evaluate_settings:
        :type evaluate_settings:

        :param dataset_name:
        :type dataset_name:

        :param batch_metric_reducer_funcs: Optional alternative dict wth reducer functions
        :type batch_metric_reducer_funcs: dict

        :param show_dataset_evaluation_progress: Optional alternative value
        :type show_dataset_evaluation_progress: bool

        :param force_no_chunking: Optional, will force not to chunk batch when set to true
        :param force_no_chunking: bool

        :return: True on success, else False
        :rtype: Bool
        """
        if not hasattr(dataset, '__iter__'):
            self._log.error(f"The given dataset {str(dataset)} is not iterable, unable to calculate metrics on dataset")
            return False

        if not can_get_and_set_items(metrics_output):
            self._log.error(f"The given metrics output variable is not valid ({metrics_output}), "
                            f"unable to calculate metrics on dataset")
            return False

        if show_dataset_evaluation_progress is None:
            show_dataset_evaluation_progress = self._show_dataset_evaluation_progress

        metric_names = list(self._batch_metric_funcs.keys())

        if show_dataset_evaluation_progress:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self._log.debug(f"Calculating metrics ({', '.join(metric_names)}) on whole "
                            f"{'' if dataset_name is None else dataset_name} dataset")

        dataset_iterator = iter(dataset)

        batch_metric_data_lists = {}

        metric_paths = None
        for dataset_batch in dataset_iterator:
            batch_metric_data_map = {}
            batch_success = self.calc_batch_metrics_for(dataset_batch,
                                                        batch_metric_data_map,
                                                        evaluate_settings,
                                                        force_no_chunking=force_no_chunking)
            if not batch_success:
                return False

            if metric_paths is None:
                metric_paths = get_key_paths(batch_metric_data_map,
                                             keys_to_consider=metric_names,
                                             keys_not_to_consider=["auxiliary_results"])

            for metric_path in metric_paths:
                batch_metric_data = get_value_at(metric_path, batch_metric_data_map)

                batch_metric_data_list = batch_metric_data_lists[metric_path] \
                    if metric_path in batch_metric_data_lists else None

                if batch_metric_data_list is None:
                    batch_metric_data_list = []
                    batch_metric_data_lists[metric_path] = batch_metric_data_list

                batch_metric_data_list += [batch_metric_data]

            if show_dataset_evaluation_progress:
                sys.stdout.write('#')
                sys.stdout.flush()

        if show_dataset_evaluation_progress:
            sys.stdout.write('\n')
            sys.stdout.flush()

        return self.reduce(batch_metric_data_lists,
                           metrics_output,
                           batch_metric_reducer_funcs=batch_metric_reducer_funcs,
                           dataset_name=dataset_name)

    def reduce(self, batch_metric_data_lists, metrics_output, batch_metric_reducer_funcs=None, dataset_name=None):
        """

        Use the `batch_metric_reducer_funcs` to reduce the lists available in the `batch_metric_data_lists` dict

        The keys of `batch_metric_data_lists` must be the key path to the reduced metrics. Example:
        batch_metric_data_lists = {
            'loss': [ ... data to use to calculated loss ... ],
            'classification.F1': [ ... data to use to calculated F1 ... ]
        }

        To get the correct key paths you can use the `get_key_paths` as a utility

        :param batch_metric_data_lists: See above.
        :type batch_metric_data_lists: Dict

        :param metrics_output: Dict to which the reduced metrics will be written
        :type metrics_output: Dict

        :param batch_metric_reducer_funcs: Optional alternative dict wth reducer functions
        :type batch_metric_reducer_funcs: dict

        :param dataset_name: Used for logging purposes
        :type dataset_name:

        :return: True on success, else False
        :rtype:
        """
        success = True

        if batch_metric_reducer_funcs is None:
            batch_metric_reducer_funcs = self._batch_metric_reducer_funcs

        for metric_path, batch_metric_data_list in batch_metric_data_lists.items():
            try:
                reducer_func = get_value_at(metric_path, batch_metric_reducer_funcs)
                set_value_at(metric_path, metrics_output, reducer_func(batch_metric_data_list))
            except Exception as e:
                _.log_exception(self._log, f"Exception occurred reducing {metric_path} for "
                                           f"{'' if dataset_name is None else dataset_name} dataset "
                                           f"batch metric data", e)
                success = False

        return success

    def _create_default_model_evaluate_func(self):
        """
        Creates function that evaluates model using the provided trainer instance.
        This needs to be implemented for any specific deep learning framework

        :return: The returned function has the following pattern:

                    f(batch, evaluate_settings) -> results

        :rtype: function (callable)
        """
        return lambda batch_data, settings: self._trainer.evaluate_loss(batch_data,
                                                                        inference_mode=True,
                                                                        evaluate_settings=settings)

    def __str__(self):
        return self.get_name()

    @staticmethod
    def check_funcs(fdict, func_names=None):
        if is_empty(fdict) or not can_get_items(fdict):
            raise InvalidParametersException("A dict with one or more functions must be provided")

        if func_names is None:
            func_names = fdict.keys()

        for func_name in func_names:
            func = fdict[func_name]

            if not callable(func):
                raise InvalidParametersException(f"The function at {func_name} is not valid. "
                                                 f"The values of the function dictionary must be functions")

    @staticmethod
    def is_valid(ev):
        return has_method(ev, 'get_metric_names') and \
               has_method(ev, 'calc_batch_metrics_for') and \
               has_method(ev, 'calc_dataset_metrics_for') and \
               has_method(ev, 'reduce')
