import sys
import abc

from typing import Iterable, Tuple, Dict, Collection

import numpy as np

from mlpug.base import Base
import basics.base_utils as _

from mlpug.batch_chunking import ChunkableBatchDataset

from mlpug.mlpug_exceptions import InvalidParametersException
from mlpug.utils import *


def gather_loss(loss, **kwargs):
    """
    Warning: this simple default implementation assumes that all batches have the same number of samples.

    :param loss:
    :param kwargs:
    :return:
    """
    return loss, 1


def default_metric_reducer_func(batch_metrics_list):
    # unzip
    _, metric_sum_list, num_samples_list = list(zip(*batch_metrics_list))

    metric_sum = sum(metric_sum_list)
    num_samples = sum(num_samples_list)

    metric = metric_sum/num_samples

    return metric, metric_sum, num_samples


def no_reduction(batch_metrics_list):
    return batch_metrics_list


class CombineBatchData(metaclass=abc.ABCMeta):

    def __init__(self, dim: int = 0):
        """
        :param dim: dimension of to concatenate the numpy arrays over, if relevant
        """
        self.dim = dim

    @abc.abstractmethod
    def __call__(self, iterable: Iterable[Collection]) -> Collection:
        """
        Assumes that the input is a iterable of batch data, where each batch data object contains items that are
        numpy arrays and/or int/float scalars.

        When a batch data item value is a numpy array, the array values for this item are concatenated
        When a batch data item value is a float/int scalar, the array values for this item are summed

        In other cases no specific combination is made, these item values will just be combined in to a list.

        :param iterable:
        :return:
        """
        raise NotImplementedError("Please implement in your child class.")

    def _concat(self, list_of_items):
        try:
            first_item = next(iter(list_of_items))
        except StopIteration:
            return list_of_items

        if isinstance(first_item, (float, int, np.number)):
            return sum(list_of_items)
        elif isinstance(first_item, np.ndarray):
            return np.concatenate(list_of_items, axis=self.dim)
        else:
            return self._handle_other_type(list_of_items, first_item=first_item)

    def _handle_other_type(self, list_of_items, first_item=None):
        return list_of_items


class CombineBatchTuples(CombineBatchData):

    def __call__(self, tuples: Iterable[Tuple]) -> Tuple:
        """

        Assumes that batch data in the iterable are tuples.
        Also see CombineBatchData.


        :param tuples:    Real world example, with first item NOT a numpy array or int/float scalar:
                          [
                              # Batch 1
                              (labels Numpy Array, predictions Numpy array, num_samples int, other data type),
                              ...
                              # Batch M
                              (labels Numpy Array, predictions Numpy array, num_samples int, other data type),
                          ]

        :return: Result for real world example:
                 (all M label numpy arrays concatenated,
                  all M prediction numpy arrays concatenated,
                  all M num_samples summed,
                  all M values of other data type in a list)

        """

        lists_of_items = zip(*tuples)

        return tuple(self._concat(list_of_items) for list_of_items in lists_of_items)


class CombineBatchDicts(CombineBatchData):

    def __call__(self, dicts: Iterable[Dict]) -> Dict:
        """

        Assumes that batch data in the iterable are dict's.
        Also see CombineBatchData.

        :param dicts:     Real world example:
                          [
                              {
                                "labels": Batch 1 Numpy Array with labels,
                                "predictions": Batch 1 Numpy Array with predictions,
                                "num_samples": Batch 1 num. samples (int),
                                "other_data": Batch 1 other data not of type Numpy Array or scalar
                              },
                              ...
                              {
                                "labels": Batch M Numpy Array with labels,
                                "predictions": Batch M Numpy Array with predictions,
                                "num_samples": Batch M num. samples (int),
                                "other_data": Batch 1 other data not of type Numpy Array or scalar
                              }
                          ]

        :return: Result for real world example:
                {
                    "labels": all M label numpy arrays concatenated,
                    "predictions": all M prediction numpy arrays concatenated,
                    "num_samples": sum of all M num_samples,
                    "other_data": all M values of other data type in a list
                }
        """
        try:
            first_dict = next(iter(dicts))
            keys = first_dict.keys()
        except StopIteration:
            return {}

        lists_of_items = zip(*[d.values() for d in dicts])

        return {key: self._concat(list_of_items) for key, list_of_items in zip(keys, lists_of_items)}


class MetricEvaluator(Base, metaclass=abc.ABCMeta):

    def __init__(self,
                 model_evaluate_func=None,
                 trainer=None,
                 gather_metric_inputs_funcs=None,
                 gather_distributed_inputs_funcs=None,
                 combine_metric_inputs_funcs=None,
                 metric_funcs=None,
                 batch_chunk_size=None,
                 show_dataset_evaluation_progress=False,
                 name="MetricEvaluator",
                 **kwargs):

        """

        A MetricEvaluator can evaluate metrics using the batch data and the training model output.
        It can do this at different levels:
         * Metrics calculation on the level of a single batch, either directly or
           by first chunking the batch in to smaller batch chunks (to reduce memory usage)

         * Metrics calculation on the level of multiple batches, e.g. over a dataset or window of batches

        To evaluate the metrics, the MetricEvaluator, needs a way to evaluate the training model.
        This is done by either providing a model_evaluate_func, or a trainer instance, since a trainer
        knows how to evaluate the training model.


        In order to accurately calculate the metrics of interest, combining all the samples in a batch, or batches,
        we call a few different types of functions in order:

         1) `gather_metric_inputs_funcs`, a dict with, for each metric, a function that can
            gather inputs, from the batch data and model output, to calculate the metric of interest with

         2) `gather_distributed_inputs_funcs`, a dict with, for each metric, a function that can
            gather the metric inputs from different devices in a distributed computing context, if applicable

         3) `combine_metric_inputs_funcs`, a dict with, for each metric, a function that can
            combine metric inputs from multiple batches, e.g. data from multiple batch chunks, batches of a dataset,
            or of a window of batches.

         4) `metric_funcs`, a dict with, for each metric, a function to
            calculate the metric based on the gathered/combined metric inputs

        In a default situation, for loss no functions need to be provided, defaults are available for all functions
        For other metrics, at least a gather_metric_inputs_func and a metric_func needs to be provided.

        In summary, for each metric the functions are called as follows:
        batch_metric_inputs = gather_metric_inputs_func(batch, evaluate_settings, loss, auxiliary_results)
        batch_metric_inputs = gather_distributed_inputs_func(batch_metric_inputs)

        In case of calculating metrics for a single batch:
            metric = metric_func(batch_metric_inputs)

        In case of calculating metrics over multiple batches, or batch chunks:
            combined_metric_inputs = combine_metric_inputs_func([batch_metric_inputs1, ..., batch_metric_inputsM])
            metric = metric_func(combined_metric_inputs)

        Hence, the data structure of batch_metric_inputs and combined_metric_inputs must be the same.


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

        :param trainer: An optional trainer instance to evaluate a model.
            You can provide this instead of a custom model_evaluate_func

        :param gather_metric_inputs_funcs: A dict with keys representing the metric names
            (e.g. "loss", "classification", etc.), the values are functions that can
            gather inputs, from the batch data and model output, to calculate the metric of interest with

            In general there is no strict signature for the functions here.
            However, for defaults the following is assumed:
                func(batch, evaluate_settings, loss, auxiliary_results) -> Tuple[Union[T, int, float]]
                Where T is a Tensor.

            Default: If no gather_metric_inputs_funcs are provided, a default implementation to only gather
            loss data is used.

            The functions will be called as follows:

                gather_metric_inputs_func(**kwargs)

                Where kwargs will contain the following keys:
                'batch', 'evaluate_settings' and the keys of the model evaluation results, see model_evaluate_func
                Usually this is 'loss' and 'auxiliary_results'.

                Example gather_batch_data_funcs dict:

                    def gather_loss_data(batch, loss, **kwargs):
                        inputs = batch[0]
                        num_samples = inputs.shape[1]

                        return loss, num_samples

                    def gather_classification_data(batch, auxiliary_results, **kwargs):
                        target = batch[1]
                        predicted = auxiliary_results[0]

                        return target, predicted

                    gather_batch_data_funcs = {
                        'loss': gather_loss_data,
                        'classification': gather_classification_data
                    }

        :param gather_distributed_inputs_funcs: A dict with keys representing the metric names
            (e.g. "loss", "classification", etc.), the values are functions that can
            gather the metric inputs from different devices in a distributed computing context, if applicable.

            In general there is no strict signature for the functions here.
            However for defaults, the following is assumed:
                gather_func(gathered_input_data: Tuple[Union[T, float, int]]) -> Tuple[Union[T, float, int]]
                Where T is a Tensor type.

            Per Deep Learning library a default implementation is provided for this parameter.

        :param combine_metric_inputs_funcs: A dict with keys representing the metric names
            (e.g. "loss", "classification", etc.), the values are functions that can
            combine metric inputs from multiple batches, e.g. data from multiple batch chunks, batches of a dataset,
            or of a window of batches.

            In general there is no strict signature for the functions here.
            However for defaults, the following is assumed:
                combine_func(gathered_input_data: Iterable[Tuple[Union[T, float, int]]])
                    -> Tuple[Union[Iterable, T, float, int]]
                Where T is a Tensor type (numpy arrays are also allowed)

            Default: If no combine_metric_inputs_funcs are provided, the defaults are as follows:
                * for 'loss': the loss and num_samples are summed over all batch data given using `CombineBatchTuples`
                * for other metrics: it is assumed that the output of the gather_batch_data_funcs are
                  tuples of Tensors of the Deep Learning library used (numpy arrays are also allowed).

                To do this `CombineBatchTuples` is also used here.

        :param metric_funcs: A dict with keys representing the metric names
            (e.g. "loss", "classification", etc.), the values are functions to
            calculate the metric based on the gathered/combined metric inputs

            In general there is no strict signature for the functions here.
            However for defaults, the following is assumed:
                metric_func(combined_input_data: Tuple[Union[Iterable, T, float, int]]) -> Union[M, Tuple[M], Dict[M]]
                Where M is a scalar float/int/numpy.number

            Default: If no metric_funcs are provided, the defaults are as follows:
                * for 'loss': the average loss is calculated; it is assumed that
                * for other metrics: it is assumed that the output of the gather_batch_data_funcs are
                  tuples of Tensors of the Deep Learning library used (numpy arrays are also allowed).

                  To do this `CombineBatchTuples` is also used here.

            Example metric_funcs dict:

                def average_loss(combined_metric_inputs):
                    loss_sum, tot_num_samples = combined_metric_inputs

                    return loss_sum/tot_num_samples

                def calc_classification_quality(combined_metric_inputs):
                    target, predicted = combined_metric_inputs

                    recall, precision = ... calculate precision and recall based in target and predicted ...

                    return {
                        'recall': recall,
                        'precision': precision
                    }

                metric_funcs = {
                    'loss': average_loss,
                    'classification': calc_classification_quality
                }

            NOTE :
             * When a tuple is returned, the LogProgress logger will always only print the first value in the tuple
             * When a dict is returned, the LogProgress logger will print values for all keys in the dict
               (e.g. recall and precision)

        :param batch_chunk_size: If given, batches will be evaluated by chunking it to
                 smaller batches of size batch_chunk_size

                 When specifying this option, you can specify batch_chunk_metric_reducer_funcs.
                 The way the batch is evaluated works as follows:

                 for all chunks in batch:
                    eval chunk
                    use eval result to calculate batch metrics with batch_metric_funcs

                 Reduce all batch metric results of the chunks using batch_chunk_metric_reducer_funcs
        :type batch_chunk_size: int

        :param show_dataset_evaluation_progress If True, the progress of dataset evaluation will be logged
        :type show_dataset_evaluation_progress
        """

        super().__init__(pybase_logger_name=name, **kwargs)

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
            if gather_metric_inputs_funcs is None:
                gather_metric_inputs_funcs = {
                    "loss": gather_loss
                }

            self.check_funcs(gather_metric_inputs_funcs)
        except InvalidParametersException as e:
            raise InvalidParametersException("The gather metric inputs funcs are invalid") from e

        if gather_distributed_inputs_funcs is None:
            gather_distributed_inputs_funcs = {}

        # For the different deep learning library backends supported, defaults should be provided
        for metric_name in gather_metric_inputs_funcs.keys():
            if metric_name not in gather_distributed_inputs_funcs:
                raise InvalidParametersException(
                    f'No gather_distributed_inputs_func provided for metric : {metric_name}')

        try:
            self.check_funcs(gather_distributed_inputs_funcs, func_names=gather_metric_inputs_funcs.keys())
        except InvalidParametersException as e:
            raise InvalidParametersException("There are issues with the provided gather_distributed_inputs_funcs") \
                from e

        # For the different deep learning library backends supported, defaults should be provided
        if combine_metric_inputs_funcs is None:
            combine_metric_inputs_funcs = {}

        for metric_name in gather_metric_inputs_funcs.keys():
            if metric_name not in combine_metric_inputs_funcs:
                raise InvalidParametersException(f'No combine_metric_inputs_func provided for metric : {metric_name}')

        try:
            self.check_funcs(combine_metric_inputs_funcs, func_names=gather_metric_inputs_funcs.keys())
        except InvalidParametersException as e:
            raise InvalidParametersException("There are issues with the provided combine_metric_inputs_funcs") \
                from e

        for metric_name in gather_metric_inputs_funcs.keys():
            if metric_name not in metric_funcs:
                raise InvalidParametersException(f'No metric_func provided for metric : {metric_name}')

        try:
            self.check_funcs(metric_funcs, func_names=gather_metric_inputs_funcs.keys())
        except InvalidParametersException as e:
            raise InvalidParametersException("There are issues with the provided metric_funcs") \
                from e

        self._gather_metric_inputs_funcs = gather_metric_inputs_funcs
        self._gather_distributed_inputs_funcs = gather_distributed_inputs_funcs
        self._combine_metric_inputs_funcs = combine_metric_inputs_funcs
        self._metric_funcs = metric_funcs

        self._batch_chunk_size = batch_chunk_size

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
        return list(self._gather_metric_inputs_funcs.keys())

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
        for metric_name, gather_metric_inputs_func in self._gather_metric_inputs_funcs.items():
            try:
                metric_inputs = gather_metric_inputs_func(**metric_func_args)

                gather_distributed_inputs_func = self._gather_distributed_inputs_funcs[metric_name]
                metric_inputs = gather_distributed_inputs_func(metric_inputs)

                metric_func = self._metric_funcs[metric_name]
                metric = metric_func(metric_inputs)

                metrics_output[metric_name] = (metric, metric_inputs)
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

        batch_metric_data_lists = {}

        metric_paths = None
        for dataset_batch in dataset:
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
