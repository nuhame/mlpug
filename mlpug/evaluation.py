import sys
import abc

import numpy as np
from basics.base import Base

import basics.base_utils as _

from mlpug.mlpug_exceptions import InvalidParametersException
from mlpug.utils import *


class MetricEvaluatorBase(Base, metaclass=abc.ABCMeta):

    def __init__(self,
                 model_evaluate_func=None,
                 trainer=None,
                 batch_metric_funcs=None,
                 batch_metric_reducer_funcs=None,
                 show_dataset_evaluation_progress=False,
                 name="MetricEvaluatorBase"):
        """

        TODO : Add more documentation

        :param model_evaluate_func: f(batch_data, evaluate_settings) -> model_output

                                    IMPORTANT: it is assumed that the `model_evaluate_func` takes all
                                    appropriate measures to disable training specific layers such as
                                    dropout and gradient calculations.

                                    Eg. for Pytorch:

                                    def eval_model(batch_data, evaluate_settings):

                                        my_training_model.eval()

                                        with torch.no_grad():
                                            return my_training_model(batch_data, evaluate_settings)

        :type model_evaluate_func: callable

        :param trainer:
        :type trainer:

        :param batch_metric_funcs: A dict with keys representing the metric names (e.g. "loss", "recall", etc.) and
                             the corresponding values are functions to calculate the metric value, or to gather
                             information to calculated a combined/averaged metric value over a window,
                             also see batch_metric_reducer_funcs

                             The functions will be called as follows:

                             metric_func(**kwargs)

                             Where kwargs will contain the following keys:
                             'batch', 'evaluate_settings' and the keys of the model evaluation results.
                             Usually that is 'loss' and 'auxiliary_results'.

                             Example batch_metric_funcs dict:

                                 def get_loss(loss, **kwargs):
                                    return loss

                                 batch_metric_funcs = {
                                    'loss': get_loss
                                 }

                             The function can also return a dict with metrics. For instance:

                                 def calc_metrics(loss, **kwargs):
                                    return {
                                        "loss": loss,
                                        "perplexity": math.exp(loss)
                                    }

                                 batch_metric_funcs = {
                                    'metrics': calc_metrics
                                 }

                             Last, the function can also gather data, to be used by the corresponding
                             reducer_func, e.g.:

                                 def get_target_and_predicted(batch, auxiliary_results, **kwargs):
                                        target = batch[1]
                                        predicted = auxiliary_results[0]

                                        return target, predicted

                                 batch_metric_funcs = {
                                    'recall': get_target_and_predicted
                                 }

                                 The corresponding reducer_func could be:

                                def calc_recall(window):
                                    target = []
                                    predicted = []
                                    # append all batch-level data
                                    for t, p in window:
                                        target.append(t)
                                        predicted.append(p)

                                    return recall_score(t, p)

                                 batch_metric_reducer_funcs = {
                                    'recall': calc_recall
                                 }
        :type batch_metric_funcs: dict

        :param batch_metric_reducer_funcs:
                             An optional dict with keys representing the metric names (e.g. "loss", "recall", etc.) and
                             the corresponding values are functions to calculate the average metric value,
                             based on metrics, or other data, gathered per batch, also see batch_metric_funcs and
                             batch_averaging_window

                             The functions will be called as follows:

                             reducer_func(window), where window is a list with metrics or other data
                             gathered per batch.

                             See `batch_metric_funcs` for example.

                             The default functions assumes the window to be a list of floats and will average the
                             values in this list

        :type batch_metric_reducer_funcs: dict

        :param show_dataset_evaluation_progress If True, the progress of dataset evaluation will be logged
        :type show_dataset_evaluation_progress
        """

        super().__init__()

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
            self.check_funcs(batch_metric_funcs)
        except InvalidParametersException as e:
            raise InvalidParametersException("The batch metrics funcs are invalid") from e

        if batch_metric_reducer_funcs is None:
            batch_metric_reducer_funcs = {}

        # Add default metric averaging funcs for metrics that don't have a metric averaging func provided:
        for metric_name in batch_metric_funcs.keys():
            if metric_name not in batch_metric_reducer_funcs:
                batch_metric_reducer_funcs[metric_name] = lambda window: np.nanmean(np.array(window))

        try:
            self.check_funcs(batch_metric_reducer_funcs, func_names=batch_metric_funcs.keys())
        except InvalidParametersException as e:
            raise InvalidParametersException("The batch metric reducer funcs are invalid") from e

        self._batch_metric_funcs = batch_metric_funcs
        self._batch_metric_reducer_funcs = batch_metric_reducer_funcs

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

    def calc_batch_metrics_for(self, batch_data, metrics_output, evaluate_settings=None, model_output=None):
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

        :return: True on success, else False
        :rtype: Bool

        """
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

    def calc_dataset_metrics_for(self, dataset, metrics_output, evaluate_settings=None, dataset_name=None):
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

        metric_names = list(self._batch_metric_funcs.keys())

        if self._show_dataset_evaluation_progress:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self._log.debug(f"Calculating metrics ({', '.join(metric_names)}) on whole "
                            f"{'' if dataset_name is None else dataset_name} dataset")

        dataset_iterator = iter(dataset)

        batch_metric_data_lists = {}

        metric_paths = None
        for dataset_batch in dataset_iterator:
            batch_metric_data_map = {}
            batch_success = self.calc_batch_metrics_for(dataset_batch, batch_metric_data_map, evaluate_settings)
            if not batch_success:
                return False

            if metric_paths is None:
                metric_paths = get_key_paths(batch_metric_data_map, keys_to_consider=metric_names)

            for metric_path in metric_paths:
                batch_metric_data = get_value_at(metric_path, batch_metric_data_map)

                batch_metric_data_list = batch_metric_data_lists[metric_path] \
                    if metric_path in batch_metric_data_lists else None

                if batch_metric_data_list is None:
                    batch_metric_data_list = []
                    batch_metric_data_lists[metric_path] = batch_metric_data_list

                batch_metric_data_list += [batch_metric_data]

            if self._show_dataset_evaluation_progress:
                sys.stdout.write('#')
                sys.stdout.flush()

        if self._show_dataset_evaluation_progress:
            sys.stdout.write('\n')
            sys.stdout.flush()

        return self.reduce(batch_metric_data_lists, metrics_output, dataset_name=dataset_name)

    def reduce(self, batch_metric_data_lists, metrics_output, dataset_name=None):
        """

        Use the `batch_metric_reducer_funcs` to reduce the lists available in the `batch_metric_data_lists` dict

        The keys of `batch_metric_data_lists` must be the key path to the reduced metrics. Example:
        batch_metric_data_lists = {
            'loss': [ ... data to use to calculated recall ... ],
            'classification.F1': [ ... data to use to calculated F1 ... ]
        }

        To get the correct key paths you can use the `get_key_paths` as a utility

        :param batch_metric_data_lists: See above.
        :type batch_metric_data_lists: Dict

        :param metrics_output: Dict to which the reduced metrics will be written
        :type metrics_output: Dict

        :param dataset_name: Used for logging purposes
        :type dataset_name:

        :return: True on success, else False
        :rtype:
        """
        success = True

        for metric_path, batch_metric_data_list in batch_metric_data_lists.items():
            try:
                reducer_func = get_value_at(metric_path, self._batch_metric_reducer_funcs)
                set_value_at(metric_path, metrics_output, reducer_func(batch_metric_data_list))
            except Exception as e:
                _.log_exception(self._log, f"Exception occurred reducing {metric_path} for "
                                           f"{'' if dataset_name is None else dataset_name} dataset "
                                           f"batch metric data", e)
                success = False

        return success

    @abc.abstractmethod
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
