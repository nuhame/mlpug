import sys

import abc
from enum import Enum

import math

import numpy as np

from mlpug.trainers.callbacks.callback import Callback
from mlpug.mlpug_exceptions import CallbackInvalidException
from mlpug.utils import has_key, SlidingWindow

import basics.base_utils as _

from mlpug.utils import get_value_at, set_value_at


class MetricsLoggingMode(Enum):
    BATCH_AND_WINDOW_AVERAGE_METRICS = "BATCH_AND_WINDOW_AVERAGE_METRICS"
    WINDOW_AVERAGE_METRICS = "WINDOW_AVERAGE_METRICS"
    WHOLE_DATASET_METRICS = "WHOLE_DATASET_METRICS"

    @staticmethod
    def will_log_window_average_metrics(mode):
        return mode in {MetricsLoggingMode.BATCH_AND_WINDOW_AVERAGE_METRICS,
                        MetricsLoggingMode.WINDOW_AVERAGE_METRICS}


class MetricsLoggerBase(Callback, metaclass=abc.ABCMeta):

    def __init__(self,
                 dataset_name,
                 batch_metric_funcs,
                 batch_level,
                 logging_mode,
                 dataset=None,
                 evaluate_settings=None,
                 batch_averaging_window=None,
                 batch_metric_reducer_funcs=None,
                 log_condition_func=None,
                 show_dataset_evaluation_progress=False,
                 name="MetricsLoggerBase",
                 **kwargs):
        """

        Different modes of operation:

        Batch level: After every batch training iteration:
            Calculate metrics over a sliding window
                log metrics per batch and over window   batch_level = True, mode = BATCH_AND_WINDOW_AVERAGE_METRICS
                log metrics over window only            batch_level = True, mode = WINDOW_AVERAGE_METRICS
            Calculate metrics over whole data set       batch_level = True, mode = WHOLE_DATASET_METRICS

        Epoch level: After every training epoch:
            Calculate metrics over whole data set       batch_level = False, mode = WHOLE_DATASET_METRICS


        Using `log_condition_func` it can be used to control if metrics are calculated


        # TODO : update documentation after refactor dd 18042020

        :param dataset_name:
        :type dataset_name:
        :param batch_metric_funcs: A dict with keys representing the metric names (e.g. "loss", "recall", etc.) and
                             the corresponding values are functions to calculate the metric value, or to gather
                             information to calculated a combined/averaged metric value over a window,
                             also see batch_metric_reducer_funcs and batch_averaging_window

                             The functions will be called as follows:

                             metric_func(**kwargs)

                             Where kwargs will contain the following keys:
                             batch, loss, auxiliary_results, evaluate_settings

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

        :type batch_metric_funcs:
        :param evaluate_settings:
        :type evaluate_settings:
        :param batch_averaging_window: Window length in batches, over which the average metric must be calculated
        :type batch_averaging_window: Int
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

        :type batch_metric_reducer_funcs:
        :param average_only: If True, only results of the batch_metric_reducer_funcs will be logged, default is false
        :type average_only: Boolean
        :param get_loss_and_aux_from_logs: If true, the model loss and the auxiliary results are retrieved from the logs
        :type get_loss_and_aux_from_logs:
        :param show_dataset_evaluation_progress
        :type show_dataset_evaluation_progress
        :param name:
        :type name:
        """
        super().__init__(name=f"{dataset_name} dataset {name}", **kwargs)

        self._dataset = dataset
        self._dataset_name = dataset_name

        self._batch_metric_funcs = batch_metric_funcs
        self._batch_level = batch_level

        self._logging_mode = logging_mode

        self._evaluate_settings = evaluate_settings

        self._batch_averaging_window = batch_averaging_window

        self._batch_metric_reducer_funcs = batch_metric_reducer_funcs or {}

        self._log_condition_func = log_condition_func or (lambda logs, dataset_batch: True)

        self._show_dataset_evaluation_progress = show_dataset_evaluation_progress

        self._name = name

        # Add default metric averaging funcs for metrics that don't have a metric averaging func provided:
        for metric_name in self._batch_metric_funcs.keys():
            if metric_name not in self._batch_metric_reducer_funcs:
                self._batch_metric_reducer_funcs[metric_name] = lambda window: np.nanmean(np.array(window))

        self._metric_windows = {}

        self._validate()

        if not self.instance_valid():
            raise CallbackInvalidException(self._name)

    def get_state(self):
        """

        :return: state, success (True or False)
        """
        return {
                    "metric_windows": {metric_path: window.get_state()
                                       for metric_path, window in self._metric_windows.items()}
               }, True

    def set_state(self, state):
        """

        :param state:
        :return: success (True or False)
        """

        if not self._check_state(state):
            self._log.error('Given state is invalid, unable to set state')
            return False

        self._metric_windows = {metric_path: SlidingWindow(state=window_state)
                                for metric_path, window_state in state["metric_windows"].items()}

        return True

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):

        if MetricsLoggingMode.will_log_window_average_metrics(self._logging_mode):
            self._init_metric_windows()

        return True

    def on_batch_training_completed(self, dataset_batch, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if not self._batch_level:
            return True

        if not self._log_condition_func(logs=logs, dataset_batch=dataset_batch):
            return True

        self._init_logs(logs)

        if self._logging_mode is MetricsLoggingMode.WHOLE_DATASET_METRICS:
            return self._calc_whole_dataset_metrics(logs, f"{self._dataset_name}.dataset")
        else:
            current = self._get_logs_base(logs)

            batch_metrics = {}
            if not self._calc_batch_metric_data_from(dataset_batch, batch_metrics, logs):
                return False

            dataset_batch_logs = current[self._dataset_name]['batch']
            if self._logging_mode is MetricsLoggingMode.BATCH_AND_WINDOW_AVERAGE_METRICS:
                # Merge in new batch level results
                dataset_batch_logs = {**dataset_batch_logs, **batch_metrics}
                current[self._dataset_name]['batch'] = dataset_batch_logs

            metric_names = list(self._batch_metric_funcs.keys())
            metric_paths = self._get_metric_paths(dataset_batch_logs, metric_names=metric_names)

            self._update_metrics_windows_for(metric_paths, dataset_batch_logs)

            # gather all window data
            batch_metrics_lists = {p: s.window for p, s in self._metric_windows.items()}
            if not self._reduce(batch_metrics_lists, current[self._dataset_name]['window_average']):
                return False

            return True

    def on_epoch_completed(self, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if self._batch_level:
            return True

        if not self._log_condition_func(logs=logs, dataset_batch=None):
            return True

        self._init_logs(logs)

        return self._calc_whole_dataset_metrics(logs, f"{self._dataset_name}.dataset")

    def _init_logs(self, logs):
        current = self._get_logs_base(logs)
        if self._dataset_name not in current:
            current[self._dataset_name] = {}

        dataset_metrics = current[self._dataset_name]
        for type in ['batch', 'window_average', 'dataset']:
            if type not in dataset_metrics:
                dataset_metrics[type] = {}

    def _calc_whole_dataset_metrics(self, logs, log_path):
        metric_names = list(self._batch_metric_funcs.keys())

        if self._show_dataset_evaluation_progress:
            self._log.debug(f"Calculating metrics ({', '.join(metric_names)}) on whole {self._dataset_name} dataset")

        self._dataset_iterator = iter(self._dataset)

        batch_metric_data_lists = {}

        metric_paths = None
        # Loop over all dataset batches, this will fill the sliding windows, finally calculate the averages
        for dataset_batch in self._dataset_iterator:
            batch_metric_data_map = {}
            if not self._calc_batch_metric_data_from(dataset_batch, batch_metric_data_map, logs):
                return False

            if metric_paths is None:
                metric_paths = self._get_metric_paths(batch_metric_data_map, metric_names=metric_names)

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

        current = self._get_logs_base(logs)
        reduced_metrics_log = get_value_at(log_path, current)

        return self._reduce(batch_metric_data_lists, reduced_metrics_log)

    def _init_metric_windows(self, reset=False):
        if reset:
            self._metric_windows = {}

        # Do nothing if metric windows are already available
        self._metric_windows = self._metric_windows or {}

        return True

    def _get_metric_paths(self, metrics, metric_names=None, base_path=None):

        if metric_names is None:
            metric_names = list(metrics.keys())

        metric_paths = []
        for name in metric_names:
            value = metrics[name]

            if base_path is not None:
                current_path = f"{base_path}.{name}"
            else:
                current_path = name

            if type(value) is dict:
                path_list = self._get_metric_paths(value, base_path=current_path)
            else:
                path_list = [current_path]

            metric_paths += path_list

        return metric_paths

    def _calc_batch_metric_data_from(self, batch, batch_metric_data, logs):
        success = True

        # Use custom settings if available, else use default settings
        evaluate_settings = get_value_at('evaluate_settings', logs, warn_on_failure=False)
        if evaluate_settings is None:
            evaluate_settings = self._evaluate_settings

        if self._dataset is None:
            current = self._get_logs_base(logs)
            loss = get_value_at(f"{self._dataset_name}.batch.loss", current)
            auxiliary_results = get_value_at(f"{self._dataset_name}.batch.auxiliary_results",
                                             current,
                                             warn_on_failure=False)
        else:
            loss, auxiliary_results = self._evaluate_loss(batch, evaluate_settings)

        metric_func_args = {
            'batch': batch,
            'loss': loss,
            'auxiliary_results': auxiliary_results,
            'evaluate_settings': evaluate_settings
        }

        for metric_name, batch_metric_func in self._batch_metric_funcs.items():
            try:
                batch_metric_data[metric_name] = batch_metric_func(**metric_func_args)
            except Exception as e:
                _.log_exception(self._log, f"Exception occurred calculating {metric_name} data for "
                                           f"{self._dataset_name} batch", e)
                success = False

        return success

    @abc.abstractmethod
    def _evaluate_loss(self, batch, evaluate_settings=None):
        """
        Always returns loss, auxiliary_results tuple
        :param batch:
        :type batch:
        :param evaluate_settings:
        :type evaluate_settings:
        :return: loss, auxiliary_results
        :rtype: tuple
        """
        pass

    def _update_metrics_windows_for(self, metric_paths, batch_metrics):
        for metric_path in metric_paths:
            metric_value = get_value_at(metric_path, batch_metrics)

            sliding_window = self._metric_windows[metric_path] if metric_path in self._metric_windows else None
            if sliding_window is None:
                self._log.debug(f"Creating sliding window for {metric_path}")

                sliding_window = SlidingWindow(length=self._batch_averaging_window)
                self._metric_windows[metric_path] = sliding_window

            sliding_window.slide(metric_value)

    def _reduce(self, batch_metric_data_lists, reduced_metrics):
        success = True

        for metric_path, batch_metric_data_list in batch_metric_data_lists.items():
            try:
                reducer_func = get_value_at(metric_path, self._batch_metric_reducer_funcs)
                set_value_at(metric_path, reduced_metrics, reducer_func(batch_metric_data_list))
            except Exception as e:
                _.log_exception(self._log, f"Exception occurred reducing {metric_path} for {self._dataset_name} dataset "
                                           f"batch metric data", e)
                success = False

        return success

    def _check_state(self, state):
        state_attributes = ['metric_windows']

        for attr in state_attributes:
            if not has_key(state, attr):
                self._log.error(f"Given state does not have a value for {attr}, state is invalid")
                return False

        return True

    def _check_validity_funcs(self, func_dict, func_type):
        valid = True

        if func_dict is None:
            return False

        for metric_name, f in func_dict.items():
            if not callable(f):
                self._log.error(f"No valid {func_type} function provided for {metric_name}, "
                                f"the {self} will not function")
                valid = False

        return valid

    def _validate(self):
        self._valid = True

        if not hasattr(self._dataset, '__iter__'):
            if MetricsLoggingMode.will_log_window_average_metrics(self._logging_mode):
                self._log.debug("Training batch evaluation results will be used to calculate and log metrics.")
            else:
                self._log.error(f"No valid dataset provided to calculated metrics on, "
                                f"the {self} will not function")
                self._valid = False

        self._valid &= self._check_validity_funcs(self._batch_metric_funcs, "batch metric")
        self._valid &= self._check_validity_funcs(self._batch_metric_reducer_funcs, "batch metric combining")

        if not type(self._batch_level) is bool:
            self._log.error(f"Batch level must be True or False, given value is: {self._batch_level}\n"
                            f"the {self} will not function")
            self._valid = False

        if self._logging_mode not in MetricsLoggingMode:
            self._log.error(f"Metric logging mode unknown ({self._logging_mode})\n"
                            f"the {self} will not function")
            self._valid = False

        if not self._batch_level and self._logging_mode is not MetricsLoggingMode.WHOLE_DATASET_METRICS:
            self._log.error(f"For epoch level metric logging, the metric logging can only be "
                            f"MetricsLoggingMode.WHOLE_DATASET_METRICS. value is :{self._logging_mode}\n"
                            f"the {self} will not function")
            self._valid = False

        if MetricsLoggingMode.will_log_window_average_metrics(self._logging_mode) and \
                (not isinstance(self._batch_averaging_window, int) or
                 self._batch_averaging_window <= 0):
            self._log.error(f"A valid batch processing window is required for "
                            f"metric logging mode ({self._logging_mode}, the {self} will not function")
            self._valid = False

        if self._log_condition_func is not None and not callable(self._log_condition_func):
            self._log.error(f"The log condition function must be callable, "
                            f"metric logging mode ({self._logging_mode}, the {self} will not function")
            self._valid = False


class TrainingMetricsLogger(MetricsLoggerBase):
    """
    By default, gets already calculated loss and auxiliary results from logs

    TODO : needs testing
    TODO : what to do for batch_level=false? Allow usage of window average of metrics to be used for epoch level?
    """

    def __init__(self,
                 batch_metric_funcs,
                 batch_level=True,
                 logging_mode=MetricsLoggingMode.BATCH_AND_WINDOW_AVERAGE_METRICS,
                 name="TrainingMetricsLogger",
                 **kwargs):

        super().__init__(dataset_name='training',
                         batch_metric_funcs=batch_metric_funcs,
                         batch_level=batch_level,
                         logging_mode=logging_mode,
                         name=name,
                         **kwargs)

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):

        if MetricsLoggingMode.will_log_window_average_metrics(self._logging_mode):
            if self._batch_averaging_window is None:
                self._log.debug(f"Set batch_averaging_window to number of batches per epoch : {num_batches_per_epoch}")
                self._batch_averaging_window = num_batches_per_epoch

            if self._batch_averaging_window == math.inf:
                self._log.error("The batch average window is infinite, unable to calculate window average. "
                                "Please set a finite batch_averaging_window during construction of this "
                                "TrainingMetricsLogger.")
                self._valid = False
                return False

        return self.on_training_start(num_epochs, num_batches_per_epoch, start_epoch, start_batch, start_update_iter)

    def _evaluate_loss(self, batch, evaluate_settings=None):
        """
        No implementation required, because loss and auxilairy results are usually available for the training set
        """
        pass


class TestMetricsLoggerBase(MetricsLoggerBase, metaclass=abc.ABCMeta):
    """
    Child class must implement _evaluate_loss method
    """

    def __init__(self,
                 dataset,
                 dataset_name,
                 batch_metric_funcs,
                 batch_level=True,
                 logging_mode=None,
                 name="TestMetricsLoggerBase",
                 **kwargs):

        if logging_mode is None:
            logging_mode = MetricsLoggingMode.BATCH_AND_WINDOW_AVERAGE_METRICS \
                if batch_level else MetricsLoggingMode.WHOLE_DATASET_METRICS

        super().__init__(dataset=dataset,
                         dataset_name=dataset_name,
                         batch_metric_funcs=batch_metric_funcs,
                         batch_level=batch_level,
                         logging_mode=logging_mode,
                         name=name,
                         **kwargs)

        self._dataset_iterator = None

    def on_epoch_start(self, logs):
        """

        :param logs:

        :return: success (True or False)
        """
        success = super().on_epoch_start(logs)

        if not success:
            self._log.error('A problem occurred, will not continue executing this hook')
            return success

        if MetricsLoggingMode.will_log_window_average_metrics(self._logging_mode) and \
                self._dataset_iterator is None:
            self._dataset_iterator = iter(self._dataset)

        return True

    def on_batch_training_completed(self, training_batch, logs):
        current = self._get_logs_base(logs)
        training_iter = current["global_iter"]

        if self._logging_mode is MetricsLoggingMode.WHOLE_DATASET_METRICS:
            return super().on_batch_training_completed(training_batch, logs)
        else:
            try:
                dataset_batch = next(self._dataset_iterator)
            except StopIteration:
                self._log.debug(f"[Iteration {training_iter}] Rewinding the {self._dataset_name} data set")
                self._dataset_iterator = iter(self._dataset)
                dataset_batch = next(self._dataset_iterator)

            return super().on_batch_training_completed(dataset_batch, logs)

    def _validate(self):
        if not hasattr(self._dataset, '__iter__'):
            self._log.error(f"The given dataset {str(self._dataset)} is not iterable, the {self} will not function")
            self._valid = False

        if self._batch_averaging_window is None:
            try:
                self._batch_averaging_window = len(self._dataset)
                if self._batch_averaging_window == math.inf:
                    self._log.error(f"The batch average window of the {self._dataset_name} data set is infinite, "
                                    f"unable to calculate window average. Please set a finite batch_processing_window "
                                    f"during construction of this TestMetricsLogger.")
                    self._valid = False
                else:
                    self._log.debug(f"Set batch averaging window to number of batches in "
                                    f"{self._dataset_name} data set : {self._batch_averaging_window}")
            except Exception as e:
                _.log_exception(self._log, f"Unable to assess data set length to set the batch averaging window, "
                                           f"{self} will not function", e)
                self._valid = False

        super()._validate()
