from enum import Enum

import math

from mlpug.trainers.callbacks.callback import Callback
from mlpug.mlpug_exceptions import CallbackInvalidException
from mlpug.utils import has_key, get_key_paths, SlidingWindow

from mlpug.evaluation import MetricEvaluatorBase

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

    def __str__(self):
        return str(self.value)


class MetricsLoggerBase(Callback):

    def __init__(self,
                 metric_evaluator,
                 dataset_name,
                 batch_level,
                 logging_mode,
                 dataset=None,
                 evaluate_settings=None,
                 batch_averaging_window=None,
                 log_condition_func=None,
                 name=None,
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


        `log_condition_func` can be used to control if metrics are calculated

        :param dataset_name:
        :type dataset_name:

        :param batch_level: See general description above
        :type batch_level:

        :param logging_mode: See general description above
        :type logging_mode:

        :param dataset: Dataset, where the items are batches, to calculate the metrics with
        :type dataset:

        :param metric_evaluator: MetricEvaluator instance, or other object with the same public interface
        :type metric_evaluator:

        :param evaluate_settings: Default evaluation settings to use, when no evaluate_settings are available in the
                                  logs object provided to the callback methods
        :type evaluate_settings:

        :param batch_averaging_window: Window length in batches, over which the average metric must be calculated
        :type batch_averaging_window: Int

        :param name:
        :type name:
        """

        if name is None:
            name = self.__class__.__name__

            logging_level = "Batch level" if batch_level else "Epoch level"
            name = f"{name}[{dataset_name} dataset][{logging_level}][{str(logging_mode)}]"

        super().__init__(name=name, **kwargs)

        self._dataset = dataset
        self._dataset_name = dataset_name

        self._metric_evaluator = metric_evaluator

        self._batch_level = batch_level

        self._logging_mode = logging_mode

        self._evaluate_settings = evaluate_settings

        self._batch_averaging_window = batch_averaging_window

        self._log_condition_func = log_condition_func or (lambda logs, dataset_batch: True)

        self._name = name

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

            base_path = f"{self._dataset_name}.batch"
            dataset_batch_logs = get_value_at(base_path, current)

            # Merge in new batch level results
            dataset_batch_logs = {**dataset_batch_logs, **batch_metrics}
            if self._logging_mode is MetricsLoggingMode.BATCH_AND_WINDOW_AVERAGE_METRICS:
                set_value_at(base_path, current, dataset_batch_logs)

            metric_names = self._metric_evaluator.get_metric_names()
            metric_paths = get_key_paths(dataset_batch_logs, keys_to_consider=metric_names)

            self._update_metrics_windows_for(metric_paths, dataset_batch_logs, base_path=base_path)

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
        for metric_level in ['batch', 'window_average', 'dataset']:
            if metric_level not in dataset_metrics:
                dataset_metrics[metric_level] = {}

    def _get_current_evaluate_settings(self, logs):
        # Use custom settings if available, else use default settings

        evaluate_settings = get_value_at('evaluate_settings', logs, warn_on_failure=False)
        if evaluate_settings is None:
            evaluate_settings = self._evaluate_settings

        return evaluate_settings

    def _calc_whole_dataset_metrics(self, logs, log_path):

        current = self._get_logs_base(logs)
        metrics_log = get_value_at(log_path, current)

        evaluate_settings = self._get_current_evaluate_settings(logs)

        return self._metric_evaluator.calc_dataset_metrics_for(self._dataset,
                                                               metrics_log,
                                                               evaluate_settings=evaluate_settings,
                                                               dataset_name=self._dataset_name)

    def _init_metric_windows(self, reset=False):
        if reset:
            self._metric_windows = {}

        # Do nothing if metric windows are already available
        self._metric_windows = self._metric_windows or {}

        return True

    def _calc_batch_metric_data_from(self, batch, batch_metrics, logs):
        evaluate_settings = self._get_current_evaluate_settings(logs)

        model_output = None
        if self._dataset is None:
            current = self._get_logs_base(logs)
            loss = get_value_at(f"{self._dataset_name}.batch.loss", current)
            auxiliary_results = get_value_at(f"{self._dataset_name}.batch.auxiliary_results",
                                             current,
                                             warn_on_failure=False)

            model_output = {
                'loss': loss,
                'auxiliary_results': auxiliary_results
            }

        return self._metric_evaluator.calc_batch_metrics_for(batch,
                                                             batch_metrics,
                                                             evaluate_settings=evaluate_settings,
                                                             model_output=model_output)

    def _update_metrics_windows_for(self, metric_paths, batch_metrics, base_path):
        for metric_path in metric_paths:
            metric_value = get_value_at(metric_path, batch_metrics)

            full_metric_path = f"{base_path}.{metric_path}"
            sliding_window = self._metric_windows[metric_path] if metric_path in self._metric_windows else None
            if sliding_window is None:
                self._log.debug(f"Creating sliding window for {full_metric_path}")

                sliding_window = SlidingWindow(length=self._batch_averaging_window, name=full_metric_path)
                self._metric_windows[metric_path] = sliding_window

            sliding_window.slide(metric_value)

    def _reduce(self, batch_metric_data_lists, reduced_metrics_log):

        return self._metric_evaluator.reduce(batch_metric_data_lists,
                                             reduced_metrics_log,
                                             dataset_name=self._dataset_name)

    def _check_state(self, state):
        state_attributes = ['metric_windows']

        for attr in state_attributes:
            if not has_key(state, attr):
                self._log.error(f"Given state does not have a value for {attr}, state is invalid")
                return False

        return True

    def _validate(self):
        self._valid = True

        if not hasattr(self._dataset, '__iter__'):
            if MetricsLoggingMode.will_log_window_average_metrics(self._logging_mode):
                self._log.debug("Training batch evaluation results will be used to calculate and log metrics.")
            else:
                self._log.error(f"No valid dataset provided to calculated metrics on, "
                                f"the {self} will not function")
                self._valid = False

        if not MetricEvaluatorBase.is_valid(self._metric_evaluator):
            self._log.error(f"The given metric evaluator is not valid, the {self} will not function : "
                            f"{self._metric_evaluator}")
            self._valid = False

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
                 metric_evaluator,
                 batch_level=True,
                 logging_mode=MetricsLoggingMode.BATCH_AND_WINDOW_AVERAGE_METRICS,
                 **kwargs):

        super().__init__(
            metric_evaluator=metric_evaluator,
            dataset_name='training',
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


class TestMetricsLogger(MetricsLoggerBase):
    """
    Child class must implement _evaluate_loss method

    TODO : We should implement a specialization for the most common case of logging the (window/dataset) loss average
    """

    def __init__(self,
                 dataset,
                 dataset_name,
                 metric_evaluator,
                 batch_level=True,
                 logging_mode=None,
                 **kwargs):

        if logging_mode is None:
            logging_mode = MetricsLoggingMode.BATCH_AND_WINDOW_AVERAGE_METRICS \
                if batch_level else MetricsLoggingMode.WHOLE_DATASET_METRICS

        super().__init__(dataset=dataset,
                         dataset_name=dataset_name,
                         metric_evaluator=metric_evaluator,
                         batch_level=batch_level,
                         logging_mode=logging_mode,
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
