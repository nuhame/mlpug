from enum import Enum

import math

from mlpug.trainers.callbacks.callback import Callback
from mlpug.mlpug_exceptions import CallbackInvalidException
from mlpug.utils import has_key, SlidingWindow, describe_data

from mlpug.evaluation import MetricEvaluator

import basics.base_utils as _

from mlpug.utils import get_value_at, set_value_at


class MetricsLoggingMode(Enum):
    BATCH_METRICS_ONLY = "BATCH_METRICS_ONLY"
    BATCH_AND_SLIDING_WINDOW_METRICS = "BATCH_AND_SLIDING_WINDOW_METRICS"
    WHOLE_DATASET_METRICS = "WHOLE_DATASET_METRICS"

    def will_log_sliding_window_metrics(self):
        return self.value is MetricsLoggingMode.BATCH_AND_SLIDING_WINDOW_METRICS.value

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
                 sliding_window_length=None,
                 log_condition_func=None,
                 sliding_window_factory=None,
                 inspect_sliding_windows=False,
                 name=None,
                 **kwargs):
        """

        Different modes of operation:

        Batch level: After every batch training iteration:
            Calculate batch metrics only
                log metrics per batch                   batch_level = True, mode = BATCH_METRICS
            Calculate metrics over a sliding window
                log metrics per batch and over window   batch_level = True, mode = BATCH_AND_SLIDING_WINDOW_METRICS
                log metrics over window only            batch_level = True, mode = SLIDING_WINDOW_METRICS
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

        :param sliding_window_length: Window length in batches, over which the average metric must be calculated
        :type sliding_window_length: Int

        :param sliding_window_factory: Optional. Function to create SlidingWindow instances

        :param inspect_sliding_windows: When set to True, information about the contents of the sliding windows
                                        will be provided. This feature is useful for debugging.

        :param name:
        :type name:
        """
        if name is None:
            name = self.__class__.__name__
            name = f"{name}[{dataset_name}]"

        super().__init__(name=name, **kwargs)

        self._dataset = dataset
        self._dataset_name = dataset_name

        self._metric_evaluator = metric_evaluator

        self._batch_level = batch_level

        self._logging_mode = logging_mode

        self._evaluate_settings = evaluate_settings

        self._sliding_window_length = sliding_window_length

        self._log_condition_func = log_condition_func or (lambda logs, dataset_batch: True)

        self._sliding_window_factory = SlidingWindow if not callable(sliding_window_factory) else sliding_window_factory

        self._inspect_sliding_windows = inspect_sliding_windows

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

        self._metric_windows = {metric_path: self._sliding_window_factory(state=window_state)
                                for metric_path, window_state in state["metric_windows"].items()}

        return True

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):

        will_log_sliding_window_metrics = self._logging_mode.will_log_sliding_window_metrics()

        if not self.instance_valid():
            if will_log_sliding_window_metrics:
                self._log.error(f"{self} is not valid, unable to set up averaging window ... ")
            else:
                self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if will_log_sliding_window_metrics and \
                (self._sliding_window_length is None or self._sliding_window_length <= 0):
            self._log.error(f"A valid batch sliding window length is required for "
                            f"metric logging mode ({self._logging_mode}, the {self} will not function")
            self._valid = False

            return False

        if will_log_sliding_window_metrics:
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

            results, success = self._calc_batch_metric_data_from(dataset_batch, logs)
            if not success:
                return False

            batch_metrics = results["metrics"]

            batch_path = f"{self._dataset_name}.batch"
            dataset_batch_logs = get_value_at(batch_path, current, {})

            # Merge in new batch level results
            dataset_batch_logs = {**dataset_batch_logs, **batch_metrics}
            set_value_at(batch_path, current, dataset_batch_logs)

            if self._logging_mode.will_log_sliding_window_metrics():
                self._update_metrics_windows_with(results["inputs"])

                sliding_window_metrics, success = self._calc_sliding_window_metrics()
                if not success:
                    return False

                sliding_window_path = f"{self._dataset_name}.sliding_window"
                dataset_sliding_window_logs = get_value_at(sliding_window_path, current, {})

                # Merge in new batch level results
                dataset_sliding_window_logs = {**dataset_sliding_window_logs, **sliding_window_metrics}
                set_value_at(sliding_window_path, current, dataset_sliding_window_logs)

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
        for metric_level in ['batch', 'sliding_window', 'dataset']:
            if metric_level not in dataset_metrics:
                dataset_metrics[metric_level] = {}

    def _get_current_evaluate_settings(self, logs):
        # Use custom settings if available, else use default settings

        evaluate_settings = get_value_at('evaluate_settings', logs, warn_on_failure=False)
        if evaluate_settings is None:
            evaluate_settings = self._evaluate_settings

        return evaluate_settings

    def _calc_whole_dataset_metrics(self, logs, dataset_metrics_log_path):
        """

        :param logs:
        :param dataset_metrics_log_path:
        :return: True on success
        """

        current = self._get_logs_base(logs)
        dataset_metrics_log = get_value_at(dataset_metrics_log_path, current, {})

        evaluate_settings = self._get_current_evaluate_settings(logs)

        results, success = self._metric_evaluator.calc_dataset_metrics_for(
            dataset=self._dataset,
            evaluate_settings=evaluate_settings,
            dataset_name=self._dataset_name)

        if not success:
            return False

        dataset_metrics = results["metrics"]

        # Merge in new batch level results
        dataset_metrics_log = {**dataset_metrics_log, **dataset_metrics}
        set_value_at(dataset_metrics_log_path, current, dataset_metrics_log)

        return True

    def _init_metric_windows(self, reset=False):
        if reset:
            self._metric_windows = {}

        # Do nothing if metric windows are already available
        self._metric_windows = self._metric_windows or {}

        return True

    def _calc_batch_metric_data_from(self, batch, logs):
        evaluate_settings = self._get_current_evaluate_settings(logs)

        model_outputs = None
        if self._dataset is None:
            current = self._get_logs_base(logs)
            model_outputs = get_value_at(f"{self._dataset_name}.batch.raw.model_outputs", current)

            # Use the model_outputs instead of the batch data
            batch = None

        return self._metric_evaluator.calc_batch_metrics_for(
            batch_data=batch,
            model_outputs=model_outputs,
            evaluate_settings=evaluate_settings,
            return_gathered_inputs=self._logging_mode.will_log_sliding_window_metrics())

    def _update_metrics_windows_with(self, batch_metric_inputs):
        for metric_name, metric_inputs in batch_metric_inputs.items():
            sliding_window = self._metric_windows[metric_name] if metric_name in self._metric_windows else None
            if sliding_window is None:
                self._log.debug(f"Creating sliding window for metric {metric_name}")

                sliding_window = self._sliding_window_factory(length=self._sliding_window_length, name=metric_name)
                self._metric_windows[metric_name] = sliding_window

            sliding_window.slide(metric_inputs)

            if self._inspect_sliding_windows:
                self._log.info(f"Sliding window for metric {metric_name}:\n"
                               f"Total length: {len(sliding_window)}\n"
                               f"Last added metric input data:\n{describe_data(metric_inputs)}\n")

    def _calc_sliding_window_metrics(self):
        """

        :return: (metric_outputs: dict, success: bool)
                  metric_output is dict with calculated metrics
        """
        # Gather batch metric inputs from sliding windows
        gathered_metric_inputs = {metric_name: sliding_window.window
                                  for metric_name, sliding_window in self._metric_windows.items()}

        combined_metric_inputs, success = self._metric_evaluator.combine_gathered_metric_inputs(
            gathered_metric_inputs,
            dataset_name=self._dataset_name,
            show_progress=False)

        if not success:
            self._log.error("Failed to combine batch metric inputs for sliding window(s), "
                            f"unable to calculate sliding window metrics for dataset {self._dataset_name}")
            return None, False

        return self._metric_evaluator.calc_metrics_using(
            combined_metric_inputs,
            dataset_name=self._dataset_name,
            show_progress=False)

    def _get_callback_properties_for_hash(self):
        """
        This is used to create the unique callback hash.

        Returns a dict with properties describing the setup of the callback.
        It should at least contain the properties that influence the callback state.

        Property values should only be simple types, such as int, float, boolean and strings.
        Convert any object and function values (or similar) into a booleans
        (True = available, False = None, not available)

        :return: dict
        """
        props = super()._get_callback_properties_for_hash()
        return {
            **props,
            "metric_evaluator": self._metric_evaluator is not None,
            "dataset_name": self._dataset_name,
            "batch_level": self._batch_level,
            "logging_mode": self._logging_mode,
            "dataset": self._dataset is not None,
            "evaluate_settings": self._evaluate_settings is not None,
            "sliding_window_length": self._sliding_window_length,
            "log_condition_func": self._log_condition_func is not None,
            "sliding_window_factory": self._sliding_window_factory is not None
        }

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
            if self._logging_mode.will_log_sliding_window_metrics():
                self._log.debug("Training batch evaluation results will be used to calculate and log metrics.")
            else:
                self._log.error(f"No valid dataset provided to calculated metrics on, "
                                f"the {self} will not function")
                self._valid = False

        if not MetricEvaluator.is_valid(self._metric_evaluator):
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

        if self._log_condition_func is not None and not callable(self._log_condition_func):
            self._log.error(f"The log condition function must be callable, "
                            f"metric logging mode ({self._logging_mode}, the {self} will not function")
            self._valid = False


class TrainingMetricsLogger(MetricsLoggerBase):
    """
    By default, gets already calculated model_output results from logs
    """

    def __init__(self,
                 metric_evaluator,
                 batch_level=True,
                 logging_mode=MetricsLoggingMode.BATCH_AND_SLIDING_WINDOW_METRICS,
                 **kwargs):

        super().__init__(
            metric_evaluator=metric_evaluator,
            dataset_name='training',
            batch_level=batch_level,
            logging_mode=logging_mode,
            **kwargs)

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):

        if self._logging_mode.will_log_sliding_window_metrics():
            if self._sliding_window_length is None:
                self._sliding_window_length = math.ceil(num_batches_per_epoch/2.0)
                self._log.debug(f"Set sliding_window_length to half of "
                                f"number of batches per epoch : {self._sliding_window_length}")

            if self._sliding_window_length == math.inf:
                self._log.error("The batch average window is infinite, unable to calculate window average. "
                                "Please set a finite sliding_window_length during construction of this "
                                "TrainingMetricsLogger.")
                self._valid = False
                return False

        return super().on_training_start(num_epochs, num_batches_per_epoch, start_epoch, start_batch, start_update_iter)


class DatasetMetricsLogger(MetricsLoggerBase):
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
            logging_mode = MetricsLoggingMode.BATCH_AND_SLIDING_WINDOW_METRICS \
                if batch_level else MetricsLoggingMode.WHOLE_DATASET_METRICS

        super().__init__(dataset=dataset,
                         dataset_name=dataset_name,
                         metric_evaluator=metric_evaluator,
                         batch_level=batch_level,
                         logging_mode=logging_mode,
                         **kwargs)

        self._dataset_iterator = None

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):

        if self._logging_mode.will_log_sliding_window_metrics():
            if self._sliding_window_length is None:
                try:
                    self._sliding_window_length = int(0.5*len(self._dataset))+1
                    if self._sliding_window_length < 2:
                        self._log.error(f"The sliding window length for the {self._dataset_name} data set is "
                                        f"too small : {self._sliding_window_length}, "
                                        f"unable to calculate metrics over sliding window.")
                        self._valid = False
                    elif self._sliding_window_length == math.inf:
                        self._log.error(f"The sliding window length for the {self._dataset_name} data set is infinite, "
                                        f"unable to calculate metrics over sliding window. "
                                        f"Please set a finite sliding_window_length during construction of "
                                        f"this {self}.")
                        self._valid = False
                    else:
                        self._log.debug(f"Set the sliding window length to half of the number of batches in "
                                        f"{self._dataset_name} data set : {self._sliding_window_length}")
                except Exception as e:
                    _.log_exception(self._log, f"Unable to assess data set length to set the "
                                               f"batch sliding window length, {self} will not function", e)
                    self._valid = False

        return super().on_training_start(num_epochs, num_batches_per_epoch, start_epoch, start_batch, start_update_iter)

    def on_epoch_start(self, logs):
        """

        :param logs:

        :return: success (True or False)
        """
        success = super().on_epoch_start(logs)

        if not success:
            self._log.error('A problem occurred, will not continue executing this hook')
            return success

        if self._logging_mode.will_log_sliding_window_metrics() and self._dataset_iterator is None:
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

        super()._validate()
