import math

import numpy as np

from mlpug.trainers.callbacks.callback import Callback
from mlpug.utils import has_key, SlidingWindow

import basics.base_utils as _


class MetricsLoggerBase(Callback):

    def __init__(self,
                 dataset_name,
                 metrics,
                 batch_averaging_window=None,
                 name="MetricsLoggerBase"):
        super().__init__(name=f"{dataset_name} dataset {name}")

        self._metrics = metrics

        self._dataset_name = dataset_name

        self._batch_averaging_window = batch_averaging_window

        self._metric_averages = None
        self._metric_windows = None

    def get_state(self):
        """

        :return: state, success (True or False)
        """
        return {
                    "batch_averaging_window": self._batch_averaging_window,
                    "metric_averages": self._metric_averages,
                    "metric_windows": {metric_name: window.get_state()
                                       for metric_name, window in self._metric_windows.items()}
               }, True

    def set_state(self, state):
        """

        :param state:
        :return: success (True or False)
        """

        if not self._check_state(state):
            self._log.error('Given state is invalid, unable to set state')
            return False

        self._batch_averaging_window = state["batch_averaging_window"]
        self._metric_averages = state["metric_averages"]
        self._metric_windows = {metric_name: SlidingWindow(state=window_state)
                                for metric_name, window_state in state["metric_windows"].items()}

        return True

    def on_epoch_start(self, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if self._dataset_name not in logs:
            logs[self._dataset_name] = {}

        if 'mean' not in logs[self._dataset_name]:
            logs[self._dataset_name]['mean'] = self._metric_averages or {}

        return True

    def on_batch_training_completed(self, dataset_batch, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if not self._calc_metrics_on(dataset_batch, logs[self._dataset_name]):
            return False

        metric_names = self._metrics.keys()
        if not self._update_metrics_windows_for(metric_names, logs[self._dataset_name]):
            return False

        if not self._calc_metric_window_averages(metric_names, logs[self._dataset_name]["mean"]):
            return False

        return True

    def _setup_metrics_averaging(self, window_length, reset=False):
        if reset:
            self._metric_averages = {}
            self._metric_windows = {}

        self._metric_averages = self._metric_averages or {}
        self._metric_windows = self._metric_windows or {}

        for metric_name in self._metrics.keys():
            if metric_name not in self._metric_windows:
                self._metric_windows[metric_name] = SlidingWindow(length=window_length)
                self._metric_averages[metric_name] = None

        return True

    def _calc_metrics_on(self, batch, metrics):
        success = True

        for metric_name, metric_func in self._metrics.items():
            try:
                metrics[metric_name] = metric_func(batch)
            except Exception as e:
                _.log_exception(self._log, f"Exception occurred calculating {metric_name} on "
                                           f"{self._dataset_name} batch", e)
                success = False

        return success

    def _update_metrics_windows_for(self, metric_names, batch_metric_values):
        success = True

        for metric_name in metric_names:
            try:
                metric_value = batch_metric_values[metric_name]
                self._metric_windows[metric_name].slide(metric_value)
            except Exception as e:
                _.log_exception(self._log, f"Exception occurred updating sliding averaging window for "
                                           f"{self._dataset_name} batch {metric_name}", e)
                success = False

        return success

    def _calc_metric_window_averages(self, metric_names, metrics_averages):
        success = True

        for metric_name in metric_names:
            try:
                window_data = self._metric_windows[metric_name].window
                metrics_averages[metric_name] = np.mean(np.array(window_data))
            except Exception as e:
                _.log_exception(self._log, f"Exception occurred calculating sliding window average for "
                                           f"{self._dataset_name} batch {metric_name}", e)
                success = False

        return success

    def _check_state(self, state):
        state_attributes = ['batch_averaging_window', 'metric_averages', 'metric_windows']

        for attr in state_attributes:
            if not has_key(state, attr):
                self._log.error(f"Given state does not have a value for {attr}, state is invalid")
                return False

        return True


class TrainingMetricsLogger(MetricsLoggerBase):

    def __init__(self, metrics, batch_averaging_window, name="MetricsLogger"):
        super().__init__(dataset_name='training',
                         metrics=metrics,
                         batch_averaging_window=batch_averaging_window,
                         name=name)

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if self._batch_averaging_window is None:
            self._log.debug(f"Set batch_averaging_window to number of batches per epoch : {num_batches_per_epoch}")
            self._batch_averaging_window = num_batches_per_epoch

        if self._batch_averaging_window == math.inf:
            self._log.error("The batch average window is infinite, unable to calculate window average. "
                            "Please set a finite batch_averaging_window during construction of this "
                            "TrainingMetricsLogger.")
            self._valid = False
            return False

        return self._setup_metrics_averaging(self._batch_averaging_window)


class TestMetricsLogger(MetricsLoggerBase):

    def __init__(self,
                 dataset,
                 dataset_name,
                 metrics,
                 batch_level=True,
                 batch_assessment_period=1,
                 batch_averaging_window=None,
                 name="MetricsLogger"):
        super().__init__(dataset_name=dataset_name,
                         metrics=metrics,
                         batch_averaging_window=batch_averaging_window,
                         name=name)

        self._dataset = dataset

        self._batch_level = batch_level
        self._batch_assessment_period = batch_assessment_period
        self._batch_averaging_window = batch_averaging_window

        self._dataset_iterator = None

        self._validate()

    def get_state(self):
        if self._batch_level:
            return super().get_state()
        else:
            return None, True

    def set_state(self, state):
        if self._batch_level:
            return super().set_state(state)
        else:
            return True

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        return self._setup_metrics_averaging(self._batch_averaging_window)

    def on_epoch_start(self, logs):
        """

        :param logs:

        :return: success (True or False)
        """
        success = super().on_epoch_start(logs)

        if not success:
            self._log.error('A problem occurred, will not continue executing this hook')
            return success

        if self._batch_level and self._dataset_iterator is None:
            self._dataset_iterator = iter(self._dataset)

        return True

    def on_batch_training_completed(self, training_batch, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if not self._batch_level:
            return True

        training_iter = logs['global_iter']

        if training_iter % self._batch_assessment_period != 0:
            return True

        try:
            dataset_batch = next(self._dataset_iterator)
        except StopIteration:
            self._log.debug(f"[Iteration {training_iter}] Rewinding the {self._dataset_name} data set")
            self._dataset_iterator = iter(self._dataset)
            dataset_batch = next(self._dataset_iterator)

        return super().on_batch_training_completed(dataset_batch, logs)

    def on_epoch_completed(self, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if self._batch_level:
            return True

        self._dataset_iterator = iter(self._dataset)

        self._setup_metrics_averaging(self._batch_averaging_window, reset=True)

        metric_names = self._metrics.keys()
        # Loop over all dataset batches, this will fill the sliding windows, finally calculate the averages
        for dataset_batch in self._dataset_iterator:
            batch_metrics = {}
            if not self._calc_metrics_on(dataset_batch, batch_metrics):
                return False

            if not self._update_metrics_windows_for(metric_names, batch_metrics):
                return False

        return self._calc_metric_window_averages(metric_names, logs[self._dataset_name]["mean"])

    def _validate(self):
        self._valid = True

        if not hasattr(self._dataset, '__iter__'):
            self._log.error(f"The given dataset {str(self._dataset)} is not iterable, the {self} will not function")
            self._valid = False

        if self._batch_averaging_window is None:
            try:
                self._batch_averaging_window = len(self._dataset)
                if self._batch_averaging_window == math.inf:
                    self._log.error(f"The batch average window of the {self._dataset_name} data set is infinite, "
                                    f"unable to calculate window average. Please set a finite batch_averaging_window "
                                    f"during construction of this TestMetricsLogger.")
                    self._valid = False
                else:
                    self._log.debug(f"Set batch averaging window to number of batches in "
                                    f"{self._dataset_name} data set : {self._batch_averaging_window}")
            except Exception as e:
                _.log_exception(self._log, f"Unable to assess data set length to set the batch averaging window, "
                                           f"{self} will not function", e)
                self._valid = False





