import sys

import abc
import math

import numpy as np

from mlpug.trainers.callbacks.callback import Callback
from mlpug.utils import has_key, SlidingWindow

import basics.base_utils as _

from mlpug.utils import get_value_at, set_value_at


class MetricsLoggerBase(Callback, metaclass=abc.ABCMeta):

    def __init__(self,
                 dataset_name,
                 metric_funcs,
                 evaluate_settings=None,
                 batch_averaging_window=None,
                 get_loss_and_aux_from_logs=False,
                 name="MetricsLoggerBase"):
        """

        :param dataset_name:
        :type dataset_name:
        :param metric_funcs: A dict with keys representing the metric names (e.g. "loss", "recall", etc.) and
                             the corresponding values are functions to calculate the metric value
                             The functions will be called as follows:

                             metric_func(**kwargs)

                             Where kwargs will contain the following keys:
                             batch, loss, auxiliary_results, evaluate_settings

                             Example metric_funcs dict:

                                 def get_loss(loss, **kwargs):
                                    return loss

                                 metric_funcs = {
                                    'loss': get_loss
                                 }

                             The function can also return a dict with metrics. For instance:

                                 def calc_metrics(loss, **kwargs):
                                    return {
                                        "loss": loss,
                                        "perplexity": math.exp(loss)
                                    }

                                 metric_funcs = {
                                    'metrics': calc_metrics
                                 }

        :type metric_funcs:
        :param evaluate_settings:
        :type evaluate_settings:
        :param batch_averaging_window:
        :type batch_averaging_window:
        :param get_loss_and_aux_from_logs: If true, the model loss and the auxiliary results are retrieved from the logs
        :type get_loss_and_aux_from_logs:
        :param name:
        :type name:
        """
        super().__init__(name=f"{dataset_name} dataset {name}")

        self._metric_funcs = metric_funcs
        self._evaluate_settings = evaluate_settings

        self._dataset_name = dataset_name

        self._batch_averaging_window = batch_averaging_window

        self._get_loss_and_aux_from_logs = get_loss_and_aux_from_logs

        self._metric_averages = None
        self._metric_windows = None

    def get_state(self):
        """

        :return: state, success (True or False)
        """
        return {
                    "batch_averaging_window": self._batch_averaging_window,
                    "metric_averages": self._metric_averages,
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

        self._batch_averaging_window = state["batch_averaging_window"]
        self._metric_averages = state["metric_averages"]
        self._metric_windows = {metric_path: SlidingWindow(state=window_state)
                                for metric_path, window_state in state["metric_windows"].items()}

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

        dataset_logs = logs[self._dataset_name]

        if not self._calc_metrics_on(dataset_batch, dataset_logs, logs):
            return False

        metric_names = list(self._metric_funcs.keys())
        metric_paths = self._get_metric_paths(dataset_logs, metric_names=metric_names)
        if not self._update_metrics_windows_for(metric_paths, dataset_logs):
            return False

        if not self._calc_metric_window_averages(metric_paths, dataset_logs["mean"]):
            return False

        return True

    def _init_metrics_averaging(self, reset=False):
        if reset:
            self._metric_averages = {}
            self._metric_windows = {}

        self._metric_averages = self._metric_averages or {}
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

    def _calc_metrics_on(self, batch, metrics, logs):
        success = True

        # Use custom settings if available, else use default settings
        evaluate_settings = get_value_at('evaluate_settings', logs, warn_on_failure=False)
        if evaluate_settings is None:
            evaluate_settings = self._evaluate_settings

        if self._get_loss_and_aux_from_logs:
            loss = get_value_at(f"{self._dataset_name}.loss", logs)
            auxiliary_results = get_value_at(f"{self._dataset_name}.auxiliary_results", logs, warn_on_failure=False)
        else:
            loss, auxiliary_results = self._evaluate_loss(batch, evaluate_settings)

        metric_func_args = {
            'batch': batch,
            'loss': loss,
            'auxiliary_results': auxiliary_results,
            'evaluate_settings': evaluate_settings
        }

        for metric_name, metric_func in self._metric_funcs.items():
            try:
                metrics[metric_name] = metric_func(**metric_func_args)
            except Exception as e:
                _.log_exception(self._log, f"Exception occurred calculating {metric_name} on "
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

    def _update_metrics_windows_for(self, metric_paths, batch_metric_values):
        success = True

        for metric_path in metric_paths:
            try:
                metric_value = get_value_at(metric_path, batch_metric_values)

                sliding_window = self._metric_windows[metric_path] if metric_path in self._metric_windows else None
                if sliding_window is None:
                    self._log.debug(f"Creating sliding window for {metric_path}")

                    sliding_window = SlidingWindow(length=self._batch_averaging_window)
                    self._metric_windows[metric_path] = sliding_window

                sliding_window.slide(metric_value)
            except Exception as e:
                _.log_exception(self._log, f"Exception occurred updating sliding averaging window for "
                                           f"{self._dataset_name} batch {metric_path}", e)
                success = False

        return success

    def _calc_metric_window_averages(self, metric_paths, metrics_averages):
        success = True

        for metric_path in metric_paths:
            try:
                window_data = self._metric_windows[metric_path].window
                set_value_at(metric_path, metrics_averages, np.mean(np.array(window_data)))
            except Exception as e:
                _.log_exception(self._log, f"Exception occurred calculating sliding window average for "
                                           f"{self._dataset_name} batch {metric_path}", e)
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
    """
    By default, gets already calculated loss and auxiliary results from logs
    """

    def __init__(self, metric_funcs, batch_averaging_window, name="MetricsLogger"):
        super().__init__(dataset_name='training',
                         metric_funcs=metric_funcs,
                         batch_averaging_window=batch_averaging_window,
                         get_loss_and_aux_from_logs=True,
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

        return self._init_metrics_averaging()

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
                 metric_funcs,
                 evaluate_settings=None,
                 batch_level=True,
                 batch_assessment_period=1,
                 batch_averaging_window=None,
                 get_loss_and_aux_from_logs=False,
                 name="MetricsLogger"):
        super().__init__(dataset_name=dataset_name,
                         metric_funcs=metric_funcs,
                         evaluate_settings=evaluate_settings,
                         batch_averaging_window=batch_averaging_window,
                         get_loss_and_aux_from_logs=get_loss_and_aux_from_logs,
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

        return self._init_metrics_averaging()

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

        self._log.info(f"Calculating metrics on {self._dataset_name} dataset")

        self._dataset_iterator = iter(self._dataset)

        self._init_metrics_averaging(reset=True)

        metric_names = list(self._metric_funcs.keys())
        metric_paths = None
        # Loop over all dataset batches, this will fill the sliding windows, finally calculate the averages
        for dataset_batch in self._dataset_iterator:
            batch_metrics = {}
            if not self._calc_metrics_on(dataset_batch, batch_metrics, logs):
                return False

            if metric_paths is None:
                metric_paths = self._get_metric_paths(batch_metrics, metric_names=metric_names)

            if not self._update_metrics_windows_for(metric_paths, batch_metrics):
                return False

            sys.stdout.write('#')
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()

        return self._calc_metric_window_averages(metric_paths, logs[self._dataset_name]["mean"])

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





