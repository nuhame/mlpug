import sys
import pprint
import datetime

from mlpug.trainers.callbacks.callback import Callback
from mlpug.utils import get_value_at, is_chunkable, describe_data

import basics.base_utils as _
from basics.logging_utils import log_exception


class LogProgress(Callback):

    def __init__(self,
                 log_period=None,
                 log_condition_func=None,
                 set_names=None,
                 batch_level=True,
                 logs_base_path="current",
                 name="LogProgress",
                 **kwargs):
        """

        Note: if there are dataset level metrics available, while logging at batch level, these
              metrics will always be logged

        :param log_period: Optional, number of batches or epochs between progress logs
                           If both log_period and log_condition_func are not given, this defaults to 200.
        :param log_condition_func: Optional, if evaluated to True progress will be logged
                                   log_condition_func(logs, dataset_batch) -> Bool
        :param set_names:
        :param batch_level:
        :param logs_base_path:
        :param name:
        :param kwargs:
        """
        super(LogProgress, self).__init__(name=name, **kwargs)

        if log_period is None and log_condition_func is None:
            log_period = 200
            self._log.debug(f"Logging progress every {log_period} batch training steps")

        if log_condition_func is None:
            def log_progress(logs, batch_data):
                current = self._get_logs_base(logs)
                batch_step = current["batch_step"]

                return batch_step % self.log_period

            log_condition_func = log_progress

        if log_condition_func is not None and not callable(log_condition_func):
            raise ValueError("log_condition_func must be callable and have "
                             "signature log_condition_func(logs, dataset_batch) -> Bool")

        self.log_period = log_period
        self.log_condition_func = log_condition_func
        self.set_names = set_names or ["training"]
        self.batch_level = batch_level
        self.logs_base_path = logs_base_path

        self.metric_level_names = {
            'batch': 'Batch',
            'window_average': "Moving average",
            'dataset': "Computed over dataset",
            'epoch': "Epoch"
        }

    def on_batch_training_completed(self, training_batch, logs):
        if not self.batch_level:
            return True

        success = True
        current = self._get_logs_base(logs)

        has_dataset_level_metrics = False
        for set_name in self.set_names:
            dataset_metrics = get_value_at(f"{set_name}.dataset", current, warn_on_failure=False)
            has_dataset_level_metrics |= type(dataset_metrics) is dict and len(dataset_metrics) > 0

            if has_dataset_level_metrics:
                break

        if has_dataset_level_metrics or self.log_condition_func(logs, training_batch):
            eta = self._calc_eta(logs)
            average_duration = self._get_average_batch_duration(logs)

            self._write('\nEpoch {:d}/{:d} - ETA: {:s}\tBatch {:d}/{:d} '
                        'Average batch training time {:s}\n'.format(current["epoch"],
                                                                    logs["final_epoch"],
                                                                    eta,
                                                                    current["batch_step"],
                                                                    logs["final_batch_step"],
                                                                    average_duration))

            for metric_level in ['batch', 'window_average', 'dataset', 'epoch']:
                self._write_metric_logs(metric_level, logs)
                self._write(f'\n')

            self._write(f'\n')

        return success

    def on_epoch_completed(self, logs):
        current = self._get_logs_base(logs)
        duration = self._get_epoch_duration(logs)
        self._write('\n')
        self._write('###############################################################################')
        self._write('\n')
        self._write('Epoch {:d}/{:d}\tREADY - Duration {:s}\n'.format(current["epoch"],
                                                                      logs["final_epoch"],
                                                                      duration))
        success = True
        for metric_level in ['window_average', 'dataset', 'epoch']:
            self._write_metric_logs(metric_level, logs)
            self._write(f'\n')

        self._write(f'\n')

        return success

    def _calc_eta(self, logs):

        current = self._get_logs_base(logs)

        eta_str = None
        try:
            training_params = current["training_params"]

            average_batch_duration = training_params["window_average"]["duration"]
            if average_batch_duration and average_batch_duration > 0:
                batch_step = current["batch_step"]
                final_batch_step = logs["final_batch_step"]
                num_batches_to_go = final_batch_step - batch_step + 1

                eta_seconds = int(round(average_batch_duration * num_batches_to_go))

                eta_str = str(datetime.timedelta(seconds=eta_seconds))
            else:
                eta_str = "[UNKNOWN]"
        except Exception as e:
            _.log_exception(self._log, "Unable to calculate epoch ETA", e)

        return eta_str

    def _get_average_batch_duration(self, logs):
        current = self._get_logs_base(logs)

        duration_str = "[UNKNOWN]"
        try:
            duration = current["training_params"]["window_average"]["duration"]
            if duration and duration > 0.0:
                duration = int(duration*1000)
                duration_str = f"{duration}ms"
        except Exception as e:
            _.log_exception(self._log, "Unable to get average batch duration", e)

        return duration_str

    def _get_epoch_duration(self, logs):
        current = self._get_logs_base(logs)

        duration_str = None
        try:
            epoch_duration = int(round(current["training_params"]["epoch"]["duration"]))
            duration_str = str(datetime.timedelta(seconds=epoch_duration))
        except Exception as e:
            _.log_exception(self._log, "Unable to get epoch duration", e)

        return duration_str

    def _write_metric_logs(self, metric_level, logs):
        metrics_log = ''
        for set_name in self.set_names:
            set_metrics_log = self._create_set_metrics_log_for(set_name, metric_level, logs)
            if set_metrics_log is None:
                continue

            metrics_log += f'{set_name:<15}: {set_metrics_log}.\n'

        if len(metrics_log) > 0:
            self._write(f'{self.metric_level_names[metric_level]}:\n')
            self._write(metrics_log)

    def _create_set_metrics_log_for(self, set_name, metric_level, logs):
        current = self._get_logs_base(logs)

        key_path = f"{set_name}.{metric_level}"
        metrics = get_value_at(key_path, current, warn_on_failure=False)
        return self._create_log_for(metrics)

    def _create_log_for(self, metrics, base_metric=None, log_depth=0):
        if not _.is_dict(metrics):
            return None

        metric_names = set(metrics.keys())
        # TODO : Make this a library level constant
        skip_metric_names = {"auxiliary_results", "duration"}

        num_metrics = len(metric_names-skip_metric_names)
        if num_metrics < 1:
            return None

        log = "\n"*int(log_depth > 0) + "\t"*log_depth
        if base_metric is not None:
            log += f"{base_metric:<15}: "

        metric_value_logs = []
        for c, (metric, value) in enumerate(metrics.items()):
            if metric in skip_metric_names:
                continue

            if type(value) is tuple:
                # use the first value as metric value, the other values are auxiliary results meant for other purposes
                value = value[0]

            if type(value) is dict:
                nested_logs = self._create_log_for(value, metric, log_depth+1)
                if nested_logs is not None:
                    metric_value_logs += ["\n" + nested_logs]
            else:
                try:
                    log_format = self._get_log_format(value)
                    metric_value_logs += [log_format.format(metric, value)]
                except Exception as e:
                    metric_value_logs += ["[UNKNOWN]"]

        if len(metric_value_logs) > 0:
            log += ', '.join(metric_value_logs)

        return log

    def _get_log_format(self, value):
        if abs(value) < 0.1:
            log_format = "{:<9s} {:.3e}"
        else:
            log_format = "{:<9s} {:>9.3f}"

        return log_format

    def _write(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()


class BatchSizeLogger(Callback):

    def __init__(self, batch_dimension=1, name="BatchSizeLogger", **kwargs):
        super().__init__(name=name, **kwargs)

        self._batch_dimension = batch_dimension

    def on_batch_training_start(self, training_batch, logs):
        """

        :param training_batch:
        :param logs:

        :return: success (True or False)
        """

        current = self._get_logs_base(logs)

        # TODO : doesn't work for Tensorflow
        current['training_params']['batch']['batch_size'] = len(training_batch) if is_chunkable(training_batch) else \
            training_batch[0].size(self._batch_dimension)

        return True


class DescribeLogsObject(Callback):

    def __init__(self,
                 batch_level=True,
                 log_condition_func=None,
                 indent=1,
                 width=120,
                 depth=8,
                 compact=True,
                 sort_dicts=False,
                 name="DescribeLogsObject",
                 **kwargs):
        """

        :param batch_level:
        :param log_condition_func: Optional, if evaluated to True a description of the logs object will be logged
                                   log_condition_func(logs, dataset_batch) -> Bool

                                   This function is only applied when batch_level=True

        PrettyPrinter params:
        :param indent:
        :param width:
        :param depth:
        :param compact:
        :param sort_dicts:

        :param name:
        :param kwargs:
        """
        super().__init__(name=name, **kwargs)

        if log_condition_func is not None and not callable(log_condition_func):
            raise ValueError("log_condition_func must be callable and have "
                             "signature log_condition_func(logs, dataset_batch) -> Bool")

        self._batch_level = batch_level
        self._log_condition_func = log_condition_func

        self._printer = pprint.PrettyPrinter(indent=indent,
                                             width=width,
                                             depth=depth,
                                             compact=compact,
                                             sort_dicts=sort_dicts)

    def on_batch_training_completed(self, training_batch, logs):
        """

        :param training_batch:
        :param logs:

        :return: success (True or False)
        """

        if not self._batch_level:
            return True

        if not self._log_condition_func(logs, training_batch):
            return True

        return self._safe_describe(logs)

    def on_epoch_completed(self, logs):
        if self._batch_level:
            return True

        return self._safe_describe(logs)

    def _safe_describe(self, logs):
        try:
            self._describe(logs)
        except Exception as e:
            log_exception(self._log, "Unable to describe and log the logs object", e)
            return False

        return True

    def _describe(self, logs):
        logs_description_data = describe_data(logs)
        logs_description = self._printer.pformat(logs_description_data)

        self._log.info(f"Current state of the logs object :\n{logs_description}\n")
