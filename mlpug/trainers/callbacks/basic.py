import sys

import datetime

from mlpug.trainers.callbacks.callback import Callback
from mlpug.utils import get_value_at, is_chunkable

import basics.base_utils as _


class LogProgress(Callback):

    def __init__(self, log_period=200, set_names=None, batch_level=True, logs_base_path="current", name="LogProgress"):
        super(LogProgress, self).__init__(name=name)

        self.log_period = log_period
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
        batch_step = current["batch_step"]

        has_dataset_level_metrics = False
        for set_name in self.set_names:
            dataset_metrics = get_value_at(f"{set_name}.dataset", current, warn_on_failure=False)
            has_dataset_level_metrics |= type(dataset_metrics) is dict and len(dataset_metrics) > 0

            if has_dataset_level_metrics:
                break

        if batch_step == 0 or batch_step % self.log_period == 0 or has_dataset_level_metrics:
            eta = self._calc_eta(logs)
            average_duration = self._get_average_batch_duration(logs)

            sys.stdout.write('\nEpoch {:d}/{:d} - ETA: {:s}\tBatch {:d}/{:d} '
                             'Average batch training time {:s}\n'.format(current["epoch"],
                                                                         logs["final_epoch"],
                                                                         eta,
                                                                         current["batch_step"],
                                                                         logs["final_batch_step"],
                                                                         average_duration))

            for metric_level in ['batch', 'window_average', 'dataset', 'epoch']:
                self._write_metric_logs(metric_level, logs)
                sys.stdout.write(f'\n')

            sys.stdout.write(f'\n')

        return success

    def on_epoch_completed(self, logs):
        current = self._get_logs_base(logs)
        duration = self._get_epoch_duration(logs)
        sys.stdout.write('\nEpoch {:d}/{:d}\tREADY - Duration {:s}\n'.format(current["epoch"],
                                                                           logs["final_epoch"],
                                                                           duration))
        success = True
        for metric_level in ['window_average', 'dataset', 'epoch']:
            self._write_metric_logs(metric_level, logs)
            sys.stdout.write(f'\n')

        sys.stdout.write(f'\n')

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
            sys.stdout.write(f'{self.metric_level_names[metric_level]}:\n')
            sys.stdout.write(metrics_log)

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

        for c, (metric, value) in enumerate(metrics.items()):
            if metric in skip_metric_names:
                continue

            if type(value) is dict:
                log += "\n" + self._create_log_for(value, metric, log_depth+1)
            elif isinstance(value, (float, int)):
                log_format = self._get_log_format(value)
                log += log_format.format(metric, value)
            else:
                log += "[UNKNOWN]"

            if c < num_metrics - 1:
                log += ', '

        return log

    def _get_log_format(self, value):
        if abs(value) < 0.1:
            log_format = "{:<9s} {:.3e}"
        else:
            log_format = "{:<9s} {:>9.3f}"

        return log_format


class BatchSizeLogger(Callback):

    def __init__(self, batch_dimension=1, name="BatchSizeLogger"):
        super().__init__(name=name)

        self._batch_dimension = batch_dimension

    def on_batch_training_start(self, training_batch, logs):
        """

        :param training_batch:
        :param logs:

        :return: success (True or False)
        """

        current = self._get_logs_base(logs)

        current['training_params']['batch']['batch_size'] = len(training_batch) if is_chunkable(training_batch) else \
            training_batch[0].size(self._batch_dimension)

        return True

