import sys

import datetime

from mlpug.trainers.callbacks.callback import Callback
from mlpug.utils import get_value_at, is_chunkable

import basics.base_utils as _


class LogProgress(Callback):

    def __init__(self, log_period=200, set_names=None, batch_level=True, name="LogProgress"):
        super(LogProgress, self).__init__(name=name)

        self.log_period = log_period
        self.set_names = set_names or ["training"]
        self.batch_level = batch_level

    def on_batch_training_completed(self, training_batch, logs):
        if not self.batch_level:
            return True

        success = True
        batch_step = logs["batch_step"]
        if batch_step == 0 or batch_step % self.log_period == 0:
            eta = self._calc_eta(logs)
            average_duration = self._get_average_batch_duration(logs)
            sys.stdout.write('Epoch {:d}/{:d} - ETA: {:s}\tBatch {:d}/{:d} '
                             'Average batch training time {:s}\n'.format(logs["epoch"],
                                                                           logs["final_epoch"],
                                                                           eta,
                                                                           logs["batch_step"],
                                                                           logs["final_batch_step"],
                                                                           average_duration))
            sys.stdout.write('Batch:\n')
            for set_name in self.set_names:
                metrics_log = self._create_set_metrics_log_for(set_name, logs, mean_metrics=False)
                if metrics_log is None:
                    success = False
                    continue

                sys.stdout.write(f'{set_name}:\t{metrics_log}.\n')

            sys.stdout.write('\nMoving average:\n')
            for set_name in self.set_names:
                metrics_log = self._create_set_metrics_log_for(set_name, logs, mean_metrics=True)
                if metrics_log is None:
                    success = False
                    continue

                sys.stdout.write(f'{set_name}:\t{metrics_log}.\n')

            sys.stdout.write(f'\n\n')

        return success

    def on_epoch_completed(self, logs):
        duration = self._get_epoch_duration(logs)
        sys.stdout.write('Epoch {:d}/{:d}\tREADY - Duration {:s}\n'.format(logs["epoch"],
                                                                           logs["final_epoch"],
                                                                           duration))
        success = True
        sys.stdout.write('Average:\n')
        for set_name in self.set_names:
            metrics_log = self._create_set_metrics_log_for(set_name, logs, mean_metrics=True)
            if metrics_log is None:
                success = False
                continue

            sys.stdout.write(f'{set_name}:\t{metrics_log}.\n')

        sys.stdout.write(f'\n')

        return success

    def _calc_eta(self, logs):

        eta_str = None
        try:
            average_batch_duration = logs["duration"]["mean"]["batch"]
            if average_batch_duration and average_batch_duration > 0:
                batch_step = logs["batch_step"]
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
        duration_str = "[UNKNOWN]"
        try:
            duration = logs["duration"]["mean"]["batch"]
            if duration and duration > 0.0:
                duration = int(duration*1000)
                duration_str = f"{duration}ms"
        except Exception as e:
            _.log_exception(self._log, "Unable to get average batch duration", e)

        return duration_str

    def _get_epoch_duration(self, logs):
        duration_str = None
        try:
            epoch_duration = int(round(logs["duration"]["epoch"]))
            duration_str = str(datetime.timedelta(seconds=epoch_duration))
        except Exception as e:
            _.log_exception(self._log, "Unable to get epoch duration", e)

        return duration_str

    def _create_set_metrics_log_for(self, set_name, logs, mean_metrics):
        key_path = set_name
        if mean_metrics:
            key_path += '.mean'
        metrics = get_value_at(key_path, logs)

        metrics_log = self._create_log_for(metrics)
        if metrics_log is None:
            self._log.error(f'No {"mean" if mean_metrics else ""} metrics data available for {key_path}, '
                            f'unable to create log for these set metrics')
            return None

        return metrics_log

    def _create_log_for(self, metrics):
        if not _.is_dict(metrics):
            return None

        metrics = metrics.copy()
        if 'mean' in metrics:
            del metrics['mean']

        num_metrics = len(metrics)

        log_format = "{:s} {:.3f}"
        log = ""
        for c, (metric, value) in enumerate(metrics.items()):
            if metric == "mean":
                continue

            log += log_format.format(metric, value)
            if c < num_metrics - 1:
                log += ', '

        return log


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

        logs['batch_size'] = len(training_batch) if is_chunkable(training_batch) else \
            training_batch[0].size(self._batch_dimension)

        return True

