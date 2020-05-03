import abc

from mlpug.utils import get_value_at, convert_to_dict
from mlpug.trainers.callbacks.callback import Callback

import basics.base_utils as _


class LRSchedulerWrapperBase(Callback, metaclass=abc.ABCMeta):

    def __init__(self,
                 schedulers,
                 batch_level=True,
                 metric_to_monitor=None,
                 name="LRSchedulerWrapper"):
        """

        Calls given schedulers, per batch or per epoch, optionally using the current model quality
        as given by `metric_to_monitor`. The updated learning rate is added to the logs object.

        :param schedulers: A single scheduler instance or a dict or list of schedulers
        :param batch_level: True if the LR schedulers should be updated after every batch, else after every epoch
        :param metric_to_monitor: key path to metric value in the log object,
                                  e.g. `validation.window_average.perplexity`, used by the schedulers. If None, it is
                                  assumed that the schedulers don't monitor any metric.
        """
        super().__init__(name=name)

        schedulers = convert_to_dict("scheduler", schedulers)

        self._schedulers = schedulers
        self._batch_level = batch_level

        self._metric_to_monitor = metric_to_monitor

    @abc.abstractmethod
    def get_state(self):
        """

        :return: state, success (True or False)
        """
        self._log.error("This method is not implemented, implement it in your child class implementation")
        return None, False

    @abc.abstractmethod
    def set_state(self, state):
        """

        :param state:
        :return: success (True or False)
        """
        self._log.error("This method is not implemented, implement it in your child class implementation")
        return None, False

    def on_batch_training_completed(self, training_batch, logs):
        if not self._batch_level:
            return True

        self._init_logs(logs)

        return self._update_lr(logs, 'global_iter')

    def on_epoch_completed(self, logs):
        if self._batch_level:
            return True

        self._init_logs(logs)

        return self._update_lr(logs, 'epoch')

    def _get_schedule_level(self):
        return 'batch' if self._batch_level else 'epoch'

    def _init_logs(self, logs):
        schedule_level = self._get_schedule_level()

        current = self._get_logs_base(logs)
        ctp = current['training_params'][schedule_level]

        if 'lr' not in ctp:
            ctp['lr'] = {}

    def _update_lr(self, logs, iter_name):
        schedule_level = self._get_schedule_level()

        current = self._get_logs_base(logs)
        ctp = current['training_params'][schedule_level]

        model_quality = self._get_model_quality(current)
        training_iter = current[iter_name]

        success = False
        try:
            success = self._exec_schedulers(training_iter, model_quality)
            if not success:
                self._log.error("Updating of the learning rate(s) failed.")
        except Exception as e:
            _.log_exception(self._log, "An unexpected error occurred, "
                                       "execution of the learning rate scheduler(s) failed", e)

        try:
            current_lr = self._get_current_lr()

            lr = get_value_at('lr', ctp, warn_on_failure=False) or {}

            ctp['lr'] = {**lr, **current_lr}
        except Exception as e:
            _.log_exception(self._log, "An unexpected error occurred, "
                                       "unable to add current learning rate values to the logs object", e)
            return False

        return success

    def _get_model_quality(self, current_logs):
        if not self._metric_to_monitor:
            return None

        return get_value_at(self._metric_to_monitor, current_logs)

    @abc.abstractmethod
    def _exec_schedulers(self, training_iter, model_quality=None):
        """

        :param training_iter
        :param model_quality:
        :return: On success : True
                 On failure : False
        """
        self._log.error("This method is not implemented, implement it in your child class implementation")
        return False

    @abc.abstractmethod
    def _get_current_lr(self):
        """
        :return: dict with learning rate, per optimizer
        """
        self._log.error("This method is not implemented, implement it in your child class implementation")
        return None



