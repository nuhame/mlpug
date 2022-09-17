from tensorflow.keras import backend

from mlpug.trainers.callbacks.lr_scheduler_wrapper import LRSchedulerWrapper as LRSchedulerWrapperBase


class LRSchedulerWrapper(LRSchedulerWrapperBase):

    def get_state(self):
        """

        :return: state, success (True or False)
        """
        return None, True

    def set_state(self, state):
        """
        :param state:
        :return: success (True or False)
        """

        return True

    def _exec_schedulers(self, training_iter, model_quality=None):
        for name, scheduler in self._schedulers.items():
            scheduler.on_epoch_begin(training_iter)

        return True

    def _get_current_lr(self):
        """
        :return: dict with learning rate, per optimizer
        """

        current_lr = {}
        for name, optimizer in self.optimizers.items():
            # https://github.com/keras-team/keras/blob/b80dd12da9c0bc3f569eca3455e77762cf2ee8ef/keras/callbacks.py#L2182
            current_lr[name] = float(backend.get_value(optimizer.lr))

        return current_lr
