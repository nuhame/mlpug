import abc

from basics.base import Base

from mlpug.utils.utils import get_value_at


class Callback(Base, metaclass=abc.ABCMeta):

    def __init__(self, name, base_logs_path="current"):
        super(Callback, self).__init__(pybase_logger_name=name)

        self.name = name

        self.base_log_path = base_logs_path

        self.training_manager = None
        self.trainer = None
        self.model_components = None
        self.optimizers = None

    def set_training_manager(self, manager):
        self.training_manager = manager
        self.trainer = manager.get_trainer()
        self.model_components = self.trainer.get_model_components()
        self.optimizers = self.trainer.get_optimizers()

        return True

    def get_name(self):
        return self.name

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

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):
        """

        :param num_epochs:
        :param num_batches_per_epoch:
        :param start_epoch:
        :param start_batch:
        :param start_update_iter:

        :return: success (True or False)
        """
        return True

    def on_epoch_start(self, logs):
        """

        :param logs:

        :return: success (True or False)
        """
        return True

    def on_batch_training_start(self, training_batch, logs):
        """

        :param training_batch:
        :param logs:

        :return: success (True or False)
        """
        return True

    def on_batch_training_failed(self, exception, logs):
        """

        :param exception:
        :param logs:

        :return: success (True or False)
        """
        return True

    def on_batch_training_completed(self, training_batch, logs):
        """

        :param training_batch:
        :param logs:

        :return: success (True or False)
        """
        return True

    def on_epoch_completed(self, logs):
        """

        :param logs:

        :return: success (True or False)
        """
        return True

    def on_training_ended(self, stopped_early, stopped_on_error, callback_calls_success):
        """

        :param stopped_early:
        :param stopped_on_error:
        :param callback_calls_success:

        :return: success (True or False)
        """
        return True

    def _get_logs_base(self, logs):
        if self.base_log_path is None:
            return logs
        else:
            return get_value_at(self.base_log_path, logs)

