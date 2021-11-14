import abc
import hashlib

from mlpug.base import Base

from mlpug.utils.utils import get_value_at


class Callback(Base, metaclass=abc.ABCMeta):

    def __init__(self, name, base_logs_path="current", **kwargs):
        super(Callback, self).__init__(pybase_logger_name=name, **kwargs)

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

    def get_hash(self):
        """
        Used to uniquely identify callback when loading and setting callback states
        :return:
        """
        hash_str = self._get_hash_string()

        return hashlib.md5(hash_str.encode('utf-8')).hexdigest()

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

    def on_training_ended(self, stopped_early, stopped_on_error, interrupted, callback_calls_success):
        """

        :param stopped_early:
        :param stopped_on_error:
        :param interrupted:
        :param callback_calls_success:

        :return: success (True or False)
        """
        return True

    def _get_logs_base(self, logs):
        """
        Get the part of the log that needs to be accessed by the callback.
        Usually this is the "current" property, providing the logs for the current batch iteration.

        :param logs:
        :return:
        """

        if self.base_log_path is None:
            return logs
        else:
            return get_value_at(self.base_log_path, logs)

    def _get_hash_string(self):
        """
        A string, uniquely describing the callback
        :return:
        """
        return f"{self.__class__.__name__}={self.get_name()}={self._get_callback_properties_for_hash()}"

    def _get_callback_properties_for_hash(self):
        """
        This is used to create the unique callback hash.

        Returns a dict with properties describing the setup of the callback.
        It should at least contain the properties that influence the callback state.

        Property values should only be simple types, such as int, float, boolean and strings.
        Convert any object and function values (or similar) into a booleans (True = available, False = None)

        :return: dict
        """
        return {}

