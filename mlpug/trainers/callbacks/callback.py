import abc

from typing import Any, Optional, Tuple, Mapping


class TrainingManager:
    pass


from mlpug.base import Base
from mlpug.utils.utils import get_value_at


class Callback(Base, metaclass=abc.ABCMeta):

    def __init__(self,
                 name: str,
                 base_logs_path: str = "current",
                 **kwargs: Any):
        super(Callback, self).__init__(pybase_logger_name=name, **kwargs)

        self.name = name

        self.base_log_path = base_logs_path

        self.training_manager = None
        self.trainer = None
        self.model_components = None
        self.optimizers = None

    def set_training_manager(self, manager: TrainingManager):
        self.training_manager = manager
        self.trainer = manager.get_trainer()
        self.model_components = self.trainer.get_model_components()
        self.optimizers = self.trainer.get_optimizers()

        return True

    def get_name(self) -> str:
        return self.name

    def get_state(self) -> Tuple[Optional[Mapping], bool]:
        """

        :return: state, success (True or False)
        """
        return None, True

    def set_state(self, state: Mapping) -> bool:
        """

        :param state:
        :return: success (True or False)
        """
        return True

    def on_training_start(self,
                          num_epochs: int,
                          num_batches_per_epoch: int,
                          start_epoch: int,
                          start_batch: int,
                          start_update_iter: int) -> bool:
        """

        :param num_epochs:
        :param num_batches_per_epoch:
        :param start_epoch:
        :param start_batch:
        :param start_update_iter:

        :return: success (True or False)
        """
        return True

    def on_epoch_start(self, logs: dict) -> bool:
        """

        :param logs:

        :return: success (True or False)
        """
        return True

    def on_batch_training_start(self, training_batch: Any, logs: dict) -> bool:
        """

        :param training_batch:
        :param logs:

        :return: success (True or False)
        """
        return True

    def on_batch_training_failed(self, exception: BaseException, logs: dict) -> bool:
        """

        :param exception:
        :param logs:

        :return: success (True or False)
        """
        return True

    def on_batch_training_completed(self, training_batch: Any, logs: dict) -> bool:
        """

        :param training_batch:
        :param logs:

        :return: success (True or False)
        """
        return True

    def on_epoch_completed(self, logs: dict) -> bool:
        """

        :param logs:

        :return: success (True or False)
        """
        return True

    def on_training_ended(self,
                          stopped_early: bool,
                          stopped_on_error: bool,
                          interrupted: bool,
                          callback_calls_success: bool) -> bool:
        """

        :param stopped_early:
        :param stopped_on_error:
        :param interrupted:
        :param callback_calls_success:

        :return: success (True or False)
        """
        return True

    def _get_logs_base(self, logs: dict) -> dict:
        if self.base_log_path is None:
            return logs
        else:
            return get_value_at(self.base_log_path, logs)
