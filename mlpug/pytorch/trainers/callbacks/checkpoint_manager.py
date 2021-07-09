import torch
import torch.distributed as dist

from mlpug.trainers.callbacks.checkpoint_manager import CheckpointManager as CheckpointManagerBase


class CheckpointManager(CheckpointManagerBase):

    def __init__(self,
                 *args,
                 model_checkpoint_filename_ext='pt',
                 training_checkpoint_filename_ext='tar',
                 is_primary=None,
                 disable_logging=None,
                 **kwargs):
        """

        Please see CheckpointManagerBase for all constructor arguments

        :param args:
        :param model_checkpoint_filename_ext:
        :param training_checkpoint_filename_ext:
        :param is_primary:
        :param disable_logging:
        :param kwargs:
        """

        if is_primary is None:
            is_primary = not dist.is_initialized() or dist.get_rank() == 0

        if disable_logging is None:
            disable_logging = not is_primary

        super().__init__(*args,
                         model_checkpoint_filename_ext=model_checkpoint_filename_ext,
                         training_checkpoint_filename_ext=training_checkpoint_filename_ext,
                         is_primary=is_primary,
                         disable_logging=disable_logging,
                         **kwargs)

    def _save_model_checkpoint(self, filename, state):
        # Do nothing if this is not the master worker
        if not self._is_primary:
            return True

        torch.save(state, filename)

    def _save_training_checkpoint(self, filename, state):
        # Do nothing if this is not the master worker
        if not self._is_primary:
            return True

        torch.save(state, filename)
