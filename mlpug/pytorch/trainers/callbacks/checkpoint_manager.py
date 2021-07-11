import torch

from mlpug.trainers.callbacks.checkpoint_manager import CheckpointManager as CheckpointManagerBase

from mlpug.pytorch.multi_processing import MultiProcessingMixin


class CheckpointManager(MultiProcessingMixin, CheckpointManagerBase):

    def __init__(self,
                 *args,
                 model_checkpoint_filename_ext='pt',
                 training_checkpoint_filename_ext='tar',
                 **kwargs):
        """

        Please see CheckpointManagerBase for all constructor arguments

        :param args:
        :param model_checkpoint_filename_ext:
        :param training_checkpoint_filename_ext:
        :param kwargs:
        """

        super().__init__(*args,
                         model_checkpoint_filename_ext=model_checkpoint_filename_ext,
                         training_checkpoint_filename_ext=training_checkpoint_filename_ext,
                         **kwargs)

    def _save_model_checkpoint(self, filename, state):
        # Do nothing if this is not the master worker
        if not self.is_primary:
            return True

        torch.save(state, filename)

    def _save_training_checkpoint(self, filename, state):
        # Do nothing if this is not the master worker
        if not self.is_primary:
            return True

        torch.save(state, filename)

    def _copy(self, source_fname, dest_fname):
        # Do nothing if this is not the master worker
        if not self.is_primary:
            return True

        return super()._copy(source_fname, dest_fname)
