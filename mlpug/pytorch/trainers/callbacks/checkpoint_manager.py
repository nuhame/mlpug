import torch

from mlpug.trainers.callbacks.checkpoint_manager import CheckpointManager as CheckpointManagerBase

from mlpug.pytorch.multi_processing import MultiProcessingMixin
from mlpug.pytorch.utils.mlpug_data import MLPugDataCleaner


class CheckpointManager(MultiProcessingMixin, CheckpointManagerBase):

    def __init__(self,
                 *args,
                 clean_logs_func=None,
                 model_checkpoint_filename_ext='pt',
                 training_checkpoint_filename_ext='tar',
                 **kwargs):
        """

        Please see CheckpointManagerBase for all constructor arguments

        :param args:
        :param clean_logs_func: Optional. Whenever the metric we monitor here improves, a copy of the
                                current logs are made and stored in the logs object using the 'best' key.
                                This custom function cleans the best logs before they are stored. This is useful
                                to, for instance, optimize the memory usage.

                                When no function is provided a MLPugDataCleaner instance will be used by default.
                                This callable instance will move any tensors to cpu and covert to Numpy arrays.
                                Scalar Numpy arrays, will be converted into scalars (e.g. a float or int)

        :param model_checkpoint_filename_ext:
        :param training_checkpoint_filename_ext:
        :param kwargs:
        """

        if clean_logs_func is None:
            clean_logs_func = MLPugDataCleaner()

        super().__init__(*args,
                         clean_logs_func=clean_logs_func,
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
