from mlpug.pytorch.trainers.callbacks.checkpoint_manager import CheckpointManager as CheckpointManagerPyTorch

import torch_xla.core.xla_model as xm


class CheckpointManager(CheckpointManagerPyTorch):

    def __init__(self,
                 *args,
                 is_primary=None,
                 **kwargs):

        if is_primary is None:
            is_primary = xm.is_master_ordinal()

        super().__init__(*args, is_primary=is_primary, **kwargs)

    def _save_model_checkpoint(self, filename, state):
        # By default only the master worker saves
        xm.save(state, filename)

    def _save_training_checkpoint(self, filename, state):
        # By default only the master worker saves
        xm.save(state, filename)
