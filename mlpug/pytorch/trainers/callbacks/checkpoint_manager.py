import sys

import torch

from mlpug.trainers.callbacks.checkpoint_manager import CheckpointManager as CheckpointManagerBase


class CheckpointManager(CheckpointManagerBase):

    def __init__(self, *args, **kwargs):

        ext_args = {
            'model_checkpoint_filename_ext': 'pt',
            'training_checkpoint_filename_ext': 'tar'
        }

        if kwargs is not None:
            for arg, val in ext_args.items():
                if arg in kwargs:
                    continue

                kwargs[arg] = val
        else:
            kwargs = ext_args

        super().__init__(*args, **kwargs)

    def _save_model_checkpoint(self, filename, state):
        torch.save(state, filename)

    def _save_training_checkpoint(self, filename, state):
        torch.save(state, filename)
