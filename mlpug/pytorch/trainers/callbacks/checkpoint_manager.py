import sys

import torch

from mlpug.trainers.callbacks.checkpoint_manager import CheckpointManagerBase

import basics.base_utils as _


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

    def _save_training_checkpoint(self, filename):
        state, success = self.training_manager.get_state()

        if state:
            if not success:
                self._log.warn("Getting the training state was not completely successful, "
                               "trying to save available state data any way ...")

            try:
                self._log.debug(f"Saving training checkpoint : {filename}")
                torch.save(state, filename)
            except Exception as e:
                _.log_exception(self._log, "Saving the training checkpoint failed", e)
                success = False

        return success

    def _save_model_checkpoint(self, filename):
        state, success = self.trainer.get_model_components_state()

        if state:
            if not success:
                self._log.warn("Getting the model components state was not completely successful, "
                               "trying to save available state data any way ...")

            if self._model_hyper_parameters:
                try:
                    state['hyper_parameters'] = self._model_hyper_parameters
                except Exception as e:
                    _.log_exception(self._log, "Unable to add model hyper parameters to model checkpoint, "
                                               "trying to save the checkpoint anyway ...", e)
                    success = False

            try:
                manager_state, manager_state_success = self.training_manager.get_state_for_model_checkpoint()

                if manager_state_success:
                    state['manager_state'] = manager_state
                else:
                    self._log.warn("Getting the manager state for the model checkpoint was not successful, "
                                   "will continue any way without this data ...")
                    success = False

            except Exception as e:
                _.log_exception(self._log, "Unable to add manager state to model checkpoint, "
                                           "trying to save the checkpoint anyway ...", e)
                success = False

            try:
                self._log.debug(f"Saving model checkpoint : {filename}")
                torch.save(state, filename)
            except Exception as e:
                _.log_exception(self._log, "Saving the model checkpoint failed", e)
                success = False

        return success
