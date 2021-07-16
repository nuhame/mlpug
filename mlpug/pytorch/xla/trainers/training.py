import torch_xla.core.xla_model as xm

from mlpug.pytorch.trainers import \
    Trainer, \
    DefaultTrainer as DefaultTrainerPyTorch, \
    TrainingManager as TrainingManagerPyTorch

from mlpug.mlpug_exceptions import TrainerInvalidException


class TrainingManager(TrainingManagerPyTorch):

    def _training_ended(self):
        if self.is_distributed:
            # Wait for all processes to finish
            xm.rendezvous("mlpug-training-ended")


class DefaultTrainer(DefaultTrainerPyTorch):

    def __init__(self, *args, use_mixed_precision=False, name="DefaultTrainer", **kwargs):
        if use_mixed_precision:
            raise TrainerInvalidException("Mixed precision training not supported with XLA devices")

        super(DefaultTrainer, self).__init__(*args,
                                             use_mixed_precision=use_mixed_precision,
                                             name=name,
                                             **kwargs)

    def _back_propagate_from(self, loss, last_chunk=False):
        super()._back_propagate_from(loss, last_chunk=last_chunk)
        # Required when evaluating batch chunks for gradient accumulation
        xm.mark_step()

    def _update_model_parameters(self):
        for optimizer in self.get_optimizers().values():
            xm.optimizer_step(optimizer)
