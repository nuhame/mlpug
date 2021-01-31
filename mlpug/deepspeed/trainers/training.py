from mlpug.pytorch.trainers import Trainer
from mlpug.pytorch.trainers import DefaultTrainer as PyTorchDefaultTrainer


class DefaultTrainer(PyTorchDefaultTrainer):

    def __init__(self, *args, **kwargs):
        super(DefaultTrainer, self).__init__(*args, **kwargs)

    def _reset_gradients(self):
        # This is done by DeepSpeed
        pass

    def _back_propagate_from(self, loss, last_chunk=False):
        self.training_model.backward(loss)

    def _update_model_parameters(self):
        self.training_model.step()

    def _after_update_model_parameters(self):
        pass
