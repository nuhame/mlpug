from mlpug.trainers.training import *

from mlpug_exceptions import TrainerInvalidException


class PTTrainerMixin():

    def _activate_inference_mode(self, inference_mode):
        is_training = self.training_model.training
        if inference_mode:
            if is_training:
                self.training_model.eval()
        else:
            if not is_training:
                self.training_model.train()

    def _get_model_state(self, model, model_name=None):
        return model.state_dict()

    def _get_optimizer_state(self, optimizer, optimizer_name=None):
        return optimizer.state_dict()

    def _set_model_state(self, model, state, model_name=None):
        model.load_state_dict(state)

    def _set_optimizer_state(self, optimizer, state, optimizer_name):
        optimizer.load_state_dict(state)


class Trainer(PTTrainerMixin, TrainerBase):
    pass


class DefaultTrainer(PTTrainerMixin, DefaultTrainerBase):

    def train_on(self, batch_data, training_settings=None):
        """
        Use batch_data to perform a training iteration

        :param batch_data: batch_data object (e.g. dict, list, tuple)
        :param training_settings: optional training_settings object (usually dict)

        :return: loss, auxiliary_results

        loss : number (e.g. float)
        auxiliary_results : can be anything, e.g dict or list with values or data items
        """

        if not self.instance_valid():
            raise TrainerInvalidException()

        self._reset_gradients()

        loss, auxiliary_results = self.evaluate_loss(batch_data,
                                                     inference_mode=False,
                                                     evaluate_settings=training_settings)

        self._back_propagate_from(loss)

        self._prepare_update_model_parameters()

        self._update_model_parameters()

        return loss.item(), auxiliary_results

    def _evaluate_loss(self, batch_data, evaluate_settings=None):
        """

        Evaluates the given training model on the  given batch_data, using the optional training_settings

        :param batch_data: batch_data object to evaluate loss on (e.g. dict, list, tuple)
        :param evaluate_settings: optional evaluate_settings object (usually dict)

        :return: loss, auxiliary_results

        loss : Tensor
        auxiliary_results : can be anything, e.g dict or list with values or data items
        """
        loss, auxiliary_results = self.training_model(batch_data, evaluate_settings)

        return loss, auxiliary_results

    def _reset_gradients(self):
        for optimizer in self.get_optimizers().values():
            optimizer.zero_grad()

    def _back_propagate_from(self, loss):
        loss.backward()

    def _prepare_update_model_parameters(self):
        pass

    def _update_model_parameters(self):
        for optimizer in self.get_optimizers().values():
            optimizer.step()
