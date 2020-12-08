import io

import tensorflow as tf
import h5py
from tensorflow.python.keras.saving import hdf5_format

import basics.base_utils as _

from mlpug.trainers.training import *
from mlpug.mlpug_exceptions import TrainerInvalidException, \
    TrainerStateInvalidException, \
    BatchNotChunkableException, \
    MLPugException, \
    LossNotAvailableException

from mlpug.utils import get_value_at


class TFTrainerMixin:

    def _activate_inference_mode(self, inference_mode):
        # No pre-evaluation mode change
        pass

    def _get_model_state(self, model, model_name=None):
        state = io.BytesIO()
        with h5py.File(state, 'w') as f:
            hdf5_format.save_weights_to_hdf5_group(f, model.layers)

        return state

    def _get_optimizer_state(self, optimizer, optimizer_name=None):
        state = io.BytesIO()
        with h5py.File(state, 'w') as f:
            hdf5_format.save_optimizer_weights_to_hdf5_group(f, optimizer)

        return state

    def _set_model_state(self, model, state, model_name=None):
        with h5py.File(state, 'r') as f:
            hdf5_format.load_weights_from_hdf5_group(f, model.layers)

    def _set_optimizer_state(self, optimizer, state, optimizer_name):
        with h5py.File(state, 'r') as f:
            weights = hdf5_format.load_optimizer_weights_from_hdf5_group(f)
            optimizer.set_weights(weights)


class Trainer(TFTrainerMixin, TrainerBase):
    pass


class DefaultTrainer(TFTrainerMixin, DefaultTrainerBase):

    def __init__(self, *args, trainable_variables=None, **kwargs):
        super(DefaultTrainer, self).__init__(*args, **kwargs)

        self.trainable_variables = trainable_variables

        if self.trainable_variables is None:
            if len(self.optimizers) > 1:
                raise TrainerInvalidException(f"No trainable variables provided per optimizer")
        else:
            self.trainable_variables = convert_to_dict("optimizer", trainable_variables)

            missing_optimizer_vars = []
            for optimizer_name in self.optimizers.keys():
                if optimizer_name not in self.trainable_variables or self.trainable_variables[optimizer_name] is None:
                    missing_optimizer_vars += [optimizer_name]

            if len(missing_optimizer_vars) > 0:
                raise TrainerInvalidException(f"Missing trainable variables for optimizer(s) : "
                                              f"{', '.join(missing_optimizer_vars)}")

        self._deferred_model_components_state = None
        self._deferred_optimizers_state = None

        self._first_batch = True

    def set_model_components_state(self, state):
        """

        :param state:
        :return: success (True or False)
        """
        if not _.is_callable(getattr(state, 'items', None)):
            self._log.error("State is invalid, unable to set model components state")
            return False

        self._deferred_model_components_state = state

        self._log.debug("Model components checkpoint state received; "
                        "deferred setting the state until training has started")

        return True

    def set_optimizers_state(self, state):
        """

        :param state:
        :return: success (True, False)
        """
        if not _.is_callable(getattr(state, 'items', None)):
            self._log.error("State is invalid, unable to set optimizers state")
            return False

        self._deferred_optimizers_state = state

        self._log.debug("Optimizers checkpoint state received; "
                        "deferred setting the state until training has started")

        return True

    def set_learning_rate_for(self, optimizer_name, lr):
        """

        Set learning rate for specific optimizer `optimizer_name` to `lr`

        :param optimizer_name:
        :param lr:

        :return: True on success, else False
        """
        optimizer = self.get_optimizer(optimizer_name)
        if not hasattr(optimizer, 'learning_rate'):
            self._log.error(f"No valid optimizer available with name {optimizer_name}, unable to set learning rate")
            return False

        try:
            optimizer.learning_rate = lr
        except Exception as e:
            _.log_exception(self._log, f"Unable to set learning rate for optimizer {optimizer_name}", e)
            return False

        self._log.debug(f"Learning rate of optimizer {optimizer_name} set to : {lr}")

        return True

    def train_on(self, batch_data, training_settings=None):
        """
        Use batch_data to perform a training iteration.

        Optionally uses `batch_chunk_size` to evaluate the loss in chunks.
        If a `batch_chunk_size` was given during construction of the trainer, the gradients are updated by evaluating
        the batch in chunks.

        *Note*
        When using chunked batch processing, the default implementation assumes that the
        loss, calculated over a chunk, is the average of the sample losses.

        :param batch_data: batch_data object to train on (e.g. dict, list, tuple)
                           When `batch_chunk_size` is given, `batch_data` must be an object that implements the
                           `__len__` and `__getitem__` methods. Here the `__getitem__` method must be able to deal
                           with slices.
        :param training_settings: optional training_settings object (usually dict)

        :return: loss, auxiliary_results

        loss : number (e.g. float)
        auxiliary_results : can be anything, e.g dict or list with values or data items
        """

        if not self.instance_valid():
            raise TrainerInvalidException()

        if self._first_batch:
            self._set_deferred_model_components_state(batch_data, training_settings)
            self._set_trainable_variables()
            self._set_deferred_optimizers_state()

            self._first_batch = False

        loss, auxiliary_results, gradients = self._calc_gradients(batch_data, training_settings=training_settings)

        gradients = self._process_gradients(gradients)

        self._apply_gradients(gradients)

        return loss.numpy(), auxiliary_results

    def _set_trainable_variables(self):
        if len(self.optimizers) > 1:
            return

        # This only needs to be done once
        # Further, this situation only occurs when there is only one optimizer

        optimizer_name = next(iter(self.optimizers))
        trainable_variables = get_value_at(optimizer_name, self.trainable_variables, warn_on_failure=False)
        if trainable_variables is None:
            trainable_variables = self.training_model.trainable_variables

            self.trainable_variables = {
                optimizer_name: trainable_variables
            }

    def _set_deferred_model_components_state(self, batch_data, training_settings):
        """
        Model component state can only be set after evaluating the model on input data
        (Crazy but true)

        :param batch_data:
        :param training_settings:
        :return:
        """

        if self._deferred_model_components_state is None:
            return

        self.evaluate_loss(batch_data,
                           inference_mode=False,
                           evaluate_settings=training_settings)

        success = super().set_model_components_state(self._deferred_model_components_state)
        if not success:
            self._log.error("Unable to set deferred model components state, weights are not loaded")

        self._deferred_model_components_state = None

    def _set_deferred_optimizers_state(self):
        if self._deferred_optimizers_state is None:
            return

        for optimizer_name, optimizer in self.optimizers.items():
            trainable_variables = get_value_at(optimizer_name, self.trainable_variables, warn_on_failure=False)
            optimizer._create_all_weights(trainable_variables)

        success = super().set_optimizers_state(self._deferred_optimizers_state)
        if not success:
            self._log.error("Unable to set deferred optimizers state, weights are not loaded")

        self._deferred_optimizers_state = None

    def _evaluate_loss(self, batch_data, evaluate_settings=None, inference_mode=None):
        """
        Evaluates the given training model on the  given batch_data, using the optional training_settings
        Depending on the Deep learning backend you might need to use inference mode here

        :param batch_data: batch_data object to evaluate loss on (e.g. dict, list, tuple)
        :param evaluate_settings: optional evaluate_settings object (usually dict)
        :param inference_mode: optional bool, important when inference mode not set in `_activate_inference_mode`
                               Pytorch:     inference_mode not required here
                               Tensorflow:  inference_mode required here

        :return: dict or tuple
            {
                "loss": <Tensor>,
                "auxiliary_results": <can be anything, e.g dict or list with values or data items>
            }

            (loss, ... auxiliary results ...)
        """

        if not (type(inference_mode) is bool):
            raise TrainerStateInvalidException("Inference mode is not set")

        return self.training_model(batch_data, evaluate_settings, training=inference_mode)

    def _calc_gradients(self, batch_data, training_settings=None):
        """

        :param batch_data:
        :param training_settings:
        :return:

        :raises LossNotAvailableException
        """

        if not self.batch_chunk_size:
            with tf.GradientTape() as tape:
                results = self.evaluate_loss(batch_data,
                                             inference_mode=False,
                                             evaluate_settings=training_settings)

            if 'loss' not in results:
                raise LossNotAvailableException()

            loss = results['loss']
            auxiliary_results = get_value_at('auxiliary_results', results, warn_on_failure=False)

            gradients = self._back_propagate_from(loss, tape)
        else:
            raise NotImplementedError("Gradient accumulation over batch chunks is not implemented")

        return loss, auxiliary_results, gradients

    def _back_propagate_from(self, loss, tape, last_chunk=False):
        gradients = {}
        for optimizer_name in self.optimizers.keys():
            trainable_variables = get_value_at(optimizer_name, self.trainable_variables, warn_on_failure=False)

            gradients[optimizer_name] = tape.gradient(loss, trainable_variables)

        return gradients

    def _process_gradients(self, gradients):
        """

        :param gradients: dict with gradients per provided optimizer
                          The simple situation, when only one optimizer is given, the structure would be:
                          {
                              'optimizer': <gradients>
                          }
        :return: processed gradients with the same dict structure
        """
        return gradients

    def _apply_gradients(self, gradients):
        for optimizer_name, optimizer in self.get_optimizers().items():
            trainable_variables = get_value_at(optimizer_name, self.trainable_variables)
            if trainable_variables is None:
                raise MLPugException("Unexpected state :  trainable variables not found. Please file an issue.")

            optimizer.apply_gradients(zip(gradients[optimizer_name], trainable_variables))