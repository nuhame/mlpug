import io

import h5py

from functools import reduce

import basics.base_utils as _

import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format

from mlpug.batch_chunking import apply_chunkable_batch_wrapper, ChunkableBatchDataset, \
    create_chunks_generator, BatchChunkingResults

from mlpug.trainers.training import *
from mlpug.trainers.training import Trainer as TrainerBase
from mlpug.trainers.training import DefaultTrainer as DefaultTrainerBase


from mlpug.mlpug_exceptions import TrainerInvalidException, \
    TrainerStateInvalidException, \
    MLPugException, \
    LossNotAvailableException, InvalidParametersException, NumSamplesNotAvailableException

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

    def __init__(self, *args,
                 eager_mode=False,
                 batch_data_signature=None,
                 training_settings_signature=None,
                 distribution_strategy=None,
                 trainable_variables=None,
                 name="DefaultTrainer",
                 **kwargs):
        """

        :param args:
        :param eager_mode: If true, the training step is not wrapped in a @tf.function
        :param batch_data_signature: Is only required when eager_mode=False

        Example, when batch data is a tuple of an input and target tensor
            (
                tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            )

            Note: When batch_chunk_size is given, the sample dimension size of the tensors given
                should either be None, or equal to the `batch_chunk_size`. In the later case you have to
                ensure that the `batch_data` size is an exact multiple of the `batch_chunk_size`.

        :param training_settings_signature: Use when you use training_settings.
            Is only used when eager_mode=False

            Default is {}.

            Note: training_settings, are the same as evaluation_settings.

        :param distribution_strategy: Optional distributed training strategy

        :param trainable_variables: Only required when using multiple optimizers

        :param kwargs:
        """
        super(DefaultTrainer, self).__init__(*args, name=name, **kwargs)

        self._eager_mode = eager_mode
        self._batch_data_signature = batch_data_signature
        self._training_settings_signature = training_settings_signature

        self._distribution_strategy = distribution_strategy

        self._trainable_variables = trainable_variables

        if not eager_mode:
            self._log.info(f"Training in graph mode.")
            # TODO: Providing the batch_data_signature is likely not required anymore; check this.
            # if self._batch_data_signature is None:
            #     raise TrainerInvalidException(f"Missing batch_data_signature such that the "
            #                                   f"training step computation graph can be traced")

            if self._training_settings_signature is None:
                self._log.info("_training_settings_signature not given, setting to empty dict, "
                               "implying that training settings won't be used.")
                self._training_settings_signature = {}

            self._call_model_tf_func = self._create_call_model_tf_func() if self._distribution_strategy is None \
                else self._create_distributed_call_model_tf_func()

            if self.batch_chunk_size is None:
                self._train_step_tf_func = self._create_training_step_tf_func() if self._distribution_strategy is None \
                    else self._create_distributed_training_step_tf_func()
            else:
                self._train_step_tf_func = self._train_step if self._distribution_strategy is None \
                    else self._create_distributed_training_step_eager()
        else:
            self._log.warn("Training in eager mode.")
            self._train_step_tf_func = self._train_step if self._distribution_strategy is None \
                else self._create_distributed_training_step_eager()

            self._call_model_tf_func = self._call_model if self._distribution_strategy is None \
                else self._create_distributed_call_model_eager()

        if self._trainable_variables is None:
            if len(self.optimizers) > 1:
                raise TrainerInvalidException(f"No trainable variables provided per optimizer")
        else:
            self._trainable_variables = convert_to_dict("optimizer", trainable_variables)

            missing_optimizer_vars = []
            for optimizer_name in self.optimizers.keys():
                if optimizer_name not in self._trainable_variables or self._trainable_variables[optimizer_name] is None:
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

        :param training_settings: optional training_settings object (usually dict)

        :return: loss, auxiliary_results

        loss : number (e.g. float)
        auxiliary_results : can be anything, e.g dict or list with values or data items
        """

        if not self.instance_valid():
            raise TrainerInvalidException()

        if self._first_batch:
            # Check if we first need to restore a checkpoint
            deferred_model_components_state_set = \
                self._set_deferred_model_components_state(batch_data, training_settings)

            if deferred_model_components_state_set:
                # To set the deferred_model_components_state at the first batch the model was evaluated
                # So we can get the trainable variables from the model and subsequently set the
                # deferred optimizer state, which needs teh trainable variables.
                self._retrieve_trainable_variables()
                self._set_deferred_optimizers_state()

            self._first_batch = False

        loss, model_outputs = self._train_on(batch_data, training_settings)

        return loss.numpy(), model_outputs

    def _create_train_step_signature(self):
        # The batch_data_signature is assumes to be the element spec
        # In case of gradient accumulation the train step function gets a chunks dataset as input
        if self.batch_chunk_size is not None:
            input_signature = [
                tf.data.DatasetSpec(self._batch_data_signature),
                self._training_settings_signature,
                tf.TensorSpec(shape=(), dtype=tf.int64),  # batch_size
                tf.TensorSpec(shape=(), dtype=tf.int64)   # num_chunks
            ]

        else:
            input_signature = [
                self._batch_data_signature,
                self._training_settings_signature
            ]

        return input_signature

    def _create_distributed_training_step_tf_func(self):
        @tf.function(input_signature=self._create_train_step_signature())
        def training_step_func(
                batch_data,
                training_settings,
                batch_size=None,
                num_chunks=None):
            return self._distribution_strategy.run(
                self._train_step,
                args=(
                    batch_data,
                    training_settings,
                    batch_size,
                    num_chunks
                ))

        return training_step_func

    def _create_training_step_tf_func(self):
        @tf.function(input_signature=self._create_train_step_signature())
        # @tf.function
        def training_step_func(
                batch_data,
                training_settings,
                batch_size=None,
                num_chunks=None):
            return self._train_step(
                batch_data,
                training_settings,
                batch_size=batch_size,
                num_chunks=num_chunks
            )

        return training_step_func

    def _create_distributed_training_step_eager(self):
        def training_step_func(
                batch_data,
                training_settings,
                batch_size=None,
                num_chunks=None):
            return self._distribution_strategy.run(
                self._train_step,
                args=(
                    batch_data,
                    training_settings,
                    batch_size,
                    num_chunks
                ))

        return training_step_func

    def _train_on(self, batch_data, training_settings):
        if not self.batch_chunk_size:
            return self._train_step_tf_func(
                batch_data,
                training_settings)
        else:
            if not is_chunkable(batch_data):
                batch_data = apply_chunkable_batch_wrapper(
                    batch_data,
                    self.chunkable_batch_wrapper)

            chunks_dataset = ChunkableBatchDataset(
                batch_data,
                batch_chunk_size=self.batch_chunk_size)

            batch_size = len(batch_data)
            num_chunks = len(chunks_dataset)

            generate_chunk_func = create_chunks_generator(chunks_dataset)

            # TODO: debug
            # print(f"self._batch_data_signature : {self._batch_data_signature}")
            chunks_tf_dataset = tf.data.Dataset.from_generator(
                generate_chunk_func,
                output_signature=self._batch_data_signature
            )

            results = self._train_step_tf_func(
                chunks_tf_dataset,
                training_settings,
                batch_size,
                num_chunks
            )

            return results

    def _train_step(self, batch_data, training_settings, batch_size=None, num_chunks=None):
        loss, model_outputs, gradients = self._calc_gradients(
            batch_data,
            training_settings,
            batch_size=batch_size,
            num_chunks=num_chunks)

        self._update_model_parameters(self._prepare_update_model_parameters(gradients))

        self._after_update_model_parameters(gradients)

        return loss, model_outputs

    def _create_call_model_signature(self):
        return [
            self._batch_data_signature,              # batch_data
            self._training_settings_signature,       # evaluate_settings
            tf.TensorSpec(shape=(), dtype=tf.bool)  # inference_mode
        ]

    def _create_distributed_call_model_tf_func(self):
        @tf.function(input_signature=self._create_call_model_signature())
        def call_model_func(batch_data, evaluate_settings, inference_mode):
            return self._distribution_strategy.run(self._call_model,
                                                   args=(batch_data, evaluate_settings, inference_mode))

        return call_model_func

    def _create_call_model_tf_func(self):
        @tf.function(input_signature=self._create_call_model_signature())
        def call_model_func(batch_data, evaluate_settings, inference_mode):
            return self._call_model(batch_data, evaluate_settings, inference_mode)

        return call_model_func

    def _create_distributed_call_model_eager(self):
        def call_model_func(batch_data, evaluate_settings, inference_mode):
            return self._distribution_strategy.run(self._call_model,
                                                   args=(batch_data, evaluate_settings, inference_mode))

        return call_model_func

    def _retrieve_trainable_variables(self):
        if len(self.optimizers) > 1:
            return

        # This only needs to be done once
        # Further, this situation only occurs when there is only one optimizer

        optimizer_name = next(iter(self.optimizers))
        trainable_variables = get_value_at(optimizer_name, self._trainable_variables, warn_on_failure=False)
        if trainable_variables is None:
            trainable_variables = self.training_model.trainable_variables

            self._trainable_variables = {
                optimizer_name: trainable_variables
            }

    def _set_deferred_model_components_state(self, batch_data, training_settings):
        """
        Model component state can only be set after evaluating the model on input data
        (Crazy but true)

        :param batch_data:
        :param training_settings:
        :return: True if set, else False
        """

        if self._deferred_model_components_state is None:
            return False

        def dry_eval_model():
            self.evaluate_loss(batch_data,
                               inference_mode=False,
                               evaluate_settings=training_settings)

        if self._distribution_strategy is not None:
            with self._distribution_strategy.scope():
                dry_eval_model()
        else:
            dry_eval_model()

        success = super().set_model_components_state(self._deferred_model_components_state)
        if not success:
            self._log.error("Unable to set deferred model components state, weights are not loaded")

        self._deferred_model_components_state = None

        return success

    def _set_deferred_optimizers_state(self):
        if self._deferred_optimizers_state is None:
            return

        def create_optimizer_weights():
            for optimizer_name, optimizer in self.optimizers.items():
                trainable_variables = get_value_at(optimizer_name, self._trainable_variables, warn_on_failure=False)
                optimizer._create_all_weights(trainable_variables)

        if self._distribution_strategy is not None:
            with self._distribution_strategy.scope():
                create_optimizer_weights()
        else:
            create_optimizer_weights()

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

        if not inference_mode:
            # @tf.function on the train step level
            return self._call_model(batch_data, evaluate_settings, inference_mode)
        else:
            return self._call_model_tf_func(batch_data, evaluate_settings, tf.constant(inference_mode, dtype=tf.bool))

    def _call_model(self, batch_data, evaluate_settings, inference_mode):
        return self.training_model(batch_data, evaluate_settings, inference_mode)

    def _calc_gradients(
            self,
            batch_data,
            training_settings,
            batch_size=None,
            num_chunks=None
    ):
        """

        :param batch_data:
        :param training_settings:

        Required when using gradient accumulation:
        :param batch_size:
        :param num_chunks:

        :return:

        :raises LossNotAvailableException
        """

        if self.batch_chunk_size is None:
            return self._calc_gradients_atomic_batch(
                batch_data,
                training_settings
            )
        else:
            if batch_size is None or num_chunks is None:
                raise InvalidParametersException(
                    f"batch_size ({batch_size}) and num_chunks ({num_chunks}) must be given to perform "
                    f"gradient accumulation."
                )

            return self._calc_gradients_chunked(
                batch_data,
                training_settings,
                batch_size,
                num_chunks
            )

    def _calc_gradients_atomic_batch(self, batch_data, training_settings):

        with tf.GradientTape() as tape:
            results = self.evaluate_loss(
                batch_data,
                inference_mode=False,
                evaluate_settings=training_settings)

        if 'loss' not in results:
            raise LossNotAvailableException()

        if self._trainable_variables is None:
            # We now have evaluated the model and the trainable variables should be available
            self._retrieve_trainable_variables()

        loss = results['loss']
        model_outputs = [results]
        gradients = self._back_propagate_from(loss, tape)

        return loss, model_outputs, gradients

    def _calc_gradients_chunked(
            self,
            chunks_tf_dataset,
            training_settings,
            batch_size,
            num_chunks
    ):
        """

        See `train_on` method.

        Calculate batch gradients by accumulating gradients over batch chunks.
        `chunks_tf_dataset` contains batch chunks which are slices of the batch data to use during this training step.
        The slices (chunks) have size `self.batch_chunk_size`.

        :param chunks_tf_dataset:
        :param training_settings:

        :param batch_size: Total batch size over all chunks
        :param num_chunks: Number of batch chunks in the chunks_tf_dataset

        :return: loss, model_outputs
                model_outputs: BatchChunkingResults: a list of tuples, one tuple per batch chunk results:
                    [{'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>},  # Chunk 1
                     ...
                     {'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>}]  # Chunk N

        :rtype: Tuple[Tensor, BatchChunkingResults[Dict]]
        """

        def process_chunk(chunk, last_chunk):
            with tf.GradientTape() as tape:
                # Chunk model outputs
                chunk_mos = self.evaluate_loss(
                    chunk,
                    inference_mode=False,
                    evaluate_settings=training_settings)

                if 'loss' not in chunk_mos:
                    raise LossNotAvailableException()

                if 'num_samples' not in chunk_mos:
                    raise NumSamplesNotAvailableException()

                # loss is assumed to be the average over the sample loss for the chunk
                # Divide through batch size to factor in that this loss is part of a larger batch.
                chunk_loss = chunk_mos['loss']
                num_chunk_samples = chunk_mos['num_samples']
                chunk_loss = float(num_chunk_samples) * chunk_loss / float(batch_size)

            if self._trainable_variables is None:
                # We now have evaluated the model and the trainable variables should be available
                self._retrieve_trainable_variables()

            chunk_grads = self._back_propagate_from(chunk_loss, tape, last_chunk=last_chunk)

            return chunk_mos, chunk_grads

        model_outputs = BatchChunkingResults()
        accumulated_grads = {}
        for chunk_idx, batch_chunk in enumerate(chunks_tf_dataset):
            chunk_model_outputs, chunk_gradients = process_chunk(
                batch_chunk,
                chunk_idx == (num_chunks-1))

            # ### ACCUMULATE CHUNK MODEL OUTPUTS ###
            model_outputs += (chunk_model_outputs,)

            # ### ACCUMULATE GRADIENTS ###
            if chunk_idx == 0:
                if self._trainable_variables is None or len(self._trainable_variables) == 0:
                    raise MLPugException("Unexpected state :  trainable variables not found. Please file an issue.")

                for optimizer_name, tvs in self._trainable_variables.items():
                    accumulated_grads[optimizer_name] = [tf.zeros_like(tv) for tv in tvs]

            for optimizer_name, opt_chunk_grads in chunk_gradients.items():
                opt_accu_grads = accumulated_grads[optimizer_name]

                accumulated_grads[optimizer_name] = [
                    (accu_param_grad + chunk_param_grad)
                    for accu_param_grad, chunk_param_grad in zip(opt_accu_grads, opt_chunk_grads)
                ]

        loss = reduce(lambda tot, mo: tot + (float(mo['num_samples']) * mo['loss']), model_outputs, 0)
        num_samples = reduce(lambda tot, mo: tot + mo['num_samples'], model_outputs, 0)

        loss /= float(num_samples)

        # loss, and model outputs for each chunk
        return loss, model_outputs, accumulated_grads

    def _back_propagate_from(self, loss, tape, last_chunk=False):
        gradients = {}
        for optimizer_name in self.optimizers.keys():
            trainable_variables = get_value_at(optimizer_name, self._trainable_variables, warn_on_failure=False)

            gradients[optimizer_name] = tape.gradient(loss, trainable_variables)

        return gradients

    def _prepare_update_model_parameters(self, gradients):
        """

        :param gradients: dict with gradients per provided optimizer
                          The simple situation, when only one optimizer is given, the structure would be:
                          {
                              'optimizer': <gradients>
                          }
        :return: processed gradients with the same dict structure
        """
        return gradients

    def _update_model_parameters(self, gradients):
        for optimizer_name, optimizer in self.get_optimizers().items():
            trainable_variables = get_value_at(optimizer_name, self._trainable_variables)
            if trainable_variables is None:
                raise MLPugException("Unexpected state :  trainable variables not found. Please file an issue.")

            optimizer.apply_gradients(zip(gradients[optimizer_name], trainable_variables))

    def _after_update_model_parameters(self, gradients):
        pass
