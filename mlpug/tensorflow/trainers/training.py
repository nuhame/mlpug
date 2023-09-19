import io
from typing import Dict, Callable, List

import h5py

from functools import reduce

import basics.base_utils as _

import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format

from mlpug.batch_chunking import (
    BatchChunkingResults,
    convert_to_chunkable_dataset,
    get_total_batch_size,
    get_num_chunks, ChunkableBatchDatasetInterface
)

from mlpug.trainers.training import *
from mlpug.trainers.training import Trainer as TrainerBase
from mlpug.trainers.training import DefaultTrainer as DefaultTrainerBase

from mlpug.mlpug_exceptions import (
    TrainerInvalidException,
    TrainerStateInvalidException,
    MLPugException,
    LossNotAvailableException,
    NumSamplesNotAvailableException
)

from mlpug.utils import get_value_at

from mlpug.tensorflow.batch_chunking import DistributedChunkableBatchDataset


def create_distributed_func(func: Callable, distribution_strategy):
    def distributed_func(*args):
        return distribution_strategy.run(
            func,
            args=args
        )

    return distributed_func


def reduce_chunk_num_samples(accumulated_model_outputs):
    return reduce(lambda tot, mo: tot + mo['num_samples'], accumulated_model_outputs, 0)


def reduce_chunk_loss(accumulated_model_outputs: BatchChunkingResults, num_samples: int):
    loss = reduce(lambda tot, mo: tot + (float(mo['num_samples']) * mo['loss']), accumulated_model_outputs, 0)

    loss /= float(num_samples)

    return loss


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

        # TODO: There are a few issues here:
        #       1) In non-eager mode, it is very hard to get batch chunking right, because it involves
        #          slicing of batches in the graph.
        #       2) When using a distribution strategy, such as MirroredStrategy in eager mode training gets stuck
        # if self._distribution_strategy is not None and self._eager_mode:
        #     raise ValueError(f"Distributed training in eager mode is not available.")
        #
        # if self.batch_chunk_size is not None and not self._eager_mode:
        #     raise ValueError(f"Gradient accumulation only available eager mode.")

        # Tensorflow likes wrapping as follows:
        # 1) strategy.run
        # 2) tf.function

        self._call_model_wrapped = self._call_model
        self._train_step_wrapped = self._train_step

        self._process_chunk_wrapped = self._process_chunk
        self._accumulate_chunk_process_results_wrapped = self._accumulate_chunk_process_results
        self._reduce_chunk_num_samples_wrapped = reduce_chunk_num_samples
        self._reduce_chunk_loss_wrapped = reduce_chunk_loss

        self._apply_gradients_wrapped = self._apply_gradients

        self._num_replicas = 1
        if distribution_strategy is not None:

            self._num_replicas = distribution_strategy.num_replicas_in_sync

            self._call_model_wrapped = create_distributed_func(
                self._call_model_wrapped,
                distribution_strategy
            )

            if self.batch_chunk_size is None:
                self._train_step_wrapped = create_distributed_func(
                    self._train_step_wrapped,
                    distribution_strategy
                )
            else:
                self._process_chunk_wrapped = create_distributed_func(
                    self._process_chunk_wrapped,
                    distribution_strategy
                )
                self._accumulate_chunk_process_results_wrapped = create_distributed_func(
                    self._accumulate_chunk_process_results_wrapped,
                    distribution_strategy
                )
                self._reduce_chunk_num_samples_wrapped = create_distributed_func(
                    self._reduce_chunk_num_samples_wrapped,
                    distribution_strategy
                )
                self._reduce_chunk_loss_wrapped = create_distributed_func(
                    self._reduce_chunk_loss_wrapped,
                    distribution_strategy
                )
                self._apply_gradients_wrapped = create_distributed_func(
                    self._apply_gradients_wrapped,
                    distribution_strategy
                )

        if not eager_mode:
            self._log.info(f"Training in graph mode.")
            if self._batch_data_signature is None:
                self._log.warning("batch_data_signature not given, "
                                  "graph compilation will be done without specifying an input signature")

            if self._training_settings_signature is None:
                self._log.info("training_settings_signature not given, setting to empty dict, "
                               "assuming that not training settings are provided.")
                self._training_settings_signature = {}

            self._call_model_wrapped = tf.function(
                self._call_model_wrapped,
                input_signature=self._create_call_model_signature()
            )

            if self.batch_chunk_size is None:
                self._train_step_wrapped = tf.function(
                    self._train_step_wrapped,
                    input_signature=self._create_train_step_signature()
                )
            else:
                self._process_chunk_wrapped = tf.function(
                    self._process_chunk_wrapped,
                    input_signature=self._create_process_chunk_signature()
                )

                self._accumulate_chunk_process_results_wrapped = tf.function(
                    self._accumulate_chunk_process_results_wrapped
                )

                self._reduce_chunk_num_samples_wrapped = tf.function(
                    self._reduce_chunk_num_samples_wrapped
                )

                self._reduce_chunk_loss_wrapped = tf.function(
                    self._reduce_chunk_loss_wrapped
                )

                # In case of batch chunking, the parent function is not wrapped in a tf.function.
                # Hence, wrapping it here.
                self._apply_gradients_wrapped = tf.function(self._apply_gradients_wrapped)
        else:
            self._log.warn("Training in eager mode.")

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

        return loss, model_outputs

    def _create_call_model_signature(self):
        if self._batch_data_signature:
            return None

        return [
            self._batch_data_signature,              # batch_data
            self._training_settings_signature,       # evaluate_settings
            tf.TensorSpec(shape=(), dtype=tf.bool)   # inference_mode
        ]

    def _create_train_step_signature(self):
        if self._batch_data_signature:
            return None

        return [
            self._batch_data_signature,
            self._training_settings_signature
        ]

    def _create_process_chunk_signature(self):
        if self._batch_data_signature:
            return None

        return [
            self._batch_data_signature,               # chunk, has same signature
            tf.TensorSpec(shape=(), dtype=tf.bool),   # last_chunk
            tf.TensorSpec(shape=(), dtype=tf.int64),  # total_batch_size
            self._training_settings_signature         # training_settings
        ]

    def _train_on(self, batch_data, training_settings):
        if self.batch_chunk_size is None:
            # Wrapped in tf.function and strategy.run if not in eager_mode
            return self._train_step_wrapped(
                batch_data,
                training_settings
            )
        else:
            if self._distribution_strategy is None:
                batch_data = convert_to_chunkable_dataset(
                    batch_data,
                    self.chunkable_batch_wrapper,
                    self.batch_chunk_size
                )
            else:
                batch_data = DistributedChunkableBatchDataset(
                    batch_data,
                    self.chunkable_batch_wrapper,
                    self.batch_chunk_size,
                    self._distribution_strategy
                )

            return self._train_step_in_chunks(
                batch_data,
                training_settings
            )

    def _train_step(self, batch_data, training_settings):
        loss, model_outputs, gradients = self._calc_gradients(
            batch_data,
            training_settings)

        # Don't need to wrap this in tf.function, since that's already done at the caller level
        self._apply_gradients(gradients)

        return loss, model_outputs

    def _train_step_in_chunks(
            self,
            chunkable_batch_dataset: ChunkableBatchDatasetInterface,
            training_settings
    ):
        loss, model_outputs, gradients = self._calc_gradients_in_chunks(
            chunkable_batch_dataset,
            training_settings)

        # Wrapped in tf.function when not eager_mode
        self._apply_gradients_wrapped(gradients)

        return loss, model_outputs

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
                               inference_mode=False,  # TODO: shouldn't this be True?
                               evaluate_settings=training_settings)

        # TODO: DRY up
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

        # TODO: DRY up
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
            # Applying the distribution strategy and @tf.function will be performed on a higher level
            return self._call_model(
                batch_data,
                evaluate_settings, inference_mode
            )
        else:
            return self._call_model_wrapped(
                batch_data,
                evaluate_settings,
                tf.constant(inference_mode, dtype=tf.bool)
            )

    def _call_model(self, batch_data, evaluate_settings, inference_mode):
        return self.training_model(batch_data, evaluate_settings, inference_mode)

    def _calc_gradients(self, batch_data, training_settings):

        with tf.GradientTape() as tape:
            results = self.evaluate_loss(
                batch_data,
                inference_mode=False,
                evaluate_settings=training_settings)

            if 'loss' not in results:
                raise LossNotAvailableException()

            # Divide by number of replicas (devices), this is > 1 when running with a distributed strategy
            results['loss'] /= float(self._num_replicas)

        if self._trainable_variables is None:
            # We now have evaluated the model and the trainable variables should be available
            self._retrieve_trainable_variables()

        loss = results['loss']
        model_outputs = [results]

        gradients = self._back_propagate_from(loss, tape)

        return loss, model_outputs, gradients

    def _calc_gradients_in_chunks(
            self,
            chunkable_batch_dataset: ChunkableBatchDatasetInterface,
            training_settings
    ):
        """

        See `train_on` method.

        Calculate batch gradients by accumulating gradients over batch chunks.
        `chunks_tf_dataset` contains batch chunks which are slices of the batch data to use during this training step.
        The slices (chunks) have size `self.batch_chunk_size`.

        :param chunkable_batch_dataset: Usually a ChunkableBatchDataset or DistributedChunkableBatchDataset
        :param training_settings:

        :return: loss, model_outputs
                model_outputs: BatchChunkingResults: a list of tuples, one tuple per batch chunk results:
                    [{'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>},  # Chunk 1
                     ...
                     {'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>}]  # Chunk N

        :rtype: Tuple[Tensor, BatchChunkingResults[Dict]]
        """

        # Could be distributed
        total_batch_size = chunkable_batch_dataset.total_batch_size
        num_chunks = len(chunkable_batch_dataset)

        accumulated_model_outputs = None
        accumulated_gradients = None
        for chunk_idx, batch_chunk in enumerate(chunkable_batch_dataset):
            # When distributed, process chunk results are per replica
            chunk_model_outputs, chunk_gradients = self._process_chunk_wrapped(
                batch_chunk,
                tf.constant(chunk_idx == (num_chunks-1), dtype=tf.bool),
                tf.constant(total_batch_size, dtype=tf.int64),
                training_settings
            )

            accumulated_model_outputs, accumulated_gradients = self._accumulate_chunk_process_results_wrapped(
                chunk_model_outputs,
                chunk_gradients,
                accumulated_model_outputs,
                accumulated_gradients
            )

        num_samples = self._reduce_chunk_num_samples_wrapped(accumulated_model_outputs)
        loss = self._reduce_chunk_loss_wrapped(accumulated_model_outputs, num_samples)

        # Tag as chunk processing results
        accumulated_model_outputs = BatchChunkingResults(accumulated_model_outputs)

        # When distributed, loss, and accumulated_gradients are per replica.
        # accumulated_model_outputs is per replica, per chunk
        return loss, accumulated_model_outputs, accumulated_gradients

    def _process_chunk(self, chunk, last_chunk: bool, total_batch_size: int, training_settings: Dict):
        with tf.GradientTape() as tape:
            # Chunk model outputs
            chunk_model_outputs = self.evaluate_loss(
                chunk,
                inference_mode=False,
                evaluate_settings=training_settings)

            if 'loss' not in chunk_model_outputs:
                raise LossNotAvailableException()

            if 'num_samples' not in chunk_model_outputs:
                raise NumSamplesNotAvailableException()

            # loss is assumed to be the average over the sample loss for the chunk
            # Divide through batch size to factor in that this loss is part of a larger batch.
            chunk_loss = chunk_model_outputs['loss']
            num_chunk_samples = chunk_model_outputs['num_samples']
            # Divide by number of replicas (devices), this is > 1 when running with a distributed strategy
            chunk_loss = float(num_chunk_samples) * chunk_loss / float(self._num_replicas*total_batch_size)

        if self._trainable_variables is None:
            # We now have evaluated the model and the trainable variables should be available
            self._retrieve_trainable_variables()

        chunk_gradients = self._back_propagate_from(chunk_loss, tape, last_chunk=last_chunk)

        return chunk_model_outputs, chunk_gradients

    def _accumulate_chunk_process_results(
            self,
            chunk_model_outputs,
            chunk_gradients,
            accumulated_model_outputs: Optional[List] = None,
            accumulated_gradients: Optional[Dict] = None
    ):
        # ### ACCUMULATE CHUNK MODEL OUTPUTS ###
        if accumulated_model_outputs is None:
            accumulated_model_outputs = []
        else:
            # Not allowed to change the input, hence using a copy
            accumulated_model_outputs = accumulated_model_outputs.copy()

        accumulated_model_outputs += [chunk_model_outputs]

        # ### ACCUMULATE GRADIENTS ###
        if accumulated_gradients is None:
            if self._trainable_variables is None or len(self._trainable_variables) == 0:
                raise MLPugException("Unexpected state :  trainable variables not found. Please file an issue.")

            accumulated_gradients = {}
            for optimizer_name, tvs in self._trainable_variables.items():
                # opt_chunk_grads = chunk_gradients[optimizer_name]
                # with tf.device(opt_chunk_grads[0].device):
                accumulated_gradients[optimizer_name] = [tf.zeros_like(tv) for tv in tvs]
        else:
            accumulated_gradients = accumulated_gradients.copy()

        for optimizer_name, opt_chunk_grads in chunk_gradients.items():
            opt_accu_grads = accumulated_gradients[optimizer_name]

            accumulated_gradients[optimizer_name] = [
                (accu_param_grad + chunk_param_grad)
                for accu_param_grad, chunk_param_grad in zip(opt_accu_grads, opt_chunk_grads)
            ]

        return accumulated_model_outputs, accumulated_gradients

    def _back_propagate_from(self, loss, tape, last_chunk=False):
        gradients = {}
        for optimizer_name in self.optimizers.keys():
            trainable_variables = get_value_at(optimizer_name, self._trainable_variables, warn_on_failure=False)

            gradients[optimizer_name] = tape.gradient(loss, trainable_variables)

        return gradients

    def _apply_gradients(self, gradients):
        self._update_model_parameters(self._prepare_update_model_parameters(gradients))

        self._after_update_model_parameters(gradients)

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
        manual_reduction_required = self._distribution_strategy is not None and self.batch_chunk_size is not None
        if manual_reduction_required:
            # When distributing training over multiple devices, and using gradient accumulation,
            # manually reduce the gradients to all devices first
            ctx = tf.distribute.get_replica_context()
            gradients = ctx.all_reduce(tf.distribute.ReduceOp.SUM, gradients)

        for optimizer_name, optimizer in self.get_optimizers().items():
            trainable_variables = get_value_at(optimizer_name, self._trainable_variables)
            if trainable_variables is None:
                raise MLPugException("Unexpected state :  trainable variables not found. Please file an issue.")

            optimizer.apply_gradients(zip(gradients[optimizer_name], trainable_variables),
                                      skip_aggregate_gradients=manual_reduction_required)

    def _after_update_model_parameters(self, gradients):
        pass
