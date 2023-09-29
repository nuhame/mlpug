import os

import contextlib

from functools import cached_property

from basics.logging_utils import log_exception
from basics.logging import get_logger

import mlpug.tensorflow as mlp

from mlpug.debugging import enable_pycharm_remote_debugging

from examples.chatbot.training_process import TrainingProcess as TrainingProcessBase
from examples.chatbot.tensorflow.collation import MultipleChoiceCollator, CollatedSampleGenerator

from examples.chatbot.shared_args import create_arg_parser, describe_args

mlp.logging.use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


try:
    import tensorflow as tf
    import keras
except Exception as e:
    log_exception(module_logger, "Please `pip install tensorflow`", e)


try:
    from transformers import TFGPT2DoubleHeadsModel, AdamWeightDecay
except Exception as e:
    log_exception(module_logger, "Please `pip install transformers`", e)


def gather_next_sentence_prediction_data(batch, auxiliary_results, **kwargs):
    # Batch is a tuple with the following items :
    # [0] input_ids_batch,
    # [1] token_type_ids_batch,
    # [2] token_labels_ids_batch,
    # [3] last_token_idx_batch,
    # [4] reply_class_batch

    targets = batch[4]

    nsp_logits = auxiliary_results["nsp_logits"]

    prediction_probability = tf.nn.softmax(nsp_logits, axis=1)
    predictions = tf.math.argmax(prediction_probability, axis=1)

    return targets, predictions


def clean_up_batch_data(model_output, **kwargs):
    loss = model_output["loss"]

    model_output["loss"] = loss.numpy()

    # We don't need the auxiliary_results anymore
    del model_output["auxiliary_results"]


# MLPug needs a TrainModel that outputs the loss
class TrainModel(tf.keras.Model):
    def __init__(self, model, lm_loss_weight):
        super(TrainModel, self).__init__()

        self.model = model
        self.lm_loss_weight = lm_loss_weight

        # TODO: tf.keras is behind keras, so using keras directly here
        #       We need it because we miss the ignore_class feature
        self.lm_loss_func = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            ignore_class=-100,
            reduction=tf.keras.losses.Reduction.SUM)

        self.mc_loss_func = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM)

    def call(self, batch_data, evaluate_settings, inference_mode=None):
        input_ids_batch, \
            token_type_ids_batch, \
            token_labels_ids_batch, \
            last_token_idx_batch, \
            reply_class_batch = batch_data

        results = self.model(
            input_ids=input_ids_batch,
            token_type_ids=token_type_ids_batch,
            mc_token_ids=last_token_idx_batch,
            training=not inference_mode)

        shifted_logits = results.logits[..., :-1, :]
        shifted_labels = token_labels_ids_batch[..., 1:]

        mc_logits = results.mc_logits

        num_samples = input_ids_batch.shape[0]

        lm_loss = self.lm_loss_func(shifted_labels, shifted_logits) / num_samples
        mc_loss = self.mc_loss_func(reply_class_batch, mc_logits) / num_samples

        loss = (lm_loss*self.lm_loss_weight + mc_loss)/(self.lm_loss_weight+1.0)

        return {
            "loss": loss,
            "num_samples": tf.convert_to_tensor(num_samples, dtype=tf.int64),
            "auxiliary_results": {
                # required to calculate next sentence prediction (classification) quality
                "nsp_logits": mc_logits
            }
        }


# See TrainingProcessBase for more information on the different methods implemented here
# Here we implement the methods that are specific to our problem and specific to our ML library, Tensorflow.
class TrainingProcess(TrainingProcessBase):

    MLPUG_MODULE = mlp

    def __init__(self, args, num_devices, name="TFTrainingProcess"):
        """

        :param args:
        :param num_devices:
        :param name:
        """
        # For Tensorflow we don't need a device rank, since the distribution works differently
        super().__init__(0, args, num_devices, name=name)

        self._distribution_strategy = None

        self._global_batch_size = self.num_devices * self._args.batch_size

    @cached_property
    def num_batches_training_set(self):
        return self._calc_num_batches(self._training_set)

    @cached_property
    def num_batches_validation_set(self):
        return self._calc_num_batches(self._validation_set)

    def _setup_compute(self):
        if self.is_distributed:
            devices = [f"GPU:{i}" for i in range(self.num_devices)]
            self._distribution_strategy = tf.distribute.MirroredStrategy(
                devices=devices,
                # TODO: required to make strategy.gather work for next_sentence_prediction_data
                #       Is this because NCCL version issues? Old GPUs/Machines?
                #       Should function properly with default cross_device_ops=None, leading to
                #       use of CollectiveAllReduce.
                #       Alternatively, this might also be because we execute some code eagerly
                #       between strategy.run code.
                cross_device_ops=tf.distribute.ReductionToOneDevice()
            )
        else:
            num_gpus_available = len(tf.config.list_physical_devices('GPU'))

            device = "/device:GPU:0" if num_gpus_available > 0 else "/CPU:0"
            self._log.debug(f"Training on device : {device}")

            self._distribution_strategy = tf.distribute.OneDeviceStrategy(device=device)
            # TODO: Debugging without single device strategy
            # self._distribution_strategy = None

    def _create_tf_dataset(self, multiple_choice_samples):
        choice_collator = MultipleChoiceCollator(
            pad_token_idx=self._hf_tokenizer.pad_token_id,
            max_sequence_length=self._opt_max_sequence_length)

        sample_generator = CollatedSampleGenerator(multiple_choice_samples, choice_collator)

        batch_dataset = tf.data.Dataset\
            .from_generator(
                sample_generator,
                # Note: reply_class must be tf.int32, because tf.int8 is not allowed when
                #       using distributed_strategy.gather
                output_types=(tf.int64, tf.int64, tf.int64, tf.int64, tf.int32)
            )\
            .shuffle(20*self._global_batch_size, reshuffle_each_iteration=True) \
            .batch(self._global_batch_size)

        return batch_dataset

    def _setup_batch_datasets(self):
        self._batch_training_set = self._create_tf_dataset(self._training_set)
        self._batch_validation_set = self._create_tf_dataset(self._validation_set)

        if self.is_distributed:
            strategy = self._distribution_strategy

            # Distribute training and validation set
            self._batch_training_set = strategy.experimental_distribute_dataset(self._batch_training_set)
            self._batch_validation_set = strategy.experimental_distribute_dataset(self._batch_validation_set)

    def _calc_num_batches(self, multiple_choice_samples):
        num_samples = len(multiple_choice_samples)
        num_batches = int(num_samples/self._global_batch_size)
        if num_samples % self._global_batch_size != 0:
            num_batches += 1

        return num_batches

    def _build_model(self):
        model_config = self._gather_model_config()

        self._log.info(f"Loading pre-trained GPT-2 model : {self._args.pretrained_model}")

        strategy = self._distribution_strategy.scope() if self._distribution_strategy is not None \
            else contextlib.suppress()

        # Building pre-trained GPT-2 model
        with strategy:
            self._model = TFGPT2DoubleHeadsModel.from_pretrained(self._args.pretrained_model, config=model_config)
            self._model.resize_token_embeddings(new_num_tokens=self._orig_num_tokens + self._num_special_tokens)

            # TODO: Is activation --activation-checkpointing available for Huggingface TF models?
            #       No it isn't: https://github.com/huggingface/transformers/issues/19095
            if self._args.activation_checkpointing:
                try:
                    self._model.gradient_checkpointing_enable()
                except Exception as e:
                    raise Exception("Activation checkpointing not available for TF GPT2 model") from e

        self._log.info(f"Configuration of loaded model : \n{model_config}")

    def _setup_training_model(self):
        strategy = self._distribution_strategy.scope() if self._distribution_strategy is not None \
            else contextlib.suppress()

        with strategy:
            self._training_model = TrainModel(
                self._model,
                self._args.lm_loss_weight)

    def _build_optimizer(self):
        strategy = self._distribution_strategy.scope() if self._distribution_strategy is not None \
            else contextlib.suppress()

        with strategy:
            # TODO: AdamWeightDecay fails at:
            # ERROR   : TrainingManager::log_exception :
            #   Exception type : <class 'tensorflow.python.framework.errors_impl.UnimplementedError'>
            # ERROR   : TrainingManager::log_exception :
            #   {{function_node __wrapped__Cast_device_/job:localhost/replica:0/task:0/device:CPU:0}}
            #       Cast string to float is not supported [Op:Cast]
            #   File "site-packages/keras/optimizers/optimizer_experimental/optimizer.py", line 1151, in weight_decay_fn
            #     wd = tf.cast(self.weight_decay, variable.dtype)

            # self._optimizer = AdamWeightDecay(
            #     learning_rate=self._args.learning_rate,
            #     weight_decay_rate=self._args.weight_decay,
            #     beta_1=0.9,
            #     beta_2=0.999,
            #     epsilon=1e-08,
            #     exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

            self._optimizer = tf.keras.optimizers.Adam(
                learning_rate=self._args.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08)

    def _get_custom_trainer_config(self):
        """
        Implementation depends on specific ML Library you are using

        :return:
        """

        # Tensorflow only needs batch_data_signature as additional argument, compared to PyTorch
        return {
            "eager_mode": False,
            "distribution_strategy": self._distribution_strategy,
            "batch_data_signature": self._batch_training_set.element_spec,
            "monitor_tracing": True
        }

    def _create_gather_classification_data_function(self):
        return gather_next_sentence_prediction_data

    def _create_gather_distributed_classification_data_function(self):
        # Use default implementation made available by MLPug
        return mlp.evaluation.GatherDistributedTensorTuple(distribution_strategy=self._distribution_strategy)

    def _create_clean_up_batch_data_func(self):
        return clean_up_batch_data

    def _get_custom_evaluator_config(self):
        """
        Implementation depends on specific ML Library you are using

        :return:
        """

        # Tensorflow only needs distribution_strategy as additional argument, compared to PyTorch
        return {
            "distribution_strategy": self._distribution_strategy,
            "monitor_tracing": True
        }

    def _add_lr_scheduler_callback(self):
        """
        Returns function that can cleanup the batch data to optimize memory use

        Implementation depends on specific ML Library you are using

        :return:
        """
        def update_lr(iter, lr):
            return lr * self._lr_scheduling_func(iter)

        self._callbacks += [mlp.callbacks.LRSchedulerWrapper({
                'warmup-scheduler': tf.keras.callbacks.LearningRateScheduler(update_lr)
            }, batch_level=True)]


if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True)

    # ############# SETUP LOGGING #############
    mlp.logging.use_fancy_colors()
    logger = get_logger(os.path.basename(__file__))
    # ########################################

    # ############## PARSE ARGS ##############
    parser = create_arg_parser()

    parser.parse_args()

    args = parser.parse_args()

    describe_args(args, logger)

    if args.remote_debug_ip:
        enable_pycharm_remote_debugging(args.remote_debug_ip)

    # ############## TRAIN MODEL ##############
    if args.distributed:
        num_gpus_available = len(tf.config.list_physical_devices('GPU'))

        if num_gpus_available < 1:
            logger.error(f"No GPUs available for data distributed training over multiple GPUs")
            exit(-1)

        num_devices = args.num_devices if args.num_devices > 0 else num_gpus_available
        if num_devices > num_gpus_available:
            logger.warn(f"Number of requested GPUs is lower than available GPUs, "
                        f"limiting training to {num_gpus_available} GPUS")
            num_devices = num_gpus_available

        logger.info(f"Distributing batch training over {num_devices} GPUs.")

        global_batch_size = args.batch_size * num_devices
    else:
        num_devices = 1
        logger.info(f"Single device mode.")
        global_batch_size = args.batch_size

    logger.info(f"Global batch size: {args.batch_size}")

    training_process = TrainingProcess(args, num_devices=num_devices)
    training_process.setup()
    training_process.start()
