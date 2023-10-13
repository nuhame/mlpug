import abc
import os

# See https://stackoverflow.com/a/57549064/889617
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import random

from functools import cached_property

import math
import numpy as np

from basics.logging_utils import log_exception
from basics.logging import get_logger

import mlpug.abstract_interface as mlp_interface
from mlpug.base import Base

from examples.chatbot.datasets.manager import DatasetManager
from examples.chatbot.datasets.multiple_choice import max_sequence_length_in

from examples.chatbot.datasets.tokenizers import HFTokenizer
from examples.chatbot.datasets.conversations import ConversationSampleFactory


module_logger = get_logger(os.path.basename(__file__))


try:
    from transformers import GPT2Tokenizer, GPT2Config
except Exception as e:
    log_exception(module_logger, "Please `pip install transformers`", e)

try:
    from sklearn.metrics import precision_recall_fscore_support
except Exception as e:
    log_exception(module_logger, "Please install the scikit-learn package, "
                                 "see https://scikit-learn.org/stable/install.html#installing-the-latest-release", e)


def find_optimal_max_sequence_length(multiple_choice_dataset, outlier_threshold=0.05):
    """
    Calculates multiple choice sample sequence lengths and discards the samples with the longest sequence lengths,
    based on outlier_threshold, to find a max sequence length that still covers most samples but is not too long
    (and hence uses too much memory)

    :param multiple_choice_dataset:
    :param outlier_threshold:

    :return: opt_max_sequence_length, max_sequence_length
    """
    sample_sequence_lengths = np.array([max_sequence_length_in(conversation_choices)
                                        for conversation_choices in multiple_choice_dataset])

    sample_sequence_lengths = np.sort(sample_sequence_lengths)

    num_samples = len(sample_sequence_lengths)
    cutoff_idx = int(num_samples * (1-outlier_threshold))

    return sample_sequence_lengths[cutoff_idx], sample_sequence_lengths[-1]


def filter_out_too_long_sequences(conversation_choices_list, max_sequence_length):
    return [conversation_choices for conversation_choices in conversation_choices_list
            if max_sequence_length_in(conversation_choices) <= max_sequence_length]


def calc_classification_quality(classification_data):
    if classification_data is None:
        return classification_data

    labels, predictions = classification_data

    num_samples = len(labels)
    accuracy = np.sum(labels == predictions)/num_samples

    return {
        "accuracy": accuracy,
        # Added for demonstration purposes
        "num_samples": num_samples
    }


class TrainingProcess(Base, metaclass=abc.ABCMeta):

    # Set this to your MLPUG implementation of choice, e.g. mlpug.pytorch, mlpug.pytorch.xla or mlpug.tensorflow:
    #
    # import mlpug.pytorch as mlp
    #
    # class TrainingProcess(TrainingProcessBase):
    #     MLPUG_MODULE = mlp
    #
    # ... instantiate and use the TrainingProcess ...
    #
    # WARNING: Below MLPUG_MODULE has been set to default value 'mlp_interface'.
    #          This will allow your IDE to find code references to the BASE classes of MLP,
    #          not of your chosen mlpug implementation!
    MLPUG_MODULE = mlp_interface

    SPECIAL_TOKENS_MAPPING = {
        'bos_token': '<bos>',
        'eos_token': '<eos>',
        'pad_token': '<pad>',
        'additional_special_tokens': ['<speaker1>', '<speaker2>']
    }

    def __init__(self, rank, args, num_devices, name="TrainingProcess"):
        logger_name, disable_logging = self.get_logger_info(rank, num_devices, name)
        super().__init__(disable_logging=disable_logging, pybase_logger_name=logger_name)

        self._args = args
        self._rank = rank
        self._num_devices = num_devices

        self._training_set = None
        self._validation_set = None

        self._opt_max_sequence_length = None

        self._batch_training_set = None
        self._batch_validation_set = None

        self._model = None
        self._training_model = None
        self._optimizer = None

        self._lr_scheduling_func = None

        self._trainer = None
        self._callbacks = None
        self._training_manager = None

    @property
    def rank(self):
        return self._rank

    @property
    def is_primary(self):
        return self.rank == 0

    @property
    def num_devices(self):
        return self._num_devices

    @property
    def is_distributed(self):
        return self.num_devices > 1

    @cached_property
    def num_batches_training_set(self):
        return len(self._batch_training_set)

    @cached_property
    def num_batches_validation_set(self):
        return len(self._batch_validation_set)

    def setup(self):
        self._set_random_seed()

        self._setup_compute()

        self._setup_tokenizer()
        self._prepare_datasets()
        self._setup_batch_datasets()

        self._build_model()
        self._setup_training_model()
        self._build_optimizer()
        self._setup_lr_scheduling_func()

        self._setup_trainer()
        self._setup_callbacks()
        self._setup_training_manager()
        self._prepare_training()

    def start(self):
        self._training_manager.start_training()

    def _set_random_seed(self):
        # For reproducibility
        random.seed(self._args.seed)
        np.random.seed(self._args.seed)

    @abc.abstractmethod
    def _setup_compute(self):
        raise NotImplementedError("Please implement in your child class")

    def _prepare_datasets(self):
        tokenizer_func = HFTokenizer(self._hf_tokenizer)

        sample_factory = ConversationSampleFactory(
            tokenizer_func,
            bos=self.SPECIAL_TOKENS_MAPPING['bos_token'],
            eos=self.SPECIAL_TOKENS_MAPPING['eos_token'],
            speaker1=self.SPECIAL_TOKENS_MAPPING['additional_special_tokens'][0],
            speaker2=self.SPECIAL_TOKENS_MAPPING['additional_special_tokens'][1])

        dataset_manager = DatasetManager(
            sample_factory,
            disable_logging=self.logging_disabled)

        self._training_set = self._generate_dataset(dataset_manager, "train")
        self._validation_set = self._generate_dataset(dataset_manager, "validation")

        outlier_threshold = self._args.sequence_length_outlier_threshold
        self._opt_max_sequence_length, max_sequence_length = find_optimal_max_sequence_length(
            self._training_set,
            outlier_threshold=outlier_threshold)

        self._log.info(f"Max. sequence length reduced from "
                       f"{max_sequence_length} tokens to "
                       f"{self._opt_max_sequence_length} tokens "
                       f"by discarding {outlier_threshold*100}% of the samples.")

        self._training_set = self._filter_conversation_samples(
            self._training_set,
            self._opt_max_sequence_length,
            "train")

        self._validation_set = self._filter_conversation_samples(
            self._validation_set,
            self._opt_max_sequence_length,
            "validation")

    def _setup_tokenizer(self):
        self._log.info(f"Loading Tokenizer for {self._args.pretrained_model} model ...")

        self._hf_tokenizer = GPT2Tokenizer.from_pretrained(self._args.pretrained_model)
        self._orig_num_tokens = len(self._hf_tokenizer.encoder)

        self._num_special_tokens = self._hf_tokenizer.add_special_tokens(self.SPECIAL_TOKENS_MAPPING)
        self._log.debug(f"Number of special tokens added to tokenizer : {self._num_special_tokens}")

    def _generate_dataset(self, manager, dataset_name):
        return manager.get_dataset_for(
            dataset_name,
            max_num_samples=self._args.max_conversations,
            num_choices_per_sample=self._args.num_choices,
            force_generate=self._args.force_generate_samples)

    def _filter_conversation_samples(self, dataset, max_sequence_length, dataset_name):
        mxsl = max_sequence_length

        self._log.info(f"[{dataset_name} set] Filtering out conversation sequences that are longer than {mxsl} tokens")
        self._log.info(f"[{dataset_name} set] [BEFORE] Number of multiple choice conversations samples : "
                       f"{len(dataset)}")

        dataset = filter_out_too_long_sequences(dataset, mxsl)

        self._log.info(f"[{dataset_name} set] [AFTER ] Number of multiple choice conversations samples : "
                       f"{len(dataset)}")

        return dataset

    @abc.abstractmethod
    def _setup_batch_datasets(self):
        """
        Sets self._batch_training_set and self._batch_validation_set.
        Implementation depends on specific ML Library you are using
        :return:
        """
        raise NotImplementedError("Please implement in your child class")

    def _gather_model_config(self):
        resid_pdrop = embd_pdrop = attn_pdrop = self._args.dropout_rate

        return GPT2Config.from_pretrained(self._args.pretrained_model, **{
            "resid_pdrop": resid_pdrop,
            "embd_pdrop": embd_pdrop,
            "attn_pdrop": attn_pdrop
        })

    @abc.abstractmethod
    def _build_model(self):
        """
        Sets self._model.
        Implementation depends on specific ML Library you are using.
        Use _gather_model_config to get the Huggingface model configuration to build

        :return:
        """
        raise NotImplementedError("Please implement in your child class")

    def _setup_lr_scheduling_func(self):
        mlp = self.MLPUG_MODULE

        if not self._args.lr_warmup_schedule:
            return

        num_iters_per_epoch = len(self._batch_training_set)
        num_warmup_iters = self._args.lr_warmup_epochs*num_iters_per_epoch
        total_iters = self._args.num_epochs*num_iters_per_epoch

        self._log.info(f"Applying LR warmup schedule: \n"
                       f"num_warmup_iters = {num_warmup_iters}\n"
                       f"total_iters      = {total_iters}")

        self._lr_scheduling_func = mlp.scheduler_funcs.LRWarmupSchedule(num_warmup_iters, total_iters)

    @abc.abstractmethod
    def _setup_training_model(self):
        """
        Sets self._training_model.
        Implementation depends on specific ML Library you are using
        :return:
        """
        raise NotImplementedError("Please implement in your child class")

    @abc.abstractmethod
    def _build_optimizer(self):
        """
        Sets self._optimizer.
        Implementation depends on specific ML Library you are using
        :return:
        """
        raise NotImplementedError("Please implement in your child class")

    def _get_custom_trainer_config(self):
        """
        Implementation depends on specific ML Library you are using

        :return:
        """
        return {}

    def _setup_trainer(self):
        mlp = self.MLPUG_MODULE

        custom_trainer_config = self._get_custom_trainer_config()

        self._trainer = mlp.trainers.DefaultTrainer(
            optimizers=self._optimizer,
            model_components=self._model,
            # In case of gradient accumulation batch_chunk_size > 0 and a wrapper function is given
            # to make the batches sliceable such that we can chunk them into smaller pieces.
            batch_chunk_size=self._args.batch_chunk_size,
            chunkable_batch_wrapper=mlp.batch_chunking.ChunkableTupleBatchDim0.wrapper,
            **custom_trainer_config)

    @abc.abstractmethod
    def _create_gather_classification_data_function(self):
        """
        Returns function that can gather classification data (prediction and target data)

        Implementation depends on specific ML Library you are using

        :return:
        """
        raise NotImplementedError("Please implement in your child class")

    @abc.abstractmethod
    def _create_gather_distributed_classification_data_function(self):
        """
        Returns function that can gather classification data, gathered per device, over multiple devices
        in a distributed training setting

        Implementation depends on specific ML Library you are using

        :return:
        """
        raise NotImplementedError("Please implement in your child class")

    @abc.abstractmethod
    def _create_clean_up_batch_data_func(self):
        """
        Returns function that can cleanup the batch data to optimize memory use

        Implementation depends on specific ML Library you are using

        :return:
        """
        raise NotImplementedError("Please implement in your child class")

    def _get_custom_evaluator_config(self):
        """
        Implementation depends on specific ML Library you are using

        :return:
        """
        return {}

    def _setup_callbacks(self):
        """
        Sets self._callbacks.

        :return:
        """
        mlp = self.MLPUG_MODULE

        # For demo purposes, during training we want to do the following:
        #
        # a. Every <progress_log_period> batches, log the batch training loss, calculated during the
        #    forward pass for training
        #
        # b. Every 10x<progress_log_period> batches, calculate and log, for one batch of the validation set:
        #    1. the loss
        #    2. the output candidate classification quality
        #
        # c. Every epoch, calculate and log above metrics over the complete training and validation set
        #
        # d. Every epoch, check if the model improved, if so, save a best model checkpoint.
        #    Also save a "latest model" checkpoint and a training checkpoint
        #
        # e. Automatically log all calculated metrics as Tensorboard logs
        #
        # f. Log the progress every <progress_log_period>, in terms of training and validation metrics, to the terminal
        #

        # parameters required to rebuild the model from a checkpoint
        model_hyper_parameters = {
            "pretrained_model": self._args.pretrained_model
        }

        # In general, the MetricEvaluator defines the following steps to calculate metrics:
        #
        # * Gather inputs from the batch data to calculate metrics of interest
        #   (using "gather_metric_inputs_funcs" dict, with a function per metric)
        #
        # * Gather metric inputs, gathered on different devices, when relevant
        #   (using "gather_distributed_inputs_funcs" dict, with a function per metric)
        #
        #   These metric inputs are used to calculate the batch metrics of interest.
        #   In our case the loss and the output candidate classification quality.
        #   (using "metric_funcs" dict, with a function per metric)
        #
        # * Clean-up batch data that is not used any more after gathering the metric inputs.
        #   This is useful to optimize memory usage.
        #
        # * Combine the gathered metric inputs per batch.
        #   This is relevant When calculating metrics over:
        #   * a complete dataset, where the dataset was broken-up in batches
        #   * a window of batches
        #   * the batch chunks of a batch when gradient accumulation was applied.
        #
        #   (using "combine_metric_inputs_funcs" dict, with a function per metric)
        #
        # * Use the combined metric inputs to calculate the metrics over a dataset and/or
        #   a sliding window of batches;
        #   (using "metric_funcs" dict, with a function per metric)
        #
        # Although you can define functions for all steps, in many cases suitable defaults are available.
        #

        custom_evaluator_config = self._get_custom_evaluator_config()

        # Every batch we will calculate the training and validation loss
        # We are using all the default loss gathering functions here.
        loss_only_evaluator = mlp.evaluation.MetricEvaluator(
            # The trainer knows how to evaluate the model
            # We also get batch_chunk_size and chunkable_batch_wrapper from the trainer, to evaluate the
            # metrics in smaller chunks, if these values were set for the trainer
            trainer=self._trainer,
            # The implementation depends on the Deep Learning library backend
            clean_up_batch_data_func=self._create_clean_up_batch_data_func(),
            **custom_evaluator_config,
            name="LossOnlyEvaluator")

        # Every epoch we will also calculate the loss and candidate output classification quality over
        # the whole dataset. Since classification quality is a custom metric, here we need to specify how
        # MLPug should gather inputs for it and how to calculate it.
        all_metrics_evaluator = mlp.evaluation.MetricEvaluator(
            gather_metric_inputs_funcs={
                # We will use the default function for loss, so we don't need to add it here
                #
                # Gather classification targets and predictions (next sentence prediction)
                # The implementation depends on the Deep Learning library backend
                'classification': self._create_gather_classification_data_function()
            },
            gather_distributed_inputs_funcs={
                # We will use the default function for loss, so we don't need to add it here
                #
                # Gather the classification data, gathered on all devices used, when using distributed training
                # The implementation depends on the Deep Learning library backend
                'classification': self._create_gather_distributed_classification_data_function()
            },
            #
            # The implementation depends on the Deep Learning library backend
            clean_up_batch_data_func=self._create_clean_up_batch_data_func(),
            #
            # We will use the defaults for combine_metric_inputs_funcs
            #
            metric_funcs={
                # We will use the default function for loss, so we don't need to add it here
                #
                # Compute the classification quality
                'classification': calc_classification_quality
            },
            #
            # The trainer knows how to evaluate the model
            # We also get batch_chunk_size and chunkable_batch_wrapper from the trainer, to evaluate the
            # metrics in smaller chunks, if these values were set for the trainer
            trainer=self._trainer,
            show_progress=True,
            **custom_evaluator_config,
            name="AllMetricsEvaluator")

        def log_metrics(logs, dataset_batch):
            # It is also possible to use logs['current']['global_iter']
            batch_step = logs['current']['batch_step']

            return batch_step % self._args.progress_log_period == 0

        # Since the validation metrics will only calculated every self._args.progress_log_period batches
        avg_window_train = math.ceil(0.5 * self.num_batches_training_set / self._args.progress_log_period)
        avg_window_validation = math.ceil(0.5 * self.num_batches_validation_set / self._args.progress_log_period)
        self._callbacks = [
            # Get training loss calculated, during forward pass, and gather+reduce it from all devices
            # The model loss and other auxiliary results are already calculated by the Trainer, so we do not need
            # to provide the the training set here.
            mlp.callbacks.TrainingMetricsLogger(metric_evaluator=loss_only_evaluator,
                                                log_condition_func=log_metrics,
                                                sliding_window_length=avg_window_train,
                                                inspect_sliding_windows=self._args.inspect_sliding_windows),
            # Calculate validation loss and classification quality, every <progress_log_period> batches
            mlp.callbacks.DatasetMetricsLogger(self._batch_validation_set,
                                               'validation',
                                               metric_evaluator=all_metrics_evaluator,
                                               log_condition_func=log_metrics,
                                               sliding_window_length=avg_window_validation,
                                               inspect_sliding_windows=self._args.inspect_sliding_windows),
            # Calculate training metrics only once per epoch over the whole dataset
            mlp.callbacks.DatasetMetricsLogger(self._batch_training_set,
                                               'training',
                                               metric_evaluator=all_metrics_evaluator,
                                               # epoch level only
                                               batch_level=False),
            # Calculate validation metrics only once per epoch over the whole dataset
            mlp.callbacks.DatasetMetricsLogger(self._batch_validation_set,
                                               'validation',
                                               metric_evaluator=all_metrics_evaluator,
                                               # epoch level only
                                               batch_level=False),
        ]

        if self._lr_scheduling_func is not None:
            self._log.debug("Adding LR scheduler callback ... ")
            self._add_lr_scheduler_callback()

        # Only primary worker needs to log progress
        if self._rank == 0:
            tb_args = {'experiment_name': self._args.experiment_name}

            self._callbacks += [
                mlp.callbacks.CheckpointManager(
                    base_checkpoint_filename=self._args.experiment_name,
                    # monitor per epoch
                    batch_level=False,
                    metric_to_monitor="validation.dataset.loss",
                    # check if there is a better model every epoch
                    metric_monitor_period=1,
                    # every epoch, this will create a latest model and training checkpoint,
                    create_checkpoint_every=1,
                    # no archiving
                    archive_last_model_checkpoint_every=0,
                    # No backups
                    backup_before_override=False,
                    # model_hyper_parameters stored with each checkpoint
                    model_hyper_parameters=model_hyper_parameters),
                mlp.callbacks.LogProgress(log_condition_func=log_metrics,
                                          set_names=['training', 'validation', 'training_params']),
                # Track metrics for all datasets of interest
                mlp.callbacks.AutoTensorboard(dataset_name='training', **tb_args),
                mlp.callbacks.AutoTensorboard(dataset_name='validation', **tb_args),
                mlp.callbacks.AutoTensorboard(dataset_name='training_params', **tb_args),
            ]

        if self._args.describe_logs_object:
            self._callbacks += [mlp.callbacks.DescribeLogsObject(log_condition_func=log_metrics)]

    @abc.abstractmethod
    def _add_lr_scheduler_callback(self):
        """
        Adds an LR Scheduler callback to self._callbacks.
        Only called when an LR scheduler func is defined.

        Implementation depends on specific ML Library you are using

        :return:
        """

    def _setup_training_manager(self):
        """
        Sets self._training_manager.

        :return:
        """

        mlp = self.MLPUG_MODULE

        # TODO: revisit the behavior of MLPug when the total number of batches is not known before hand.
        self._training_manager = mlp.trainers.TrainingManager(self._trainer,
                                                              self._batch_training_set,
                                                              num_epochs=self._args.num_epochs,
                                                              callbacks=self._callbacks,
                                                              num_batches_per_epoch=self.num_batches_training_set,
                                                              experiment_data={
                                                                  "args": self._args
                                                              })

    def _prepare_training(self):
        self._trainer.set_training_model(self._training_model)

    @staticmethod
    def get_logger_info(rank, num_devices, name):
        """

        :param rank:
        :param num_devices:
        :param name:

        :return: (Logger name: String, disable_logging: Boolean)
        """
        return name, False
