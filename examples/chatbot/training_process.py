import abc
import os
import random
import numpy as np

import mlpug.abstract_interface as mlp_interface

from mlpug.base import Base
from mlpug.evaluation import ConcatBatchTuplesWithNumpyArrays

from basics.logging_utils import log_exception
from basics.logging import get_logger

from examples.chatbot.tokenizers import HFTokenizer
from examples.chatbot.multiple_choice_dataset import MultipleConversationChoicesDataset
from examples.chatbot.conversation_sample_factory import ConversationSampleFactory

module_logger = get_logger(os.path.basename(__file__))

try:
    from datasets import load_dataset
except Exception as e:
    log_exception(module_logger, "Please `pip install datasets`", e)

try:
    from transformers import GPT2Tokenizer, GPT2Config
except Exception as e:
    log_exception(module_logger, "Please `pip install transformers`", e)

try:
    from sklearn.metrics import precision_recall_fscore_support
except Exception as e:
    log_exception(module_logger, "Please install the scikit-learn package, "
                                 "see https://scikit-learn.org/stable/install.html#installing-the-latest-release", e)


def calc_classification_quality(batch_labels_predictions):
    concat = ConcatBatchTuplesWithNumpyArrays()

    labels, predictions = concat(batch_labels_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


class TrainingProcess(Base, metaclass=abc.ABCMeta):

    # Set this to your MLPUG implementation of choice, e.g. mlpug.pytorch, mlpug.pytorch.xla or mlpug.tensorflow:
    #
    # import mlpug.pytorch as mlp
    # TrainingProcess.MLPUG_MODULE = mlp
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

        self._hf_tokenizer = None
        self._orig_num_tokens = None
        self._num_special_tokens = None

        self._sample_training_set = None
        self._sample_validation_set = None

        self._batch_training_set = None
        self._batch_validation_set = None

        self._model = None
        self._training_model = None
        self._optimizer = None

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
        return self._num_devices > 1

    @property
    def is_distributed(self):
        return self.num_devices > 1

    def setup(self):
        self._set_random_seed()

        self._setup_compute()

        self._load_and_prepare_data()
        self._setup_batch_datasets()

        self._build_model()
        self._setup_training_model()
        self._build_optimizer()

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

    def _load_dataset(self):
        return load_dataset("bavard/personachat_truecased")

    def _load_and_prepare_data(self):
        self._log.info("Loading Personachat dataset ...")
        dataset = self._load_dataset()

        self._log.info(f"Loading Tokenizer for {self._args.pretrained_model} model ...")
        self._hf_tokenizer = GPT2Tokenizer.from_pretrained(self._args.pretrained_model)

        self._orig_num_tokens = len(self._hf_tokenizer.encoder)
        self._num_special_tokens = self._hf_tokenizer.add_special_tokens(self.SPECIAL_TOKENS_MAPPING)
        self._log.debug(f"Number of special tokens added to tokenizer : {self._num_special_tokens }")

        tokenizer_func = HFTokenizer(self._hf_tokenizer)

        sample_factory = ConversationSampleFactory(
            tokenizer_func,
            bos=self.SPECIAL_TOKENS_MAPPING['bos_token'],
            eos=self.SPECIAL_TOKENS_MAPPING['eos_token'],
            speaker1=self.SPECIAL_TOKENS_MAPPING['additional_special_tokens'][0],
            speaker2=self.SPECIAL_TOKENS_MAPPING['additional_special_tokens'][1])

        self._sample_training_set = MultipleConversationChoicesDataset(dataset["train"],
                                                                       sample_factory,
                                                                       name="TrainingSampleGenerator")
        self._sample_training_set.initialize()

        self._sample_validation_set = MultipleConversationChoicesDataset(dataset["validation"],
                                                                         sample_factory,
                                                                         name="ValidationSampleGenerator")
        self._sample_validation_set.initialize()

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

    def _setup_trainer(self):
        mlp = self.MLPUG_MODULE

        self._trainer = mlp.trainers.DefaultTrainer(optimizers=self._optimizer, model_components=self._model)

    @abc.abstractmethod
    def _create_gather_loss_function(self):
        """
        Returns function that can gather and reduce loss from one or more devices (when performing distributed training)
        Implementation depends on specific ML Library you are using

        :return:
        """
        raise NotImplementedError("Please implement in your child class")

    @abc.abstractmethod
    def _create_gather_classification_data_function(self):
        """
        Returns function that can gather and reduce classification data (prediction and target data)
        from one or more devices (when performing distributed training)

        Implementation depends on specific ML Library you are using

        :return:
        """
        raise NotImplementedError("Please implement in your child class")

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

        loss_gather_func = self._create_gather_loss_function()

        # Every batch we will calculate the training and validation loss
        # In distributed training mode, this will be done by gathering the loss calculated on all devices
        loss_only_evaluator = mlp.evaluation.MetricEvaluator(
            batch_metric_funcs={
                'loss': loss_gather_func
            },
            # The trainer knows how to evaluate the model
            trainer=self._trainer,
            batch_chunk_size=self._args.batch_chunk_size,
            name="LossOnlyEvaluator")

        # Every epoch we will also calculated the candidate output classification quality
        all_metrics_evaluator = mlp.evaluation.MetricEvaluator(
            batch_metric_funcs={
                'loss': loss_gather_func,
                # gather classification targets and predictions (next sentence prediction)
                'classification': self._create_gather_classification_data_function()
            },
            # The trainer knows how to evaluate the model
            trainer=self._trainer,
            batch_metric_reducer_funcs={
                # Use default reducer for loss (not shown here)
                # Use the gathered classification targets and predictions to calculate Precision, Recall and F1
                'classification': calc_classification_quality
            },
            # When batch_chunk_size is given, perform gradient accumulation in chunks with batch_chunk_size samples
            batch_chunk_size=self._args.batch_chunk_size,
            # Concatenate the classification targets and predictions, gathered per batch chunk, such that
            # the concatenated output can be used by the registered batch_metric_reducer_funcs['classification'] to
            # calculate the classification quality metrics.
            batch_chunk_metric_reducer_funcs={
                'classification': ConcatBatchTuplesWithNumpyArrays(),
            },
            show_dataset_evaluation_progress=True,
            name="AllMetricsEvaluator")

        def log_training_metrics(logs, dataset_batch):
            global_iter = logs['current']['global_iter']

            return global_iter % self._args.progress_log_period == 0

        def log_validation_metrics(logs, dataset_batch):
            global_iter = logs['current']['global_iter']

            return global_iter % (10*self._args.progress_log_period) == 0

        # Length of window
        avg_window_training = int(0.5 * len(self._batch_training_set) / self._args.progress_log_period)
        avg_window_validation = int(0.5 * len(self._batch_validation_set) / (10 * self._args.progress_log_period))
        self._callbacks = [
            # Get training loss calculated, during forward pass, and gather+reduce it from all devices
            # The model loss and other auxiliary results are already calculated by the Trainer, so we do not need
            # to provide the the training set here.
            mlp.callbacks.TrainingMetricsLogger(metric_evaluator=loss_only_evaluator,
                                                log_condition_func=log_training_metrics,
                                                batch_averaging_window=avg_window_training),
            # Calculate validation loss and classification quality, every 10x<progress_log_period> batches
            mlp.callbacks.TestMetricsLogger(self._batch_validation_set,
                                            'validation',
                                            metric_evaluator=all_metrics_evaluator,
                                            log_condition_func=log_validation_metrics,
                                            batch_averaging_window=avg_window_validation),
            # Calculate training metrics only once per epoch over the whole dataset
            mlp.callbacks.TestMetricsLogger(self._batch_training_set,
                                            'training',
                                            metric_evaluator=all_metrics_evaluator,
                                            # epoch level only
                                            batch_level=False),
            # Calculate validation metrics only once per epoch over the whole dataset
            mlp.callbacks.TestMetricsLogger(self._batch_validation_set,
                                            'validation',
                                            metric_evaluator=all_metrics_evaluator,
                                            # epoch level only
                                            batch_level=False),
        ]

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
                mlp.callbacks.LogProgress(
                    log_period=self._args.progress_log_period,
                    set_names=['training', 'validation']),
                # Track metrics for all datasets of interest
                mlp.callbacks.AutoTensorboard(dataset_name='training', **tb_args),
                mlp.callbacks.AutoTensorboard(dataset_name='validation', **tb_args),
                mlp.callbacks.AutoTensorboard(dataset_name='training_params', **tb_args)
            ]

    def _setup_training_manager(self):
        """
        Sets self._training_manager.

        :return:
        """

        mlp = self.MLPUG_MODULE

        self._training_manager = mlp.trainers.TrainingManager(self._trainer,
                                                              self._batch_training_set,
                                                              num_epochs=self._args.num_epochs,
                                                              callbacks=self._callbacks,
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
