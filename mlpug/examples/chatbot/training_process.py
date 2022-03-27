import abc
import os
import random
import numpy as np

from mlpug.base import Base

from basics.logging_utils import log_exception
from basics.logging import get_logger

module_logger = get_logger(os.path.basename(__file__))

try:
    from datasets import load_dataset
except Exception as e:
    log_exception(module_logger, "Please `pip install datasets`", e)


def calc_classification_quality():
    pass


class TrainingProcess(Base, metaclass=abc.ABCMeta):

    # Every ML library specialization uses a different mlpug specialization
    MLPUG_MODULE = None

    @staticmethod
    def get_logger_info(rank, world_size, name):
        """

        :param rank:
        :param world_size:
        :param name:

        :return: (Logger name: String, disable_logging: Boolean)
        """
        return name, False

    def __init__(self, rank, args, world_size, name="TrainingProcess"):
        logger_name, disable_logging = self.get_logger_info(rank, world_size, name)
        super().__init__(disable_logging=disable_logging, pybase_logger_name=logger_name)

        self._rank = rank
        self._args = args
        self._world_size = world_size

        self._raw_training_set = None
        self._raw_validation_set = None

        self._batch_training_set = None
        self._batch_validation_set = None

        self._model = None
        self._training_model = None
        self._optimizer = None

        self._trainer = None
        self._callbacks = None
        self._training_manager = None

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

    def _load_and_prepare_data(self):
        dataset = load_dataset("bavard/personachat_truecased")

        self._raw_training_set = dataset["training"]
        self._raw_validation_set = dataset["validation"]

    @abc.abstractmethod
    def _setup_batch_datasets(self):
        """
        Sets self._batch_training_set and self._batch_validation_set.
        Implementation depends on specific ML Library you are using
        :return:
        """
        raise NotImplementedError("Please implement in your child class")

    @abc.abstractmethod
    def _build_model(self):
        """
        Sets self._model.
        Implementation depends on specific ML Library you are using
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

        # During training we want to do the following:
        # a. Every <progress_log_period> batches, log the batch training loss, calculated during the
        #    forward pass for training
        # b. Every <metric_monitoring_period> batches, calculate and log, for one batch of the validation set,
        #    1. the loss
        #    2. the output candidate classification quality
        # c. Every epoch, calculate and log above metrics over the complete training and validation set
        # d. Every epoch, check if the model improved, if so, save a best model checkpoint.
        #    Also save a "latest model" checkpoint and a training checkpoint
        # e. Log the progress every <progress_log_period>, in terms of training and validation metrics, to the terminal

        # parameters required to rebuild the model from a checkpoint
        model_hyper_parameters = {
            "pretrained_model": self._args.pretrained_model
        }

        loss_gather_func = self._create_gather_loss_function()

        # Every batch we will calculate the training and validation loss
        loss_only_evaluator = mlp.evaluation.MetricEvaluator(
            batch_metric_funcs={
                'loss': loss_gather_func
            },
            trainer=self._trainer)

        # Every epoch we will also calculated the candidate output classification quality
        all_metrics_evaluator = mlp.evaluation.MetricEvaluator(
            batch_metric_funcs={
                'loss': loss_gather_func,
                'classification': self._create_gather_classification_data_function()
            },
            # The trainer knows how to evaluate the model
            trainer=self._trainer,
            batch_metric_reducer_funcs={
                # Use the gathered classification targets and predictions to calculate Recall, Precision and F1
                'classification': None,  # TODO : calc_classification_quality
            },
            batch_chunk_size=self._args.batch_chunk_size,
            batch_chunk_metric_reducer_funcs={
                'classification': flatten_gathered_classification_data,
            },
            show_dataset_evaluation_progress=True)

        def log_condition_func(logs, dataset_batch):
            global_iter = logs['current']['global_iter']

            return global_iter % self._args.metric_monitoring_period == 0

        self._callbacks = [
            # Get training loss calculated, during forward pass, and gather+reduce it from all devices
            mlp.callbacks.TrainingMetricsLogger(metric_evaluator=loss_only_evaluator),

            # Calculate validation loss and classification quality, every <metric_monitoring_period> batches
            mlp.callbacks.TestMetricsLogger(self._batch_validation_set,
                                            'validation',
                                            metric_evaluator=all_metrics_evaluator,
                                            log_condition_func=log_condition_func),
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
            mlp.callbacks.CheckpointManager(base_checkpoint_filename=self._args.experiment_name,
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
                                            model_hyper_parameters=model_hyper_parameters)
        ]

        # Only primary worker needs to log progress
        if self._rank == 0:
            self._callbacks += [
                mlp.callbacks.LogProgress(log_period=self._args.progress_log_period,
                                          set_names=['training', 'validation']),
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
