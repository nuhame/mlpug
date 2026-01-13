"""
Abstract TrainingProcess base class.

Provides a generic, reusable training orchestration layer that sits above
Trainer/TrainingManager. Framework-specific implementations (PyTorch, JAX, etc.)
derive from this class.
"""
import abc
import random
from typing import Any

import numpy as np

from mlpug.base import Base
import mlpug.abstract_interface as mlp_interface
from mlpug.lr_scheduler_configs import LRSchedulerConfig
from mlpug.scheduler_funcs import create_lr_schedule, LRScheduleFunc
from mlpug.trainers.callbacks.callback import Callback
from mlpug.trainers.training import Trainer


class TrainingProcess(Base, metaclass=abc.ABCMeta):
    """
    Abstract training process with common lifecycle and configuration.

    Framework-agnostic base class. Derive for specific ML frameworks
    (PyTorch, JAX, etc.) to implement framework-specific methods.

    The setup() method orchestrates the training initialization in order:
    1. _set_random_seed() - reproducibility
    2. _setup_compute() - device/distributed setup
    3. _setup_datasets() - data loading
    4. _build_model() - model creation
    5. _build_training_model() - training model wrapper creation
    6. _setup_training_model() - move to device, distributed setup
    7. _build_optimizer() - optimizer creation
    8. _setup_lr_scheduler() - LR scheduling
    9. _setup_trainer() - trainer creation
    10. _setup_callbacks() - callback setup (includes LR scheduler callback, if configured)
    11. _setup_training_manager() - training manager setup
    12. _prepare_training() - final preparation hook

    To create a training model wrapper, implement _build_training_model().
    """

    # Override in framework-specific class with actual mlpug module
    # e.g., MLPUG_MODULE = mlpug.pytorch
    MLPUG_MODULE: Any = mlp_interface

    def __init__(
        self,
        rank: int,
        num_devices: int,
        *,
        # Distributed training
        distributed: bool = False,
        # Training hyperparameters
        batch_size: int = 64,
        micro_batch_size: int | None = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        num_epochs: int = 10,
        # LR scheduling
        lr_scheduler_config: LRSchedulerConfig | None = None,
        # Logging & checkpointing
        experiment_name: str = "training",
        log_frequency: int = 30,
        # Hardware settings
        use_mixed_precision: bool = True,
        eager_mode: bool = False,
        activation_checkpointing: bool = False,
        num_dataloader_workers: int = 2,
        # Reproducibility
        seed: int = 42,
        # Base class params
        name: str = "TrainingProcess",
    ):
        """
        :param rank: Device rank (0 for single device, 0..N-1 for distributed).
        :param num_devices: Total number of devices (1 for single device).
        :param distributed: Enable distributed training.
        :param batch_size: Effective batch size (samples per optimizer step).
        :param micro_batch_size: Memory batch size. If None, equals batch_size.
        :param learning_rate: Peak learning rate.
        :param weight_decay: Weight decay coefficient.
        :param num_epochs: Number of training epochs.
        :param lr_scheduler_config: LR scheduler configuration. None for no scheduling.
        :param experiment_name: Name for logging and checkpoints.
        :param log_frequency: Log every N batches.
        :param use_mixed_precision: Enable automatic mixed precision.
        :param eager_mode: Disable graph compilation (if applicable).
        :param activation_checkpointing: Enable gradient checkpointing.
        :param num_dataloader_workers: Number of DataLoader workers.
        :param seed: Random seed for reproducibility.
        :param name: Logger name.
        """
        logger_name, disable_logging = self.get_logger_info(rank, num_devices, name)
        super().__init__(disable_logging=disable_logging, pybase_logger_name=logger_name)

        self._rank = rank
        self._num_devices = num_devices
        self._distributed = distributed

        # Training hyperparameters
        self._batch_size = batch_size
        self._micro_batch_size = micro_batch_size if micro_batch_size is not None else batch_size
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._num_epochs = num_epochs

        # LR scheduling
        self._lr_scheduler_config = lr_scheduler_config

        # Logging & checkpointing
        self._experiment_name = experiment_name
        self._log_frequency = log_frequency

        # Hardware settings
        self._use_mixed_precision = use_mixed_precision
        self._eager_mode = eager_mode
        self._activation_checkpointing = activation_checkpointing
        self._num_dataloader_workers = num_dataloader_workers

        # Reproducibility
        self._seed = seed

        # Components (set during setup via return values)
        self._device: object | None = None
        self._training_set: object | None = None
        self._validation_set: object | None = None
        self._model: object | None = None
        self._training_model: object | None = None
        self._optimizer: object | None = None
        self._lr_scheduling_func: LRScheduleFunc | None = None
        self._trainer: object | None = None
        self._callbacks: list[Callback] | None = None
        self._training_manager: object | None = None

    @staticmethod
    def get_logger_info(
        rank: int,
        num_devices: int,
        name: str,
    ) -> tuple[str, bool]:
        """
        Get logger name and disable flag based on rank.

        Only rank 0 logs in distributed training.

        :param rank: Device rank.
        :param num_devices: Total devices.
        :param name: Base logger name.

        :return: (logger_name, disable_logging)
        """
        is_distributed = num_devices > 1
        is_primary = rank == 0

        if is_distributed:
            logger_name = f"[Device {rank}] {name}"
            disable_logging = not is_primary
        else:
            logger_name = name
            disable_logging = False

        return logger_name, disable_logging

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def rank(self) -> int:
        """Device rank (0 for primary)."""
        return self._rank

    @property
    def num_devices(self) -> int:
        """Total number of devices."""
        return self._num_devices

    @property
    def is_primary(self) -> bool:
        """True if this is the primary device (rank 0)."""
        return self._rank == 0

    @property
    def is_distributed(self) -> bool:
        """True if distributed training is configured."""
        return self._distributed

    @property
    def batch_size(self) -> int:
        """Effective batch size (samples per optimizer step)."""
        return self._batch_size

    @property
    def micro_batch_size(self) -> int:
        """Memory batch size (samples per forward pass)."""
        return self._micro_batch_size

    @property
    def learning_rate(self) -> float:
        """Peak learning rate."""
        return self._learning_rate

    @property
    def num_epochs(self) -> int:
        """Number of training epochs."""
        return self._num_epochs

    @property
    def device(self) -> object:
        """The compute device being used."""
        if self._device is None:
            raise RuntimeError("Call setup() before accessing device")
        return self._device

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def setup(self) -> None:
        """
        Initialize all training components.

        Call this after construction before start().
        """
        self._set_random_seed()

        # Setup compute - returns device
        self._device = self._setup_compute()
        if self._device is None:
            raise RuntimeError("_setup_compute() must return a device")

        # Setup datasets - returns (training_set, validation_set)
        self._training_set, self._validation_set = self._setup_datasets()
        if self._training_set is None:
            raise RuntimeError("_setup_datasets() must return a training set")

        # Build model - returns model
        self._model = self._build_model()
        if self._model is None:
            raise RuntimeError("_build_model() must return a model")

        # Build training model wrapper
        training_model = self._build_training_model()
        if training_model is None:
            raise RuntimeError("_build_training_model() must return a training model")

        # Setup training model (move to device, DDP, etc.)
        self._training_model = self._setup_training_model(training_model)
        if self._training_model is None:
            raise RuntimeError("_setup_training_model() must return a training model")

        # Build optimizer - returns optimizer
        self._optimizer = self._build_optimizer()
        if self._optimizer is None:
            raise RuntimeError("_build_optimizer() must return an optimizer")

        # Setup LR scheduler - may return None
        self._lr_scheduling_func = self._setup_lr_scheduler()

        # Setup trainer - returns trainer
        self._trainer = self._setup_trainer()
        if self._trainer is None:
            raise RuntimeError("_setup_trainer() must return a trainer")

        # Setup callbacks - returns list
        self._callbacks = self._setup_callbacks()
        if self._callbacks is None:
            raise RuntimeError("_setup_callbacks() must return a list")

        # Setup training manager - returns training manager
        self._training_manager = self._setup_training_manager()
        if self._training_manager is None:
            raise RuntimeError("_setup_training_manager() must return a training manager")

        self._prepare_training()

    def start(self) -> None:
        """Start training. Call setup() first."""
        if self._training_manager is None:
            raise RuntimeError("Call setup() before start()")
        self._training_manager.start_training()

    # -------------------------------------------------------------------------
    # Protected methods (common implementation)
    # -------------------------------------------------------------------------

    def _set_random_seed(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self._seed)
        np.random.seed(self._seed)
        # Framework-specific seeding should be done in _setup_compute()

    def _setup_lr_scheduler(self) -> LRScheduleFunc | None:
        """
        Setup LR scheduler from config.

        :return: LR scheduling function, or None if not configured.
        """
        if self._lr_scheduler_config is None:
            self._log.info("No LR scheduler configured")
            return None

        total_steps = self._calculate_total_steps()
        self._log.info(
            f"Setting up LR scheduler: {type(self._lr_scheduler_config).__name__}\n"
            f"  total_steps = {total_steps}"
        )

        return create_lr_schedule(
            self._lr_scheduler_config,
            total_steps,
        )

    def _setup_lr_scheduler_callbacks(self) -> list[Callback]:
        """
        Create LR scheduler callback(s).

        Override in framework-specific class to return appropriate callback.
        Call this from your _setup_callbacks() implementation and add the
        returned callbacks to your callback list.

        :return: List of LR scheduler callbacks (usually 0 or 1).
        """
        return []

    def _setup_training_manager(self) -> object:
        """
        Setup MLPug TrainingManager.

        :return: TrainingManager instance.
        """
        mlp = self.MLPUG_MODULE

        return mlp.trainers.TrainingManager(
            trainer=self._trainer,
            training_dataset=self._training_set,
            num_epochs=self._num_epochs,
            callbacks=self._callbacks,
        )

    def _prepare_training(self) -> None:
        """
        Final preparation before training starts.

        Override to add custom preparation logic.
        """
        pass

    # -------------------------------------------------------------------------
    # Abstract methods (must implement in derived class)
    # -------------------------------------------------------------------------

    @abc.abstractmethod
    def _setup_compute(self) -> object:
        """
        Setup compute devices and distributed training.

        Framework-specific: Initialize device, DDP, set framework random seeds.

        :return: Device object (e.g., torch.device).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_datasets(self) -> tuple[object, object | None]:
        """
        Setup training and validation datasets.

        :return: (training_set, validation_set). validation_set may be None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_model(self) -> object:
        """
        Build the model instance.

        :return: Model instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_training_model(self, training_model: object) -> object:
        """
        Setup training model wrapper.

        Implement in framework-specific class to add device movement and
        distributed setup.

        :param training_model: Training model from _build_training_model().

        :return: Training model wrapper (possibly modified).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_training_model(self) -> object:
        """
        Build the training model wrapper.

        Creates a module that wraps self._model and computes the training loss.

        :return: Training model wrapper.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_optimizer(self) -> object:
        """
        Build optimizer.

        Typically uses self._model.parameters().

        :return: Optimizer instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_trainer(self) -> Trainer:
        """
        Setup MLPug trainer.

        :return: Trainer instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_callbacks(self) -> list[Callback]:
        """
        Setup training callbacks.

        Call _setup_lr_scheduler_callbacks() and add the returned callbacks
        to your list before returning.

        :return: List of callbacks.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _calculate_total_steps(self) -> int:
        """
        Calculate total training steps for LR scheduling.

        :return: Total number of optimizer steps across all epochs.
        """
        raise NotImplementedError
