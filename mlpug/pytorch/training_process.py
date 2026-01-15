"""
PyTorch-specific TrainingProcess.

Handles device setup (CUDA/MPS/CPU), DDP initialization, optimizer creation,
and trainer setup. Derives from the abstract TrainingProcess base class.
"""
import abc
import os
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

import mlpug.pytorch as mlp
from mlpug.training_process import TrainingProcess as TrainingProcessBase
from mlpug.trainers import ModelWrapperFunc
from mlpug.trainers.callbacks.callback import Callback
from mlpug.trainers.training import Trainer
from mlpug.pytorch.model_wrappers.ddp import DDPModelWrapper


class TrainingProcess(TrainingProcessBase, metaclass=abc.ABCMeta):
    """
    PyTorch-specific training process.

    Implements device setup, DDP, optimizer creation, and trainer setup.
    Task-specific classes derive from this to implement dataset and model setup.
    """

    MLPUG_MODULE = mlp

    # Default AdamW configuration (used in default _build_optimizer)
    DEFAULT_ADAMW_OPTIMIZER_CONFIG: dict = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }

    def __init__(
        self,
        rank: int,
        num_devices: int,
        *,
        # PyTorch-specific settings
        force_on_cpu: bool = False,
        optimizer_config: dict | None = None,
        # Base class params
        **kwargs,
    ):
        """
        :param rank: Device rank (0 for single device, 0..N-1 for distributed).
        :param num_devices: Total number of devices.
        :param force_on_cpu: Force CPU even if GPU available.
        :param optimizer_config: Optional dict of optimizer parameters. The default
            _build_optimizer uses AdamW and merges this with DEFAULT_OPTIMIZER_CONFIG.
            Subclasses can use this config however they need.
        :param kwargs: Arguments passed to base TrainingProcess.
        """
        super().__init__(rank, num_devices, **kwargs)

        self._force_on_cpu = force_on_cpu
        self._optimizer_config = optimizer_config

        # Set during _setup_compute()
        self._device: torch.device | None = None
        self._model_wrapper_func: Callable | None = None

    @property
    def device(self) -> torch.device:
        """The torch device being used."""
        if self._device is None:
            raise RuntimeError("Call setup() before accessing device")
        return self._device

    # -------------------------------------------------------------------------
    # Implemented methods (PyTorch-specific)
    # -------------------------------------------------------------------------

    def _set_random_seed(self) -> None:
        """Set random seeds including PyTorch."""
        super()._set_random_seed()
        torch.manual_seed(self._seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)

    def _setup_compute(self) -> torch.device:
        """
        Setup device (CUDA/MPS/CPU) and DDP if distributed.

        :return: The torch device to use.
        """
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()

        # Optimizations for CUDA
        if cuda_available and not self._force_on_cpu:
            torch.set_float32_matmul_precision('high')

        if self.is_distributed:
            device = self._setup_distributed(cuda_available)
        else:
            device = self._setup_single_device(cuda_available, mps_available)

        return device

    def _setup_distributed(self, cuda_available: bool) -> torch.device:
        """
        Setup distributed training with DDP.

        :return: The torch device to use.
        """
        self._log.info("Distributed Data Parallel (DDP) mode")

        # Set master address/port if not already set (e.g., by torchrun)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"

        if cuda_available and not self._force_on_cpu:
            torch.cuda.set_device(self.rank)
            device = torch.device("cuda")
            backend = "nccl"
            self._log.info(f"Using GPU {self.rank}/{self.num_devices}")
        else:
            device = torch.device("cpu")
            backend = "gloo"
            self._log.info(f"Using CPU worker {self.rank}/{self.num_devices}")

        dist.init_process_group(
            backend=backend,
            rank=self.rank,
            world_size=self.num_devices,
        )
        self._log.info(f"DDP backend: {backend}")

        return device

    def _setup_single_device(
        self,
        cuda_available: bool,
        mps_available: bool,
    ) -> torch.device:
        """
        Setup single device (no DDP).

        :return: The torch device to use.
        """
        if cuda_available and not self._force_on_cpu:
            device = torch.device("cuda")
            self._log.info("Single device mode: Using GPU")
        elif mps_available and not self._force_on_cpu:
            device = torch.device("mps")
            self._log.info("Single device mode: Using Apple MPS")
        else:
            device = torch.device("cpu")
            self._log.info("Single device mode: Using CPU")

        return device

    def _build_optimizer(self) -> Optimizer:
        """
        As a default implementation, builds AdamW optimizer with sensible defaults.

        Excludes bias and LayerNorm parameters from weight decay following
        best practices for transformer training.

        Merges DEFAULT_ADAMW_OPTIMIZER_CONFIG with self._optimizer_config,
        allowing users to override defaults (e.g., betas=(0.9, 0.95) for LLMs).

        :return: AdamW optimizer.
        """
        # Assert narrows type from object to Module (base class uses object for
        # framework independence), and guards against incorrect setup() ordering
        assert isinstance(self._model, Module)

        # Separate parameters into weight decay and no weight decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue

            # Skip weight decay for biases and LayerNorm
            if name.endswith(".bias") or "layernorm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self._weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Merge default config with user-provided config
        optimizer_config = {
            **self.DEFAULT_ADAMW_OPTIMIZER_CONFIG,
            **(self._optimizer_config or {}),
        }

        return AdamW(
            param_groups,
            lr=self._learning_rate,
            **optimizer_config,
        )

    def _build_ddp_model_wrapper(self) -> ModelWrapperFunc:
        """
        Build DDP model wrapper function.

        Override this method to customize DDP behavior, e.g.:

            from functools import partial

            def _build_ddp_model_wrapper(self) -> ModelWrapperFunc:
                wrapper = DDPModelWrapper(self.rank, self._device)
                return partial(wrapper, gradient_as_bucket_view=True)

        :return: Model wrapper function for DDP.
        """
        return DDPModelWrapper(self.rank, self._device)

    def _setup_training_model(self, training_model: Module) -> Module:
        """
        Setup training model wrapper: move to device and configure DDP.

        :param training_model: Training model from _build_training_model().

        :return: Training model wrapper (on device).
        """
        # Move to device
        training_model.to(self._device)

        # Create DDP wrapper function if distributed
        if self.is_distributed:
            self._model_wrapper_func = self._build_ddp_model_wrapper()

        return training_model

    def _setup_trainer(self) -> Trainer:
        """
        Setup MLPug DefaultTrainer.

        :return: Trainer instance.
        """
        custom_config = self._get_custom_trainer_config()

        return mlp.trainers.DefaultTrainer(
            optimizers=self._optimizer,
            model_components=self._model,
            model_wrapper_func=self._model_wrapper_func,
            batch_size=self._batch_size,
            micro_batch_size=self._micro_batch_size,
            use_mixed_precision=self._use_mixed_precision,
            autocast_dtype=self._autocast_dtype,
            use_loss_scaling=self._use_loss_scaling,
            eager_mode=self._eager_mode,
            **custom_config,
        )

    def _setup_lr_scheduler_callbacks(self) -> list[Callback]:
        """
        Create LR scheduler callback if scheduling is configured.

        :return: List containing LR scheduler callback, or empty list.
        """
        if self._lr_scheduling_func is None:
            return []

        # Assert narrows type from object to Optimizer (base class uses object for
        # framework independence), and guards against incorrect setup() ordering
        assert isinstance(self._optimizer, Optimizer)

        # Wrap in tensor for torch.compile compatibility
        # See https://github.com/pytorch/pytorch/issues/120934#issuecomment-1973390203
        def lr_lambda(step: int) -> torch.Tensor:
            return torch.tensor(self._lr_scheduling_func(step), requires_grad=False)

        scheduler = LambdaLR(self._optimizer, lr_lambda)

        return [
            mlp.callbacks.LRSchedulerWrapper(
                {"lr-scheduler": scheduler},
                batch_level=True,
            )
        ]

    def _get_custom_trainer_config(self) -> dict:
        """
        Get custom trainer configuration.

        Override to add compile_kwargs or other trainer settings.

        :return: Dict of additional trainer kwargs.
        """
        return {}

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def execute_for_primary_device_first(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute function on primary device first, then on others.

        Useful for distributed training where primary downloads/generates
        data and others wait, then load from cache.

        :param func: Function to execute.
        :param args: Positional arguments for func.
        :param kwargs: Keyword arguments for func.

        :return: Result from func.
        """
        # Non-primary waits for primary to finish
        if self.is_distributed and not self.is_primary:
            dist.barrier()

        result = func(*args, **kwargs)

        # Primary waits for others to catch up
        if self.is_distributed and self.is_primary:
            dist.barrier()

        return result

    # -------------------------------------------------------------------------
    # Abstract methods (still need task-specific implementation)
    # -------------------------------------------------------------------------

    @abc.abstractmethod
    def _setup_datasets(self) -> tuple[object, object | None]:
        """
        Setup training and validation datasets.

        For distributed training, use DistributedSampler.

        :return: (training_set, validation_set). validation_set may be None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_model(self) -> Module:
        """
        Build the model architecture.

        :return: The model instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _build_training_model(self) -> Module:
        """
        Build the training model wrapper.

        Creates a module that wraps self._model and computes the training loss.

        :return: Training model wrapper.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_callbacks(self) -> list[Callback]:
        """
        Setup training callbacks.

        Call _setup_lr_scheduler_callbacks() and add the returned callbacks
        to your list. For distributed training, add DistributedSamplerManager.

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
