"""
NTP (Next-Token Prediction) training process for PyTorch.

Derives from the PyTorch TrainingProcess to implement NTP-specific dataset loading,
model initialization, and callback setup for training language models from scratch.

This module expects tokenized and indexed data created using Data Forager.
See examples/agentic_llm_pretraining/datasets/tokenize_dataset.py for creating
the required tokenized datasets.
"""
from typing import Tuple

import math
from functools import partial
from pathlib import Path

import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

# Optional: Liger Kernel for memory-efficient cross-entropy
# See docs/liger-kernel-memory-efficient-cross-entropy.md for details
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False

import mlpug.pytorch as mlp
from mlpug.pytorch.training_process import TrainingProcess
from mlpug.pytorch.model_wrappers.ddp import DDPModelWrapper
from mlpug.evaluation import GatherLoss
from mlpug.pytorch.evaluation import GatherLossDistributed
from mlpug.lr_scheduler_configs import LRSchedulerConfig, CosineDecayConfig
from mlpug.trainers import ModelWrapperFunc
from mlpug.trainers.callbacks.callback import Callback

from data_forager.datasets.common import SubsampledDataset

from examples.agentic_llm_pretraining.datasets.loading import (
    load_tokens_dataset,
    get_sample_properties,
)
from examples.agentic_llm_pretraining.training.pytorch.datasets import PyTorchTokensDataset
from examples.agentic_llm_pretraining.training.pytorch.model import NTPTrainModel


# Default model for NTP training
DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

# Default paths (relative to repo root)
DEFAULT_CHECKPOINT_DIR = "../trained-models"
DEFAULT_LOG_DIR = "../training-logs"

# Default LR scheduler for LLM pretraining
DEFAULT_LR_SCHEDULER_CONFIG = CosineDecayConfig(
    warmup_ratio=0.1,
    min_lr_ratio=0.01,
)

# AdamW config for LLM pretraining (lower beta2)
DEFAULT_LLM_OPTIMIZER_CONFIG = {
    "betas": (0.9, 0.95),
    "eps": 1e-8,
}


def perplexity(loss_data: Tuple[float, int] | None) -> float | None:
    """
    Calculate perplexity from aggregated loss data.

    Perplexity = exp(cross_entropy_loss), a standard metric for language models.
    Lower perplexity indicates better prediction of the next token.

    :param loss_data: Tuple of (loss_sum, num_samples) from GatherLoss,
        or None if no data available (e.g., non-primary device in DDP).

    :return: Perplexity value, or None if input is None.
    """
    if loss_data is None:
        return None

    loss_sum, tot_num_samples = loss_data
    avg_loss = loss_sum / tot_num_samples

    # Cap at infinity for numerical stability (exp(100) ≈ 2.7e43)
    return math.exp(avg_loss) if avg_loss < 100 else float('inf')


class NTPTrainingProcess(TrainingProcess):
    """
    Training process for Next-Token Prediction (NTP).

    Loads tokenized data from Data Forager, initializes a Qwen3 model from scratch,
    and trains using standard NTP loss (cross-entropy on next-token prediction).

    This class expects tokenized and indexed data created using Data Forager.
    See examples/agentic_llm_pretraining/datasets/tokenize_dataset.py for creating
    the required tokenized datasets.

    :param rank: Device rank (0 for single device, 0..N-1 for distributed).
    :param num_devices: Total number of devices.
    :param train_data_path: Path to tokenized training data directory (must contain
        a Data Forager index created by tokenize_dataset.py).
    :param val_data_path: Path to tokenized validation data directory (must contain
        a Data Forager index created by tokenize_dataset.py).
    :param train_fraction: Fraction of training data to use (0-1). If None, use all data.
        Requires seed to be set for reproducible subsampling across distributed processes.
    :param model_name: HuggingFace model name for config (default: Qwen/Qwen3-1.7B-Base).
    :param use_liger_kernel: Enable Liger Kernel for memory-efficient cross-entropy.
        Reduces logits memory by ~97% by fusing linear + cross-entropy computation.
        Requires liger-kernel package: pip install liger-kernel
    :param checkpoint_dir: Directory for saving checkpoints.
    :param log_dir: Directory for training logs.
    :param kwargs: Additional arguments passed to base TrainingProcess.
    """

    def __init__(
        self,
        rank: int,
        num_devices: int,
        *,
        # NTP-specific parameters
        train_data_path: str,
        val_data_path: str | None = None,
        train_fraction: float | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        use_liger_kernel: bool = True,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
        log_dir: str = DEFAULT_LOG_DIR,
        # Base class parameters with NTP-specific defaults
        lr_scheduler_config: LRSchedulerConfig | None = DEFAULT_LR_SCHEDULER_CONFIG,
        optimizer_config: dict | None = None,
        **kwargs,
    ):
        # Set default optimizer config for LLM training if not provided
        if optimizer_config is None:
            optimizer_config = DEFAULT_LLM_OPTIMIZER_CONFIG

        super().__init__(
            rank,
            num_devices,
            lr_scheduler_config=lr_scheduler_config,
            optimizer_config=optimizer_config,
            **kwargs,
        )

        self._train_data_path = train_data_path
        self._val_data_path = val_data_path
        self._train_fraction = train_fraction
        self._model_name = model_name
        self._use_liger_kernel = use_liger_kernel
        self._checkpoint_dir = checkpoint_dir
        self._log_dir = log_dir

        # Set during _setup_datasets
        self._context_length: int | None = None
        self._token_dtype: np.dtype | None = None
        self._training_sampler: DistributedSampler | None = None
        self._validation_sampler: DistributedSampler | None = None

    # -------------------------------------------------------------------------
    # Implemented abstract methods
    # -------------------------------------------------------------------------

    def _setup_datasets(self) -> tuple[DataLoader, DataLoader | None]:
        """
        Load tokenized datasets from Data Forager and create DataLoaders.

        :return: (training_dataloader, validation_dataloader)
        """
        # Load training dataset
        self._log.info(f"Loading training data from: {self._train_data_path}")
        train_tokens_dataset = load_tokens_dataset(self._train_data_path)

        # Apply subsampling if requested
        if self._train_fraction is not None:
            if self._seed is None:
                raise ValueError(
                    "train_fraction requires a seed for reproducible subsampling across "
                    "distributed processes. Please specify --seed."
                )
            self._log.info(f"  Subsampling training data to {self._train_fraction:.1%} "
                           f"(seed={self._seed})")
            train_tokens_dataset = SubsampledDataset(
                train_tokens_dataset,
                subsample_factor=self._train_fraction,
                seed=self._seed,
            )

        train_dataset = PyTorchTokensDataset(train_tokens_dataset)
        self._log.info(f"  Training samples: {len(train_dataset):,}")

        # Get sample properties from first dataset
        self._context_length, self._token_dtype = get_sample_properties(train_tokens_dataset)

        # Load validation dataset if provided
        val_dataset = None
        if self._val_data_path:
            self._log.info(f"Loading validation data from: {self._val_data_path}")
            val_tokens_dataset = load_tokens_dataset(self._val_data_path)
            val_dataset = PyTorchTokensDataset(val_tokens_dataset)
            self._log.info(f"  Validation samples: {len(val_dataset):,}")

        # Create samplers for distributed training
        if self.is_distributed:
            self._training_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.num_devices,
                rank=self.rank,
                shuffle=True,
            )
            if val_dataset is not None:
                self._validation_sampler = DistributedSampler(
                    val_dataset,
                    num_replicas=self.num_devices,
                    rank=self.rank,
                    shuffle=True,  # Shuffle for representative batch-level sliding window metrics
                )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._micro_batch_size,
            shuffle=(self._training_sampler is None),
            sampler=self._training_sampler,
            num_workers=self._num_dataloader_workers,
            pin_memory=True,
            drop_last=True,  # Ensure consistent batch sizes for torch.compile
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self._micro_batch_size,
                shuffle=(self._validation_sampler is None),  # Shuffle when no sampler
                sampler=self._validation_sampler,
                num_workers=self._num_dataloader_workers,
                pin_memory=True,
                drop_last=True,
            )

        self._log.info(f"Context length: {self._context_length}")
        self._log.info(f"Token dtype: {self._token_dtype}")

        return train_loader, val_loader

    def _build_model(self) -> Module:
        """
        Initialize Qwen3 model from config (random weights, training from scratch).

        :return: Initialized model.
        """
        self._log.info(f"Initializing model from config: {self._model_name}")

        # Apply Liger Kernel patches if requested
        if self._use_liger_kernel:
            if LIGER_KERNEL_AVAILABLE:
                self._log.info("Applying Liger Kernel patches for memory-efficient cross-entropy")
                # Patch must happen BEFORE model creation
                apply_liger_kernel_to_qwen3(fused_linear_cross_entropy=True)
            else:
                self._log.warning(
                    "Liger Kernel requested but not available. "
                    "Install with: pip install liger-kernel"
                )

        # Load config from HuggingFace (architecture parameters only)
        config = self.execute_for_primary_device_first(
            AutoConfig.from_pretrained,
            self._model_name,
            trust_remote_code=True,
        )

        self._log.info(f"Model config:\n"
                       f"  hidden_size: {config.hidden_size}\n"
                       f"  num_hidden_layers: {config.num_hidden_layers}\n"
                       f"  num_attention_heads: {config.num_attention_heads}\n"
                       f"  num_key_value_heads: {config.num_key_value_heads}\n"
                       f"  vocab_size: {config.vocab_size}\n"
                       f"  max_position_embeddings: {config.max_position_embeddings}")

        # Initialize model with random weights (not from pretrained checkpoint)
        # Wrap in execute_for_primary_device_first because trust_remote_code=True
        # may trigger downloading model code from HuggingFace
        model = self.execute_for_primary_device_first(
            AutoModelForCausalLM.from_config,
            config,
            trust_remote_code=True,
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        return model

    def _build_training_model(self) -> Module:
        """
        Wrap model with NTPTrainModel for loss computation.

        :return: NTPTrainModel wrapper.
        """
        # Assert narrows type and validates setup ordering
        assert isinstance(self._model, PreTrainedModel), \
            f"Expected PreTrainedModel, got {type(self._model)}"

        return NTPTrainModel(
            self._model,
            device=self._device,
            activation_checkpointing=self._activation_checkpointing,
        )

    def _build_ddp_model_wrapper(self) -> ModelWrapperFunc:
        """
        Build DDP model wrapper with memory optimization.

        Uses gradient_as_bucket_view=True to reduce memory by avoiding
        gradient copy to communication buckets.

        :return: Model wrapper function for DDP.
        """
        wrapper = DDPModelWrapper(self.rank, self._device)
        return partial(wrapper, gradient_as_bucket_view=True)

    def _setup_callbacks(self) -> list[Callback]:
        """
        Setup training callbacks for logging, checkpointing, and TensorBoard.

        :return: List of callbacks.
        """
        callbacks = []

        # Add DistributedSampler managers first (if distributed)
        if self.is_distributed:
            if self._training_sampler is not None:
                callbacks.append(
                    mlp.callbacks.DistributedSamplerManager(
                        self._training_sampler,
                        name="DistributedSamplerManager[training]",
                    )
                )
            if self._validation_sampler is not None:
                callbacks.append(
                    mlp.callbacks.DistributedSamplerManager(
                        self._validation_sampler,
                        name="DistributedSamplerManager[validation]",
                    )
                )

        # Create metric evaluator for loss and perplexity tracking
        # Both metrics use the same GatherLoss function to extract (loss_sum, num_samples),
        # but apply different final calculations (average_loss vs exp(average_loss))
        metric_evaluator = mlp.evaluation.MetricEvaluator(
            trainer=self._trainer,
            gather_metric_inputs_funcs={
                # 'loss' uses default GatherLoss (provided automatically when not specified)
                'perplexity': GatherLoss(requester="MetricEvaluator"),
            },
            gather_distributed_inputs_funcs={
                # 'loss' uses default GatherLossDistributed (provided automatically)
                'perplexity': GatherLossDistributed(requester="MetricEvaluator"),
            },
            metric_funcs={
                # 'loss' uses default average_loss (provided automatically when not specified)
                'perplexity': perplexity,
            },
            clean_up_batch_data_func=self._create_clean_up_batch_data_func(),
            eager_mode=self._eager_mode,
            name="MetricEvaluator",
        )

        # Condition function for batch-level logging
        def log_condition(logs, dataset_batch):
            batch_step = logs['current']['batch_step']
            return batch_step % self._log_frequency == 0

        # Calculate sliding window length for metrics averaging
        # The sliding window averages metrics over recent batches to smooth noise.
        # Formula: avg_window_length ≈ log_frequency / 2
        #
        # Rationale:
        # - We log every `log_frequency` batches
        # - Setting window to ~half the log frequency means:
        #   - ~50% of batches in each window are new since last log (shows trends)
        #   - ~50% overlap provides continuity and noise smoothing
        # - This balances responsiveness to changes vs stability of metrics
        #
        # Example: log_frequency=100, batches_per_epoch=1000
        #   → logs_per_epoch = 10
        #   → avg_window_length = 1000 / (2 * 10) = 50 batches
        num_batches_per_epoch = self._calculate_batches_per_epoch()
        num_logs_per_epoch = max(1, num_batches_per_epoch // self._log_frequency)
        avg_window_length = max(1, num_batches_per_epoch // (2 * num_logs_per_epoch))

        # Training metrics logger (batch-level)
        callbacks.append(
            mlp.callbacks.TrainingMetricsLogger(
                metric_evaluator=metric_evaluator,
                log_condition_func=log_condition,
                sliding_window_length=avg_window_length,
            )
        )

        # Validation metrics logger (batch-level, if validation set exists)
        if self._validation_set is not None:
            callbacks.append(
                mlp.callbacks.DatasetMetricsLogger(
                    self._validation_set,
                    'validation',
                    metric_evaluator=metric_evaluator,
                    log_condition_func=log_condition,
                    sliding_window_length=avg_window_length,
                )
            )

            # Validation metrics over full dataset (epoch-level)
            callbacks.append(
                mlp.callbacks.DatasetMetricsLogger(
                    self._validation_set,
                    'validation',
                    metric_evaluator=metric_evaluator,
                    batch_level=False,  # epoch level only
                )
            )

        # Add LR scheduler callback
        callbacks.extend(self._setup_lr_scheduler_callbacks())

        # Primary device only: checkpointing, logging, TensorBoard
        if self.is_primary:
            # Ensure directories exist
            checkpoint_path = Path(self._checkpoint_dir) / self._experiment_name
            checkpoint_path.mkdir(parents=True, exist_ok=True)

            log_path = Path(self._log_dir) / self._experiment_name
            log_path.mkdir(parents=True, exist_ok=True)

            # Checkpoint manager
            callbacks.append(
                mlp.callbacks.CheckpointManager(
                    base_checkpoint_filename=self._experiment_name,
                    checkpoints_path=str(checkpoint_path),
                    batch_level=False,  # monitor per epoch
                    metric_to_monitor="validation.dataset.loss" if self._validation_set else None,
                    metric_monitor_period=1,  # check every epoch
                    create_checkpoint_every=1,  # save every epoch
                    archive_last_model_checkpoint_every=0,  # no archiving
                    backup_before_override=False,
                )
            )

            # Progress logging
            set_names = ['training', 'training_params']
            if self._validation_set is not None:
                set_names.append('validation')

            callbacks.append(
                mlp.callbacks.LogProgress(
                    log_condition_func=log_condition,
                    set_names=set_names,
                )
            )

            # TensorBoard logging
            tb_args = {
                'experiment_name': self._experiment_name,
                'log_dir': str(log_path),
            }

            callbacks.append(mlp.callbacks.AutoTensorboard(dataset_name='training', **tb_args))
            callbacks.append(mlp.callbacks.AutoTensorboard(dataset_name='training_params', **tb_args))

            if self._validation_set is not None:
                callbacks.append(mlp.callbacks.AutoTensorboard(dataset_name='validation', **tb_args))

        return callbacks

    def _calculate_total_steps(self) -> int:
        """
        Calculate total training steps for LR scheduling.

        :return: Total number of optimizer steps across all epochs.
        """
        batches_per_epoch = self._calculate_batches_per_epoch()
        total_steps = batches_per_epoch * self._num_epochs

        self._log.info(f"Total training steps: {total_steps:,} "
                       f"({batches_per_epoch:,} batches/epoch × {self._num_epochs} epochs)")

        return total_steps

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _calculate_batches_per_epoch(self) -> int:
        """
        Calculate number of effective batches (optimizer steps) per epoch.

        :return: Number of batches per epoch.
        """
        if not isinstance(self._training_set, DataLoader):
            raise RuntimeError("Call _setup_datasets() before _calculate_batches_per_epoch()")

        num_micro_batches = len(self._training_set)
        gradient_accumulation_steps = self._batch_size // self._micro_batch_size

        return num_micro_batches // gradient_accumulation_steps

    def _create_clean_up_batch_data_func(self):
        """
        Create function to clean up batch data after metric computation.

        :return: Cleanup function.
        """
        def clean_up_batch_data(model_output, **kwargs):
            loss = model_output["loss"]
            model_output["loss"] = loss.cpu().item()
            # Remove auxiliary results to free memory
            if "auxiliary_results" in model_output:
                del model_output["auxiliary_results"]

        return clean_up_batch_data
