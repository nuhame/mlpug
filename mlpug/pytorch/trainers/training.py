from contextlib import nullcontext
from typing import Optional, Dict, Tuple

import torch

from torch.amp import autocast
import torch.distributed as dist

import basics.base_utils as _

from mlpug.mlpug_exceptions import TrainerInvalidException, LossNotAvailableException, TrainerStateInvalidException

from mlpug.trainers.training import TrainingManager as TrainingManagerBase
from mlpug.trainers.training import Trainer as TrainerBase
from mlpug.trainers.training import DefaultTrainer as DefaultTrainerBase
from mlpug.trainers.training import MicroBatchResults

from mlpug.pytorch.utils import SlidingWindow

from mlpug.pytorch.multi_processing import MultiProcessingMixin


class TrainingManager(MultiProcessingMixin, TrainingManagerBase):
    def __init__(self, *args, sliding_window_factory=SlidingWindow, **kwargs):
        super().__init__(*args, sliding_window_factory=sliding_window_factory, **kwargs)

    def _training_ended(self):
        if self.is_distributed:
            # Wait for all processes to finish
            dist.barrier()


class PTTrainerMixin:

    def _activate_inference_mode(self, inference_mode):
        if inference_mode:
            self.training_model.eval()
        else:
            self.training_model.train()

    def _get_model_state(self, model, model_name=None):
        return model.state_dict()

    def _get_optimizer_state(self, optimizer, optimizer_name=None):
        return optimizer.state_dict()

    def _set_model_state(self, model, state, model_name=None):
        model.load_state_dict(state)

    def _set_optimizer_state(self, optimizer, state, optimizer_name):
        optimizer.load_state_dict(state)


class Trainer(MultiProcessingMixin, PTTrainerMixin, TrainerBase):
    pass


class DefaultTrainerMixin(PTTrainerMixin, DefaultTrainerBase):

    def __init__(
            self,
            *args,
            scaler=None,
            amp_init_scale: int = 2**10,
            compile_kwargs: Optional[Dict] = None,
            name="DefaultTrainer",
            **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)

        self._scaler = scaler

        if self.use_mixed_precision:
            if scaler is None:
                self._log.debug("Creating default scaler instance for automatic mixed precision ...")
                # Default 65536 causes overflow on some platforms (e.g., ROCm 7.1.1)
                # Lower initial scale (1024) provides better stability
                self._scaler = torch.amp.GradScaler('cuda', init_scale=amp_init_scale)

            self._log.info(f"Using scalar instance for automatic mixed precision : {self._scaler}")

        self.no_grad_sync_available = False

        self.compile_kwargs = compile_kwargs if compile_kwargs is not None else {}

        self._training_step_func = None
        self._optimizer_step_funcs = {}

        # Validate: AMP with multiple optimizers is not supported
        if self.use_mixed_precision and len(self.get_optimizers()) > 1:
            raise ValueError(
                "Mixed precision (AMP) with multiple optimizers is not supported. "
                "GradScaler tracking requires a single optimizer."
            )

    def _apply_model_wrapper_func(self, model):
        """
        Apply the model wrapper function with PyTorch-specific kwargs.

        Injects eager_mode and compile_kwargs so the wrapper can handle compilation
        before wrapping (e.g., compile model then wrap with DDP).

        :param model: The model to wrap

        :return: The wrapped model
        """
        return self._model_wrapper_func(
            model,
            eager_mode=self.eager_mode,
            compile_kwargs=self.compile_kwargs
        )

    def set_training_model(self, model):
        # Base class applies model wrapper if provided
        super().set_training_model(model)

        # Get the (possibly wrapped) model
        model = self.training_model

        # Detect DDP features for gradient sync optimization
        self.no_grad_sync_available = hasattr(model, 'no_sync') and callable(model.no_sync)

        # Determine training step function based on configuration
        if callable(self._model_wrapper_func):
            # It is assumed that the wrapper handled model compilation, training step runs in eager mode
            self._training_step_func = self._training_step
            self._log.info("Model wrapper provided, no additional model compilation performed.")
        elif self.eager_mode:
            self._training_step_func = self._training_step
            self._log.warn("Training in eager mode.")
        else:
            # No wrapper, not eager - compile the entire training step
            self._training_step_func = torch.compile(self._training_step, **self.compile_kwargs)
            self._log.info("Compiled training step.")

        # Setup optimizer step functions (compiled if not in eager mode)
        self._setup_optimizer_step_funcs()

    def set_learning_rate_for(self, optimizer_name, lr):
        """

        Set learning rate for specific optimizer `optimizer_name` to `lr`

        :param optimizer_name:
        :param lr:

        :return: True on success, else False
        """
        optimizer = self.get_optimizer(optimizer_name)
        if not hasattr(optimizer, 'param_groups'):
            self._log.error(f"No valid optimizer available with name {optimizer_name}, unable to set learning rate")
            return False

        try:
            for group in optimizer.param_groups:
                group['lr'] = lr
        except Exception as e:
            _.log_exception(self._log, f"Unable to set learning rate for optimizer {optimizer_name}", e)
            return False

        self._log.debug(f"Learning rate of optimizer {optimizer_name} set to : {lr}")

        return True

    def evaluate_loss(self, batch_data, inference_mode, evaluate_settings=None):

        if self.use_mixed_precision:
            self._activate_inference_mode(inference_mode)

            with autocast('cuda'):
                results = self._evaluate_loss(batch_data, evaluate_settings, inference_mode)

                return self.normalize_evaluation(results)
        else:
            return super().evaluate_loss(batch_data, inference_mode, evaluate_settings)

    # ========== Public Methods ==========

    def train_on(self, micro_batch, training_settings=None) -> Tuple[MicroBatchResults, bool]:
        """
        Use micro_batch to perform a training iteration.

        Note that the micro-batch data is simply the batch data when batch_size equals
        micro_batch_size (i.e. no gradient accumulation).

        The trainer accumulates gradients across multiple micro-batches (when
        gradient_accumulation_steps > 1) and steps the optimizer when the
        accumulation boundary is reached.

        :param micro_batch: Micro-batch data object to train on (e.g. dict, list, tuple)
        :param training_settings: Optional training settings object (usually dict)

        :return: Tuple (micro_batch_results, did_update)
            micro_batch_results: MicroBatchResults containing results for all micro-batches
                in the current accumulation window so far.
            did_update: Boolean indicating if optimizer stepped.
        """
        if not self.instance_valid():
            raise TrainerInvalidException()

        if not callable(self._training_step_func):
            raise TrainerStateInvalidException(
                "_training_step_func is not callable. "
                "You must first call `my_trainer.set_training_model(my_model)`"
            )

        self._accumulation_counter += 1
        is_boundary = (self._accumulation_counter >= self.gradient_accumulation_steps)

        # Use no_sync context OUTSIDE compiled function to avoid torch.compile issues
        # with context managers when using dynamic=True
        context = nullcontext() if is_boundary else self._get_no_sync_context()

        with context:
            # Core computation in compiled call (no context managers inside)
            results, did_update = self._training_step_func(micro_batch, training_settings, is_boundary)

        self._micro_batch_results.append(results)
        accumulated_results = MicroBatchResults(self._micro_batch_results)

        if is_boundary:
            self._reset_accumulation_state()

        return accumulated_results, did_update

    def epoch_complete(self) -> bool:
        """
        Called when an epoch ends. Steps optimizer if mid-accumulation.

        :return: did_update - True if optimizer was stepped, False otherwise.
        """
        if self._accumulation_counter == 0:
            return False

        # Force gradient sync and step optimizer with partial accumulation
        self._prepare_update_model_parameters()
        did_update = self._update_model_parameters()
        self._after_update_model_parameters(did_update)
        self._reset_accumulation_state()

        return did_update

    # ========== Protected Methods ==========

    def _reset_gradients(self):
        for optimizer in self.get_optimizers().values():
            optimizer.zero_grad()

    def _setup_optimizer_step_funcs(self):
        """
        Setup optimizer step functions for all optimizers.

        For AMP: wraps scaler.step(optimizer)
        For non-AMP: uses optimizer.step

        If not in eager mode, applies torch.compile with fullgraph=False
        to allow graph breaks at control flow points (e.g., inf/nan checking in GradScaler).
        """
        def make_scaler_step(opt, sc):
            def step():
                sc.step(opt)
            return step

        for name, optimizer in self.get_optimizers().items():
            if self.use_mixed_precision:
                step_func = make_scaler_step(optimizer, self._scaler)
            else:
                step_func = optimizer.step

            if not self.eager_mode:
                step_func = torch.compile(step_func, fullgraph=False)
                self._log.debug(f"Compiled optimizer step for '{name}'")

            self._optimizer_step_funcs[name] = step_func

    def _training_step(self, micro_batch, training_settings, is_boundary) -> Tuple[Dict, bool]:
        """
        Training step for a single micro-batch.

        Handles gradient calculation and optimizer step (at accumulation boundary).

        Compilation behavior:
        - If model_wrapper_func provided: This function runs in eager mode.
          The wrapper is responsible for model compilation (e.g., compile then wrap with DDP).
        - If no wrapper and not eager_mode: This entire function is compiled via torch.compile.

        Note: The no_sync() context is managed OUTSIDE this function (in train_on)
        to avoid torch.compile issues with context managers.

        :param micro_batch: Micro-batch data
        :param training_settings: Training settings
        :param is_boundary: Whether this is the accumulation boundary (optimizer step)

        :return: Tuple (results, did_update)
            results: Normalized results dict with 'loss', 'num_samples', 'auxiliary_results'
            did_update: Boolean indicating if optimizer stepped
        """
        results = self.evaluate_loss(
            micro_batch,
            inference_mode=False,
            evaluate_settings=training_settings
        )

        if 'loss' not in results:
            raise LossNotAvailableException()

        loss = results['loss']

        # Scale loss by accumulation factor
        scaled_loss = loss / self.gradient_accumulation_steps
        self._back_propagate_from(scaled_loss)

        # Reduce memory usage
        loss.detach_()

        did_update = False
        if is_boundary:
            self._prepare_update_model_parameters()
            did_update = self._update_model_parameters()
            self._after_update_model_parameters(did_update)

        return results, did_update

    def _get_no_sync_context(self):
        """Get DDP no_sync context if available, otherwise nullcontext."""
        if self.no_grad_sync_available:
            return self.training_model.no_sync()
        return nullcontext()

    def _back_propagate_from(self, loss):
        if self.use_mixed_precision:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def _prepare_update_model_parameters(self):
        pass

    def _update_model_parameters(self):
        did_update = True

        for name in self.get_optimizers().keys():
            self._optimizer_step_funcs[name]()

        # AMP scale tracking (single optimizer enforced in __init__)
        if self.use_mixed_precision:
            old_scale = self._scaler.get_scale()
            self._scaler.update()
            new_scale = self._scaler.get_scale()

            did_update = new_scale >= old_scale
            if not did_update:
                self._log.debug("Optimizer did not update, AMP scaling factor too high ...")

        # Reset gradients after optimizer step
        if did_update:
            self._reset_gradients()

        return did_update

    def _after_update_model_parameters(self, did_update):
        pass


class DefaultTrainer(MultiProcessingMixin, DefaultTrainerMixin):
    pass