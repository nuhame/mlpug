from contextlib import nullcontext
from typing import Optional, Dict, Tuple

import torch

from torch.cuda.amp import autocast
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
            compile_kwargs: Optional[Dict] = None,
            name="DefaultTrainer",
            **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)

        self._scaler = scaler

        if self.use_mixed_precision:
            if scaler is None:
                self._log.debug("Creating default scaler instance for automatic mixed precision ...")
                self._scaler = torch.cuda.amp.GradScaler()

            self._log.info(f"Using scalar instance for automatic mixed precision : {self._scaler}")

        self.no_grad_sync_available = False

        self.compile_kwargs = compile_kwargs if compile_kwargs is not None else {}

        self._calc_gradients_func = None

    def set_training_model(self, model):
        super().set_training_model(model)

        self.no_grad_sync_available = hasattr(model, 'no_sync') and callable(model.no_sync)

        if not self.eager_mode:
            self._calc_gradients_func = torch.compile(self._calc_gradients, **self.compile_kwargs)
        else:
            self._calc_gradients_func = self._calc_gradients
            self._log.warn("Training in eager mode.")

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

            with autocast():
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

        if not callable(self._calc_gradients_func):
            raise TrainerStateInvalidException(
                "_calc_gradients_func is not callable. "
                "You must first call `my_trainer.set_training_model(my_model)`"
            )

        self._accumulation_counter += 1
        is_boundary = (self._accumulation_counter >= self.gradient_accumulation_steps)

        # Compute gradients for this micro-batch
        results = self._calc_gradients_func(micro_batch, training_settings, sync_gradients=is_boundary)
        self._micro_batch_results.append(results)
        accumulated_results = MicroBatchResults(self._micro_batch_results)

        did_update = False
        if is_boundary:
            self._prepare_update_model_parameters()
            did_update = self._update_model_parameters()
            self._after_update_model_parameters(did_update)
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

    def _calc_gradients(self, micro_batch, training_settings, sync_gradients: bool) -> Dict:
        """
        Compute gradients for a single micro-batch.

        :param micro_batch: Micro-batch data
        :param training_settings: Training settings
        :param sync_gradients: If True, allow gradient synchronization (DDP).
            If False, use no_sync context to skip synchronization.

        :return: Normalized results dict with 'loss', 'num_samples', 'auxiliary_results'
        """
        # Use no_sync context when not at accumulation boundary (DDP optimization)
        context = self._get_no_sync_context() if not sync_gradients else nullcontext()

        with context:
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

        return results

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
        for name, optimizer in self.get_optimizers().items():
            optimizer_did_update = self._execute_optimizer(optimizer)
            if not optimizer_did_update:
                self._log.debug(f"Optimizer '{name}' did not update, AMP scaling factor too high ...")

            did_update &= optimizer_did_update

        # Reset gradients after optimizer step
        if did_update:
            self._reset_gradients()

        return did_update

    def _execute_optimizer(self, optimizer) -> bool:
        did_update = True
        if self.use_mixed_precision:
            self._scaler.step(optimizer)

            old_scale = self._scaler.get_scale()
            self._scaler.update()
            new_scale = self._scaler.get_scale()

            did_update = new_scale >= old_scale
        else:
            optimizer.step()

        return did_update

    def _after_update_model_parameters(self, did_update):
        pass


class DefaultTrainer(MultiProcessingMixin, DefaultTrainerMixin):
    pass