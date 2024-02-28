import math
from typing import Optional, Dict

import torch

import contextlib

from torch.cuda.amp import autocast
import torch.distributed as dist

import basics.base_utils as _

from mlpug.mlpug_exceptions import TrainerInvalidException, LossNotAvailableException, TrainerStateInvalidException

from mlpug.trainers.training import TrainingManager as TrainingManagerBase
from mlpug.trainers.training import Trainer as TrainerBase
from mlpug.trainers.training import DefaultTrainer as DefaultTrainerBase

from mlpug.pytorch.utils import SlidingWindow
from mlpug.batch_chunking import BatchChunkingResults, is_chunkable, apply_chunkable_batch_wrapper

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

        self._compile_kwargs = compile_kwargs if compile_kwargs is not None else {}

        self._training_step_func = None

    def set_training_model(self, model):
        super().set_training_model(model)

        self.no_grad_sync_available = hasattr(model, 'no_sync') and callable(model.no_sync)

        if not self.eager_mode:
            self._training_step_func = torch.compile(self._training_step, **self._compile_kwargs)
        else:
            self._training_step_func = self._training_step
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

                           When `batch_chunk_size` is given, `batch_data` must be an object that implements the
                           `__len__` and `__getitem__` methods. Here the `__getitem__` method must be able to deal
                           with slices.

                           Alternatively, a chunkable_batch_wrapper can be provided at construction to convert
                           any batch into a chunkable batch

        :param training_settings: optional training_settings object (usually dict)

        :return: model_outputs

                 model_outputs is a
                    Single normalized results dict:
                        {'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>}
                 or
                    BatchChunkingResults: a list of tuples, one tuple per batch chunk results:
                        [{'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>},  # Chunk 1
                         ...
                         {'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>}]  # Chunk N

        :rtype: Union[Dict, BatchChunkingResults[Dict]]

        """

        if not self.instance_valid():
            raise TrainerInvalidException()

        if not callable(self._training_step_func):
            raise TrainerStateInvalidException("Training_step_func is not callable. "
                                               "You must first call` my_trainer.set_training_model(my_model)`")

        return self._training_step_func(batch_data, training_settings=training_settings)

    def _training_step(self, batch_data, training_settings=None):
        """
        See train_on, for documentation.
        This function will be compiled when not training in eager mode
        """

        self._reset_gradients()

        model_outputs = self._calc_gradients(batch_data, training_settings=training_settings)

        self._prepare_update_model_parameters()

        self._update_model_parameters()

        self._after_update_model_parameters()

        return model_outputs

    def _reset_gradients(self):
        for optimizer in self.get_optimizers().values():
            optimizer.zero_grad()

    def _calc_gradients(self, batch_data, training_settings=None):
        """

        :param batch_data:
        :type batch_data:
        :param training_settings:
        :type training_settings:

        :return: loss, model_outputs

                 model_outputs is a
                    Single normalized results dict:
                        {'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>}
                 or
                    BatchChunkingResults: a list of tuples, one tuple per batch chunk results:
                        [{'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>},  # Chunk 1
                         ...
                         {'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>}]  # Chunk N

        :rtype: Union[Dict, BatchChunkingResults[Dict]]

        :raises MLPugException, LossNotAvailableException
        """

        return self._calc_gradients_single_batch(batch_data, training_settings) \
            if not self.batch_chunk_size else self._calc_gradients_chunked(batch_data, training_settings)

    def _calc_gradients_single_batch(self, batch_data, training_settings=None):
        """

        :param batch_data:
        :param training_settings:

       :return: model_outputs a single normalized results dict:
                        {'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>}

        :rtype: Dict

        """
        results = self.evaluate_loss(batch_data,
                                     inference_mode=False,
                                     evaluate_settings=training_settings)

        if 'loss' not in results:
            raise LossNotAvailableException()

        loss = results['loss']

        self._back_propagate_from(loss)

        # Reduce memory usage
        loss.detach_()

        # single model output
        return results

    def _calc_gradients_chunked(self, batch_data, training_settings=None):
        """
        See `train_on` method.

        This method slices the `batch_data` in slices of size `self.batch_chunk_size`. For each slice the loss is
        calculated and the gradients are updated through back prop.

        :return: model_outputs: BatchChunkingResults: a list of tuples, one tuple per batch chunk results:
                    [{'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>},  # Chunk 1
                     ...
                     {'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>}]  # Chunk N

        :rtype: BatchChunkingResults[Dict]
        """

        if not is_chunkable(batch_data):
            batch_data = apply_chunkable_batch_wrapper(
                batch_data,
                self.chunkable_batch_wrapper)

        model_outputs = BatchChunkingResults()

        batch_size = len(batch_data)
        num_chunks = math.ceil(batch_size / self.batch_chunk_size)

        def process_chunk(chunk_idx, chunk_results):
            chunk_start = chunk_idx * self.batch_chunk_size
            chunk_end = min((chunk_idx + 1) * self.batch_chunk_size, batch_size)

            chunk_len = chunk_end - chunk_start

            chunk = batch_data[chunk_start:chunk_end]

            results = self.evaluate_loss(chunk, inference_mode=False, evaluate_settings=training_settings)

            if 'loss' not in results:
                raise LossNotAvailableException()

            loss = results['loss']

            # loss is assumed to be the average over the sample loss for the chunk
            # Divide through batch size to factor in that this loss is part of a larger batch.
            last_chunk = chunk_idx == (num_chunks - 1)
            self._back_propagate_from(chunk_len * loss / batch_size, last_chunk=last_chunk)

            # Reduce memory usage
            loss.detach_()

            chunk_results += [results]

            return chunk_results

        # Speed up processing by disabling gradient syncing for all batch chunk backward operations
        # except for the last
        no_sync = self.training_model.no_sync() if self.no_grad_sync_available else contextlib.suppress()

        with no_sync:
            for c_idx in range(num_chunks-1):
                model_outputs = process_chunk(c_idx, model_outputs)

        # sync gradients
        model_outputs = process_chunk(num_chunks-1, model_outputs)

        # model outputs for each chunk
        return model_outputs

    def _back_propagate_from(self, loss, last_chunk=False):
        if self.use_mixed_precision:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def _prepare_update_model_parameters(self):
        pass

    def _update_model_parameters(self):
        for optimizer in self.get_optimizers().values():
            self._execute_optimizer(optimizer)

    def _execute_optimizer(self, optimizer):
        if self.use_mixed_precision:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def _after_update_model_parameters(self):
        if self.use_mixed_precision:
            self._scaler.update()


class DefaultTrainer(MultiProcessingMixin, DefaultTrainerMixin):
    pass