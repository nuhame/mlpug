import torch

from torch.cuda.amp import autocast
import torch.distributed as dist

from functools import reduce

import basics.base_utils as _

from mlpug.trainers.training import *
from mlpug.trainers.training import TrainingManager as TrainingManagerBase

from mlpug.mlpug_exceptions import TrainerInvalidException, BatchNotChunkableException, LossNotAvailableException
from mlpug.pytorch.utils import is_chunkable, SlidingWindow

from mlpug.pytorch.multi_processing import MultiProcessingMixin


class TrainingManager(MultiProcessingMixin, TrainingManagerBase):
    def __init__(self, *args, sliding_window_factory=SlidingWindow, **kwargs):
        super().__init__(*args, sliding_window_factory=sliding_window_factory, **kwargs)

    def _training_ended(self):
        if self.is_distributed:
            # Wait for all processes to finish
            dist.barrier()


class PTTrainerMixin(MultiProcessingMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


class Trainer(PTTrainerMixin, TrainerBase):
    pass


class DefaultTrainer(PTTrainerMixin, DefaultTrainerBase):

    def __init__(self, *args, scaler=None, name="DefaultTrainer", **kwargs):
        super(DefaultTrainer, self).__init__(*args, name=name, **kwargs)

        self._scaler = scaler

        if self.use_mixed_precision:
            if scaler is None:
                self._log.debug("Creating default scaler instance for automatic mixed precision ...")
                self._scaler = torch.cuda.amp.GradScaler()

            self._log.info(f"Using scaler instance for automatic mixed precision : {self._scaler}")

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
                return normalize_evaluation(results)
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
        :param training_settings: optional training_settings object (usually dict)

        :return: loss, auxiliary_results

        loss : number (e.g. float)
        auxiliary_results : can be anything, e.g dict or list with values or data items
        """

        if not self.instance_valid():
            raise TrainerInvalidException()

        self._reset_gradients()

        loss, auxiliary_results = self._calc_gradients(batch_data, training_settings=training_settings)

        self._prepare_update_model_parameters()

        self._update_model_parameters()

        self._after_update_model_parameters()

        return loss, auxiliary_results

    def _reset_gradients(self):
        for optimizer in self.get_optimizers().values():
            optimizer.zero_grad()

    def _calc_gradients(self, batch_data, training_settings=None):
        """

        :param batch_data:
        :type batch_data:
        :param training_settings:
        :type training_settings:
        :return:
        :rtype:

        :raises LossNotAvailableException
        """

        if not self.batch_chunk_size:
            results = self.evaluate_loss(batch_data,
                                         inference_mode=False,
                                         evaluate_settings=training_settings)

            if 'loss' not in results:
                raise LossNotAvailableException()

            loss = results['loss']
            auxiliary_results = get_value_at('auxiliary_results', results, warn_on_failure=False)

            self._back_propagate_from(loss)
        else:
            chunk_losses, chunk_aux_results, chunk_lengths = self._calc_gradients_chunked(batch_data, training_settings)

            loss, auxiliary_results = self._combine_chunk_results(chunk_losses, chunk_aux_results, chunk_lengths)

        return loss, auxiliary_results

    def _calc_gradients_chunked(self, batch_data, training_settings=None):
        """
        See `train_on` method.

        This method slices the `batch_data` in slices of size `self.batch_chunk_size`. For each slice the loss is
        calculated and the gradients are updated through back prop.

        return: chunk_losses, chunk_aux_results, chunk_lengths
                All three outputs are lists
        """

        if not is_chunkable(batch_data):
            raise BatchNotChunkableException()

        chunk_losses = []
        chunk_aux_results = BatchChunkingResults()
        chunk_lengths = []

        batch_size = len(batch_data)
        num_chunks = math.ceil(batch_size / self.batch_chunk_size)
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx*self.batch_chunk_size
            chunk_end = min((chunk_idx+1)*self.batch_chunk_size, batch_size)

            chunk_len = chunk_end-chunk_start

            chunk = batch_data[chunk_start:chunk_end]

            results = self.evaluate_loss(chunk, inference_mode=False, evaluate_settings=training_settings)

            if 'loss' not in results:
                raise LossNotAvailableException()

            loss = results['loss']
            aux_results = get_value_at('auxiliary_results', results, warn_on_failure=False)

            # loss is assumed to be the average over the sample loss for the chunk
            # Divide through batch size to factor in that this loss is part of a larger batch.
            last_chunk = chunk_idx == (num_chunks-1)
            self._back_propagate_from(chunk_len*loss/batch_size, last_chunk=last_chunk)

            chunk_losses += [loss]
            chunk_aux_results += [aux_results]
            chunk_lengths += [chunk_len]

        return chunk_losses, chunk_aux_results, chunk_lengths

    def _combine_chunk_results(self, chunk_losses, chunk_aux_results, chunk_lengths):
        """
        This default implementation assumes that the loss for a chunk is the average loss of all samples in the chunk.
        There is no specific combination logic to combine the chunk auxiliary results

        :returns loss, auxiliary_results
                    loss: weighted average of chunk losses
                    auxiliary_results: list of dicts:
                                        [
                                            ...
                                            {
                                                "results": chunk aux. results,
                                                "num_samples": num samples in chunk
                                            }
                                            ...
                                        ]
        """

        loss = reduce(lambda tot, c: tot+(c[1]*c[0]), zip(chunk_losses, chunk_lengths), 0)
        num_samples = reduce(lambda tot, l: tot+l, chunk_lengths, 0)

        loss /= num_samples

        auxiliary_results = [{
            "results": aux_results,
            "num_samples": chunk_length
        } for aux_results, chunk_length in zip(chunk_aux_results, chunk_lengths)]

        auxiliary_results = BatchChunkingResults(auxiliary_results)

        return loss, auxiliary_results

    def _back_propagate_from(self, loss, last_chunk=False):
        if self.use_mixed_precision:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def _prepare_update_model_parameters(self):
        pass

    def _update_model_parameters(self):
        for optimizer in self.get_optimizers().values():
            if self.use_mixed_precision:
                self._scaler.step(optimizer)
            else:
                optimizer.step()

    def _after_update_model_parameters(self):
        if self.use_mixed_precision:
            self._scaler.update()
