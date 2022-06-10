import math
import torch

from torch.cuda.amp import autocast
import torch.distributed as dist

from functools import reduce

import basics.base_utils as _

from mlpug.utils import get_value_at, has_key

from mlpug.trainers.training import normalize_evaluation, extend_auxiliary_results
from mlpug.trainers.training import TrainingManager as TrainingManagerBase
from mlpug.trainers.training import Trainer as TrainerBase
from mlpug.trainers.training import DefaultTrainer as DefaultTrainerBase

from mlpug.mlpug_exceptions import MLPugException, TrainerInvalidException, BatchNotChunkableException, LossNotAvailableException
from mlpug.pytorch.utils import is_chunkable, SlidingWindow

from mlpug.pytorch.multi_processing import MultiProcessingMixin
from mlpug.batch_chunking import BatchChunkingResults


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

        self._num_samples_provided_by_user = None

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

        :return: loss, model_outputs

                 model_outputs is a
                    List with single tuple:
                        [(loss, auxiliary_results, num_samples)]
                 or
                    BatchChunkingResults: a list of tuples, one tuple per batch chunk results:
                        [(loss, auxiliary_results, num_samples),  # Chunk 1
                         ...
                         (loss, auxiliary_results, num_samples)]  # Chunk N

        :rtype: Tuple[Tensor, Union[List[Tuple], BatchChunkingResults[Tuple]]

        """

        if not self.instance_valid():
            raise TrainerInvalidException()

        self._reset_gradients()

        loss, model_outputs = self._calc_gradients(batch_data, training_settings=training_settings)

        self._prepare_update_model_parameters()

        self._update_model_parameters()

        self._after_update_model_parameters()

        return loss, model_outputs

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
                    List with single tuple:
                        [(loss, auxiliary_results, num_samples)]
                 or
                    BatchChunkingResults: a list of tuples, one tuple per batch chunk results:
                        [(loss, auxiliary_results, num_samples),  # Chunk 1
                         ...
                         (loss, auxiliary_results, num_samples)]  # Chunk N

        :rtype: Tuple[Tensor, Union[List[Tuple], BatchChunkingResults[Tuple]]

        :raises MLPugException, LossNotAvailableException
        """

        return self._calc_gradients_single_batch(batch_data, training_settings) \
            if not self.batch_chunk_size else self._calc_gradients_chunked(batch_data, training_settings)

    def _calc_gradients_single_batch(self, batch_data, training_settings=None):
        results = self.evaluate_loss(batch_data,
                                     inference_mode=False,
                                     evaluate_settings=training_settings)

        if 'loss' not in results:
            raise LossNotAvailableException()

        loss = results['loss']
        auxiliary_results = get_value_at('auxiliary_results', results, warn_on_failure=False)

        # At this point we don't know how to evaluate bacth size.
        # So, by default we are assuming that each batch has the same size
        num_samples = 1
        if has_key(auxiliary_results, 'num_samples'):
            if self._num_samples_provided_by_user is None:
                self._num_samples_provided_by_user = True

            # Use the batch size provided by the user
            num_samples = auxiliary_results['num_samples']

        elif self._num_samples_provided_by_user is True:
            raise MLPugException("Unexpected: user has provided the num_samples in a batch before through "
                                 "auxiliary_results['num_samples'], however in this batch, the num_samples are "
                                 "not provided by the user.")

        self._back_propagate_from(loss)

        # Reduce memory usage
        loss.detach_()

        # loss, and single model output
        return loss, [(loss, auxiliary_results, num_samples)]

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

        model_outputs = BatchChunkingResults()

        num_samples_warning_given = False

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

            if has_key(aux_results, 'num_samples') and not num_samples_warning_given:
                self._log.warning("The given 'num_samples' in auxiliary_results will not be used when "
                                  "chunking batches for gradient accumulation, because we need the total number of "
                                  "samples of the complete, unchunked, batch to be known beforehand."
                                  "The chunk length will be used, instead of the given num_samples.")
                num_samples_warning_given = True

            # loss is assumed to be the average over the sample loss for the chunk
            # Divide through batch size to factor in that this loss is part of a larger batch.
            last_chunk = chunk_idx == (num_chunks-1)
            self._back_propagate_from(chunk_len*loss/batch_size, last_chunk=last_chunk)

            model_outputs += (loss.detach_(),  # Detach from graph to reduce memory
                              aux_results,
                              chunk_len)

        loss = reduce(lambda tot, mo: tot + (mo[2] * mo[0]), model_outputs, 0)
        num_samples = reduce(lambda tot, mo: tot + mo[2], model_outputs, 0)

        loss /= num_samples

        # loss, and model outputs for each chunk
        return loss, model_output

    def _combine_chunk_results(self, chunk_losses, chunk_aux_results, chunk_lengths):
        """
        This default implementation assumes that the loss for a chunk is the average loss of all samples in the chunk.
        There is no specific combination logic to combine the chunk auxiliary results

        :returns loss, auxiliary_results
                    loss: weighted average of chunk losses
                    auxiliary_results: list of dicts or tuples extended with number of samples
                                       used per chunk ("chunk_length"):
                                        [
                                            ...

                                            {
                                                ... chunk aux. results ...,
                                                "chunk_length": num samples in chunk
                                            }
                                            or
                                            (... chunk aux. results ..., <chunk_length>)

                                            ...
                                        ]
        """

        loss = reduce(lambda tot, c: tot+(c[1]*c[0]), zip(chunk_losses, chunk_lengths), 0)
        num_samples = reduce(lambda tot, l: tot+l, chunk_lengths, 0)

        loss /= num_samples

        auxiliary_results = [extend_auxiliary_results(aux, chunk_length, key="chunk_length")
                             for aux, chunk_length in zip(chunk_aux_results, chunk_lengths)]

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
