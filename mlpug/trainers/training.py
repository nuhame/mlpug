import os
import sys
import abc
from typing import Any, Optional, Tuple

import math
import time

from statistics import mean

from mlpug.base import Base
import basics.base_utils as _
from basics.logging import get_logger

from mlpug.utils import convert_to_dict, get_value_at, set_value_at, has_key, SlidingWindow

from mlpug.mlpug_exceptions import TrainerInvalidException, ModelWrapperException
from mlpug.trainers.model_wrappers import ModelWrapperFunc


logger = get_logger(os.path.basename(__file__))


class MicroBatchResults(list):
    """
    List of results from micro-batches within an accumulation window.

    Each element is a normalized results dict with keys:
        - 'loss': The loss tensor for the micro-batch
        - 'num_samples': Number of samples in the micro-batch
        - 'auxiliary_results': Any additional results from the model
    """
    pass


class NormalizeEvaluationResults(Base):

    NUM_SAMPLES_WARNING_MESSAGE = "You need to return the number of samples used in your batch. " \
                                  "A default of num_samples=1 will be added to your results, " \
                                  "this will result in correct average metrics if " \
                                  "all batches have the same sample size."

    def __init__(self, do_warn=True, name=None):
        super().__init__(pybase_logger_name=name)

        self.did_warn = not do_warn

    def __call__(self, results):
        """

        Ensures that the training model evaluation results are normalized as a dict
        containing the required loss and num_samples, and the optional auxiliary_results value

        :param results: Dict, Tuple or only single Tensor loss value

        :return: Normalized results dict
                 {
                    'loss': <loss tensor>,
                    'num_samples': <int>,
                    'auxiliary_results': <Any>
                }

        :raises ValueError when the loss value is missing

        """
        if type(results) is dict:
            if 'loss' not in results:
                raise ValueError("Your training model evaluation results dict must contain a 'loss' key and value")

            if 'num_samples' not in results:
                results['num_samples'] = 1
                if not self.did_warn:
                    self.did_warn = True
                    self._log.warning(
                        "'num_samples' key and value not found in your training model evaluation results. " +
                        self.NUM_SAMPLES_WARNING_MESSAGE
                    )

            if 'auxiliary_results' not in results:
                results['auxiliary_results'] = None

            return results
        elif type(results) is tuple:
            if len(results) == 0:
                raise ValueError("Results are empty: your training model evaluation results tuple "
                                 "must at least contain a loss value.")

            loss = results[0]

            if len(results) > 1:
                num_samples = results[1]
            else:
                num_samples = 1
                if not self.did_warn:
                    self.did_warn = True
                    logger.warning("No second value found in your training model evaluation results tuple, "
                                   "representing the num_samples in your batch. " + self.NUM_SAMPLES_WARNING_MESSAGE)

            auxiliary_results = None
            if len(results) > 2:
                auxiliary_results = results[2:]

                if len(auxiliary_results) == 1:
                    # If exactly three tuple values were given, the third value is assumed to be the auxiliary results
                    auxiliary_results = auxiliary_results[0]

            return {
                "loss": loss,
                "num_samples": num_samples,
                "auxiliary_results": auxiliary_results
            }
        else:
            # Assuming only the loss was returned
            if not self.did_warn:
                self.did_warn = True
                logger.warning("No second value found in your training model evaluation results, "
                               "representing the num_samples in your batch. " + self.NUM_SAMPLES_WARNING_MESSAGE)

            return {
                "loss": results,
                "num_samples": 1,
                "auxiliary_results": None
            }


def calc_gradient_accumulation_steps(batch_size, micro_batch_size):
    """
    Calculate gradient accumulation steps from batch_size and micro_batch_size.

    :param batch_size: The effective batch size (semantic batch for metrics/logging).
        Can be None if no accumulation is needed.
    :param micro_batch_size: The micro-batch size (what fits in GPU memory).
        Can be None if no accumulation is needed.

    :return: Tuple of (batch_size, micro_batch_size, gradient_accumulation_steps).
        All values are normalized (None values are resolved).

    :raises ValueError: If micro_batch_size is provided without batch_size,
        or if batch_size < micro_batch_size, or if batch_size is not divisible
        by micro_batch_size.
    """
    if batch_size is None and micro_batch_size is None:
        # No accumulation
        return None, None, 1
    elif batch_size is not None and micro_batch_size is None:
        # batch_size given, assume micro_batch_size = batch_size (no accumulation)
        return batch_size, batch_size, 1
    elif micro_batch_size is not None and batch_size is None:
        # micro_batch_size given without batch_size - cannot calculate accumulation steps
        raise ValueError(
            "micro_batch_size is provided but batch_size is not. "
            "Cannot calculate gradient_accumulation_steps. "
            "Please provide batch_size or omit micro_batch_size."
        )
    else:
        # Both provided - validate and calculate
        if batch_size < micro_batch_size:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= micro_batch_size ({micro_batch_size})"
            )
        if batch_size % micro_batch_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by micro_batch_size ({micro_batch_size})"
            )
        return batch_size, micro_batch_size, batch_size // micro_batch_size


class TrainingManager(Base, metaclass=abc.ABCMeta):

    def __init__(self,
                 trainer,
                 training_dataset,
                 num_epochs=50,
                 num_batches_per_epoch=None,
                 callbacks=None,
                 experiment_data=None,
                 sliding_window_factory=None,
                 **kwargs):
        """
        Implements a simple training loop in the `_train` method. Although simple in nature, the use of callbacks
        (see the `Callback` class), provide the means to monitor, control and customize the training.

        During each epoch a new `logs` dict is created that is passed from callback event to callback event during
        batch-wise training. Before calling the given `Trainer` instance to train on a batch, `training_settings` are
        retrieved from the `logs` dict, if available. These `training_settings` are then passed to the trainer `train_on`
        method. Also see the `Trainer` class.

        The provided `training_dataset` should be iterable, and provide a batch per iteration.

        :param trainer:
        :param training_dataset:
        :param num_epochs:
        :param num_batches_per_epoch:
        :param callbacks:
        :param experiment_data: Data object that will be added to the training manager state.
                                This could contain additional training/model experiment data
        :param sliding_window_factory: Optional. Function to create SlidingWindow instances

        """
        super(TrainingManager, self).__init__(**kwargs)

        self.trainer = trainer

        # Must implement __len__, __iter__
        self.training_dataset = training_dataset

        self.num_epochs = num_epochs
        self.num_batches_per_epoch = num_batches_per_epoch

        self.global_iter = 0
        self.micro_batch_step = 0  # Counts dataloader iterations (micro-batches)
        self.batch_step = 0  # Counts effective batches (after accumulation)
        self.epoch = 0

        # Will be set in _assess_num_batches_per_epoch
        self.num_micro_batches_per_epoch = None

        self.callbacks = callbacks or []
        self.cb_names = None
        self.cb_hashes = None

        self.experiment_data = experiment_data

        self.sliding_window_factory = SlidingWindow if not callable(sliding_window_factory) else sliding_window_factory

        self.init_logs = None
        self.logs = {
            "cb_calls_success": True
        }

        self._previous_batch_duration = None

        self._metric_windows = {}

        self._stop_training = False

        if not self._validate():
            return

        self._assess_num_batches_per_epoch()

        try:
            # TODO : better deal with num_batches_per_epoch special cases (e.g. Numpy number)
            # TODO : deal with the case when we don't have SlidingWindows
            if float(self.num_batches_per_epoch) != math.inf and self.num_batches_per_epoch > 0:
                self._metric_windows['training_params.batch.duration'] = \
                    self.sliding_window_factory(length=self.num_batches_per_epoch,
                                                name='training_params.batch.duration')

        except Exception as e:
            self._valid = False
            _.log_exception(self._log,
                            "Unable to setup sliding window for training loss and batch training duration", e)

        # populates cb_names and cb_hashes
        self._check_callback_hashes()

        if not self._call_callbacks("set_training_manager", self):
            self._valid = False
            self.logs["cb_calls_success"] = False
            self._log.error("One or more issues occurred while providing this training manager instance "
                            "to the provided callbacks")

    def get_trainer(self):
        return self.trainer

    def start_training(self):
        if not self.instance_valid():
            self._log.error('TrainingManager is not valid, unable to start training')
            return

        try:
            self._train()
        except KeyboardInterrupt:
            self._update_cb_success(self._call_callbacks('on_training_ended',
                                                         stopped_early=False,
                                                         stopped_on_error=False,
                                                         interrupted=True,
                                                         callback_calls_success=self.logs["cb_calls_success"]))

            sys.stdout.write("\n")
            self._log.info('Training process interrupted by you ... 🤷🏻‍♂️\n')
            sys.stdout.flush()

            self.stop_training()

        self._training_ended()

    def stop_training(self):
        self._stop_training = True

    def get_state(self):
        """

        :return: state, success
        """

        if not self.instance_valid():
            self._log.error('TrainingManager is not valid, unable to get training manager state')
            return None, False

        state = {
            "manager": {
                "epoch": self.epoch,
                "micro_batch_step": self.micro_batch_step,
                "batch_step": self.batch_step,
                "global_iter": self.global_iter,
                "logs": self.logs,
                "metric_windows": self._get_metric_windows_state()
            },

            "callbacks": {},

            "experiment_data": self.experiment_data
        }

        success = True
        for cb_idx, callback in enumerate(self.callbacks):
            cb_name = self.cb_names[cb_idx]
            cb_hash = self.cb_hashes[cb_idx]

            try:
                cb_state, cb_success = callback.get_state()

                success &= cb_success

                if cb_hash in state["callbacks"] and cb_state != state["callbacks"][cb_hash]:
                    self._log.error(f"There is already a callback {cb_name} in the TrainingManager state "
                                    f"with exactly the same hash, this state will be overridden now. "
                                    f"Please ensure that all your callbacks have a unique names/hashes.\n"
                                    f"Callback index = {cb_idx}\n"
                                    f"Callback hash  = {cb_hash}\n")
                    success = False

                state["callbacks"][cb_hash] = cb_state
            except Exception as e:
                _.log_exception(self._log, f"Failed to get state of callback {cb_name}, "
                                           f"unable to add callback state to Training Manager state.\n"
                                           f"Callback index = {cb_idx}\n"
                                           f"Callback hash  = {cb_hash}\n", e)
                success = False

        state["trainer"], get_trainer_state_success = self.trainer.get_state()
        success &= get_trainer_state_success

        return state, success

    def get_state_for_model_checkpoint(self):

        if not self.instance_valid():
            self._log.error('TrainingManager is not valid, unable to get training manager state for model checkpoint')
            return None, False

        return {
                   "epoch": self.epoch,
                   "micro_batch_step": self.micro_batch_step,
                   "batch_step": self.batch_step,
                   "global_iter": self.global_iter,
                   "logs": self.logs,
                   "experiment_data": self.experiment_data
               }, True

    def set_state(self, state, allow_missing_callbacks=False):
        """
        WARNING: the start batch_step in the state only controls how many batches will be trained on in the
                 the current epoch, it does not control the exact batch data that will be used for training.

        :param state:
        :param allow_missing_callbacks:
        :return: success (True or False)
        """
        if not self.instance_valid():
            self._log.error('TrainingManager is not valid, unable to set training manager state')
            return False

        if not self._check_state(state):
            self._log.error("Invalid state object, unable to set state")
            return False

        self.init_logs = state["manager"]["logs"]

        self.epoch = state["manager"]["epoch"]
        self.batch_step = state["manager"]["batch_step"]
        self.global_iter = state["manager"]["global_iter"]

        # Backwards compatibility: old checkpoints may not have micro_batch_step
        # In that case, estimate from batch_step and accumulation_steps
        accumulation_steps = getattr(self.trainer, 'gradient_accumulation_steps', 1)
        if "micro_batch_step" in state["manager"]:
            self.micro_batch_step = state["manager"]["micro_batch_step"]
        else:
            # Estimate: assume we're at the end of the accumulation window
            self.micro_batch_step = self.batch_step * accumulation_steps

        # start at next iteration
        self.global_iter += 1
        self.micro_batch_step += 1
        self.batch_step += 1
        if self.micro_batch_step >= self.num_micro_batches_per_epoch:
            self.epoch += 1
            self.micro_batch_step = 0
            self.batch_step = 0

        self.init_logs["epoch"] = self.epoch
        self.init_logs["micro_batch_step"] = self.micro_batch_step
        self.init_logs["batch_step"] = self.batch_step
        self.init_logs["global_iter"] = self.global_iter

        success = self._set_metric_windows_states(state)

        # TODO : legacy from <= 0.0.14
        if "training_loss_window" in state["manager"]:
            self._log.debug("Loading deprecated metric window state training_loss_window ...")
            self._set_metric_window_state("training.batch.loss", state["manager"]["training_loss_window"])

        if "batch_duration_window" in state["manager"]:
            self._log.debug("Loading deprecated metric window state batch_duration_window ...")
            self._set_metric_window_state("training_params.batch.duration", state["manager"]["batch_duration_window"])

        try:
            callbacks_map = {cb_hash: callback for cb_hash, callback in zip(self.cb_hashes, self.callbacks)}
            callback_names = {cb_hash: cb_name for cb_hash, cb_name in zip(self.cb_hashes, self.cb_names)}

            callbacks_state = state["callbacks"] or {}
            for cb_hash, cb_state in callbacks_state.items():
                if cb_hash not in callbacks_map and not allow_missing_callbacks:
                    self._log.warn(f"No registered callback found for available callback state in "
                                   f"Training Manager state. Unable to set callback state, skipping...\n"
                                   f"Callback state hash = {cb_hash}")
                    continue

                callback = callbacks_map[cb_hash]
                cb_name = callback_names[cb_hash]
                try:
                    success &= callback.set_state(cb_state)
                except Exception as e:
                    _.log_exception(self._log, f"Unable to set state for callback {cb_name}. "
                                               f"Callback hash = {cb_hash}", e)
                    success = False

        except Exception as e:
            _.log_exception(self._log, f"Unable to create callbacks map, unable to set callback state", e)
            success = False

        try:
            success &= self.trainer.set_state(state["trainer"])
        except Exception as e:
            _.log_exception(self._log, f"Unable to set state for trainer", e)
            success = False

        return success

    def _check_callback_hashes(self):
        """
        Check if we can get the names and hashes of the registered callbacks
        Store the names and hashes for later use.
        """
        self.cb_names = []
        self.cb_hashes = []
        max_cb_name_len = -1
        for cb_idx, callback in enumerate(self.callbacks):
            cb_name = "[UNKNOWN]"
            try:
                cb_name = callback.get_name()
            except Exception as e:
                _.log_exception(self._log, f"Failed to get callback name (callback index {cb_idx}), "
                                           f"this is not good, but it is not fatal ...", e)

            try:
                cb_hash = callback.get_hash()
            except Exception as e:
                raise Exception(f"Failed to get hash of callback {cb_name} (callback index {cb_idx}), "
                                f"unable to set callback state from Training Manager state") from e

            len_cb_name = len(cb_name)
            if max_cb_name_len < len_cb_name:
                max_cb_name_len = len_cb_name

            self.cb_names += [cb_name]
            self.cb_hashes += [cb_hash]

        self._log.debug(f"Hashes of registered callbacks:")
        for cb_idx, callback in enumerate(self.callbacks):
            self._log.debug(f"{cb_idx}\t: {self.cb_names[cb_idx]:{max_cb_name_len}}\t: {self.cb_hashes[cb_idx]}")

    def _get_metric_windows_state(self):
        state = {}
        for metric_path, metric_window in self._metric_windows.items():
            state[metric_path] = metric_window.get_state()

        return state

    def _set_metric_windows_states(self, state):
        success = True
        try:
            metric_windows_state = get_value_at("manager.metric_windows", state)
            if metric_windows_state is None:
                return success

            for metric_path, window_state in metric_windows_state.items():
                success &= self._set_metric_window_state(metric_path, window_state)

        except Exception as e:
            _.log_exception(self._log, f"Unable to set metric windows state, skipped ...", e)
            success = False

        return success

    def _set_metric_window_state(self, metric_path, window_state):
        success = True
        try:
            window_length = window_state['length']

            if self.num_batches_per_epoch and (window_length != self.num_batches_per_epoch):
                # override when different number of batches per epoch is given (or calculated)
                # during construction
                window = self.sliding_window_factory(length=self.num_batches_per_epoch,
                                                     init_window_values=window_state['window'],
                                                     name=metric_path)
            else:
                window = self.sliding_window_factory(state=window_state)

            self._metric_windows[metric_path] = window
        except Exception as e:
            _.log_exception(self._log, f"Unable to set metric window state for {metric_path}, skipped ...", e)
            success = False

        return success

    def _check_state(self, state):
        state_attributes = ['manager',
                            'manager.epoch',
                            'manager.batch_step',
                            'manager.global_iter',
                            'manager.logs',
                            'manager.metric_windows',
                            'callbacks',
                            'trainer']

        for attr in state_attributes:
            v = get_value_at(attr, state, warn_on_failure=False)

            if v is None:
                if attr == "manager.metric_windows":
                    v = get_value_at('manager.training_loss_window', state, warn_on_failure=False)
                    if v is not None:
                        self._log.debug("Legacy state detected (training_loss_window) ... ")
                        continue

                self._log.error(f"Given state does not have a value for {attr}, state is invalid")
                return False

        return True

    def _train(self):
        training_stopped_early = False
        epoch_stopped_early = False
        training_stopped_on_error = False

        if self.init_logs is not None:
            self._log.debug("Setting provided initial log object ...")

            cb_calls_success = self.logs["cb_calls_success"]
            self.logs = self.init_logs
            self.logs["cb_calls_success"] &= cb_calls_success

            self.init_logs = None

        # override log state for certain parameters given at construction
        self.logs["final_epoch"] = self.num_epochs - 1
        self.logs["final_micro_batch_step"] = self.num_micro_batches_per_epoch - 1
        self.logs["final_batch_step"] = self.num_batches_per_epoch - 1

        update = self._update_cb_success
        call_cb = self._call_callbacks

        update(call_cb('on_training_start',
                       self.num_epochs,
                       self.num_batches_per_epoch,
                       self.epoch,
                       self.batch_step,
                       self.global_iter))

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch

            if self._stop_training:
                training_stopped_early = True
                self._log.warn("Training stopped early")
                break

            # Keep anything in the logs from the previous epoch that is non-standard
            # Reset all other values
            self.logs = {**self.logs, **{
                "epoch": self.epoch,
                "current": None,
            }}

            training_dataset = self._prepare_training_dataset()

            logs = self.logs  # just a bit shorter
            current = None
            epoch_started = True
            epoch_stopped_early = False
            # Track effective batch timing (across all micro-batches in accumulation window)
            batch_training_start_time = None
            for training_micro_batch in training_dataset:
                # Start timing at beginning of accumulation window (first micro-batch)
                if batch_training_start_time is None:
                    batch_training_start_time = time.time()

                if self._stop_training:
                    # Epoch was not finished yet
                    epoch_stopped_early = True
                    self._log.warn("Training epoch stopped early")
                    break

                # We are at the start of a new micro-batch iteration: reset current dict
                current = self._init_current_logs()
                logs["current"] = current

                # on_epoch_start needs to be called in the loop, because some callbacks might need the current object
                if epoch_started:
                    update(call_cb('on_epoch_start', self.logs))
                    epoch_started = False

                update(call_cb('on_batch_training_start', training_micro_batch, logs))

                # Previous batch duration is only available on_batch_training_start
                # In this way it is not double recorded in Tensorboard
                set_value_at("training_params.batch.duration", current, None)
                try:
                    # Ensure training_settings exists for callbacks to modify
                    if not has_key(logs, "training_settings"):
                        logs["training_settings"] = {}

                    model_outputs, did_update_model = self.trainer.train_on(
                        training_micro_batch, logs["training_settings"]
                    )

                    set_value_at("training.batch.raw.model_outputs", current, model_outputs)
                    set_value_at("training.batch.did_update_model", current, did_update_model)
                except Exception as e:
                    if isinstance(e, TrainerInvalidException):
                        err_msg = f"Trainer {self.trainer} is misconfigured, unable to train on batch, " \
                                  f"will stop training ..."
                    else:
                        err_msg = f"Exception occurred while calling {self.trainer} to train on batch, " \
                                  f"will stop training ..."

                    _.log_exception(self._log, err_msg, e)

                    update(call_cb('on_batch_training_failed', e, logs))
                    training_stopped_on_error = True

                if training_stopped_on_error:
                    break

                # Fire micro-batch callback after every train_on call
                update(call_cb('on_micro_batch_completed', training_micro_batch, logs))

                # Fire batch callback only at accumulation boundary (semantic batch)
                if did_update_model:
                    # Calculate effective batch duration (time for all micro-batches)
                    batch_duration = time.time() - batch_training_start_time
                    set_value_at("training_params.batch.duration", current, batch_duration)
                    self._update_window("training_params.batch.duration")
                    self._previous_batch_duration = batch_duration
                    # Reset for next effective batch
                    batch_training_start_time = None

                    update(call_cb('on_batch_training_completed', training_micro_batch, logs))
                    self.batch_step += 1
                    self.global_iter += 1

                self.micro_batch_step += 1

                # Situation 1: When the start micro_batch_step given was > 0, go to next epoch when epoch finished
                # Situation 2: When num_batches_per_epoch was given to simulate epochs in an
                #              infinite batch streaming situation
                if self.micro_batch_step >= self.num_micro_batches_per_epoch:
                    break

            if not (epoch_stopped_early or training_stopped_on_error):
                # Handle any partial accumulation at epoch end
                try:
                    did_final_update = self.trainer.epoch_complete()
                    if did_final_update:
                        # Final partial batch was processed - fire callbacks
                        # Calculate duration for the partial batch
                        if batch_training_start_time is not None:
                            batch_duration = time.time() - batch_training_start_time
                            set_value_at("training_params.batch.duration", current, batch_duration)
                            self._update_window("training_params.batch.duration")
                            self._previous_batch_duration = batch_duration

                        set_value_at("training.batch.did_update_model", current, did_final_update)
                        update(call_cb('on_batch_training_completed', training_micro_batch, logs))
                        self.batch_step += 1
                        self.global_iter += 1
                except Exception as e:
                    _.log_exception(self._log, "Exception occurred during epoch_complete()", e)

                mean_batch_duration = self._calc_window_average("training_params.batch.duration")
                epoch_duration = self._calc_window_sum("training_params.batch.duration")

                set_value_at("training_params.sliding_window.duration", current, mean_batch_duration)
                set_value_at("training_params.epoch.duration", current, epoch_duration)

                update(call_cb('on_epoch_completed', logs))

            if training_stopped_on_error:
                break

            self.micro_batch_step = 0
            self.batch_step = 0

        update(call_cb('on_training_ended',
                       stopped_early=epoch_stopped_early or training_stopped_early,
                       stopped_on_error=training_stopped_on_error,
                       interrupted=False,
                       callback_calls_success=self.logs["cb_calls_success"]))

        if not training_stopped_on_error:
            if self.logs["cb_calls_success"]:
                self._log.info("Training completed. All good! ❤️")
            else:
                self._log.warn("Training completed, but one or more callback errors occurred. "
                               "What's up with that? 🧐")
        else:
            self._log.warn("Training stopped on error. 🥺️")

    def _init_current_logs(self):
        return {
            "epoch": self.epoch,
            "micro_batch_step": self.micro_batch_step,  # Dataloader iteration count
            "batch_step": self.batch_step,  # Effective batch count (after accumulation)
            "global_iter": self.global_iter,

            "training": {},

            "training_params": {
                "batch": {
                    "duration": self._previous_batch_duration,
                },
                "sliding_window": {
                    "duration": self._calc_window_average("training_params.batch.duration")
                },
                "epoch": {
                    "duration": None
                }
            }
        }

    def _call_callbacks(self, event, *args, **kwargs):
        success = True
        for callback in self.callbacks:
            func_name = event
            try:
                func = getattr(callback, func_name)
                func_success = func(*args, **kwargs)
                if not _.is_bool(func_success):
                    self._log.warn(f"Success value returned by hook {func_name} by callback {callback} "
                                   f"is not boolean ({func_success}), please correct this."
                                   f"Call to hook interpreted a successful, continuing ...")
                    func_success = True

                success &= func_success
            except Exception as e:
                _.log_exception(self._log, f"Call {func_name} failed for callback {callback}", e)
                success = False

        return success

    def _update_cb_success(self, cb_calls_success):
        self.logs["cb_calls_success"] &= cb_calls_success

    def _update_window(self, metric_path):
        try:
            window = self._metric_windows[metric_path]
            if window is None:
                return

            value = get_value_at(metric_path, self.logs["current"])

            window.slide(value)
        except Exception as e:
            _.log_exception(self._log, f"Exception occurred updating sliding window {metric_path}, skipped...", e)

    def _calc_window_average(self, metric_path):
        try:
            window = self._metric_windows[metric_path]
            if window is None:
                return None

            window_data = window.window
            return mean(window_data) if len(window_data) > 0 else None
        except Exception as e:
            _.log_exception(self._log, f"Exception occurred calculating sliding window average for {metric_path}.", e)
            return None

    def _calc_window_sum(self, metric_path):
        try:
            window = self._metric_windows[metric_path]
            if window is None:
                return None

            window_data = window.window
            return sum(window_data) if len(window_data) > 0 else None
        except Exception as e:
            _.log_exception(self._log, f"Exception occurred calculating sliding window average for {metric_path}.", e)
            return None

    def _prepare_training_dataset(self):
        return self.training_dataset

    def _assess_num_batches_per_epoch(self):
        """
        Determines the number of effective batches per epoch.

        With gradient accumulation, the dataloader yields micro-batches, but we want to
        track and display progress in terms of effective batches (optimizer steps).

        Sets:
            - num_micro_batches_per_epoch: Number of dataloader iterations
            - num_batches_per_epoch: Number of effective batches (optimizer steps)
        """
        # Get gradient accumulation steps from trainer (default to 1)
        accumulation_steps = getattr(self.trainer, 'gradient_accumulation_steps', 1)

        # If num_batches_per_epoch was explicitly provided, use it as-is
        # (user is specifying effective batches, not micro-batches)
        if self.num_batches_per_epoch:
            self.num_micro_batches_per_epoch = self.num_batches_per_epoch * accumulation_steps
            return

        # Try to determine from dataset length
        self.num_micro_batches_per_epoch = math.inf
        self.num_batches_per_epoch = math.inf

        err_msg = "Evaluation of training data set length failed, number of batches per epoch is " \
                  "set to Infinite. If this is not what you want, either provide the num_batches_per_epoch " \
                  "argument during construction of the TrainingManager, or ensure that the training_dataset " \
                  "given can provide the number of batches by providing an implementation of __len__ that " \
                  "does not fail."
        try:
            if hasattr(self.training_dataset, '__len__'):
                self.num_micro_batches_per_epoch = len(self.training_dataset)
                # Calculate effective batches (round up to handle partial final batch)
                self.num_batches_per_epoch = math.ceil(self.num_micro_batches_per_epoch / accumulation_steps)
                self._log.debug(f"Number of micro-batches per epoch : {self.num_micro_batches_per_epoch}")
                self._log.debug(f"Number of batches per epoch : {self.num_batches_per_epoch}")
            else:
                self._log.warn(err_msg)
        except Exception as e:
            _.log_exception(self._log, err_msg, e)

    def _training_ended(self):
        pass

    def _validate(self):
        # TODO
        self._valid = True

        return self._valid


class Trainer(Base, metaclass=abc.ABCMeta):

    def __init__(self,
                 model_components,
                 optimizers,
                 warn_about_evaluation_results_issues=True,
                 name="Trainer",
                 **kwargs):
        """

        :param model_components: dict with model components
        :param optimizers: dict with optimizers
        :param warn_about_evaluation_results_issues: If True, any issues with normalizing of the
            training model evaluation results, which can be overcome, will be logged as a warning.

        """
        super(Trainer, self).__init__(pybase_logger_name=name, **kwargs)

        self.model_components = model_components
        self.optimizers = optimizers

        self.normalize_evaluation = NormalizeEvaluationResults(do_warn=warn_about_evaluation_results_issues)

        self._validate()

    def get_state(self):
        """

        :return: state, success
        """

        model_components_state, mcs_success = self.get_model_components_state()
        optimizers_state, os_success = self.get_optimizers_state()

        return {
            "model_components": model_components_state,
            "optimizers": optimizers_state,
        }, mcs_success & os_success

    def set_state(self, state):
        """

        :param state:
        :return: success (True or False)
        """
        if not self.instance_valid():
            self._log.error('Trainer is not valid, unable to set trainer state')
            return False

        if not self._check_state(state):
            self._log.error("Invalid state object, unable to set state")
            return False

        success = self.set_model_components_state(state["model_components"])
        success = self.set_optimizers_state(state["optimizers"]) & success

        return success

    def get_optimizers(self):
        return self.optimizers

    def get_optimizer(self, name):
        return get_value_at(name, self.get_optimizers())

    def get_model_components(self):
        return self.model_components

    def get_model_component(self, name):
        return get_value_at(name, self.get_model_components())

    def get_model_components_state(self):
        """

        :return: state, success (True or False)
        """
        state = {}

        model_components = self.get_model_components()

        success = True
        for name, model in model_components.items():
            try:
                state[name] = self._get_model_state(model, name)
            except Exception as e:
                _.log_exception(self._log, f"Unable to get state for model {name}", e)
                success = False

        return state, success

    def set_model_components_state(self, state):
        """

        :param state:
        :return: success (True or False)
        """
        if not _.is_callable(getattr(state, 'items', None)):
            self._log.error("State is invalid, unable to set model components state")
            return False

        success = True
        for name, model_state in state.items():
            model = self.get_model_component(name)
            if model is None:
                self._log.error(f"No {name} model not found, unable to set state")
                success = False
                continue

            try:
                self._set_model_state(model, model_state, name)
            except Exception as e:
                _.log_exception(self._log, f"Unable to set state for model {name}", e)
                success = False

        return success

    def get_optimizers_state(self):
        """

        :return: state, success (True of False)
        """
        state = {}

        optimizers = self.get_optimizers()

        success = True
        for name, optimizer in optimizers.items():
            try:
                state[name] = self._get_optimizer_state(optimizer, name)
            except Exception as e:
                _.log_exception(self._log, f"Unable to get state for optimizer {name}", e)
                success = False

        return state, success

    def set_optimizers_state(self, state):
        """

        :param state:
        :return: success (True, False)
        """
        if not _.is_callable(getattr(state, 'items', None)):
            self._log.error("State is invalid, unable to set optimizers state")
            return False

        success = True
        for name, optimizer_state in state.items():
            optimizer = self.get_optimizer(name)
            if optimizer is None:
                self._log.error(f"No {name} optmizer not found, unable to set state")
                success = False
                continue

            try:
                self._set_optimizer_state(optimizer, optimizer_state, name)
            except Exception as e:
                _.log_exception(self._log, f"Unable to set state for optimizer {name}", e)
                success = False

        return success

    def set_learning_rate(self, lr):
        """
        Convenience method to set the learning rate of all optimizers to `lr`

        :param lr: Learning rate to set
        :return: True on success else False
        """
        success = True

        optimizer_names = self.get_optimizers().keys()
        for opt_name in optimizer_names:
            try:
                success &= self.set_learning_rate_for(opt_name, lr)
            except Exception as e:
                _.log_exception(self._log, f"Unable to set learning rate for optimizer {opt_name}", e)
                success = False

        return success

    @abc.abstractmethod
    def set_learning_rate_for(self, optimizer_name, lr):
        """
        Set learning rate for specific optimizer `optimizer_name` to `lr`

        :param optimizer_name:
        :param lr:

        :return: True on success, else False
        """
        raise NotImplementedError("Please implement this method in your child class")

    @abc.abstractmethod
    def train_on(self, batch_data, training_settings=None) -> Tuple[MicroBatchResults, bool]:
        """
        Use batch_data to perform a training iteration on a single micro-batch.

        The trainer potentially accumulates gradients across multiple micro-batches
        (when gradient_accumulation_steps > 1) and steps the optimizer when the
        accumulation boundary is reached.

        :param batch_data: Micro-batch data object (e.g. dict, list, tuple)
        :param training_settings: Optional training settings object (usually dict)

        :return: Tuple (micro_batch_results, did_update)
            micro_batch_results: MicroBatchResults containing results for all micro-batches
                in the current accumulation window so far. Each element is a dict:
                {'loss': <loss tensor>, 'num_samples': <int>, 'auxiliary_results': <Any>}

                When gradient_accumulation_steps=1 (no accumulation), this is a list
                with a single element.

            did_update: Boolean indicating if all model weights, assigned to optimizers,
                were updated. If there are multiple optimizers for different parameter
                groups, did_update is True only if all optimizers updated their
                respective model parameters.

                did_update can be False even at accumulation boundary, for instance when
                using mixed precision training and the loss scaling factor results in
                inf/nan values. In such cases one can skip, for instance, updating an
                LR scheduler.
        """
        raise NotImplementedError("Please implement this method in your child class")

    @abc.abstractmethod
    def epoch_complete(self) -> bool:
        """
        Called when an epoch ends. Can be used to perform any necessary cleanup
        or finalization, e.g. handling partial accumulation windows.

        :return: did_update - True if model parameters were updated as a result
            of this call, False otherwise.
        """
        raise NotImplementedError("Please implement this method in your child class")

    def evaluate_loss(self, batch_data, inference_mode, evaluate_settings=None):
        """

        :param batch_data: batch_data object to evaluate loss on (e.g. dict, list, tuple)
        :param inference_mode: If True the loss will be evaluated in inference mode (e.g. no Dropout).
                               If False the loss will be evaluated in training mode
        :param evaluate_settings: optional evaluate_settings object (usually dict)

        :return: dict:
            {
                "loss": <Tensor>,
                "num_samples": <int>,
                "auxiliary_results": <can be anything, e.g dict or list with values or data items>
            }

        """

        self._activate_inference_mode(inference_mode)

        if evaluate_settings is None:
            evaluate_settings = {}

        results = self._evaluate_loss(batch_data, evaluate_settings, inference_mode)
        return self.normalize_evaluation(results)

    @abc.abstractmethod
    def _evaluate_loss(self, batch_data, evaluate_settings=None, inference_mode=None):
        """
        Evaluates the given training model on the given batch_data, using the optional training_settings.
        Depending on the Deep learning backend you might need to use inference mode here.

        :param batch_data: batch_data object to evaluate loss on (e.g. dict, list, tuple)
        :param evaluate_settings: optional evaluate_settings object (usually dict)
        :param inference_mode: bool, important when inference mode not set in
            `_activate_inference_mode`

        :return: dict or tuple
            {
                "loss": <Tensor>,
                "num_samples": <int>,
                "auxiliary_results": <can be anything, e.g dict or list with values or data items>
            }

            (loss, num_samples, ... auxiliary results ...)
        """
        raise NotImplemented("Please implement this method in your child class")

    @abc.abstractmethod
    def _activate_inference_mode(self, inference_mode):
        raise NotImplemented("Please implement this method in your child class")

    @abc.abstractmethod
    def _get_model_state(self, model, model_name=None):
        raise NotImplemented("Please implement this method in your child class")

    @abc.abstractmethod
    def _get_optimizer_state(self, optimizer, optimizer_name=None):
        raise NotImplemented("Please implement this method in your child class")

    @abc.abstractmethod
    def _set_model_state(self, model, state, model_name=None):
        raise NotImplemented("Please implement this method in your child class")

    @abc.abstractmethod
    def _set_optimizer_state(self, optimizer, state, optimizer_name):
        raise NotImplemented("Please implement this method in your child class")

    def _check_state(self, state):
        state_attributes = ['model_components', 'optimizers']

        for attr in state_attributes:
            v = get_value_at(attr, state, warn_on_failure=False)
            if v is None:
                self._log.error(f"Given state does not have a value for {attr}, state is invalid")
                return False

        return True

    def _validate(self):
        # TODO
        self._valid = True

        return self._valid


class DefaultTrainer(Trainer, metaclass=abc.ABCMeta):

    def __init__(self,
                 optimizers,
                 model_components=None,
                 model_wrapper_func: Optional[ModelWrapperFunc] = None,
                 batch_size: Optional[int] = None,
                 micro_batch_size: Optional[int] = None,
                 eager_mode: bool = False,
                 use_mixed_precision: bool = False,
                 autocast_dtype: Optional[str] = None,
                 use_loss_scaling: Optional[bool] = None,
                 name="DefaultTrainer",
                 **kwargs):
        """
        Simple trainer based on a training_model, that evaluates the loss on batch data.

        Supports gradient accumulation: when batch_size is a multiple of micro_batch_size,
        the received batch data is assumed to be a micro-batch. The trainer will accumulate
        gradients over multiple micro-batches until reaching the effective batch_size.

        :param optimizers: Dict or list with optimizer(s), or a single optimizer instance
        :param model_components: Dict or list with model components(s), or a single model instance
        :param model_wrapper_func: Optional callable that wraps the model (e.g., for DDP, FSDP).
            The wrapper is applied in set_training_model() before any backend-specific processing.
            See ModelWrapperFunc protocol for the expected signature.
        :param batch_size: Optional effective batch size for optimization. If provided with
            micro_batch_size, must be divisible by micro_batch_size.
            gradient_accumulation_steps = batch_size / micro_batch_size
        :param micro_batch_size: Optional size of micro-batches received by train_on().
            This is the memory unit - what fits in AI accelerator (e.g. GPU) memory.
            If micro_batch_size is provided without batch_size, a ValueError is raised.
        :param eager_mode: If True, the training step is not compiled
        :param use_mixed_precision: Convenience flag for float16 + loss scaling. When True,
            defaults autocast_dtype to 'float16' and enables loss scaling.
        :param autocast_dtype: Dtype for autocast (e.g., 'float16', 'bfloat16'). When set
            without use_mixed_precision, only autocasting is applied (no loss scaling by default).
        :param use_loss_scaling: Whether to use loss/gradient scaling. None means use default
            behavior (enabled only when use_mixed_precision=True). Explicit False with
            use_mixed_precision=True will log a warning and be overridden.
        """

        model_components = convert_to_dict("model", model_components)
        optimizers = convert_to_dict("optimizer", optimizers)

        self._model_wrapper_func = model_wrapper_func

        super().__init__(model_components, optimizers, name=name, **kwargs)

        # Determine gradient accumulation steps based on provided parameters
        self.batch_size, self.micro_batch_size, self.gradient_accumulation_steps = \
            calc_gradient_accumulation_steps(batch_size, micro_batch_size)

        if self.gradient_accumulation_steps > 1:
            self._log.info(
                f"Gradient accumulation enabled: batch_size={self.batch_size}, "
                f"micro_batch_size={self.micro_batch_size}, "
                f"gradient_accumulation_steps={self.gradient_accumulation_steps}"
            )

        # Accumulation state
        self._accumulation_counter = 0
        self._micro_batch_results: MicroBatchResults = MicroBatchResults()

        # TODO: add property getters, disable setting
        self.eager_mode = eager_mode

        # Mixed precision configuration
        self._autocast_dtype, self._use_loss_scaling = self._resolve_mixed_precision(
            use_mixed_precision, autocast_dtype, use_loss_scaling
        )

        self.training_model = None

    def _resolve_mixed_precision(
        self,
        use_mixed_precision: bool,
        autocast_dtype: Optional[str],
        use_loss_scaling: Optional[bool],
    ) -> Tuple[Any, bool]:
        """
        Resolve mixed precision configuration from the input parameters.

        :param use_mixed_precision: Convenience flag (True = float16 + loss scaling)
        :param autocast_dtype: Explicit dtype string or None
        :param use_loss_scaling: Whether to use loss scaling (None = use default)

        :return: Tuple of (framework_dtype, use_loss_scaling)
        """
        # use_mixed_precision is the convenience flag for float16 + scaling
        if use_mixed_precision:
            if autocast_dtype is None:
                autocast_dtype = 'float16'
            if use_loss_scaling is False:
                self._log.warning(
                    "use_mixed_precision=True but use_loss_scaling=False is conflicting. "
                    "Enabling loss scaling as required by use_mixed_precision."
                )
            use_loss_scaling = True
            self._log.info(
                f"use_mixed_precision=True: autocast_dtype='{autocast_dtype}', "
                f"use_loss_scaling=True"
            )
        elif use_loss_scaling is None:
            # Default: no scaling unless explicitly requested
            use_loss_scaling = False

        # Convert to framework-specific dtype
        framework_dtype = self._get_dtype(autocast_dtype)

        # Log configuration when using autocast without use_mixed_precision
        if framework_dtype is not None and not use_mixed_precision:
            self._log.info(
                f"Autocasting enabled: autocast_dtype='{autocast_dtype}', "
                f"use_loss_scaling={use_loss_scaling}"
            )

        return framework_dtype, use_loss_scaling

    @abc.abstractmethod
    def _get_dtype(self, dtype_name: Optional[str]) -> Any:
        """
        Convert dtype name string to framework-specific dtype.

        :param dtype_name: Dtype name (e.g., 'float16', 'bfloat16') or None.

        :return: Framework-specific dtype object, or None if dtype_name is None.

        :raises ValueError: If dtype_name is not supported by this framework.
        """
        ...

    def set_training_model(self, model):
        """
        Set the training model, optionally applying a model wrapper.

        If a model_wrapper_func was provided at construction, it is applied here
        before any backend-specific processing.

        :param model: nn.Module that returns the loss based on the given batch
            The forward method of the training model must be callable and the following signature:

            model(self, batch_data, training_settings) -> {
                "loss": <Tensor>,
                "auxiliary_results": <can be anything, e.g dict or list with values or data items>
            }

            or

            model(self, batch_data, training_settings) -> [<loss_tensor>, ... auxiliary_results ...]

        :return:
        """

        # Apply model wrapper if provided
        if callable(self._model_wrapper_func):
            try:
                model = self._apply_model_wrapper_func(model)
            except Exception as e:
                raise ModelWrapperException() from e

        self.training_model = model

        instance_valid = self.instance_valid()
        self._valid = self._validate_model() & instance_valid

    @abc.abstractmethod
    def _apply_model_wrapper_func(self, model):
        """
        Apply the model wrapper function with backend-specific kwargs.

        Subclasses must override to inject relevant trainer constructor parameters.
        For example, PyTorch injects eager_mode and compile_kwargs.

        :param model: The model to wrap

        :return: The wrapped model
        """
        ...

    def _reset_accumulation_state(self):
        """Reset the gradient accumulation state."""
        self._accumulation_counter = 0
        self._micro_batch_results = MicroBatchResults()

    def _evaluate_loss(self, batch_data, evaluate_settings=None, inference_mode=None):
        """
        Evaluates the given training model on the given batch_data, using the optional training_settings.
        Depending on the Deep learning backend you might need to use inference mode here.

        :param batch_data: batch_data object to evaluate loss on (e.g. dict, list, tuple)
        :param evaluate_settings: optional evaluate_settings object (usually dict)
        :param inference_mode: optional bool, important when inference mode not set in
            `_activate_inference_mode`

        :return: dict or tuple
            {
                "loss": <Tensor>,
                "num_samples": <int>,
                "auxiliary_results": <can be anything, e.g dict or list with values or data items>
            }

            (loss, num_samples, ... auxiliary results ...)
        """
        return self.training_model(batch_data, evaluate_settings, inference_mode)

    def _validate_model(self):
        # TODO : this is framework dependent
        model_valid = callable(self.training_model)
        if not model_valid:
            self._log.error(f"Given training model {self.training_model} is invalid, trainer will not work")

        return model_valid
