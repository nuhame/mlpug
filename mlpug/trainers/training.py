import sys
import abc
import math
import time

from statistics import mean

from basics.base import Base
import basics.base_utils as _

from mlpug.utils import convert_to_dict, get_value_at, set_value_at, has_key, SlidingWindow

from mlpug.mlpug_exceptions import TrainerInvalidException


class TrainingManager(Base, metaclass=abc.ABCMeta):

    def __init__(self,
                 trainer,
                 training_dataset,
                 num_epochs=50,
                 num_batches_per_epoch=None,
                 callbacks=None,
                 experiment_data=None):
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

        """
        super(TrainingManager, self).__init__()

        self.trainer = trainer

        # Must implement __len__, __iter__
        self.training_dataset = training_dataset

        self.num_epochs = num_epochs
        self.num_batches_per_epoch = num_batches_per_epoch

        self.global_iter = 0
        self.batch_step = 0
        self.epoch = 0

        self.callbacks = callbacks or []
        self.callbacks_map = None

        self.experiment_data = experiment_data

        self.init_logs = None
        self.logs = {}

        self._previous_batch_duration = None

        self._metric_windows = {}

        self._stop_training = False

        self._callback_calls_successful = True

        if not self._validate():
            return

        self._assess_num_batches_per_epoch()

        if type(self.num_batches_per_epoch) == int and self.num_batches_per_epoch > 0:
            self._metric_windows['training.batch.loss'] = \
                SlidingWindow(length=self.num_batches_per_epoch, name='training.batch.loss')

            self._metric_windows['training_params.batch.duration'] = \
                SlidingWindow(length=self.num_batches_per_epoch, name='training_params.batch.duration')

        if not self._call_callbacks("set_training_manager", self):
            self._callback_calls_successful = False
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
            sys.stdout.write("\n\n")
            self._log.info('Training process interrupted by you ... 🤷🏻‍♂️\n')
            # TODO : deal with the distributed training case, wait to return until
            #  all workers have stopped training
            self.stop_training()

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
                "batch_step": self.batch_step,
                "global_iter": self.global_iter,
                "logs": self.logs,
                "metric_windows": self._get_metric_windows_state()
            },

            "callbacks": {},

            "experiment_data": self.experiment_data
        }

        success = True
        for callback in self.callbacks:
            cb_name = None
            try:
                cb_name = str(callback)
                cb_state, cb_success = callback.get_state()

                success &= cb_success

                if cb_name in state["callbacks"] and cb_state != state["callbacks"][cb_name]:
                    self._log.error(f"There is already a callback in the TrainingManager state "
                                    f"with the name {cb_name}, this state will be overridden now. "
                                    f"Please ensure that all your callbacks have a unique name")
                    success = False

                state["callbacks"][cb_name] = cb_state
            except Exception as e:
                _.log_exception(self._log, f"Failed to get state of callback {cb_name}, "
                                           f"unable to add callback state to Training Manager state", e)
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
                   "batch_step": self.batch_step,
                   "global_iter": self.global_iter,
                   "logs": self.logs,
                   "experiment_data": self.experiment_data
               }, True

    def set_state(self, state):
        """
        WARNING: the start batch_step in the state only controls how many batches will be trained on in the
                 the current epoch, it does not control the exact batch data that will be used for training.

        :param state:
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

        # start at next iteration
        self.global_iter += 1
        self.batch_step += 1
        if self.batch_step > self.num_batches_per_epoch-1:
            self.epoch += 1
            self.batch_step = 0

        self.init_logs["epoch"] = self.epoch
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
            callbacks_map = {str(callback): callback for callback in self.callbacks}

            callbacks = state["callbacks"] or {}
            for cb_name, cb_state in callbacks.items():
                if cb_name not in callbacks_map:
                    self._log.warn(f"Callback {cb_name} not given, unable to set callback state, skipping ...")
                    continue

                callback = callbacks_map[cb_name]
                try:
                    success &= callback.set_state(cb_state)
                except Exception as e:
                    _.log_exception(self._log, f"Unable to set state for callback {cb_name}", e)
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
                window = SlidingWindow(length=self.num_batches_per_epoch,
                                       init_window_values=window_state['window'],
                                       name=metric_path)
            else:
                window = SlidingWindow(state=window_state)

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

        cb_calls_success = self._callback_calls_successful

        cb_calls_success &= self._call_callbacks('on_training_start',
                                                 self.num_epochs,
                                                 self.num_batches_per_epoch,
                                                 self.epoch,
                                                 self.batch_step,
                                                 self.global_iter)

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch

            if self._stop_training:
                training_stopped_early = True
                self._log.warn("Training stopped early")
                break

            if self.init_logs is not None:
                self._log.debug("Setting provided initial logs object ...")
                self.logs = self.init_logs

                # override log state for certain parameters given at construction
                self.logs["final_epoch"] = self.num_epochs-1
                self.logs["final_batch_step"] = self.num_batches_per_epoch - 1
                self.init_logs = None
            else:
                # Keep anything in the logs from the previous epoch that is non-standard
                self.logs = self.logs or {}
                self.logs = {**self.logs, **{
                    "final_epoch": self.num_epochs - 1,
                    "final_batch_step": self.num_batches_per_epoch - 1,

                    "epoch": self.epoch,

                    "current": None,

                    "cb_calls_success": cb_calls_success
                }}

            # It's a bit shorter
            logs = self.logs

            logs["cb_calls_success"] &= self._call_callbacks('on_epoch_start', logs)

            epoch_stopped_early = False

            training_dataset = self._prepare_training_dataset()
            current = None
            ctl = None
            for training_batch in iter(training_dataset):
                batch_training_start_time = time.time()

                if self._stop_training:
                    # Epoch was not finished yet
                    epoch_stopped_early = True
                    self._log.warn("Training epoch stopped early")
                    break

                logs["current"] = self._init_current_logs()

                logs["cb_calls_success"] &= self._call_callbacks('on_batch_training_start', training_batch, logs)

                current = logs["current"]
                ctl = current["training"]

                try:
                    training_settings = logs["training_settings"] if has_key(logs, "training_settings") else None
                    ctl["batch"]["loss"], ctl["batch"]["auxiliary_results"] = self.trainer.train_on(training_batch,
                                                                                                    training_settings)

                    self._update_window("training.batch.loss")
                    ctl["window_average"]["loss"] = self._calc_window_average("training.batch.loss")
                except Exception as e:
                    if isinstance(e, TrainerInvalidException):
                        err_msg = f"Trainer {self.trainer} is misconfigured, unable to train on batch, " \
                                  f"will stop training ..."
                    else:
                        err_msg = f"Exception occurred while calling {self.trainer} to train on batch, " \
                                  f"will stop training ..."

                    _.log_exception(self._log, err_msg, e)

                    logs["cb_calls_success"] &= self._call_callbacks('on_batch_training_failed', e, logs)
                    training_stopped_on_error = True

                if training_stopped_on_error:
                    break

                logs["cb_calls_success"] &= self._call_callbacks('on_batch_training_completed', training_batch, logs)

                self.batch_step += 1
                self.global_iter += 1

                batch_training_end_time = time.time()

                batch_duration = batch_training_end_time - batch_training_start_time

                set_value_at("training_params.batch.duration", current, batch_duration)

                self._update_window("training_params.batch.duration")

                self._previous_batch_duration = batch_duration

                # Situation 1: When the start_batch given was > 0, go to next epoch when epoch finished
                # Situation 2: When num_batches_per_epoch was given to simulate epochs in an
                #              infinite batch streaming situation
                if self.batch_step >= self.num_batches_per_epoch:
                    break

            if not (epoch_stopped_early or training_stopped_on_error):
                ctpl = current["training_params"]

                ctpl["window_average"]["duration"] = self._calc_window_average("training_params.batch.duration")
                ctpl["epoch"]["duration"] = self._calc_window_sum("training_params.batch.duration")

                logs["cb_calls_success"] &= self._call_callbacks('on_epoch_completed', logs)

            cb_calls_success &= logs["cb_calls_success"]

            if training_stopped_on_error:
                break

            self.batch_step = 0

        cb_calls_success &= self._call_callbacks('on_training_ended',
                                                 stopped_early=epoch_stopped_early or training_stopped_early,
                                                 stopped_on_error=training_stopped_on_error,
                                                 callback_calls_success=cb_calls_success)

        if not training_stopped_on_error:
            if cb_calls_success:
                self._log.info("Training completed. All good! ❤️")
            else:
                self._log.warn("Training completed, but one or more callback errors occurred. "
                               "What's up with that? 🧐")
        else:
            self._log.warn("Training stopped on error. 🥺️")

    def _init_current_logs(self):
        return {
            "epoch": self.epoch,
            "batch_step": self.batch_step,
            "global_iter": self.global_iter,

            "training": {
                "batch": {},
                "window_average": {
                    "loss": self._calc_window_average("training.batch.loss")
                },
                "dataset": {},
            },

            "training_params": {
                "batch": {
                    "duration": self._previous_batch_duration,
                },
                "window_average": {
                    "duration": self._calc_window_average("training_params.batch.duration")
                },
                "epoch": {
                    "duration": self._calc_window_sum("training_params.batch.duration")
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
            return sum(window_data) if len(window_data) > 0 else 0
        except Exception as e:
            _.log_exception(self._log, f"Exception occurred calculating sliding window average for {metric_path}.", e)
            return None

    def _prepare_training_dataset(self):
        return self.training_dataset

    def _assess_num_batches_per_epoch(self):
        if self.num_batches_per_epoch:
            # num_batches_per_epoch given, nothing to do
            return

        self.num_batches_per_epoch = math.inf
        err_msg = "Evaluation of training data set length failed, number of batches per epoch is " \
                  "set to Infinite. If this is not what you want, either provide the num_batches_per_epoch " \
                  "argument during construction of the TrainingManager, or ensure that the training_dataset " \
                  "given can provide the number of batches by providing an implementation of __len__ that " \
                  "does not fail."
        try:
            if hasattr(self.training_dataset, '__len__'):
                self.num_batches_per_epoch = len(self.training_dataset)
                self._log.debug(f"Number of batches per epoch : {self.num_batches_per_epoch}")
            else:
                self._log.warn(err_msg)
        except Exception as e:
            _.log_exception(self._log, err_msg, e)

    def _validate(self):
        # TODO
        self._valid = True

        return self._valid


class TrainerBase(Base, metaclass=abc.ABCMeta):

    def __init__(self, model_components, optimizers):
        """

        :param model_components: dict with model components
        :param optimizers: dict with optimizers

        """
        super(TrainerBase, self).__init__()

        self.model_components = model_components
        self.optimizers = optimizers

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

    @abc.abstractmethod
    def train_on(self, batch_data, training_settings=None):
        """
        TODO : Should this also return only one dict with 'loss' and 'auxiliary_results' keys?
               (Just like evaluate_loss)

        Use batch_data to perform a training iteration

        :param batch_data: batch_data object (e.g. dict, list, tuple)
        :param training_settings: optional training_settings object (usually dict)

        :return: loss, auxiliary_results

        loss : number (e.g. float)
        auxiliary_results : can be anything, e.g dict or list with values or data items
        """
        pass

    def evaluate_loss(self, batch_data, inference_mode, evaluate_settings=None):
        """

        :param batch_data: batch_data object to evaluate loss on (e.g. dict, list, tuple)
        :param inference_mode: If True the loss will be evaluated in inference mode (e.g. no Dropout).
                               If False the loss will be evaluated in training mode
        :param evaluate_settings: optional evaluate_settings object (usually dict)

        :return: dict:
            {
                "loss": <Tensor>,
                "auxiliary_results": <can be anything, e.g dict or list with values or data items>
            }

        """

        self._activate_inference_mode(inference_mode)

        return self._evaluate_loss(batch_data, evaluate_settings)

    @abc.abstractmethod
    def _evaluate_loss(self, batch_data, evaluate_settings=None):
        """

        :param batch_data: batch_data object to evaluate loss on (e.g. dict, list, tuple)
        :param evaluate_settings: optional evaluate_settings object (usually dict)

        :return: dict:
            {
                "loss": <Tensor>,
                "auxiliary_results": <can be anything, e.g dict or list with values or data items>
            }
        """

        pass

    @abc.abstractmethod
    def _activate_inference_mode(self, inference_mode):
        pass

    @abc.abstractmethod
    def _get_model_state(self, model, model_name=None):
        pass

    @abc.abstractmethod
    def _get_optimizer_state(self, optimizer, optimizer_name=None):
        pass

    @abc.abstractmethod
    def _set_model_state(self, model, state, model_name=None):
        pass

    @abc.abstractmethod
    def _set_optimizer_state(self, optimizer, state, optimizer_name):
        pass

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


class DefaultTrainerBase(TrainerBase, metaclass=abc.ABCMeta):

    def __init__(self, optimizers, model_components=None, batch_chunk_size=None):
        """
        Simple trainer based on a training_model, that evaluates the loss on batch data

        :param optimizers: dict or list with optimizer(s), or a single optimizer instance
        :param model_components: dict or list with model components(s), or a single model instance
        :param batch_chunk_size: optional batch chunk size (int)
                                 If given, batches are processed in chunks of size `batch_chunk_size` samples to
                                 calculate the gradients. The last chunk can be smaller than `batch_chunk_size` if
                                 there is not an exact multiple that is equal to the `batch_data` size

                                 Note 1.
                                 Chunked processing of a batch only works when the `batch_data` object, received
                                 by the `train_on` method, implements the `__len__` and `__getitem__` methods.
                                 Here the `__getitem__` method must be able to deal with slices.

                                 Note 2.
                                 When using chunked batch processing, the default implementation assumes that the
                                 loss, calculated over a chunk, is the average of the sample losses
        """

        model_components = convert_to_dict("model", model_components)
        optimizers = convert_to_dict("optimizer", optimizers)

        super().__init__(model_components, optimizers)

        self.batch_chunk_size = batch_chunk_size
        if self.batch_chunk_size is not None:
            self._log.info(f"Will train on batches by slicing the batch is chunks of {batch_chunk_size} samples.")

        self.training_model = None

    def set_training_model(self, model):
        """
        :param model: nn.Module that returns the loss based on the given batch
                      The forward method of the training model must be callable and the following signature:

                      model(self, batch_data, training_settings) -> {
                        "loss": <Tensor>,
                        "auxiliary_results": <can be anything, e.g dict or list with values or data items>
                      }

        :param model:
        :return:
        """

        self.training_model = model

        instance_valid = self.instance_valid()
        self._valid = self._validate_model() & instance_valid

    def _evaluate_loss(self, batch_data, evaluate_settings=None):
        """

        Evaluates the given training model on the  given batch_data, using the optional training_settings

        :param batch_data: batch_data object to evaluate loss on (e.g. dict, list, tuple)
        :param evaluate_settings: optional evaluate_settings object (usually dict)


        :return: dict:
            {
                "loss": <Tensor>,
                "auxiliary_results": <can be anything, e.g dict or list with values or data items>
            }
        """
        return self.training_model(batch_data, evaluate_settings)

    def _validate_model(self):
        # TODO : this is framework dependent
        model_valid = callable(self.training_model)
        if not model_valid:
            self._log.error(f"Given training model {self.training_model} is invalid, trainer will not work")

        return model_valid
