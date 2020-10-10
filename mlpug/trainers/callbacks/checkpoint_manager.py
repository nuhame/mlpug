import os
import time
import json

import abc

from shutil import copyfile

from mlpug.utils import get_value_at
from mlpug.trainers.callbacks.callback import Callback

import basics.base_utils as _


class CheckpointManagerBase(Callback, metaclass=abc.ABCMeta):

    def __init__(self,
                 model_hyper_parameters=None,
                 checkpoints_path="../trained-models/",
                 batch_level=True,
                 metric_to_monitor="validation.window_average.perplexity",
                 metric_opt_mode='min',
                 metric_monitor_period=200,  # batches or epochs
                 metric_checkpoint_threshold=None,
                 create_checkpoint_every=200,  # batches or epochs
                 archive_last_model_checkpoint_every=2000,  # batches or epochs
                 force_monitoring_on_epoch=True,
                 base_checkpoint_filename=time.strftime("%d-%m-%Y_%H-%M-%S"),
                 model_checkpoint_filename_ext="model",
                 training_checkpoint_filename_ext="state",
                 backup_before_override=True,
                 disable_saving_checkpoints=False,
                 name="CheckpointManager"):
        """

        :param model_hyper_parameters: Dict with hyper parameters of the model, this will be saved as part of
                                       the model checkpoint
        :param checkpoints_path:
        :param batch_level: True if monitoring happens on the scale of batches. If False, monitoring is done
                            on the epoch level
        :param metric_to_monitor: key path to metric value in the log object, e.g. `validation.batch.perplexity`
        :param metric_opt_mode: 'max', 'min'
        :param metric_monitor_period: The period between checks for model quality improvement.
                                      This is in number of batches if `batch_level = True`, else it is a
                                      number of epochs.
        :param metric_checkpoint_threshold: when given, the model quality must be below (or above, depending on
                                            metric_opt_mode) this threshold before a new best model checkpoint can be
                                            saved

        :param create_checkpoint_every: period before saving next training/model checkpoint
                                        (will be overridden after each period)
        :param archive_last_model_checkpoint_every: period before the last available model checkpoint is archived.
                                                    Period must be multiple of create_checkpoint_every period
        :param force_monitoring_on_epoch: When True, the given metric will also be monitored on every epoch
                                          in the case that monitoring level is batch level
        :param base_checkpoint_filename:
        :param model_checkpoint_filename_ext:
        :param training_checkpoint_filename_ext:
        :param backup_before_override: When True the training and model checkpoints are backed up before they are
                                       overridden by a new version. If backing up fails, no new version will be written
                                       to disk. This gives the user a chance to fix a problem, e.g. disk full, without
                                       interruption of the training process.
        :param disable_saving_checkpoints: When True no checkpoints at all will be saved, e.g. for testing training

        """
        super().__init__(name=name)

        self._model_hyper_parameters = model_hyper_parameters

        self._checkpoints_path = checkpoints_path
        self._model_checkpoint_filename_ext = model_checkpoint_filename_ext
        self._training_checkpoint_filename_ext = training_checkpoint_filename_ext
        self._base_filename = base_checkpoint_filename

        self._batch_level = batch_level

        self._metric_to_monitor = metric_to_monitor
        self._metric_opt_mode = metric_opt_mode

        self._metric_monitor_period = metric_monitor_period
        self._metric_checkpoint_threshold = metric_checkpoint_threshold
        self._create_checkpoint_every = create_checkpoint_every
        self._archive_last_model_checkpoint_every = archive_last_model_checkpoint_every

        self._disable_saving_checkpoints = disable_saving_checkpoints

        self._force_monitoring_on_epoch = force_monitoring_on_epoch

        self._best_model_quality = float('Inf') if self._metric_opt_mode == 'min' else -float('Inf')
        self._best_model_iter = None

        self._backup_before_override = backup_before_override

        self._check_settings()

        self._describe_setup()

    def get_state(self):
        return {
            "best_model_quality": self._best_model_quality,
            "best_model_iter": self._best_model_iter
        }, True

    def set_state(self, state):
        """

        :param state Dict with saved checkpoint state to continue tracking the best model during training
        :return:
        """
        success = True
        if not _.is_dict(state):
            self._log.debug("No initial checkpoint state given, starting with clean slate ...")
            return success

        self._log.debug("Using given initial checkpoint state: \n\n%s" % json.dumps(state, indent=4))

        try:
            self._best_model_quality = state['best_model_quality']
            self._best_model_iter = state['best_model_iter']
        except Exception as e:
            _.log_exception(self._log, "Unable to set checkpoint manager state", e)
            success = False

        return success

    def on_batch_training_completed(self, training_batch, logs):
        if not self._batch_level:
            return True

        return self._monitor('global_iter', logs)

    def on_epoch_completed(self, logs):
        force = False
        iter_name = 'epoch'

        force_monitoring = False
        if self._batch_level:
            if self._force_monitoring_on_epoch:
                self._log.debug("Epoch completed : checking if model improved and creating checkpoints ... ")
                iter_name = 'global_iter'
                force_monitoring = True
            else:
                return True

        return self._monitor(iter_name, logs, force_monitoring=force_monitoring)

    def on_training_ended(self, stopped_early, stopped_on_error, callback_calls_success):
        if abs(self._best_model_quality) == float('Inf') or self._best_model_iter is None:
            return True

        self._log.info(f"Best model quality reached {self._metric_to_monitor}={self._best_model_quality} "
                       f"at global iteration {self._best_model_iter}")
        return True

    def training_checkpoint_file_name(self):
        b_fn = self._base_filename
        ext = self._training_checkpoint_filename_ext

        return os.path.join(self._checkpoints_path, f'{b_fn}-training-checkpoint.{ext}')

    def current_model_file_name(self, training_iter):
        b_fn = self._base_filename
        ext = self._model_checkpoint_filename_ext

        return os.path.join(self._checkpoints_path, f'{b_fn}-model-checkpoint-{training_iter}.{ext}')

    def latest_model_file_name(self):
        b_fn = self._base_filename
        ext = self._model_checkpoint_filename_ext

        return os.path.join(self._checkpoints_path, f'{b_fn}-latest-model-checkpoint.{ext}')

    def best_model_file_name(self):
        b_fn = self._base_filename
        ext = self._model_checkpoint_filename_ext

        return os.path.join(self._checkpoints_path, f'{b_fn}-best-model-checkpoint.{ext}')

    def _describe_setup(self):
        self._log.info(f"Metric to monitor : {self._metric_to_monitor}")
        self._log.info(f"Metric monitor period : {self._metric_monitor_period}")

        time_scale = 'batches' if self._batch_level else 'epochs'
        self._log.info(f"Create last training & model checkpoints every "
                       f"{self._archive_last_model_checkpoint_every} {time_scale}")
        self._log.info(f"Archive last model checkpoint every "
                       f"{self._archive_last_model_checkpoint_every} {time_scale}")

    def _get_model_quality(self, current_logs):
        return get_value_at(self._metric_to_monitor, current_logs)

    # TODO : it would maybe be better to split the monitoring and the checkpointing in to two separated methods
    def _monitor(self, iter_name, logs, force_monitoring=False):
        current = self._get_logs_base(logs)

        training_iter = current[iter_name]

        if not force_monitoring and training_iter == 0:
            return True

        best_model_fname = None
        success = True
        data_saved = False
        if force_monitoring or \
                ((self._metric_monitor_period > 0) and (training_iter % self._metric_monitor_period == 0)):
            model_quality = self._get_model_quality(current)

            model_quality_valid = model_quality is not None
            model_quality_good_enough = True
            if model_quality_valid and self._metric_checkpoint_threshold is not None:
                if self._metric_opt_mode == 'min':
                    model_quality_good_enough = model_quality <= self._metric_checkpoint_threshold
                elif self._metric_opt_mode == 'max':
                    model_quality_good_enough = model_quality >= self._metric_checkpoint_threshold

            if not model_quality_good_enough:
                self._log.debug("Iter : %s : model quality not good enough : %s : %3e " % (training_iter,
                                                                                           self._metric_to_monitor,
                                                                                           model_quality))

            if model_quality_valid and model_quality_good_enough:
                model_improved = ((self._metric_opt_mode == 'min') and (model_quality < self._best_model_quality)) or \
                                 ((self._metric_opt_mode == 'max') and (model_quality > self._best_model_quality))

                if model_improved:
                    self._log.debug("Iter : %s : Model improved : %s : %3e " % (training_iter,
                                                                                self._metric_to_monitor,
                                                                                model_quality))

                    if (self._best_model_iter is not None) and \
                            (iter_name == 'epoch') and \
                            (training_iter < self._best_model_iter):
                        self._log.warn(f"Inconsistency: according to the current training iter ({training_iter}), "
                                       f"current best model training iter ({self._best_model_iter}) is in the future. "
                                       f"Was the right training checkpoint loaded?")

                    best_model_fname = self._disable_saving_checkpoints or self._save_current_model_as_best()
                    if best_model_fname:
                        self._best_model_quality = model_quality
                        self._best_model_iter = training_iter

                        data_saved = not self._disable_saving_checkpoints
                    else:
                        self._log.error("Unable to save improved model checkpoint")
                        success = False

                self._update_logs(model_improved, logs, current)
            elif not model_quality_valid:
                self._log.error(f"No model quality available, unable to check if we need to save a checkpoint, "
                                f"skipping ...")
                success = False

        if not self._disable_saving_checkpoints and \
                (self._create_checkpoint_every > 0) and \
                (training_iter % self._create_checkpoint_every == 0):
            # Just copy best model if available
            latest_model_fname = self._create_model_checkpoint(file_to_copy=best_model_fname)

            latest_model_saved = (latest_model_fname is not None)
            success &= latest_model_saved

            data_saved |= latest_model_saved

            if (self._archive_last_model_checkpoint_every > 0) and \
                    (training_iter % self._archive_last_model_checkpoint_every == 0):
                if latest_model_saved:
                    copy_success = self._copy(latest_model_fname, self.current_model_file_name(training_iter))

                    success &= copy_success

                    data_saved |= copy_success
                else:
                    self._log.error("Unable to create checkpoint for latest model, unable to archive latest model")
                    success = False

            training_checkpoint_success = self._create_training_checkpoint()
            data_saved |= training_checkpoint_success

            success &= training_checkpoint_success

        if data_saved:
            self._log.debug("Saving of data done.\n\n")

        return success

    def _save_current_model_as_best(self):

        model_fn = self.best_model_file_name()

        if self._backup_before_override:
            try:
                self._backup_checkpoint(model_fn)
            except Exception as e:
                _.log_exception(self._log, f"A problem occurred backing up the last best model checkpoint, "
                                           f"will not override override model checkpoint with new one", e)
                return None

        try:
            self._save_model_checkpoint(model_fn)
        except Exception as e:
            _.log_exception(self._log, f"A problem occurred saving the current model as best model", e)
            return None

        return model_fn

    def _update_logs(self, model_improved, logs, current):
        if model_improved:
            logs["best"] = current.copy()

        current["is_best"] = model_improved

    def _create_model_checkpoint(self, file_to_copy=None):
        model_fn = self.latest_model_file_name()
        if file_to_copy:
            if not self._copy(file_to_copy, model_fn):
                self._log.error(f"Unable to create model checkpoint based on file : {file_to_copy}")
                return None
        else:
            try:
                self._save_model_checkpoint(model_fn)
            except Exception as e:
                _.log_exception(self._log, f"A problem occurred saving the latest model as checkpoint", e)
                return None

        return model_fn

    def _create_training_checkpoint(self):
        checkpoint_fname = self.training_checkpoint_file_name()

        if self._backup_before_override:
            try:
                if not self._backup_checkpoint(checkpoint_fname):
                    self._log.error("Unable to backup last training checkpoint, "
                                    "will not override override training checkpoint with new one")
                    return False
            except Exception as e:
                _.log_exception(self._log, f"A problem occurred backing up the last training checkpoint, "
                                           f"will not override override training checkpoint with new one", e)
                return False

        success = False
        try:
            success = self._save_training_checkpoint(checkpoint_fname)
        except Exception as e:
            _.log_exception(self._log, f"Unable to save training checkpoint", e)

        return success

    @abc.abstractmethod
    def _save_training_checkpoint(self, filename):
        self._log.error("This method is not implemented, implement it in your child class implementation")

    @abc.abstractmethod
    def _save_model_checkpoint(self, filename):
        self._log.error("This method is not implemented, implement it in your child class implementation")

    def _backup_checkpoint(self, filename):
        if os.path.isfile(filename):
            return self._copy(filename, f"{filename}.backup")
        else:
            return True

    def _copy(self, source_fname, dest_fname):
        try:
            self._log.debug("Copying model: [%s] ==> [%s]" % (source_fname, dest_fname))
            copyfile(source_fname, dest_fname)
        except Exception as e:
            _.log_exception(self._log, "Unable to copy [%s]" % source_fname, e)
            return False

        return True

    def _check_settings(self):
        if not os.path.exists(self._checkpoints_path):
            self._log.info(f"Creating checkpoint directory : {self._checkpoints_path}")
            os.makedirs(self._checkpoints_path)

        if (self._create_checkpoint_every < 0) and (self._archive_last_model_checkpoint_every > 0):
            self._log.error("archive_last_model_checkpoint_every can't be > 0 while _create_checkpoint_every < 0, "
                            "disabling archiving ... ")
            self._archive_last_model_checkpoint_every = -1

        if (self._create_checkpoint_every > 0) and \
           (self._archive_last_model_checkpoint_every > 0) and \
           (self._archive_last_model_checkpoint_every % self._create_checkpoint_every != 0):

            self._archive_last_model_checkpoint_every = 10 * self._create_checkpoint_every
            self._log.warn(f"archive_last_model_checkpoint_every must be "
                           f"exact multiple of _create_checkpoint_every, "
                           f"changing archive_last_model_checkpoint_every to "
                           f"[{self._archive_last_model_checkpoint_every}]")
