import os
import sys
import time
import json

import abc

import pickle

from shutil import copyfile

from mlpug.utils import get_value_at
from mlpug.trainers.callbacks.callback import Callback

import basics.base_utils as _


class CheckpointManager(Callback, metaclass=abc.ABCMeta):

    def __init__(self,
                 model_hyper_parameters=None,
                 checkpoints_path="../trained-models/",
                 batch_level=True,
                 metric_to_monitor="validation.window_average.perplexity",
                 metric_opt_mode='min',
                 metric_monitor_period=None,  # batches or epochs
                 metric_checkpoint_threshold=None,
                 allow_checkpointing_at_start=False,
                 create_checkpoint_every=200,  # batches or epochs
                 archive_last_model_checkpoint_every=2000,  # batches or epochs
                 create_latest_model_checkpoint=True,
                 create_training_checkpoint=False,
                 base_checkpoint_filename=time.strftime("%d-%m-%Y_%H-%M-%S"),
                 model_checkpoint_filename_ext="m-ckp",
                 training_checkpoint_filename_ext="t-ckp",
                 backup_before_override=True,
                 warn_on_model_quality_not_available=None,
                 name="CheckpointManager",
                 **kwargs):
        """

        :param model_hyper_parameters: Dict with hyper parameters of the model, this will be saved as part of
                                       the model checkpoint
        :param checkpoints_path:
        :param batch_level: True if monitoring happens on the scale of batches. If False, monitoring is done
                            on the epoch level
        :param metric_to_monitor: key path to metric value in the log object, e.g. `validation.batch.perplexity`
        :param metric_opt_mode: 'max', 'min'
        :param metric_monitor_period: If give, this is the period between checks for model quality improvement.
                                      This is in number of batches if `batch_level = True`, else it is a
                                      number of epochs.

                                      If not given, every time a value is available for the given metric_to_monitor
                                      the metric will be checked for improvement.

        :param metric_checkpoint_threshold: when given, the model quality must be below (or above, depending on
                                            metric_opt_mode) this threshold before a new best model checkpoint can be
                                            saved

        :param allow_checkpointing_at_start: Default False. When True: when a value for the metric_to_monitor is
                                             available at global batch iteration 0, checkpoints will be stored.

        :param create_checkpoint_every: period before saving latest training/model checkpoint
                                        (will be overridden after each period)
                                        A 0 value disables this feature.

        :param archive_last_model_checkpoint_every: period before the last available model checkpoint is archived.
                                                    Period must be multiple of create_checkpoint_every period

                                                    A 0 value disables this feature.

        :param create_latest_model_checkpoint:   If False, no latest model checkpoint will be created periodically
                                                 (also see `create_checkpoint_every`)

        :param create_training_checkpoint: If False, no training checkpoint will be created periodically
                                           (also see `create_checkpoint_every`)

        :param force_monitoring_on_epoch: When True, the given metric will also be monitored on every epoch
                                          in the case that monitoring level is batch level
        :param base_checkpoint_filename:
        :param model_checkpoint_filename_ext:
        :param training_checkpoint_filename_ext:
        :param backup_before_override: When True the training and model checkpoints are backed up before they are
                                       overridden by a new version. If backing up fails, no new version will be written
                                       to disk. This gives the user a chance to fix a problem, e.g. disk full, without
                                       interruption of the training process.

        :param warn_on_model_quality_not_available: Default is True when metric_monitor_period given else False.
                                                    If set to true and the metric_to_monitor is not available in
                                                    the logs, a warning will be logged.

                                                    This is useful for debugging purposes

        """
        super().__init__(name=name, **kwargs)

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
        self._allow_checkpointing_at_start = allow_checkpointing_at_start
        self._create_checkpoint_every = create_checkpoint_every
        self._archive_last_model_checkpoint_every = archive_last_model_checkpoint_every

        self._do_create_latest_model_checkpoint = create_latest_model_checkpoint
        self._do_create_training_checkpoint = create_training_checkpoint

        self._backup_before_override = backup_before_override

        if warn_on_model_quality_not_available is None:
            warn_on_model_quality_not_available = self._metric_monitor_period is not None

        self._warn_on_model_quality_not_available = warn_on_model_quality_not_available

        self._best_model_quality = float('Inf') if self._metric_opt_mode == 'min' else -float('Inf')
        self._best_model_iter = None

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
        create_latest_checkpoint = True
        if not self._batch_level:
            iter_name = 'epoch'
        else:
            # Monitor on epoch for batch level because checkpointing might be driven by
            # availability of metric_to_monitor value (which could also have been calculated specifically
            # at epoch ends.
            iter_name = 'global_iter'
            # ... but when monitoring on batch_level latest checkpoint might already be created
            create_latest_checkpoint = False

        self._log.debug("Epoch complete, checking if model quality improved ...")
        return self._monitor(iter_name, logs, create_latest_checkpoint=create_latest_checkpoint)

    def on_training_ended(self, stopped_early, stopped_on_error, interrupted, callback_calls_success):
        status = 'ended'
        if stopped_early:
            status = 'stopped early'

        if stopped_on_error:
            status = 'stopped on error'

        if interrupted:
            status = 'interrupted'

        sys.stdout.write("\n")
        sys.stdout.flush()
        self._log.info(f"Training {status}")
        success = True
        if self._create_checkpoint_every > 0:
            if self._do_create_latest_model_checkpoint:
                self._log.info(f"... storing latest model ...")
                latest_model_fname = self._create_model_checkpoint()
                success = (latest_model_fname is not None)

            if self._do_create_training_checkpoint and (status == 'ended' or stopped_early or interrupted):
                self._log.info(f"... storing training checkpoint ...")
                training_checkpoint_fname = self._create_training_checkpoint()
                success &= (training_checkpoint_fname is not None)

        if abs(self._best_model_quality) != float('Inf') and self._best_model_iter is not None:
            iter_name = "global iteration" if self._batch_level else "epoch"

            self._log.info(f"Best model quality reached {self._metric_to_monitor}={self._best_model_quality} "
                           f"at {iter_name} {self._best_model_iter}")
        else:
            self._log.warn(f"No good enough model found, no best model checkpoint saved.")

        return success

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
        if self._metric_monitor_period is not None:
            self._log.info(f"Metric monitor period : {self._metric_monitor_period}")
        else:
            self._log.info(f"Metric monitor period : Metric checked whenever it is available")

            if self._allow_checkpointing_at_start:
                self._log.info(f"Will create checkpoints at first global iteration, if a metric value is available")

        time_scale = 'batches' if self._batch_level else 'epochs'
        if self._create_checkpoint_every > 0:
            if self._do_create_latest_model_checkpoint:
                self._log.info(f"Create latest model checkpoints every "
                               f"{self._create_checkpoint_every} {time_scale}")

                self._log.info(f"Archive latest model checkpoint every "
                               f"{self._archive_last_model_checkpoint_every} {time_scale}")

            if self._do_create_training_checkpoint:
                self._log.info(f"Create latest training checkpoints every "
                               f"{self._create_checkpoint_every} {time_scale}")
        else:
            self._log.info(f"No latest model checkpoints and/or training checkpoint will be created")

    def _get_model_quality(self, current_logs):
        model_quality = get_value_at(self._metric_to_monitor,
                                     current_logs,
                                     warn_on_failure=self._warn_on_model_quality_not_available)

        if type(model_quality) is tuple:
            # use the first value as metric value, the other values are auxiliary results meant for other purposes
            model_quality = model_quality[0]

        return model_quality

    # TODO : it would maybe be better to split the monitoring and the checkpointing in to two separated methods
    def _monitor(self, iter_name, logs, create_latest_checkpoint=True):
        monitor_success = True
        data_saved = False

        current = self._get_logs_base(logs)
        training_iter = current[iter_name]
        if not self._allow_checkpointing_at_start and iter_name == "global_iter" and training_iter == 0:
            self._log.debug("Skipping checkpointing at first iteration")
            return monitor_success

        success, best_model_fname = self._monitor_for_best_checkpoint(iter_name, logs)
        monitor_success &= success
        data_saved |= best_model_fname is not None

        if create_latest_checkpoint:
            success, \
                latest_model_fname, \
                training_checkpoint_fname = self._monitor_for_latest_checkpoint(iter_name,
                                                                                logs,
                                                                                best_model_fname=best_model_fname)
            monitor_success &= success
            data_saved |= latest_model_fname is not None
            data_saved |= training_checkpoint_fname is not None

            # latest_model_fname can be None when the user doesn't want to create latest checkpoints
            if success and latest_model_fname is not None:
                success, \
                    archived_model_fname = self._monitor_for_checkpoint_archiving(iter_name,
                                                                                  logs,
                                                                                  latest_model_fname=latest_model_fname)
                monitor_success &= success
                data_saved |= archived_model_fname is not None
            elif not success:
                self._log.error("Unable to create checkpoint for latest model, will not archive latest model")

        if data_saved:
            self._log.debug("Saving of data done.\n\n")

        return monitor_success

    def _monitor_for_best_checkpoint(self, iter_name, logs):
        current = self._get_logs_base(logs)

        training_iter = current[iter_name]

        pretty_iter_name = self.pretty_iter_name(iter_name)

        best_model_fname = None
        success = True

        check_model_quality = self._metric_monitor_period is None or \
            ((self._metric_monitor_period > 0) and (training_iter % self._metric_monitor_period == 0))

        if not check_model_quality:
            return success, best_model_fname

        model_quality = self._get_model_quality(current)

        model_quality_available = model_quality is not None
        model_quality_good_enough = True
        if model_quality_available and self._metric_checkpoint_threshold is not None:
            if self._metric_opt_mode == 'min':
                model_quality_good_enough = model_quality <= self._metric_checkpoint_threshold
            elif self._metric_opt_mode == 'max':
                model_quality_good_enough = model_quality >= self._metric_checkpoint_threshold

        if not model_quality_good_enough:
            self._log.debug("%s : %s : model quality not good enough : %s : %3e " % (pretty_iter_name,
                                                                                     training_iter,
                                                                                     self._metric_to_monitor,
                                                                                     model_quality))

        if model_quality_available and model_quality_good_enough:
            model_improved = ((self._metric_opt_mode == 'min') and (model_quality < self._best_model_quality)) or \
                             ((self._metric_opt_mode == 'max') and (model_quality > self._best_model_quality))

            if model_improved:
                sys.stdout.write("\n")
                sys.stdout.flush()

                self._log.info("%s : %s : Model improved : %s : %3e " % (pretty_iter_name,
                                                                         training_iter,
                                                                         self._metric_to_monitor,
                                                                         model_quality))

                if (self._best_model_iter is not None) and (training_iter < self._best_model_iter):
                    self._log.warn(f"Inconsistency: according to the current training {pretty_iter_name.lower()} "
                                   f"({training_iter}), current best model training {pretty_iter_name.lower()} "
                                   f"({self._best_model_iter}) is in the future. "
                                   f"Was the right training checkpoint loaded?")

                best_model_fname = self._save_current_model_as_best()
                if best_model_fname is not None:
                    self._best_model_quality = model_quality
                    self._best_model_iter = training_iter
                else:
                    self._log.error("Unable to save improved model checkpoint")
                    success = False

            self._update_logs(model_improved, logs, current)

        return success, best_model_fname

    def _monitor_for_latest_checkpoint(self, iter_name, logs, best_model_fname=None):
        current = self._get_logs_base(logs)
        training_iter = current[iter_name]

        success = True
        save_latest_model = (self._create_checkpoint_every > 0) and (training_iter % self._create_checkpoint_every == 0)

        if not save_latest_model:
            return success, None, None

        latest_model_fname = None
        if self._do_create_latest_model_checkpoint:
            # Just copy best model if available
            latest_model_fname = self._create_model_checkpoint(file_to_copy=best_model_fname)
            success &= latest_model_fname is not None

        training_checkpoint_fname = None
        if self._do_create_training_checkpoint:
            # Also save latest training checkpoint
            training_checkpoint_fname = self._create_training_checkpoint()
            training_checkpoint_success = (training_checkpoint_fname is not None)

            success &= training_checkpoint_success

        return success, latest_model_fname, training_checkpoint_fname

    def _monitor_for_checkpoint_archiving(self, iter_name, logs, latest_model_fname):
        if latest_model_fname is None:
            self._log.error("Unable to archive latest model, no latest checkpoint provided")
            return False, None

        current = self._get_logs_base(logs)
        training_iter = current[iter_name]

        archive_last_model = (self._archive_last_model_checkpoint_every > 0) and \
                             (training_iter > 0) and \
                             (training_iter % self._archive_last_model_checkpoint_every == 0)

        success = True
        if not archive_last_model:
            return success, None

        archived_model_fname = None
        sys.stdout.write("\n")
        sys.stdout.flush()

        model_fname = self.current_model_file_name(training_iter)
        copy_success = self._copy(latest_model_fname, model_fname)
        if copy_success:
            archived_model_fname = model_fname

        success &= copy_success

        return success, archived_model_fname

    def _save_current_model_as_best(self):
        model_fn = self.best_model_file_name()

        if self._backup_before_override:
            err_msg = "A problem occurred backing up the last best model checkpoint, " \
                      "will not override override model checkpoint with new one. \n\n" \
                      "*** Please ensure there is enough disk space to store the backups and checkpoints ***\n\n"
            try:
                if not self._backup_checkpoint(model_fn):
                    self._log.error(err_msg)
                    return None
            except Exception as e:
                _.log_exception(self._log, err_msg, e)
                return None

        return self._create_model_checkpoint(model_fn=model_fn)

    def _update_logs(self, model_improved, logs, current):
        if model_improved:
            logs["best"] = current.copy()

        current["is_best"] = model_improved

    def _create_model_checkpoint(self, model_fn=None, file_to_copy=None):
        if model_fn is None:
            model_fn = self.latest_model_file_name()

        if file_to_copy:
            if not self._copy(file_to_copy, model_fn):
                self._log.error(f"Unable to create model checkpoint based on file : {file_to_copy}")
                return None
        else:
            try:
                state, success = self._gather_model_checkpoint_data()
                if state is not None:
                    if not success:
                        self._log.warn("Gathering the model checkpoint data was not completely successful, "
                                       "will save available checkpoint data anyway ...")

                    self._log.debug(f"Saving model checkpoint : {model_fn}")
                    self._save_model_checkpoint(model_fn, state)
                else:
                    return None
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
                    return None
            except Exception as e:
                _.log_exception(self._log, f"A problem occurred backing up the last training checkpoint, "
                                           f"will not override override training checkpoint with new one", e)
                return None

        try:
            state, success = self._gather_training_checkpoint_data()
            if state is not None:
                if not success:
                    self._log.warn("Gathering the training checkpoint data was not completely successful, "
                                   "will save available checkpoint data anyway ...")

                self._log.debug(f"Saving training checkpoint : {checkpoint_fname}")
                self._save_training_checkpoint(checkpoint_fname, state)
            else:
                return None
        except Exception as e:
            _.log_exception(self._log, f"Unable to save training checkpoint", e)
            return None

        return checkpoint_fname

    def _gather_model_checkpoint_data(self):
        """

        :return: state, success
        """
        state, success = self.trainer.get_model_components_state()

        if state is not None:
            if not success:
                self._log.warn("Getting the model components state was not completely successful, "
                               "continuing anyway ...")

            if self._model_hyper_parameters is not None:
                state['hyper_parameters'] = self._model_hyper_parameters

            try:
                manager_state, manager_state_success = self.training_manager.get_state_for_model_checkpoint()
                state['manager_state'] = manager_state

                if not manager_state_success:
                    self._log.warn("Getting the manager state for the model checkpoint was not successful, "
                                   "will continue anyway ...")
                    success = False

            except Exception as e:
                _.log_exception(self._log, "Unable to add manager state to model checkpoint, "
                                           "continuing anyway ...", e)
                success = False
        else:
            # In any case when the state is None, gathering the model checkpoint data is not successful
            success = False

        return state, success

    def _gather_training_checkpoint_data(self):
        """

        :return: state, success
        """
        return self.training_manager.get_state()

    def _save_model_checkpoint(self, filename, state):
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def _save_training_checkpoint(self, filename, state):
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def _backup_checkpoint(self, filename):
        if os.path.isfile(filename):
            return self._copy(filename, f"{filename}.backup")
        else:
            return True

    def _copy(self, source_fname, dest_fname):
        try:
            self._log.debug("Copying model:\n[%s] ==> \n[%s]\n" % (source_fname, dest_fname))
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

    def _get_callback_properties_for_hash(self):
        """
        This is used to create the unique callback hash.

        Returns a dict with properties describing the setup of the callback.
        It should at least contain the properties that influence the callback state.

        Property values should only be simple types, such as int, float, boolean and strings.
        Convert any object and function values (or similar) into a booleans (True = available, False = None)

        :return: dict
        """
        props = super()._get_callback_properties_for_hash()
        return {
            **props,
            "batch_level": self._batch_level,
            "metric_to_monitor": self._metric_to_monitor,
            "metric_opt_mode": self._metric_opt_mode,
            "metric_monitor_period": self._metric_monitor_period,
            "metric_checkpoint_threshold": self._metric_checkpoint_threshold,
        }

    @staticmethod
    def pretty_iter_name(iter_name):
        if iter_name == "epoch":
            return "Epoch"
        elif iter_name == "global_iter":
            return "Global iter"
        else:
            return iter_name
