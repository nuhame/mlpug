# TODO : remove PyTorch dependency
import os
from torch.utils.tensorboard import SummaryWriter

from mlpug.trainers.callbacks.callback import Callback
from mlpug.utils import get_value_at as get_value_at_func

import basics.base_utils as _


# Tensorboard writer types
METRIC_WRITER = 'metrics'
METRIC_MEANS_WRITER = 'metric_means'


def get_value_at(path, obj):
    return get_value_at_func(path, obj, warn_on_failure=False)


class Tensorboard(Callback):

    def __init__(self,
                 metric_paths,
                 experiment_name=None,
                 dataset_name=None,
                 batch_level=True,
                 metrics_are_averages=False,
                 metric_names=None,
                 batch_log_period=1,
                 flush_period=50,
                 log_dir='../training-logs/',
                 tensorboard_options=None,
                 ignore_missing_metrics=False,
                 name="Tensorboard"):
        """

        Shows batch-level or epoch-level metrics, as defined by the metric key paths in `metric_paths`.

        For instance, assume that the `logs` object, as received by the training manager, is as follows:
        ```
        {
            ...
            'duration': {
                'batch': 0.534
                'epoch': 623.23
            },
            ...
            'batch_size': 200
            ...
        }
        ```

        Then, the following `Tensorboard`
        ```
        Tensorboard(['duration.batch', 'batch_size'], 'experiment1')
        ```
        Will show a figure with the batch duration and the batch size for each batch.

        If `dataset_name` is given, it looks for the given `metric_paths` in the path with `dataset_name` first.
        Only, if the a given path is not available within the `dataset_name` path, we try to retrieve it as is.

        For example, assume that the `logs` object, as received by the training manager, is as follows:
        ```
        {
            ...
            'validation': {
                'mean': {
                    'loss': 0.234
                }
            },
            'duration': {
                'batch': 0.534,
                'mean' : {
                    'batch' : 0.531
                }
            }
        }
        ```
        The following `Tensorboard`
        ```
        Tensorboard(['mean.loss', 'duration.mean.batch'], 'experiment1', 'validation')
        ```
        Will display a figure with `validation.mean.loss`, because `mean.loss`is available in `validation`.

        If `metrics_are_averages=True` the metrics are written
        to Tensorboard using another writer, such that a metric and its average can be shown in one figure.

        For instance,

        ```
        metric_names = {
            'loss': 'cross_entropy',
            'mean.loss': 'cross_entropy'
        }

        Tensorboard([loss], 'experiment1', 'validation', metric_names = metric_names)
        ```
        Displays a figure with the `validation.loss` at every batch as `cross_entropy`

        When also adding the following `Tensorboard`:
        ```
        Tensorboard([mean.loss], 'experiment1', 'validation', metric_names = metric_names, metrics_are_averages=True)
        ```

        Adds the `validation.mean.loss` averages to the batch-level `cross_entropy` figure.

        To show the Tensorboard start is as follows:

        `[nohup] tensorboard --logdir <log_dir>/<experiment_name> [--port <my port>] [--bind_all] [&]`

        E.g.

        `tensorboard --logdir ../training-logs/my-first-exp-01012020-0`

        Note : only provide the path up to the `<log_dir>/<experiment_name>`

        :param metric_paths: {list} The key paths in the logs object, specifying the metrics to show.

        :param experiment_name: See log_dir

        :param dataset_name:    See log_dir. Further, when provided, the callback will look for metrics in the
                                <dataset_name>.* path first of the logs object provided by the training manager.

        :param batch_level:     {boolean} If True, the given metrics will be logged per batch, else per epoch

        :param metrics_are_averages: {boolean}
                                     If True, the given metrics are written by a separate writer dedicated
                                     to metrics means

        :param metric_names:    {dict}
                                Dict mapping metric paths (keys), to metric names for the metrics (values).

                                Example:
                                {
                                    'loss': 'cross_entropy',
                                    'special.my_metric': 'my_awesome_metrics'
                                }

        :param batch_log_period: {int}
                                 period in batch iterations, after which the `batch_metrics` are logged

        :param flush_period : {int}
                                 period in batch iterations, after which the tensorboard writers are flushed

        :param tensorboard_options: {None|dict}
                                    Optional dictionary with options that need to be used when instantiating the
                                    Tensorboard writers.

        :param log_dir:         Logging directory. When experiment_name and dataset_name are provided, the final
                                logging directory will be:

                                log_dir/experiment_name/dataset_name

                                When only the dataset_name is given:

                                log_dir/dataset_name

        :param ignore_missing_metrics {Boolean}

        :param name:
        """

        super().__init__(name=name)

        self._metric_paths = metric_paths

        self._experiment_name = experiment_name
        self._dataset_name = dataset_name

        self._batch_level = batch_level
        self._metrics_are_averages = metrics_are_averages

        self._metric_names = metric_names

        self._batch_log_period = batch_log_period

        self._flush_period = flush_period

        self._log_dir = log_dir

        self._tensorboard_options = tensorboard_options or {}

        self._ignore_missing_metrics = ignore_missing_metrics

        self._writer_type = METRIC_MEANS_WRITER if self._metrics_are_averages else METRIC_WRITER

        self._writer = None

    def _setup(self):
        try:
            success = self._setup_writer_for(self._writer_type)
        except Exception as e:
            _.log_exception(self._log, "An exception occurred setting up the Tensorboard callback", e)
            success = False

        self._valid = success

        return self._valid

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):

        return self._setup()

    def on_batch_training_completed(self, dataset_batch, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if not self._batch_level:
            return True

        global_iter = logs['global_iter']

        if global_iter > 0 and (global_iter % self._batch_log_period != 0):
            return True

        return self._write(logs, global_iter)

    def on_epoch_completed(self, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if self._batch_level:
            return True

        epoch = logs['epoch']

        return self._write(logs, epoch)

    def on_training_ended(self, stopped_early, stopped_on_error, callback_calls_success):
        self._writer.close()
        return True

    def _setup_tensorboard_log_dir(self, writer_type):
        """
        TODO : Put in TensorBase class so it can be reused by all Tensorboard types implemented

        :param writer_type:
        :return:
        """
        paths = [self._log_dir]
        if self._experiment_name:
            paths.append(self._experiment_name)

        subpath = ""
        if self._dataset_name:
            subpath = f'{self._dataset_name}_'

        subpath += writer_type
        paths.append(subpath)

        tensorboard_log_dir = os.path.join(*paths)

        self._log.debug(f'Tensorboard directory for {writer_type}: {tensorboard_log_dir}')

        if not os.path.exists(tensorboard_log_dir):
            self._log.debug(f'Directory doesn\'t exist, creating directory')
            os.makedirs(tensorboard_log_dir)

        return tensorboard_log_dir

    def _setup_writer_for(self, writer_type):
        """
        TODO : Put in TensorBase class so it can be reused by all Tensorboard types implemented

        :param writer_type:
        :return:
        """
        self._log.debug(f'Setting up Tensorboard writer for {writer_type} ...')

        tensorboard_log_dir = self._setup_tensorboard_log_dir(writer_type)

        tensorboard_options = {**{
            'log_dir': tensorboard_log_dir
        }, **self._tensorboard_options}

        self._writer = SummaryWriter(**tensorboard_options)

        return True

    def _write(self, logs, training_iter):
        metrics, success = self._get_metrics_from(logs)

        for tag, metric in metrics.items():
            self._writer.add_scalar(tag, metric, global_step=training_iter)

        if training_iter % self._flush_period == 0:
            self._writer.flush()

        return success

    def _get_metrics_from(self, logs):
        if not _.is_sequence(self._metric_paths) or len(self._metric_paths) == 0:
            self._log.error(f'No valid metrics to show, nothing to get')
            return None, False

        return self._get_specific_metrics_from(logs, self._metric_paths)

    def _get_specific_metrics_from(self, logs, metric_paths):
        success = True
        metrics = {}

        def _try_add_metric(metric_path, metric):
            if not metric:
                return None

            tag = self._get_tag(metric_path)
            metrics[tag] = metric

            return metric

        for metric_path in metric_paths:
            if self._dataset_name:
                set_metric_path = f'{self._dataset_name}.{metric_path}'
                metric = get_value_at(set_metric_path, logs)

                if _try_add_metric(metric_path, metric):
                    continue

            metric = get_value_at(metric_path, logs)

            if not _try_add_metric(metric_path, metric) and not self._ignore_missing_metrics:
                self._log.error(f'Unable to find metric for metric path {metric_path}')
                success = False

        return metrics, success

    def _get_tag(self, metric_name):
        prefix = 'batch' if self._batch_level else 'epoch'

        tag = self._metric_names[metric_name] if (metric_name in self._metric_names) else metric_name
        tag = f"{prefix}_{tag}"

        return tag


class AutoTensorboard(Callback):

    def __init__(self,
                 dataset_name,
                 experiment_name=None,
                 show_batch_level=True,
                 show_batch_means=True,
                 show_epoch_level=True,
                 metric_names=None,
                 batch_log_period=1,
                 flush_period=50,
                 log_dir='../training-logs/',
                 tensorboard_options=None,
                 name="AutoTensorboard"):
        """

        Shows batch-level metrics, and/or averaged batch-level (metrics), and/or epoch level metrics.

        All available metrics for `dataset_name` in the `logs` object received from the
        training manager are logged

        Example, if `dataset_name='validation'` and the received `logs` object contains:

        {
            ...

            'validation': {
                            'loss': 0.65,
                            'mean': {
                                      'loss': 0.63
                                    }
                           }
            ...
        }

        When `show_batch_level` is True, then `validation.loss` will be shown in
        the batch-level `loss` figure.

        When `show_batch_means` is True, then `validation.mean.loss` will be shown in
        the batch-level `loss` figure.

        When `show_epoch_level` is True, then `validation.mean.loss` will be shown in
        the epoch-level `loss` figure.


        To show the Tensorboard start is as follows:

        `[nohup] tensorboard --logdir <log_dir>/<experiment_name> [--port <my port>] [--bind_all] [&]`

        E.g.

        `tensorboard --logdir ../training-logs/my-first-exp-01012020-0`

        Note : only provide the path up to `<log_dir>/<experiment_name>`

        :param experiment_name: Experiment name, used in the log_dir, see log_dir.

        :param dataset_name:    See log_dir. Further, when provided, the callback will look for metrics in the
                                <dataset_name>.* path first of the logs object provided by the training manager.

        :param show_batch_level: {Boolean} If True, batch-level metrics will be logged to Tensorboard
        :param show_batch_means: {Boolean} If True, (sliding-window) averaged batch-level metrics will be logged
                                           to Tensorboard
        :param show_epoch_level: {Boolean} If True, epoch-level metrics will be logged to Tensorboard

        :param metric_names:    {dict}
                                Dict mapping metric paths (keys), to metric names for the metrics (values).

                                Example:
                                {
                                    'loss': 'cross_entropy',
                                    'special.my_metric': 'my_awesome_metrics'
                                }

                                In this case `validation.loss` and `validation.mean.loss` are named 'cross_entropy'

        :param batch_log_period: {int}
                                 period in batch iterations, after which the `batch_metrics` are logged

        :param flush_period : {int}
                                 period in batch iterations, after which the tensorboard writers are flushed

        :param log_dir:         Logging directory. When experiment_name and dataset_name are provided, the final
                                logging directory will be:

                                log_dir/experiment_name/dataset_name

                                When only the dataset_name is given:

                                log_dir/dataset_name

        :param tensorboard_options: {None|dict}
                                    Optional dictionary with options that need to be used when instantiating the
                                    Tensorboard writers.

        :param name:
        """
        super().__init__(name=name)

        self._experiment_name = experiment_name
        self._dataset_name = dataset_name

        self._show_batch_level = show_batch_level
        self._show_batch_means = show_batch_means
        self._show_epoch_level = show_epoch_level

        self._metric_names = metric_names

        self._batch_log_period = batch_log_period

        self._flush_period = flush_period

        self._log_dir = log_dir

        self._tensorboard_options = tensorboard_options or {}

        self._writers = dict()

    def _setup(self):
        try:

            # Metrics writer is used for batch level and epoch level
            success = self._setup_writer_for(METRIC_WRITER)

            if self._show_batch_level and self._show_batch_means:
                success &= self._setup_writer_for(METRIC_MEANS_WRITER)

        except Exception as e:
            _.log_exception(self._log, "An exception occurred setting up the Tensorboard callback", e)
            success = False

        self._valid = success

        return self._valid

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):

        return self._setup()

    def on_batch_training_completed(self, dataset_batch, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        global_iter = logs['global_iter']

        if global_iter > 0 and (global_iter % self._batch_log_period != 0):
            return True

        success = True
        if self._show_batch_level:
            success &= self._write(logs, global_iter, writer_type=METRIC_WRITER, batch_level=True)

        if self._show_batch_means:
            success &= self._write(logs, global_iter, writer_type=METRIC_MEANS_WRITER, batch_level=True)

        return success

    def on_epoch_completed(self, logs):

        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if not self._show_epoch_level:
            return

        epoch = logs['epoch']

        return self._write(logs, epoch, writer_type=METRIC_WRITER, batch_level=False)

    def on_training_ended(self, stopped_early, stopped_on_error, callback_calls_success):
        for writer in self._writers.values():
            writer.close()
        return True

    def _setup_tensorboard_log_dir(self, writer_type):
        paths = [self._log_dir]
        if self._experiment_name:
            paths.append(self._experiment_name)

        subpath = ""
        if self._dataset_name:
            subpath = f'{self._dataset_name}_'

        subpath += writer_type
        paths.append(subpath)

        tensorboard_log_dir = os.path.join(*paths)

        self._log.debug(f'Tensorboard directory for {writer_type}: {tensorboard_log_dir}')

        if not os.path.exists(tensorboard_log_dir):
            self._log.debug(f'Directory doesn\'t exist, creating directory')
            os.makedirs(tensorboard_log_dir)

        return tensorboard_log_dir

    def _setup_writer_for(self, writer_type):
        self._log.debug(f'Setting up Tensorboard writer for {writer_type} ...')

        tensorboard_log_dir = self._setup_tensorboard_log_dir(writer_type)

        tensorboard_options = {**{
            'log_dir': tensorboard_log_dir
        }, **self._tensorboard_options}

        self._writers[writer_type] = SummaryWriter(**tensorboard_options)

        return True

    def _write(self, logs, training_iter, writer_type, batch_level):
        metrics, success = self._get_metrics_from(logs, writer_type, batch_level)

        for tag, metric in metrics.items():
            self._writers[writer_type].add_scalar(tag, metric, global_step=training_iter)

        if training_iter % self._flush_period == 0:
            self._writers[writer_type].flush()

        return success

    def _get_metrics_from(self, logs, writer_type, batch_level):
        # Get all metrics available
        if self._dataset_name:
            base_path = self._dataset_name
            if (writer_type == METRIC_MEANS_WRITER) or (not batch_level):
                base_path += '.mean'

            return self._get_all_metrics_from(base_path, logs, batch_level)
        else:
            self._log.error(f'Don\'t know how to get metrics to log without a dataset name or '
                            f'a list of metrics to show')
            return None, False

    def _get_all_metrics_from(self, base_path, logs, batch_level):
        dataset_metrics = get_value_at(base_path, logs)

        metrics = {}

        def _add_metrics(m):
            for metric_name, metric in m.items():
                if metric is None:
                    continue

                if _.is_dict(metric):
                    continue

                tag = self._get_tag(metric_name, batch_level)

                metrics[tag] = metric

        if _.is_dict(dataset_metrics):
            _add_metrics(dataset_metrics)

            if len(metrics) > 0:
                return metrics, True
            else:
                self._log.error(f'No metrics found at {base_path}')
                return metrics, False
        else:
            self._log.error(f'No valid metrics found for {base_path}')
            return None, False

    def _get_tag(self, metric_name, batch_level):
        prefix = 'batch' if batch_level else 'epoch'

        tag = self._metric_names[metric_name] if (metric_name in self._metric_names) else metric_name
        tag = f"{prefix}_{tag}"

        return tag
