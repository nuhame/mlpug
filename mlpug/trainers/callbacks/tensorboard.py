import os

from mlpug.trainers.callbacks.callback import Callback
from mlpug.utils import get_value_at as get_value_at_func
from mlpug.utils import has_key

import basics.base_utils as _
from basics.logging_utils import log_exception
from basics.logging import get_logger

module_logger = get_logger(os.path.basename(__file__))

try:
    from tensorboardX import SummaryWriter
except Exception as e:
    log_exception(module_logger, "Please `pip install transformers`", e)


# Tensorboard writer types
METRIC_WRITER = 'metrics'
WINDOW_AVERAGED_METRICS_WRITER = 'window-averaged-metrics'


def get_value_at(path, obj):
    return get_value_at_func(path, obj, warn_on_failure=False)


def tagify(label):
    return label.replace(' ', '_').lower()


class Tensorboard(Callback):

    def __init__(self,
                 metric_paths,
                 experiment_name=None,
                 dataset_name=None,
                 label_name=None,
                 batch_level=True,
                 track_on_start=False,
                 track_epoch_level_metrics_on_batch_level=False,
                 metrics_are_averages=False,
                 metric_names=None,
                 batch_log_period=1,
                 flush_period=50,
                 log_dir='../training-logs/',
                 tensorboard_options=None,
                 ignore_missing_metrics=False,
                 name="Tensorboard",
                 **kwargs):
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
        Tensorboard(['batch.duration', 'batch_size'], 'experiment1')
        ```
        Will show a figure with the batch duration and the batch size for each batch.




        ## Using `dataset_name`

        If `dataset_name` is given, it looks for the given `metric_paths` in the path with `dataset_name` first.
        Only, if the a given path is not available within the `dataset_name` path, we try to retrieve it as is.

        For example, assume that the `logs` object, as received by the training manager, is as follows:
        ```
        {
            ...
            'training': {
                'window_average': {
                    'loss': 0.201
                }
            },
            'validation': {
                'window_average': {
                    'loss': 0.234
                }
            },
            'training_params': {
                'batch': { 'duration': 0.534 },
                'window_average': { 'duration':  0.531 }
            }
        }
        ```
        The following `Tensorboard`
        ```
        Tensorboard(['window_average.loss', 'window_average.duration'], 'experiment1', 'validation')
        ```
        Will display a figure with `validation.window_average.loss`, because `window_average.loss` is
        available in `validation`.


        ### Showing metrics for different datasets in one graph using `dataset_name`

        By creating multiple `Tensorboard` instances with different data set names, using `dataset_name`, a metric
        path, defined in all instances will be shown in one graph.

        ```
        t_train = Tensorboard(['window_average.loss'], 'experiment1', 'training')
        t_val = Tensorboard(['window_average.loss'], 'experiment1', 'validation')
        ```

        Registering the above tensorboards will plot the window_average Loss for the training set and validation set in
        one graph




        ## Using `label_name` and `metric_names`

        Using `label_name`, in combination with `metric_names`, allows us to combine multiple plots for the
        same metric, but for different categories or labels, in one figure.

        for instance, if we have the following data in a log object:
        {
            ...
           'training': {
                'recall': {
                    'hamburger' : 0.646
                    'hotdog': 0.873
                },
                'precision': {
                    'hamburger' : 0.931
                    'hotdog': 0.829
                }
            },
            ...
        }

        ```
        metric_names = {
            'recall.hamburger': 'Recall',
            'recall.hotdog': 'Recall',
            'precision.hamburger': 'Precision',
            'precision.hotdog': 'Precision'
        }
        ```

        we can create the following tensorboard callbacks
        ```
        t_hamburger = Tensorboard(['recall.hamburger', 'precision.hamburger'],
                                   experiment_name = 'experiment1',
                                   dataset_name = 'training',
                                   label_name = 'hamburger',
                                   metric_names = metric_names)
        t_hotdog = Tensorboard(['recall.hotdog', 'precision.hotdog'],
                                   experiment_name = 'experiment1',
                                   dataset_name = 'training',
                                   label_name = 'hotdog',
                                   metric_names = metric_names)
        ```

        When registering these boards the `hamburger` and `hotdog` recall plots will be combined in one recall plot.
        Further, the `hamburger` and `hotdog` precision plots will be combined in one precision plot.


        ## Using `metrics_are_averages` and `metric_names`

        If `metrics_are_averages=True` the metrics are written to Tensorboard using another writer, such that
        a metric and its average can be shown in one figure.

        For instance,

        ```
        metric_names = {
            'batch.loss': 'cross_entropy',
            'window_average.loss': 'cross_entropy'
        }

        Tensorboard([loss], 'experiment1', 'validation', metric_names = metric_names)
        ```
        Displays a figure with the `validation.batch.loss` at every batch as `cross_entropy`

        When also adding the following `Tensorboard`:
        ```
        Tensorboard([window_average.loss],
                    'experiment1',
                    'validation',
                    metric_names = metric_names,
                    metrics_are_averages=True)
        ```

        Adds the `validation.window_average.loss` averages to the same batch-level `cross_entropy` figure.





        To show the Tensorboard start is as follows:

        `[nohup] tensorboard --logdir <log_dir>/<experiment_name> [--port <my port>] [--bind_all] [&]`

        E.g.

        `tensorboard --logdir ../training-logs/my-first-exp-01012020-0`

        Note : only provide the path up to the `<log_dir>/<experiment_name>`

        :param metric_paths: {list} The key paths in the logs object, specifying the metrics to show.

        :param experiment_name: See log_dir

        :param dataset_name:    See log_dir. Further, when provided, the callback will look for metrics in the
                                <dataset_name>.* path first of the logs object provided by the training manager.

        :param dataset_name:    See log_dir. Further, when provided

        :param batch_level:     {boolean} If True, the given metrics will be logged per batch, else per epoch

        :param track_on_start: {boolean} If True, track on start of batch or start of epoch instead of on
                                         batch training completed or epoch training completed
                                         (depends on batch_level parameter)

        :param track_epoch_level_metrics_on_batch_level: {boolean} If True, and batch_level = True, epoch level metrics
                                        will also be written to batch level graphs.

        :param metrics_are_averages: {boolean}
                                     If True, the given metrics are written by a separate writer dedicated
                                     to metrics window averages

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

        :param log_dir:         Logging directory. When `experiment_name`, `dataset_name` and `label_name` are
                                provided, the final logging directory will be:

                                log_dir/<experiment_name>/<label_name>-<dataset-name>-[metric or metric-window-averages]

                                Any parameters not provided will not be used to build up the log_dir

        :param ignore_missing_metrics {Boolean}

        :param name:
        """

        super().__init__(name=name, **kwargs)

        self._metric_paths = metric_paths

        self._experiment_name = experiment_name
        self._dataset_name = dataset_name
        self._label_name = label_name

        self._batch_level = batch_level
        self._track_on_start = track_on_start
        self._track_epoch_level_metrics_on_batch_level = track_epoch_level_metrics_on_batch_level
        self._metrics_are_averages = metrics_are_averages

        self._metric_names = metric_names

        self._batch_log_period = batch_log_period

        self._flush_period = flush_period

        self._log_dir = log_dir

        self._tensorboard_options = tensorboard_options or {}

        self._ignore_missing_metrics = ignore_missing_metrics

        self._writer_type = WINDOW_AVERAGED_METRICS_WRITER if self._metrics_are_averages else METRIC_WRITER

        self._writer = None

    def _setup(self):
        try:
            success = self._setup_writer_for(self._writer_type)
        except Exception as e:
            _.log_exception(self._log, "An exception occurred setting up the Tensorboard callback", e)
            success = False

        self._valid = success

        return self._valid

    def get_dataset_name(self):
        return self._dataset_name

    def on_training_start(self,
                          num_epochs,
                          num_batches_per_epoch,
                          start_epoch,
                          start_batch,
                          start_update_iter):

        return self._setup()

    def on_epoch_start(self, logs):
        if self._track_on_start:
            return self._write_epoch_metrics(logs)
        else:
            return True

    def on_batch_training_start(self, dataset_batch, logs):
        if self._track_on_start:
            return self._write_batch_metrics(dataset_batch, logs)
        else:
            return True

    def on_batch_training_completed(self, dataset_batch, logs):
        if not self._track_on_start:
            return self._write_batch_metrics(dataset_batch, logs)
        else:
            return True

    def on_epoch_completed(self, logs):
        if not self._track_on_start:
            return self._write_epoch_metrics(logs)
        else:
            return True

    def on_training_ended(self, stopped_early, stopped_on_error, interrupted, callback_calls_success):
        self._writer.close()
        return True

    def _write_batch_metrics(self, dataset_batch, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if not self._batch_level:
            return True

        current = self._get_logs_base(logs)
        global_iter = current['global_iter']

        if global_iter > 0 and (global_iter % self._batch_log_period != 0):
            return True

        return self._write(current, global_iter)

    def _write_epoch_metrics(self, logs):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if self._batch_level:
            if self._track_epoch_level_metrics_on_batch_level:
                # There are dataset level metrics calculated that are followed on batch level.
                return self.on_batch_training_completed(None, logs)
            else:
                return True

        current = self._get_logs_base(logs)
        epoch = current['epoch']

        return self._write(current, epoch)

    def _setup_tensorboard_log_dir(self, writer_type):
        """
        TODO : Put in TensorBase class so it can be reused by all Tensorboard types implemented

        :param writer_type:
        :return:
        """
        paths = [self._log_dir]
        if self._experiment_name:
            paths.append(self._experiment_name)

        writer_name = []
        if self._label_name:
            writer_name.append(self._label_name)

        if self._dataset_name:
            writer_name.append(self._dataset_name)

        writer_name.append(writer_type)

        writer_name = '-'.join(writer_name)

        paths.append(writer_name)

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

    def _write(self, current_logs, training_iter):
        metrics, success = self._get_metrics_from(current_logs)

        for label, metric in metrics.items():
            if type(metric) is tuple:
                # use the first value as metric value, the other values are auxiliary results meant for other purposes
                metric = metric[0]

            tag = tagify(label)
            self._writer.add_scalar(tag, metric, global_step=training_iter, display_name=label)

        if training_iter % self._flush_period == 0:
            self._writer.flush()

        return success

    def _get_metrics_from(self, current_logs):
        if not _.is_sequence(self._metric_paths) or len(self._metric_paths) == 0:
            self._log.error(f'No valid metrics to show, nothing to get')
            return None, False

        return self._get_specific_metrics_from(current_logs, self._metric_paths)

    def _get_specific_metrics_from(self, current_logs, metric_paths):
        success = True
        metrics = {}

        def _try_add_metric(metric_path, metric):
            if metric is None:
                return None

            label = self._get_label(metric_path)
            metrics[label] = metric

            return metric

        for metric_path in metric_paths:
            if self._dataset_name:
                set_metric_path = f'{self._dataset_name}.{metric_path}'
                metric = get_value_at(set_metric_path, current_logs)

                if _try_add_metric(metric_path, metric) is not None:
                    continue

            metric = get_value_at(metric_path, current_logs)

            if not _try_add_metric(metric_path, metric) and not self._ignore_missing_metrics:
                self._log.error(f'Unable to find metric for metric path {metric_path}')
                success = False

        return metrics, success

    def _get_label(self, metric_name):
        iter_level = 'batch' if self._batch_level else 'epoch'

        label = self._metric_names[metric_name] if (metric_name in self._metric_names) else metric_name
        label = label.replace('.', ' ').replace('_', ' ').title()
        label = f"{label} - per {iter_level}"

        return label


class AutoTensorboard(Callback):

    def __init__(self,
                 dataset_name,
                 experiment_name=None,
                 show_batch_level=True,
                 show_batch_window_averages=True,
                 show_epoch_level=True,
                 metric_names=None,
                 batch_log_period=1,
                 flush_period=50,
                 log_dir='../training-logs/',
                 tensorboard_options=None,
                 debug=False,
                 name="AutoTensorboard",
                 **kwargs):
        """

        Shows batch-level metrics, and/or averaged batch-level (metrics), and/or epoch level metrics.

        All available metrics for `dataset_name` in the `logs` object received from the
        training manager are logged

        Example, if `dataset_name='validation'` and the received `logs` object contains:

        {
            ...

            'validation': {
                            'batch': {
                                'loss': 0.65,
                            },
                            'window_average': {
                                      'loss': 0.63
                                    }
                           }
            ...
        }

        When `show_batch_level` is True, then `validation.batch.loss` will be shown in
        the batch-level `loss` figure.

        When `show_batch_window_averages` is True, then `validation.window_averages.loss` will be shown in
        the batch-level `loss` figure.

        When `show_epoch_level` is True, then `validation.dataset.loss` will be shown in
        the epoch-level `loss` figure. Further, if `validation.dataset.loss` is not available,
        `validation.window_average.loss` will be tried


        To show the Tensorboard start is as follows:

        `[nohup] tensorboard --logdir <log_dir>/<experiment_name> [--port <my port>] [--bind_all] [&]`

        E.g.

        `tensorboard --logdir ../training-logs/my-first-exp-01012020-0`

        Note : only provide the path up to `<log_dir>/<experiment_name>`

        :param experiment_name: Experiment name, used in the log_dir, see log_dir.

        :param dataset_name:    See log_dir. Further, when provided, the callback will look for metrics in the
                                <dataset_name>.* path first of the logs object provided by the training manager.

        :param show_batch_level: {Boolean} If True, batch-level metrics will be logged to Tensorboard
        :param show_batch_window_averages: {Boolean} If True, (sliding-window) averaged batch-level metrics will be logged
                                           to Tensorboard
        :param show_epoch_level: {Boolean} If True, epoch-level metrics will be logged to Tensorboard

        :param metric_names:    {dict}
                                Dict mapping metric paths (keys), to metric names for the metrics (values).

                                Example:
                                {
                                    'loss': 'cross_entropy',
                                    'special.my_metric': 'my_awesome_metrics'
                                }

                                In this case `validation.loss` and `validation.window_average.loss` are
                                named 'cross_entropy'

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

        :param :param debug {Boolean}, If True, will provide feedback which paths did not provide any metrics

        :param name:
        """
        super().__init__(name=name, **kwargs)

        self._experiment_name = experiment_name
        self._dataset_name = dataset_name

        self._show_batch_level = show_batch_level
        self._show_batch_window_averages = show_batch_window_averages
        self._show_epoch_level = show_epoch_level

        self._metric_names = metric_names

        self._batch_log_period = batch_log_period

        self._flush_period = flush_period

        self._log_dir = log_dir

        self._tensorboard_options = tensorboard_options or {}

        self._debug = debug

        self._writers = dict()

    def _setup(self):
        try:
            # Metrics writer is used for batch level and epoch level
            success = self._setup_writer_for(METRIC_WRITER)

            if self._show_batch_level and self._show_batch_window_averages:
                success &= self._setup_writer_for(WINDOW_AVERAGED_METRICS_WRITER)

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

    def on_epoch_start(self, logs):
        return self._write_epoch_metrics(logs, at_start=True)

    def on_batch_training_start(self, dataset_batch, logs):
        return self._write_batch_metrics(dataset_batch, logs, at_start=True)

    def on_batch_training_completed(self, dataset_batch, logs):
        return self._write_batch_metrics(dataset_batch, logs, at_start=False)

    def on_epoch_completed(self, logs):
        return self._write_epoch_metrics(logs, at_start=False)

    def on_training_ended(self, stopped_early, stopped_on_error, interrupted, callback_calls_success):
        for writer in self._writers.values():
            writer.close()
        return True

    def _write_batch_metrics(self, dataset_batch, logs, at_start):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        current = self._get_logs_base(logs)
        global_iter = current['global_iter']

        if global_iter > 0 and (global_iter % self._batch_log_period != 0):
            return True

        success = True
        if self._show_batch_level:
            success &= self._write(current,
                                   global_iter,
                                   writer_type=METRIC_WRITER,
                                   batch_level=True,
                                   at_start=at_start)

        if self._show_batch_window_averages:
            success &= self._write(current,
                                   global_iter,
                                   writer_type=WINDOW_AVERAGED_METRICS_WRITER,
                                   batch_level=True,
                                   at_start=at_start)

        return success

    def _write_epoch_metrics(self, logs, at_start):
        if not self.instance_valid():
            self._log.error(f"{self} is not valid, skipping this hook ... ")
            return False

        if not self._show_epoch_level:
            return True

        current = self._get_logs_base(logs)
        epoch = current['epoch']

        return self._write(current, epoch, writer_type=METRIC_WRITER, batch_level=False, at_start=at_start)

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

    def _get_call_info(self, batch_level, at_start):
        metric_level = 'batch' if batch_level else 'epoch'
        timing = 'start' if at_start else 'completion'

        return f"{timing} of {metric_level}"

    def _write(self, current_logs, training_iter, writer_type, batch_level, at_start):
        metrics, success = self._get_metrics_from(current_logs, writer_type, batch_level, at_start=at_start)

        if not success:
            return True

        for label, metric in metrics.items():
            tag = tagify(label)

            self._writers[writer_type].add_scalar(tag, metric, global_step=training_iter, display_name=label)

        if training_iter % self._flush_period == 0:
            self._writers[writer_type].flush()

        return success

    def _get_metrics_from(self, current_logs, writer_type, batch_level, at_start):
        # Get all metrics available
        if self._dataset_name:
            if batch_level:
                if writer_type != WINDOW_AVERAGED_METRICS_WRITER:
                    possible_base_paths = [f"{self._dataset_name}.batch"]
                else:
                    possible_base_paths = [f"{self._dataset_name}.window_average"]
            else:
                possible_base_paths = [f"{self._dataset_name}.dataset",
                                       f"{self._dataset_name}.epoch"]

            metrics = None
            success = False
            for base_path in possible_base_paths:
                metrics, success = self._get_all_metrics_from(base_path, current_logs, batch_level)

                if success:
                    break

            if not success and self._debug:
                self._log.debug(f"No metrics found at {self._get_call_info(batch_level, at_start)} "
                                f"for following paths : {possible_base_paths}")

            return metrics, success
        else:
            self._log.error(f'Don\'t know how to get metrics to log without a dataset name or '
                            f'a list of metrics to show')
            return None, False

    def _get_all_metrics_from(self, base_path, current_logs, batch_level):
        dataset_metrics = get_value_at(base_path, current_logs)

        # TODO : Make this a library level constant
        skip_metric_names = {"auxiliary_results"}

        metrics = {}

        def _add_metrics(m, base_path=None):
            for metric_name, metric in m.items():
                if metric_name in skip_metric_names:
                    continue

                if metric is None:
                    continue

                if base_path is not None:
                    metric_name = f"{base_path}.{metric_name}"

                if type(metric) is tuple and len(metric) > 0:
                    # use the first value as metric value, the other values are auxiliary results meant for
                    # other purposes
                    metric = metric[0]

                if _.is_dict(metric):
                    _add_metrics(metric, metric_name)
                    continue

                label = self._get_label(metric_name, batch_level)

                metrics[label] = metric

        if _.is_dict(dataset_metrics):
            _add_metrics(dataset_metrics)

        return metrics, len(metrics) > 0

    def _get_label(self, metric_name, batch_level):
        iter_level = 'batch' if batch_level else 'epoch'

        label = self._metric_names[metric_name] if has_key(self._metric_names, metric_name) else metric_name
        label = label.replace('.', ' ').replace('_', ' ').title()
        label = f"{label} - per {iter_level}"

        return label
