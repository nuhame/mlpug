from typing import Callable

import tensorflow as tf

from mlpug.base import Base


class MonitorTracingWrapper(Base):

    def __init__(self, func: Callable, monitor_tracing=False):
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)

        name = func_name
        if monitor_tracing:
            name = f"tracing_monitored::{name}"

        super().__init__(pybase_logger_name=name)

        self._func_name = func_name
        self._func = func

        self._monitor_tracing = monitor_tracing

    @property
    def func_name(self):
        return self._func_name

    def __call__(self, *args, **kwargs):
        if self._monitor_tracing:
            replica_context = tf.distribute.get_replica_context()
            replica_id = replica_context.replica_id_in_sync_group if replica_context is not None \
                else "[OUTSIDE REPLICA CONTEXT]"

            self._log.debug(f"Tracing for replica\t: {replica_id}")

        return self._func(*args, **kwargs)


class DistributedFuncWrapper(Base):

    def __init__(self, func: Callable, distribution_strategy):
        func_name = func.__name__ if hasattr(func, "__name__") else str(func)
        name = f"distributed::{func_name}"

        super().__init__(pybase_logger_name=name)

        self._func_name = func_name
        self._func = func
        self._distribution_strategy = distribution_strategy

    @property
    def func_name(self):
        return self._func_name

    def __call__(self, *args, **kwargs):
        return self._distribution_strategy.run(
            self._func,
            args=args,
            kwargs=kwargs
        )