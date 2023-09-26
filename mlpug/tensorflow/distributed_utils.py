import os

from typing import Optional, Tuple, Union, List, Callable

from basics.logging import get_logger
from basics.validation_utils import is_callable

import tensorflow as tf

from tensorflow.python.distribute import values
from tensorflow.python.types.distribute import DistributedValues


module_logger = get_logger(os.path.basename(__file__))


def create_distributed_func(func: Callable, distribution_strategy, logger=None, monitor_tracing=False):
    if logger is None:
        logger = module_logger

    if not is_callable(func):
        raise ValueError(f"No valid function given (not callable), "
                         f"unable to create distributed function for: {func}")

    func_name = func.__name__ if hasattr(func, "__name__") else str(func)

    def monitored_func(*args, **kwargs):
        if monitor_tracing:
            logger.debug(f"Function {func_name}: tracing for "
                         f"replica\t: {tf.distribute.get_replica_context().replica_id_in_sync_group}")

        return func(*args, **kwargs)

    def distributed_func(*args, **kwargs):
        return distribution_strategy.run(
            monitored_func,
            args=args,
            kwargs=kwargs
        )

    if monitor_tracing:
        logger.debug(f"Wrapped function {func_name} as distributed function with "
                     f"ID\t: {hex(id(distributed_func))}")

    return distributed_func


def contains_per_replica_data(data):
    if isinstance(data, (list, tuple)):
        return len(data) > 0 and contains_per_replica_data(data[0])

    return isinstance(data, DistributedValues)


def unpack_per_replica_and_map(
    map_func: Callable,
    per_replica_data: Optional[Tuple[tf.distribute.DistributedValues]] = None,
    distribution_strategy: Optional[tf.distribute.Strategy] = None,
    unpacked_replica_data: Optional[Union[Tuple, List]] = None
) -> Tuple:
    if per_replica_data is None and unpacked_replica_data is None:
        raise ValueError(f"Provide per_replica_data ({per_replica_data}) or "
                         f"unpacked_replica_data ({unpacked_replica_data})")

    if per_replica_data is not None and unpacked_replica_data is not None:
        raise ValueError("Either per_replica_data or unpacked_replica_data not both.")

    if per_replica_data is not None and unpacked_replica_data is None:
        if distribution_strategy is None:
            raise ValueError("Please provide a distribution_strategy in order to "
                             "unpack the provided per_replica_data")
        # unpack
        unpacked_replica_data = distribution_strategy.experimental_local_results(per_replica_data)

    return tuple(map(map_func, unpacked_replica_data))


def pack_per_replica(unpacked_replica_data: Union[Tuple, List]) -> Tuple:
    return tuple(values.PerReplica(t) for t in zip(*unpacked_replica_data))
