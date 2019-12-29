import os
import basics.base_utils as _
from basics.logging import get_logger

logger = get_logger(os.path.basename(__file__))


def convert_to_dict(type, components):
    if _.is_sequence(components):
        if len(components) == 1:
            components = {
                type: components[0]
            }
        elif len(components) > 1:
            components = {f"{type}_{i}": component for i, component in enumerate(components)}
    elif not _.is_dict(components):
        # Assuming single optimizer
        components = {
            type: components
        }

    return components


def get_value_at(key_path, nested_data, default=None, warn_on_failure=True):
    """
    Safe way to get value from nested data structure (e.g. nested dict) based on a key path

    TODO migrate to PyBase

    :param key_path:
    :param nested_data:
    :param default:
    :param warn_on_failure:

    :return:
    """
    keys = key_path.split(".")
    value = nested_data
    for key in keys:
        if has_key(value, key):
            value = value[key]
        else:
            if warn_on_failure:
                logger.warn(f"Key path {key_path} not found in given data")
            value = None
            break

    if value is None:
        value = default

    return value


def has_key(o, key):
    """
    TODO migrate to PyBase
    :param o:
    :param key:
    :return:
    """
    return hasattr(o, '__iter__') and (key in o)


def is_chunkable(batch):
    return (batch is not None) and \
           hasattr(batch, "__len__") and callable(batch.__len__) and \
           hasattr(batch, "__getitem__") and callable(batch.__getitem__)
