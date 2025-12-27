import numpy as np

from mlpug.utils.utils import has_method


class ValueDescription:

    def __init__(self, value_type, value_size, device=None):
        self.value_type = value_type
        self.value_size = value_size
        self.device = device

    def __str__(self):
        s = f'{self.value_type}<{self.value_size}>'
        if self.device is not None:
            s += f'<{self.device}>'

        return s

    def __repr__(self):
        return str(self)


def describe_data(data):
    """
    Creates description of data, in terms of type, size and device of collection-like values in the data.
    The description output recursively follows the structure of dicts, tuples and MicroBatchResults

    The description of None, int, float, bool and str will just be the value itself.
    Otherwise the descriptions are represented by ValueDescription objects

    MicroBatchResults will be represented by a standard list

    Can, for instance, be useful to describe the training log object.

    :param data:
    :return:
    """
    if isinstance(data, dict):
        # tas = Type And Shape
        tas = {}

        for k, v in data.items():
            tas[k] = describe_data(v)

        return tas

    if isinstance(data, tuple):
        tas = ()
        for v in data:
            tas += (describe_data(v), )

        return tas

    if isinstance(data, list):
        # Could also be a MicroBatchResults
        tas = data.__class__()
        for v in data:
            tas += [describe_data(v)]

        return tas

    if data is None or isinstance(data, (float, int, bool, str)):
        return data

    t = type(data)
    s = None

    if hasattr(data, 'shape'):
        s = data.shape
    elif hasattr(data, 'size'):
        s = data.size
    elif has_method(data, '__len__'):
        s = len(data)

    d = None
    if hasattr(data, 'device'):
        d = data.device

    return ValueDescription(t, s, d)


class MLPugDataCleaner:

    def __call__(self, data):

        if isinstance(data, dict):
            values = {}

            for k, v in data.items():
                values[k] = self(v)

            return values

        if isinstance(data, tuple):
            values = ()
            for v in data:
                values += (self(v),)

            return values

        if isinstance(data, list):
            # Could also be a MicroBatchResults
            values = data.__class__()
            for v in data:
                values += [self(v)]

            return values

        if isinstance(data, np.ndarray):
            # If there is only one element
            if np.prod(data.shape) == 1:
                return data.item()

        if data is None or isinstance(data, (float, int, bool, str)):
            return data

        return self._handle_other_data_type(data)

    def _handle_other_data_type(self, data):
        return data
