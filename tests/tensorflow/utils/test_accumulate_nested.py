import os
import time

import unittest

from basics.logging import get_logger

import tensorflow as tf

from mlpug.debugging import enable_pycharm_remote_debugging

from mlpug.tensorflow.utils.accumulate_nested import NestedTensorsAccumulator


module_logger = get_logger(os.path.basename(__file__))


def generate_nested_tensors():
    num_samples = 40
    voc_size = 50000
    return {
        "loss": tf.random.normal(()),
        "num_samples": tf.constant(num_samples, dtype=tf.int32),
        "auxiliary_results": {
            # required to calculate next sentence prediction (classification) quality
            "nsp_logits": tf.random.normal((num_samples, voc_size)),
        }
    }


def is_equal(nested_tensor_1, nested_tensor_2):
    if isinstance(nested_tensor_1, tf.Tensor) and isinstance(nested_tensor_2, tf.Tensor):
        return tf.equal(nested_tensor_1, nested_tensor_2)
    else:
        if isinstance(nested_tensor_1, tuple) and isinstance(nested_tensor_2, tuple):
            is_equal_func = is_equal_iterable
        elif isinstance(nested_tensor_1, list) and isinstance(nested_tensor_2, list):
            is_equal_func = is_equal_iterable
        elif isinstance(nested_tensor_1, dict) and isinstance(nested_tensor_2, dict):
            is_equal_func = is_equal_dict
        else:
            return False

        return is_equal_func(nested_tensor_1, nested_tensor_2)


def is_equal_iterable(nested_tensor_1, nested_tensor_2):
    for v1, v2 in zip(nested_tensor_1, nested_tensor_2):
        return is_equal(v1, v2)


def is_equal_dict(nested_tensor_1, nested_tensor_2):
    for (k1, v1), (k2, v2) in zip(nested_tensor_1.items(), nested_tensor_2.items()):
        if k1 != k2:
            return False

        return is_equal(v1, v2)


class NestedTensorsAccumulatorTests(unittest.TestCase):

    def test_zip_in(self):
        num_chunks = 100
        acc = NestedTensorsAccumulator(num_chunks)

        nested_tensors_list_original = [generate_nested_tensors() for _ in range(num_chunks)]

        a = time.time()
        for nested_tensor in nested_tensors_list_original:
            acc.zip_in(nested_tensor)
        b = time.time()
        print(f"Accumulation time: {b-a} seconds")

        a = time.time()
        nested_tensors_list_rebuild = acc.unzip()
        b = time.time()
        print(f"Unzip time       : {b - a} seconds")

        for nested_tensor_original, nested_tensor_rebuild in zip(
                nested_tensors_list_original,
                nested_tensors_list_rebuild
        ):
            self.assertTrue(is_equal(nested_tensor_original, nested_tensor_rebuild))


if __name__ == "__main__":
    remote_debug_ip = os.environ.get('REMOTE_DEBUG_IP', None)
    if remote_debug_ip is not None:
        enable_pycharm_remote_debugging(remote_debug_ip)

    unittest.main()
