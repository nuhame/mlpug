import numpy as np
import unittest

from mlpug.evaluation import ConcatBatchTuples, ConcatBatchDicts


class ConcatNumpyArrays(unittest.TestCase):

    def test_concat_batch_tuples_with_numpy_arrays(self):

        gathered_batches = [(np.array([0, 2, 1]), np.array([1, 2, 1])),
                            (np.array([2, 1, 1]), np.array([2, 0, 1]))]

        concat_batches = ConcatBatchTuples()

        concatenated_batches = concat_batches(gathered_batches)

        self.assertTrue(np.all(concatenated_batches[0] == np.array([0, 2, 1, 2, 1, 1])))
        self.assertTrue(np.all(concatenated_batches[1] == np.array([1, 2, 1, 2, 0, 1])))

    def test_concat_batch_dicts_with_numpy_arrays(self):

        gathered_batches = [
            {
                "labels": np.array([0, 2, 1]),
                "predictions": np.array([1, 2, 1])
            },
            {
                "labels": np.array([2, 1, 1]),
                "predictions": np.array([2, 0, 1])
            }
        ]

        concat_batches = ConcatBatchDicts()

        concatenated_batches = concat_batches(gathered_batches)

        self.assertTrue(np.all(concatenated_batches["labels"] == np.array([0, 2, 1, 2, 1, 1])))
        self.assertTrue(np.all(concatenated_batches["predictions"] == np.array([1, 2, 1, 2, 0, 1])))


if __name__ == '__main__':
    unittest.main()
