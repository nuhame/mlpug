import numpy as np
import unittest

from mlpug.evaluation import CombineBatchTuples, CombineBatchDicts


class ConcatNumpyArrays(unittest.TestCase):

    def test_concat_batch_tuples(self):
        gathered_batches = [(np.array([0, 2, 1]), np.array([1, 2, 1]), 10, 'a'),
                            (np.array([2, 1, 1]), np.array([2, 0, 1]), 20, 'b')]

        combine_batches = CombineBatchTuples()

        combined_batches = combine_batches(gathered_batches)

        self.assertTrue(np.all(combined_batches[0] == np.array([0, 2, 1, 2, 1, 1])))
        self.assertTrue(np.all(combined_batches[1] == np.array([1, 2, 1, 2, 0, 1])))
        self.assertTrue(combined_batches[2] == 30)
        self.assertTrue(all([v0 == v1 for v0, v1 in zip(combined_batches[3], ['a', 'b'])]))

    def test_concat_batch_dicts(self):

        gathered_batches = [
            {
                "labels": np.array([0, 2, 1]),
                "predictions": np.array([1, 2, 1]),
                "num_samples": 10,
                "other_data": 'a'
            },
            {
                "labels": np.array([2, 1, 1]),
                "predictions": np.array([2, 0, 1]),
                "num_samples": 20,
                "other_data": 'b'
            }
        ]

        concat_batches = CombineBatchDicts()

        combined_batches = concat_batches(gathered_batches)

        self.assertTrue(np.all(combined_batches["labels"] == np.array([0, 2, 1, 2, 1, 1])))
        self.assertTrue(np.all(combined_batches["predictions"] == np.array([1, 2, 1, 2, 0, 1])))
        self.assertTrue(combined_batches["num_samples"] == 30)
        self.assertTrue(all([v0 == v1 for v0, v1 in zip(combined_batches["other_data"], ['a', 'b'])]))


if __name__ == '__main__':
    unittest.main()
