import itertools

import tensorflow as tf

from basics.base import Base

import mlpug.tensorflow as mlp

from mlpug.debugging import enable_pycharm_remote_debugging
from mlpug.tensorflow.batch_chunking import (
    is_chunkable,
    ChunkableTupleBatch,
    ChunkableTupleBatchDim0,
    ChunkableBatchDataset
)
from mlpug.mlpug_exceptions import BatchNotChunkableException


# class ChunkableTupleBatchDim0(ChunkableTupleBatch):
#
#     @property
#     def batch_size(self):
#         # get batch size
#         tf.print(f"Batch shape: {tf.shape(self._batch[0])}")
#         return tf.shape(self._batch[0])[0]
#
#     def __len__(self):
#         return self._batch[0].shape()[0]
#
#     def __getitem__(self, sample_slice):
#         return tuple((v[sample_slice, ...] for v in self._batch))
#
#
# class TFChunkableBatchDataset(Base):
#
#     def __init__(self, batch, batch_chunk_size):
#         """
#         Turns a chunkable batch in to an iterable dataset
#
#         :param batch: A chunkable batch must implement the `__len__` and `__getitem__` methods.
#                       len(batch) must return the number of batch samples
#                       Here the `__getitem__` method must be able to deal with slices.
#
#         :param batch_chunk_size:
#                       The sample size of each batch chunk
#         """
#         super().__init__()
#
#         self._batch = batch
#         self._batch_chunk_size = batch_chunk_size
#
#         if not is_chunkable(batch):
#             raise BatchNotChunkableException()
#
#         self._batch_size = batch.batch_size
#         self._num_chunks = self._calc_num_chunks()
#         self._chunk_idx = -1
#
#     def _calc_num_chunks(self):
#         return tf.math.ceil(self._batch_size / self._batch_chunk_size)
#
#     def __len__(self):
#         return self._num_chunks
#
#     def __iter__(self):
#         self._chunk_idx = -1
#         return self
#
#     def __next__(self):
#         self._chunk_idx += 1
#
#         if self._chunk_idx >= self._num_chunks:
#             raise StopIteration()
#
#         chunk_start = self._chunk_idx * self._batch_chunk_size
#         chunk_end = min((self._chunk_idx + 1) * self._batch_chunk_size, self._batch_size)
#
#         return self._batch[chunk_start:chunk_end]


@tf.function(input_signature=[
    (
        tf.TensorSpec(shape=None, dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.float32)
    )
])
#@tf.function()
def generate_chunks(chunks_dataset):
    s = 0.0
    for chunk in chunks_dataset:
        s += tf.math.reduce_sum(chunk[0])

    return s


if __name__ == '__main__':
    # enable_pycharm_remote_debugging("192.168.178.85:57491")

    t1 = tf.random.normal((12, 5))
    t2 = tf.random.normal((12, 3))

    batch = ChunkableTupleBatchDim0(t1, t2)

    chunks = ChunkableBatchDataset(batch, batch_chunk_size=4)
    def generate_chunk():
        for chunk in chunks:
            yield chunk

    chunks_dataset = tf.data.Dataset.from_generator(
        generate_chunk,
        output_signature=(
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.float32)
        )
    )

    s = generate_chunks(chunks_dataset)

    print(s)
