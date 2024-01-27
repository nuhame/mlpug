import itertools

import tensorflow as tf

from basics.base import Base

from mlpug.debugging import enable_pycharm_remote_debugging
from mlpug.tensorflow.batch_chunking import is_chunkable, ChunkableTupleBatch, ChunkableTupleBatchDim0, ChunkableBatchDataset
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


def generate_batches():
    for _ in range(1000):
        yield (
            tf.random.uniform(shape=(40, 2, 252), minval=0, maxval=50000, dtype=tf.int64),
            tf.random.uniform(shape=(40, 2, 252), minval=0, maxval=50000, dtype=tf.int64),
            tf.random.uniform(shape=(40, 2, 252), minval=0, maxval=50000, dtype=tf.int64),
            tf.random.uniform(shape=(40, 2), minval=0, maxval=50000, dtype=tf.int64),
            tf.cast(tf.random.uniform(shape=(40,), minval=0, maxval=1, dtype=tf.int64), tf.int8),
        )


def create_process_chunks_func(dataset_input_signature):
    @tf.function(input_signature=[
        dataset_input_signature,
        {},
        tf.TensorSpec(shape=(), dtype=tf.int32),  # batch_size
        tf.TensorSpec(shape=(), dtype=tf.int32)   # num_chunks
    ])
    # @tf.function
    def process_chunks_func(chunks_dataset, training_settings, batch_size, num_chunks):

        s = tf.constant(0, dtype=tf.int64)
        for chunk in chunks_dataset:
            for t in chunk[:-1]:
                if s is None:
                    s = tf.math.reduce_sum(t)
                else:
                    s += tf.math.reduce_sum(t)

        return s

    return process_chunks_func


if __name__ == '__main__':
    # enable_pycharm_remote_debugging("192.168.178.85:57491")

    batch_dataset = tf.data.Dataset.from_generator(
        generate_batches,
        output_signature=(
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.int8)
        )
    )

    process_chunks = create_process_chunks_func(tf.data.DatasetSpec.from_value(batch_dataset))

    for batch in batch_dataset:

        batch = ChunkableTupleBatchDim0(*batch)

        chunks = ChunkableBatchDataset(batch, batch_chunk_size=1)

        def generate_chunk():
            for chunk in chunks:
                yield chunk

        chunks_dataset = tf.data.Dataset.from_generator(
            generate_chunk,
            output_signature=batch_dataset.element_spec
        )

        s = process_chunks(chunks_dataset, {}, len(batch), len(chunks))

        print(s)
