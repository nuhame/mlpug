import math

import abc

from base import Base
from mlpug_exceptions import BatchNotChunkableException
from trainers import BatchChunkingResults
from utils import is_chunkable


def has_batch_chunking_results(batch_metrics_list):
    return type(batch_metrics_list) is list and \
           len(batch_metrics_list) > 0 and \
           type(batch_metrics_list[0]) is BatchChunkingResults


class ChunkableBatch(Base, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def source(self):
        raise NotImplementedError("Please implement in your child class")

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError("Please implement in your child class")

    @abc.abstractmethod
    def __getitem__(self, sample_slice):
        raise NotImplementedError("Please implement in your child class")


class ChunkableTupleBatch(ChunkableBatch, metaclass=abc.ABCMeta):
    def __init__(self, *batch):
        super().__init__()

        self._batch = batch

    def source(self):
        return self._batch


class ChunkableTupleBatchDim0(ChunkableTupleBatch):

    def __len__(self):
        # get batch size
        return self._batch[0].size(0)

    def __getitem__(self, sample_slice):
        return (v[sample_slice, ...] for v in self._batch)


class ChunkableTupleBatchDim1(ChunkableTupleBatch):

    def __len__(self):
        # get batch size
        return self._batch[0].size(1)

    def __getitem__(self, sample_slice):
        return (v[:, sample_slice, ...] for v in self._batch)


class ChunkableBatchDataset(Base):

    def __init__(self, batch, batch_chunk_size):
        """
        Turns a chunkable batch in to an iterable dataset

        :param batch: A chunkable batch must implement the `__len__` and `__getitem__` methods.
                      len(batch) must return the number of batch samples
                      Here the `__getitem__` method must be able to deal with slices.

        :param batch_chunk_size:
                      The sample size of each batch chunk
        """
        super().__init__()

        self._batch = batch
        self._batch_chunk_size = batch_chunk_size

        if not is_chunkable(batch):
            raise BatchNotChunkableException()

        self._batch_size = len(batch)
        self._num_chunks = math.ceil(self._batch_size / self._batch_chunk_size)
        self._chunk_idx = -1

    def __iter__(self):
        self._chunk_idx = -1
        return self

    def __next__(self):
        self._chunk_idx += 1

        if self._chunk_idx >= self._num_chunks:
            raise StopIteration()

        chunk_start = self._chunk_idx * self._batch_chunk_size
        chunk_end = min((self._chunk_idx + 1) * self._batch_chunk_size, self._batch_size)

        return self._batch[chunk_start:chunk_end]