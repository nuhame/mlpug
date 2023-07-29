from typing import Protocol, Any, Callable

import abc

import math

from mlpug.base import Base
from mlpug.mlpug_exceptions import BatchNotChunkableException


def create_chunks_generator(chunks_dataset):

    def generate_chunk():
        for chunk in chunks_dataset:
            yield chunk

    return generate_chunk


class ChunkableBatchProtocol(Protocol):

    def __len__(self):
        ...

    def __getitem__(self, sample_slice):
        ...


ChunkableBatchWrapper = Callable[[Any], ChunkableBatchProtocol]


def apply_chunkable_batch_wrapper(batch_data, wrapper: ChunkableBatchWrapper):
    if callable(wrapper):
        try:
            return wrapper(batch_data)
        except Exception as e:
            raise BatchNotChunkableException(
                f"Failed to wrap the given batch as a chunkable batch using the "
                f"given `chunkable_batch_wrapper`: {wrapper}"
            ) from e
    else:
        raise BatchNotChunkableException(
            "Given batch is not chunkable and no callable `chunkable_batch_wrapper` provided"
        )


class ChunkableBatch(Base, metaclass=abc.ABCMeta):

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

    @classmethod
    def wrapper(cls, batch):
        return cls(*batch)


class ChunkableTupleBatchDim0(ChunkableTupleBatch):

    def __len__(self):
        # get batch size
        return self._batch[0].shape[0]

    def __getitem__(self, sample_slice):
        return tuple((v[sample_slice, ...] for v in self._batch))


class ChunkableTupleBatchDim1(ChunkableTupleBatch):

    def __len__(self):
        # get batch size
        return self._batch[0].shape[1]

    def __getitem__(self, sample_slice):
        return tuple((v[:, sample_slice, ...] for v in self._batch))


def is_chunkable(batch):
    return batch is not None and \
           not isinstance(batch, (tuple, list, dict)) and \
           hasattr(batch, "__len__") and callable(batch.__len__) and \
           hasattr(batch, "__getitem__") and callable(batch.__getitem__)


def has_batch_chunking_results(batch_metrics_list):
    return type(batch_metrics_list) is list and \
           len(batch_metrics_list) > 0 and \
           type(batch_metrics_list[0]) is BatchChunkingResults


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

    @property
    def total_batch_size(self):
        return self._batch_size

    def __len__(self):
        return self._num_chunks

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


class BatchChunkingResults(list):
    pass
