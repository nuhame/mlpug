from functools import cached_property
from typing import List

from mlpug.base import Base
from mlpug.batch_chunking import (
    convert_to_chunkable_dataset,
    get_total_batch_size,
    get_num_chunks,
    ChunkableBatchWrapper
)

from tensorflow.python.distribute import values


class DistributedChunkableBatchDataset(Base):

    def __init__(
            self,
            per_replica_batch,
            chunkable_batch_wrapper: ChunkableBatchWrapper,
            batch_chunk_size: int,
            # TODO: unable to import Strategy from tensorflow.distribute
            distribution_strategy):
        """

        :param batch: batch with PerReplica objects containing batch per replica (device)
        :param chunkable_batch_wrapper: Converts a batch to something that is chunkable (can be sliced)
        :param batch_chunk_size:
        :param distribution_strategy:
        """
        super().__init__()

        per_replica_batch = distribution_strategy.experimental_local_results(per_replica_batch)
        first_batch_replica = per_replica_batch[0]

        if not isinstance(first_batch_replica, (list, tuple)):
            raise ValueError(f"The per replica batch must be a list or tuple of tensors. "
                             f"Type of first batch replica is: {type(first_batch_replica)}")

        self._per_replica_dataset = [convert_to_chunkable_dataset(b, chunkable_batch_wrapper, batch_chunk_size)
                                     for b in per_replica_batch]

        self._distribution_strategy = distribution_strategy

        self._per_replica_iter = None
        self._num_chunks = -1
        self._chunk_idx = -1

    @cached_property
    def total_batch_size(self):
        total_batch_size = [get_total_batch_size(dataset) for dataset in self._per_replica_dataset]
        return total_batch_size[0]

    def __len__(self):
        num_chunks = [get_num_chunks(dataset) for dataset in self._per_replica_dataset]
        return num_chunks[0]

    def __iter__(self):
        self._num_chunks = len(self)
        self._chunk_idx = -1
        self._per_replica_iter = [iter(dataset) for dataset in self._per_replica_dataset]

        return self

    def __next__(self):
        self._chunk_idx += 1
        if self._chunk_idx >= self._num_chunks:
            raise StopIteration()

        # IMPORTANT This assumes that the batch is a tuple or list of tensors
        per_replica_batch = tuple(values.PerReplica(t) for t in zip(*[next(it) for it in self._per_replica_iter]))
        return per_replica_batch
