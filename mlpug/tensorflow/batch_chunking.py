from functools import cached_property

import tensorflow as tf
from tensorflow.types.experimental.distributed import PerReplica

from mlpug.base import Base
from mlpug.batch_chunking import convert_to_chunkable_dataset, get_total_batch_size, get_num_chunks, ChunkableBatchWrapper


def get_first_replica_result(func, distribution_strategy, per_replica_input):
    per_replica_results = distribution_strategy.run(
        func,
        args=(per_replica_input,)
    )

    per_replica_results = distribution_strategy.experimental_local_results(per_replica_results)

    return per_replica_results[0]


class DistributedChunkableBatchDataset(Base):

    def __init__(
            self,
            batch,
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

        self._per_replica_dataset = distribution_strategy.run(
            convert_to_chunkable_dataset,
            args=(batch, chunkable_batch_wrapper, batch_chunk_size)
        )

        self._distribution_strategy = distribution_strategy

        self._per_replica_iter = None
        self._num_chunks = -1
        self._chunk_idx = -1

    @cached_property
    def total_batch_size(self):
        # All replicas will have the same total_batch_size
        return get_first_replica_result(
            get_total_batch_size,
            self._distribution_strategy,
            self._per_replica_dataset
        )

    def __len__(self):
        return get_first_replica_result(
            get_num_chunks,
            self._distribution_strategy,
            self._per_replica_dataset
        )

    def __iter__(self):
        self._num_chunks = len(self)
        self._chunk_idx = -1
        self._per_replica_iter = self._distribution_strategy.run(
            iter,
            args=(self._per_replica_dataset,)
        )

        return self

    def __next__(self):
        self._chunk_idx += 1
        if self._chunk_idx >= self._num_chunks:
            raise StopIteration()

        return self._distribution_strategy.run(
            next,
            args=(self._per_replica_iter,)
        )
