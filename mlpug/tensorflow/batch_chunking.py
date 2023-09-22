from functools import cached_property, partial

from mlpug.base import Base
from mlpug.batch_chunking import (
    convert_to_chunkable_dataset,
    get_total_batch_size,
    get_num_chunks,
    ChunkableBatchWrapper
)

from mlpug.tensorflow.distributed_utils import unpack_per_replica_and_map, pack_per_replica


class DistributedChunkableBatchDataset(Base):

    def __init__(
            self,
            per_replica_batch,
            chunkable_batch_wrapper: ChunkableBatchWrapper,
            batch_chunk_size: int,
            # TODO: unable to import Strategy from tensorflow.distribute
            distribution_strategy,
            name=None
    ):
        """

        :param batch: batch with PerReplica objects containing batch per replica (device)
        :param chunkable_batch_wrapper: Converts a batch to something that is chunkable (can be sliced)
        :param batch_chunk_size:
        :param distribution_strategy:
        """
        super().__init__(pybase_logger_name=name)

        unpacked_replica_data = distribution_strategy.experimental_local_results(per_replica_batch)
        first_batch_replica = unpacked_replica_data[0]

        if not isinstance(first_batch_replica, (list, tuple)):
            raise ValueError(f"The per replica batch must be a list or tuple of tensors. "
                             f"Type of first batch replica is: {type(first_batch_replica)}")

        self._per_replica_dataset = unpack_per_replica_and_map(
            map_func=partial(
                convert_to_chunkable_dataset,
                chunkable_batch_wrapper=chunkable_batch_wrapper,
                batch_chunk_size=batch_chunk_size
            ),
            distribution_strategy=distribution_strategy,
            unpacked_replica_data=unpacked_replica_data
        )

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

        return pack_per_replica([next(it) for it in self._per_replica_iter])