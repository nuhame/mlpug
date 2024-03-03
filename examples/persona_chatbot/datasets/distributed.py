from typing import Optional, Any
from collections.abc import Iterable, Sized, Sequence

import os

from functools import partial

import multiprocessing as mp

from basics.logging import get_logger
from basics.logging_utils import log_exception

from mlpug.base import Base


module_logger = get_logger(os.path.basename(__file__))


try:
    from tqdm import tqdm
except Exception as e:
    log_exception(module_logger, "Please `pip install tqdm`", e)


def generate_sample_worker_func(idx, sample_generator):
    """
    Used by DistributedSampleGenerator.

    Calling sample_generator[idx] triggers the generator to build the requested sample

    :param idx:
    :param sample_generator:

    :return:
    """
    return sample_generator[idx]


class DistributedSampleGenerator(Base, Iterable, Sized):

    def __init__(self,
                 sample_generator: Sequence,
                 num_workers: Optional[int] = None,
                 chunk_factor: int = 20,
                 name: Optional[str] = None):
        """

        :param sample_generator:

        :param num_workers:
        :param chunk_factor: How many chunks to process per worker

        :param name:
        """
        super().__init__(pybase_logger_name=name)

        self._sample_generator = sample_generator

        if num_workers is None:
            num_workers = mp.cpu_count()

        self._num_workers = num_workers
        self._chunk_factor = chunk_factor

        self._generate_sample_func = partial(
            generate_sample_worker_func,
            sample_generator=self._sample_generator)

        self._worker_pool = mp.Pool(self._num_workers)

        self._sample_iter = None

        self._log.info(f"Distributed generation of {len(self)} samples with {self._num_workers} workers.")

    def __len__(self):
        return len(self._sample_generator)

    def __iter__(self):
        num_samples = len(self)

        chunksize = max(int(num_samples / (self._num_workers * self._chunk_factor)), 1)

        self._sample_iter = self._worker_pool.imap(
            self._generate_sample_func,
            range(num_samples),
            chunksize=chunksize)

        return self

    def __next__(self):
        return next(self._sample_iter)

    def __del__(self):
        if self._worker_pool is not None:
            self._worker_pool.close()
            self._worker_pool.join()

