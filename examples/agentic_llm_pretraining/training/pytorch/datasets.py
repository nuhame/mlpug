"""
PyTorch dataset wrapper for Data Forager datasets.

Provides a generic wrapper around Data Forager datasets that inherits from
torch.utils.data.Dataset for proper type compatibility with PyTorch's
DataLoader and DistributedSampler.
"""
from typing import Callable, Generic, TypeVar

from torch.utils.data import Dataset as TorchDataset

from data_forager.datasets.common import Dataset as ForagerDataset, SubsampledDataset


T = TypeVar('T')


class PyTorchForagerDataset(TorchDataset, Generic[T]):
    """
    Generic PyTorch-compatible wrapper for Data Forager datasets.

    Wraps any Forager dataset (or SubsampledDataset) and uses an injected
    copy function to produce writable samples for PyTorch. The type parameter
    T represents the sample type (e.g., np.ndarray for tokens, or
    Dict[str, np.ndarray] for tokens with auxiliary data).

    Copies are necessary because Forager returns memory-mapped arrays
    (read-only), and PyTorch requires writable tensors.

    :param dataset: The Data Forager dataset to wrap.
    :param copy_sample_func: Function that copies a sample from the dataset.
        Must return a writable copy of the sample. The function signature
        and return type should match T.
    """

    def __init__(
        self,
        dataset: ForagerDataset | SubsampledDataset,
        copy_sample_func: Callable[..., T],
    ):
        super().__init__()
        self._dataset = dataset
        self._copy_sample_func = copy_sample_func

    def __getitem__(self, idx: int) -> T:
        """
        Get a sample by index.

        :param idx: Sample index.

        :return: Copied sample of type T.
        """
        return self._copy_sample_func(self._dataset[idx])

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        :return: Number of samples.
        """
        return len(self._dataset)

    @property
    def forager_dataset(self) -> ForagerDataset | SubsampledDataset:
        """Access the underlying Forager dataset."""
        return self._dataset
