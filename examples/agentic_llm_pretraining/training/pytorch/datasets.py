"""
PyTorch dataset wrappers for Data Forager datasets.

Provides thin wrappers around Data Forager datasets that inherit from
torch.utils.data.Dataset for proper type compatibility with PyTorch's
DataLoader and DistributedSampler.
"""
from torch.utils.data import Dataset as TorchDataset

from data_forager.datasets.tokens import TokensDataset


class PyTorchTokensDataset(TorchDataset):
    """
    PyTorch-compatible wrapper for Data Forager's TokensDataset.

    This wrapper inherits from torch.utils.data.Dataset to satisfy type
    requirements of PyTorch's DataLoader and DistributedSampler, while
    delegating all operations to the underlying TokensDataset.

    :param tokens_dataset: The Data Forager TokensDataset to wrap.
    """

    def __init__(self, tokens_dataset: TokensDataset):
        super().__init__()
        self._dataset = tokens_dataset

    def __getitem__(self, idx: int):
        """
        Get a sample by index.

        :param idx: Sample index.

        :return: Token IDs as numpy array of shape (context_length,).
        """
        return self._dataset[idx]

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        :return: Number of samples.
        """
        return len(self._dataset)

    @property
    def tokens_dataset(self) -> TokensDataset:
        """Access the underlying TokensDataset."""
        return self._dataset
