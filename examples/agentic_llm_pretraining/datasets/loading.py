"""
Backend-agnostic dataset loading utilities.

Functions for loading tokenized datasets from Data Forager indexes.
These functions are framework-independent and can be used with PyTorch, JAX, etc.
"""
import numpy as np

from data_forager.datasets.tokens import TokensDataset
from data_forager.datasets.tokens_with_aux import TokensWithAuxDataset
from data_forager.index_stores.fs_based import IndexStore


def load_tokens_dataset(
    data_path: str,
    token_dtype: np.dtype = np.uint32,
) -> TokensDataset:
    """
    Load a TokensDataset from a Data Forager index.

    The data_path should contain tokenized samples created by Data Forager's
    tokenization pipeline. See examples/agentic_llm_pretraining/datasets/tokenize_dataset.py
    for creating tokenized datasets.

    :param data_path: Path to tokenized data directory containing the Data Forager index.
    :param token_dtype: NumPy dtype for tokens. Use np.uint32 for vocabularies > 65535
        (e.g., Qwen3 with ~152K vocab).

    :return: TokensDataset instance ready for random access.

    :raises FileNotFoundError: If no Data Forager index exists at data_path.
    """
    # TODO: This check should be part of TokensDataset.create_from_index_on_filesystem()
    #       See: https://github.com/visionscaper/data-forager
    index_store = IndexStore(base_path=data_path)
    if not index_store.exists():
        raise FileNotFoundError(
            f"No Data Forager index found at: {data_path}\n"
            f"Create one using: python -m examples.agentic_llm_pretraining.datasets.tokenize_dataset"
        )

    # Load and return dataset
    return TokensDataset.create_from_index_on_filesystem(
        data_path,
        token_dtype=token_dtype,
    )


def load_tokens_with_aux_dataset(data_path: str) -> TokensWithAuxDataset:
    """
    Load a TokensWithAuxDataset from a Data Forager index.

    The data_path should contain tokenized samples with auxiliary data (e.g.,
    loss masks) created by Data Forager's tokenization pipeline with aux support.
    See examples/agentic_llm_pretraining/datasets/v2/tokenize_dataset.py.

    :param data_path: Path to tokenized data directory containing the Data Forager index.

    :return: TokensWithAuxDataset instance ready for random access.

    :raises FileNotFoundError: If no Data Forager index exists at data_path.
    """
    index_store = IndexStore(base_path=data_path)
    if not index_store.exists():
        raise FileNotFoundError(
            f"No Data Forager index found at: {data_path}\n"
            f"Create one using: "
            f"python -m examples.agentic_llm_pretraining.datasets.v2.tokenize_dataset"
        )

    return TokensWithAuxDataset.create_from_index_on_filesystem(data_path)


def get_sample_properties(dataset: TokensDataset) -> tuple[int, np.dtype]:
    """
    Get properties of samples in a TokensDataset.

    Reads the first sample to determine context length and token dtype.

    :param dataset: TokensDataset to inspect.

    :return: Tuple of (context_length, token_dtype).
    """
    sample: np.ndarray = dataset[0]
    context_length = len(sample)
    token_dtype = sample.dtype

    return context_length, token_dtype
