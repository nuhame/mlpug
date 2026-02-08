#!/usr/bin/env python3
"""
V1: Tokenize datasets for NTP training using Data Forager.

Tokenizes JSONL files into fixed-length token samples and creates a byte-offset
index for O(1) random access during training.

Output format: tokens only (no auxiliary data like loss masks).

Usage (from repo root):
    # Tokenize all splits with default settings (4K context)
    python -m examples.agentic_llm_pretraining.datasets.v1.tokenize_dataset \
        --splits-dir ../data/agentic_llm_pretraining/full-08012026/splits \
        --output-dir ../data/agentic_llm_pretraining/full-08012026/tokenized

    # Tokenize only train split with custom context length
    python -m examples.agentic_llm_pretraining.datasets.v1.tokenize_dataset \
        --splits-dir ../data/agentic_llm_pretraining/full-08012026/splits \
        --output-dir ../data/agentic_llm_pretraining/full-08012026/tokenized \
        --splits train \
        --context-length 4096

    # Force re-tokenization (clears existing index)
    python -m examples.agentic_llm_pretraining.datasets.v1.tokenize_dataset \
        --splits-dir ../data/agentic_llm_pretraining/full-08012026/splits \
        --output-dir ../data/agentic_llm_pretraining/full-08012026/tokenized \
        --force

    # Use pad token as end-of-document separator (recommended for Qwen3)
    python -m examples.agentic_llm_pretraining.datasets.v1.tokenize_dataset \
        --splits-dir ../data/agentic_llm_pretraining/full-08012026/splits \
        --output-dir ../data/agentic_llm_pretraining/full-08012026/tokenized \
        --eod-token pad

Output structure:
    tokenized/
    ├── train/
    │   ├── tokenized-samples/
    │   │   └── train-tokenized-samples.bin
    │   └── index/
    │       ├── sample_locations.bin
    │       └── file_location.txt
    ├── val/
    │   └── ...
    └── test/
        └── ...
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
from transformers import AutoTokenizer

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from mlpug.utils.git_logging import log_git_state

from data_forager.index_stores.fs_based import IndexStore
from data_forager.indexers.tokenization_indexer import create_tokenize_and_index_jsonl_text_func
from data_forager.datasets.tokens import TokensDataset


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


# Default tokenizer for Qwen3
DEFAULT_TOKENIZER = "Qwen/Qwen3-1.7B"

# Default context length (4K for baseline, following staged training approach)
DEFAULT_CONTEXT_LENGTH = 4096

# Token dtype for storage (uint16 supports vocab up to 65K, uint32 for larger)
# Qwen3 has ~152K vocab, so we need uint32
DEFAULT_TOKEN_DTYPE = np.uint32

# Default splits to process
DEFAULT_SPLITS = ["train", "val", "test"]


def describe_config(
    splits_dir: str,
    output_dir: str,
    splits: list[str],
    context_length: int,
    tokenizer: str,
    eod_token: str,
    force: bool,
    inspect_samples: bool,
) -> None:
    """Log script configuration."""
    module_logger.info("Configuration:")
    module_logger.info(f"  splits_dir: {splits_dir}")
    module_logger.info(f"  output_dir: {output_dir}")
    module_logger.info(f"  splits: {splits}")
    module_logger.info(f"  context_length: {context_length}")
    module_logger.info(f"  tokenizer: {tokenizer}")
    module_logger.info(f"  eod_token: {eod_token}")
    module_logger.info(f"  force: {force}")
    module_logger.info(f"  inspect_samples: {inspect_samples}")


def get_text_from_sample(
    jsonl_bytes: bytes,
    text_key: str = "text",
    text_encoding: str = "utf-8",
) -> str:
    """
    Extract text from a JSONL sample.

    The dataset format is: {"source": "...", "index": N, "text": "..."}
    We only need the "text" field for tokenization.
    """
    jsonl_text = jsonl_bytes.decode(text_encoding)
    data = json.loads(jsonl_text)
    return data[text_key]


def load_tokenizer(model_name: str):
    """
    Load tokenizer from HuggingFace.

    :param model_name: HuggingFace model name (e.g., "Qwen/Qwen3-1.7B").

    :return: Tokenizer instance.
    """
    module_logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    module_logger.info(f"  vocab_size: {tokenizer.vocab_size}")
    module_logger.info(f"  eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    module_logger.info(f"  pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    return tokenizer


def resolve_eod_token_id(tokenizer, eod_token: str) -> int:
    """
    Resolve the end-of-document token ID from the --eod-token argument.

    :param tokenizer: HuggingFace tokenizer.
    :param eod_token: One of:
        - "eos" - use tokenizer.eos_token_id
        - "pad" - use tokenizer.pad_token_id
        - An integer string (e.g., "151643") - use that ID directly
        - A token string (e.g., "<|endoftext|>") - encode and use that ID

    :return: Token ID to use as end-of-document separator.
    """
    if eod_token == "eos":
        token_id = tokenizer.eos_token_id
        if token_id is None:
            raise ValueError("Tokenizer has no eos_token, cannot use --eod-token eos")
        module_logger.info(f"Using eos_token as EOD: {tokenizer.eos_token} (id={token_id})")
    elif eod_token == "pad":
        token_id = tokenizer.pad_token_id
        if token_id is None:
            raise ValueError("Tokenizer has no pad_token, cannot use --eod-token pad")
        module_logger.info(f"Using pad_token as EOD: {tokenizer.pad_token} (id={token_id})")
    elif eod_token.isdigit():
        token_id = int(eod_token)
        if token_id < 0 or token_id >= tokenizer.vocab_size:
            raise ValueError(
                f"Token ID {token_id} is out of range (vocab_size={tokenizer.vocab_size})"
            )
        module_logger.info(f"Using explicit token ID as EOD: {token_id}")
    else:
        # Treat as token string, encode it
        encoded = tokenizer.encode(eod_token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Token string '{eod_token}' encodes to {len(encoded)} tokens ({encoded}), "
                f"expected exactly 1 token"
            )
        token_id = encoded[0]
        module_logger.info(f"Using token string as EOD: {eod_token} (id={token_id})")

    return token_id


def create_tokenizer_func(tokenizer) -> Callable[[str], list[int]]:
    """
    Create tokenizer function compatible with Data Forager.

    :param tokenizer: HuggingFace tokenizer.

    :return: Function that takes text and returns list of token IDs.
    """
    def tokenize(text: str) -> list[int]:
        # Use encode without special tokens - EOS is added by Data Forager
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenize


def tokenize_split(
    split_name: str,
    input_path: Path,
    output_dir: Path,
    tokenizer_func: Callable[[str], list[int]],
    eod_idx: int,
    context_length: int,
    token_dtype: np.dtype,
    force: bool,
    logger=None,
) -> dict:
    """
    Tokenize a single split.

    :param split_name: Name of the split (train, val, test).
    :param input_path: Path to the input JSONL file.
    :param output_dir: Base output directory.
    :param tokenizer_func: Function to tokenize text.
    :param eod_idx: End-of-document token ID.
    :param context_length: Fixed context length for samples.
    :param token_dtype: NumPy dtype for token storage.
    :param force: Whether to force re-tokenization.
    :param logger: Optional logger.

    :return: Stats dict with sample counts and timing.
    """
    if logger is None:
        logger = module_logger

    # Output structure: output_dir/split_name/
    split_output_dir = output_dir / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if index already exists
    index_store = IndexStore(base_path=str(split_output_dir))

    if index_store.exists():
        if force:
            logger.info(f"{split_name}: clearing existing index (--force)")
            index_store.clear()
            # Also clear tokenized samples
            tokenized_samples_dir = split_output_dir / "tokenized-samples"
            if tokenized_samples_dir.exists():
                shutil.rmtree(tokenized_samples_dir)
        else:
            logger.info(f"{split_name}: index already exists, skipping (use --force to rebuild)")
            # Load existing index to get sample count
            sample_index = index_store.load()
            return {
                "split": split_name,
                "status": "skipped",
                "num_samples": len(sample_index.sample_locations),
            }

    logger.info(f"{split_name}: tokenizing {input_path}")
    logger.info(f"  context_length: {context_length}")
    logger.info(f"  output_dir: {split_output_dir}")

    start_time = time.time()

    # Create the tokenization and indexing pipeline
    # Note: Data Forager uses "eos_idx" but we pass our "eod_idx" (end-of-document)
    # The library uses this token as document separator regardless of naming
    tokenize_and_index = create_tokenize_and_index_jsonl_text_func(
        tokenizer_func=tokenizer_func,
        eos_idx=eod_idx,
        input_file_paths=[str(input_path)],
        output_base_path=str(split_output_dir),
        process_text_line_func=get_text_from_sample,
        logger=logger,
        name=f"tokenize-{split_name}",
        sample_size=context_length,
        token_dtype=token_dtype,
    )

    # Run tokenization and indexing
    tokenize_and_index()

    elapsed = time.time() - start_time

    # Load the index to get sample count
    index_store = IndexStore(base_path=str(split_output_dir))
    sample_index = index_store.load()
    num_samples = len(sample_index.sample_locations)

    logger.info(f"{split_name}: tokenized {num_samples} samples in {elapsed:.1f}s")

    return {
        "split": split_name,
        "status": "success",
        "num_samples": num_samples,
        "elapsed_seconds": elapsed,
    }


def verify_tokenized_data(
    split_name: str,
    output_dir: Path,
    context_length: int,
    token_dtype: np.dtype,
    tokenizer=None,
    inspect_samples: bool = False,
    num_samples_to_check: int = 5,
    logger=None,
) -> None:
    """
    Verify tokenized data by loading a few samples.

    :param split_name: Name of the split.
    :param output_dir: Base output directory.
    :param context_length: Expected context length.
    :param token_dtype: Expected token dtype.
    :param tokenizer: HuggingFace tokenizer for decoding (required if inspect_samples=True).
    :param inspect_samples: If True, decode and log sample contents.
    :param num_samples_to_check: Number of samples to verify.
    :param logger: Optional logger.
    """
    if logger is None:
        logger = module_logger

    if inspect_samples and tokenizer is None:
        raise ValueError("tokenizer is required when inspect_samples=True")

    split_output_dir = output_dir / split_name

    try:
        dataset = TokensDataset.create_from_index_on_filesystem(
            str(split_output_dir),
            token_dtype=token_dtype,
        )

        logger.info(f"{split_name}: verification - dataset has {len(dataset)} samples")

        # Check a few samples
        for i in range(min(num_samples_to_check, len(dataset))):
            sample = dataset[i]
            if len(sample) != context_length:
                logger.warning(
                    f"{split_name}: sample {i} has length {len(sample)}, "
                    f"expected {context_length}"
                )
            else:
                logger.debug(f"{split_name}: sample {i} OK (length={len(sample)})")

            if inspect_samples:
                decoded_text = tokenizer.decode(sample.tolist())
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"SAMPLE {i} (split={split_name}, length={len(sample)})")
                logger.info("=" * 80)
                logger.info(f"\n{decoded_text}")
                logger.info("=" * 80)

        logger.info(f"{split_name}: verification passed")

    except Exception as e:
        logger.error(f"{split_name}: verification failed: {e}")
        raise


def print_summary(results: list[dict], logger=None) -> None:
    """Print summary of tokenization results."""
    if logger is None:
        logger = module_logger

    logger.info("")
    logger.info("=" * 60)
    logger.info("Tokenization Summary")
    logger.info("=" * 60)

    total_samples = 0
    total_time = 0

    for result in results:
        split = result["split"]
        status = result["status"]
        num_samples = result.get("num_samples", 0)
        elapsed = result.get("elapsed_seconds", 0)

        if status == "skipped":
            logger.info(f"  {split}: {num_samples:,} samples (skipped - already exists)")
        elif status == "success":
            rate = num_samples / elapsed if elapsed > 0 else 0
            logger.info(
                f"  {split}: {num_samples:,} samples in {elapsed:.1f}s "
                f"({rate:.0f} samples/s)"
            )
            total_time += elapsed

        total_samples += num_samples

    logger.info("-" * 60)
    logger.info(f"  Total: {total_samples:,} samples")
    if total_time > 0:
        logger.info(f"  Total time: {total_time:.1f}s")
    logger.info("=" * 60)


def main():
    log_git_state()

    parser = argparse.ArgumentParser(
        description="Tokenize datasets for NTP training using Data Forager"
    )
    parser.add_argument(
        "--splits-dir",
        required=True,
        help="Directory containing split JSONL files (train.jsonl, val.jsonl, test.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for tokenized output",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help=f"Splits to tokenize (default: {DEFAULT_SPLITS})",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH,
        help=f"Context length for packed samples (default: {DEFAULT_CONTEXT_LENGTH})",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HuggingFace tokenizer model (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--eod-token",
        default="eos",
        help=(
            "End-of-document token: 'eos' (tokenizer.eos_token), 'pad' (tokenizer.pad_token), "
            "an integer token ID, or a token string. For Qwen3, use 'pad' to get <|endoftext|>. "
            "(default: eos)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-tokenization even if index exists",
    )
    parser.add_argument(
        "--inspect-samples",
        action="store_true",
        help="Decode and log sample contents during verification (for inspection)",
    )
    args = parser.parse_args()

    config = {
        "splits_dir": args.splits_dir,
        "output_dir": args.output_dir,
        "splits": args.splits,
        "context_length": args.context_length,
        "tokenizer": args.tokenizer,
        "eod_token": args.eod_token,
        "force": args.force,
        "inspect_samples": args.inspect_samples,
    }
    describe_config(**config)

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    tokenizer_func = create_tokenizer_func(tokenizer)
    eod_idx = resolve_eod_token_id(tokenizer, args.eod_token)

    # Determine token dtype based on vocab size
    if tokenizer.vocab_size > 65535:
        token_dtype = np.uint32
        module_logger.info(f"Using uint32 for tokens (vocab_size={tokenizer.vocab_size} > 65535)")
    else:
        token_dtype = np.uint16
        module_logger.info(f"Using uint16 for tokens (vocab_size={tokenizer.vocab_size} <= 65535)")

    # Process each split
    results = []
    for split_name in args.splits:
        input_path = splits_dir / f"{split_name}.jsonl"

        if not input_path.exists():
            module_logger.warning(f"{split_name}: file not found at {input_path}, skipping")
            continue

        result = tokenize_split(
            split_name=split_name,
            input_path=input_path,
            output_dir=output_dir,
            tokenizer_func=tokenizer_func,
            eod_idx=eod_idx,
            context_length=args.context_length,
            token_dtype=token_dtype,
            force=args.force,
            logger=module_logger,
        )
        results.append(result)

        # Verify if tokenization was successful
        if result["status"] == "success":
            verify_tokenized_data(
                split_name=split_name,
                output_dir=output_dir,
                context_length=args.context_length,
                token_dtype=token_dtype,
                tokenizer=tokenizer,
                inspect_samples=args.inspect_samples,
                logger=module_logger,
            )

    print_summary(results, logger=module_logger)


if __name__ == "__main__":
    main()
