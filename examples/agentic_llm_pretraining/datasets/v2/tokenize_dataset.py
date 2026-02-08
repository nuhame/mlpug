#!/usr/bin/env python3
"""
V2: Tokenize datasets with auxiliary data (loss masks) using Data Forager.

Tokenizes JSONL files with parts format into fixed-length token samples with
loss masks, and creates a byte-offset index for O(1) random access during training.

Input format: {"parts": [{"type": "...", "text": "..."}, ...]}
Output format: tokens + loss_mask arrays per sample

Usage (from repo root):
    # Tokenize all splits with default settings (4K context)
    python -m examples.agentic_llm_pretraining.datasets.v2.tokenize_dataset \
        --splits-dir ../data/agentic_llm_pretraining/v2/splits \
        --output-dir ../data/agentic_llm_pretraining/v2/tokenized

    # Tokenize only train split with custom context length
    python -m examples.agentic_llm_pretraining.datasets.v2.tokenize_dataset \
        --splits-dir ../data/agentic_llm_pretraining/v2/splits \
        --output-dir ../data/agentic_llm_pretraining/v2/tokenized \
        --splits train \
        --context-length 4096

    # Force re-tokenization (clears existing index)
    python -m examples.agentic_llm_pretraining.datasets.v2.tokenize_dataset \
        --splits-dir ../data/agentic_llm_pretraining/v2/splits \
        --output-dir ../data/agentic_llm_pretraining/v2/tokenized \
        --force

Output structure:
    tokenized/
    ├── train/
    │   ├── tokenized-samples/
    │   │   └── train-tokenized-samples.bin  (tokens + loss_mask concatenated)
    │   └── index/
    │       ├── sample_locations.bin
    │       ├── file_location.txt
    │       └── sample_schema.json
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
from typing import List

import numpy as np
from transformers import AutoTokenizer

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from mlpug.utils.git_logging import log_git_state

from data_forager.indexers.tokenization_indexer import create_tokenize_and_index_with_aux_func
from data_forager.datasets.tokens_with_aux import TokensWithAuxDataset
from data_forager.sample_generators.aux import Part as ForagerPart, LossMaskGenerator


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


# Default tokenizer for Qwen3
DEFAULT_TOKENIZER = "Qwen/Qwen3-1.7B"

# Default context length (4K for baseline, following staged training approach)
DEFAULT_CONTEXT_LENGTH = 4096

# Token dtype for storage (uint32 for Qwen3's ~152K vocab)
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


def parse_parts_from_jsonl(line_bytes: bytes) -> List[ForagerPart]:
    """
    Parse a JSONL line into a list of ForagerPart objects.

    Expected format: {"parts": [{"type": "...", "text": "..."}, ...]}

    :param line_bytes: Raw bytes of the JSONL line.

    :return: List of ForagerPart objects (data_forager.sample_generators.aux.Part).
    """
    data = json.loads(line_bytes.decode("utf-8"))
    parts = data.get("parts", [])
    return [ForagerPart(type=p["type"], text=p["text"]) for p in parts]


def tokenize_split(
    split_name: str,
    splits_dir: Path,
    output_dir: Path,
    tokenizer,
    context_length: int,
    eod_token_id: int,
    force: bool = False,
) -> int:
    """
    Tokenize a single split with loss masks.

    :param split_name: Name of the split (train, val, test).
    :param splits_dir: Directory containing split JSONL files.
    :param output_dir: Base output directory.
    :param tokenizer: HuggingFace tokenizer instance.
    :param context_length: Fixed context length for samples.
    :param eod_token_id: End-of-document token ID.
    :param force: If True, clear existing output and re-tokenize.

    :return: Number of samples created.
    """
    input_path = splits_dir / f"{split_name}.jsonl"
    if not input_path.exists():
        module_logger.warning(f"Split file not found: {input_path}, skipping")
        return 0

    split_output_dir = output_dir / split_name

    # Check if already tokenized
    index_dir = split_output_dir / "index"
    if index_dir.exists() and not force:
        module_logger.info(f"{split_name}: already tokenized, skipping (use --force to re-tokenize)")
        # Load existing to get count
        try:
            dataset = TokensWithAuxDataset.create_from_index_on_filesystem(
                str(split_output_dir)
            )
            return len(dataset)
        except Exception:
            pass
        return 0

    # Clear existing output if force
    if split_output_dir.exists() and force:
        module_logger.info(f"{split_name}: clearing existing output (--force)")
        shutil.rmtree(split_output_dir)

    split_output_dir.mkdir(parents=True, exist_ok=True)

    module_logger.info(f"{split_name}: tokenizing from {input_path}")
    start_time = time.time()

    # Create loss mask generator
    loss_mask_generator = LossMaskGenerator(
        masked_types={"system", "prompt"},
        mask_eos=False,  # Train on EOS tokens
    )

    # Create tokenization function
    def tokenize_text(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    # Create the indexer with aux data support
    indexer = create_tokenize_and_index_with_aux_func(
        process_parts_func=parse_parts_from_jsonl,
        tokenizer_func=tokenize_text,
        eos_idx=eod_token_id,
        aux_generators={"loss_mask": loss_mask_generator},
        input_file_paths=[str(input_path)],
        output_base_path=str(split_output_dir),
        sample_size=context_length,
        token_dtype=DEFAULT_TOKEN_DTYPE,
    )

    # Run tokenization
    indexer()

    duration = time.time() - start_time

    # Load the created dataset to get sample count
    dataset = TokensWithAuxDataset.create_from_index_on_filesystem(
        str(split_output_dir)
    )
    num_samples = len(dataset)

    module_logger.info(
        f"{split_name}: created {num_samples} samples in {duration:.1f}s"
    )

    return num_samples


def inspect_tokenized_samples(
    output_dir: Path,
    split_name: str,
    tokenizer,
    num_samples: int = 3,
) -> None:
    """
    Inspect a few tokenized samples for verification.

    :param output_dir: Base output directory.
    :param split_name: Name of the split to inspect.
    :param tokenizer: HuggingFace tokenizer for decoding.
    :param num_samples: Number of samples to inspect.
    """
    split_dir = output_dir / split_name

    try:
        dataset = TokensWithAuxDataset.create_from_index_on_filesystem(str(split_dir))
    except Exception as e:
        module_logger.warning(f"Could not load {split_name} for inspection: {e}")
        return

    module_logger.info(f"\n{'='*60}")
    module_logger.info(f"Inspecting {split_name} samples ({len(dataset)} total)")
    module_logger.info(f"{'='*60}")

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        tokens = sample["tokens"]
        loss_mask = sample["loss_mask"]

        # Decode tokens
        text = tokenizer.decode(tokens, skip_special_tokens=False)

        # Count masked vs unmasked tokens
        masked_count = int(loss_mask.sum())
        unmasked_count = len(loss_mask) - masked_count

        module_logger.info(f"\n--- Sample {i} ---")
        module_logger.info(f"Tokens: {len(tokens)}, Masked: {masked_count}, Unmasked: {unmasked_count}")
        module_logger.info(f"Text preview (first 500 chars):\n{text[:500]}...")

        # Show mask transitions (where mask changes value)
        transitions = []
        for j in range(1, len(loss_mask)):
            if loss_mask[j] != loss_mask[j-1]:
                transitions.append((j, int(loss_mask[j])))
        if transitions:
            module_logger.info(f"Mask transitions (position, new_value): {transitions[:10]}...")


def main():
    log_git_state()

    parser = argparse.ArgumentParser(
        description="Tokenize datasets with loss masks (v2: parts format input)"
    )
    parser.add_argument(
        "--splits-dir",
        required=True,
        help="Directory containing split JSONL files (train.jsonl, val.jsonl, test.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write tokenized output",
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
        help=f"Context length for samples (default: {DEFAULT_CONTEXT_LENGTH})",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HuggingFace tokenizer to use (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--eod-token",
        choices=["eos", "pad"],
        default="pad",
        help="Token to use as end-of-document separator (default: pad, recommended for Qwen3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-tokenization even if output exists",
    )
    parser.add_argument(
        "--inspect-samples",
        action="store_true",
        help="Inspect a few samples after tokenization",
    )
    args = parser.parse_args()

    describe_config(
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        context_length=args.context_length,
        tokenizer=args.tokenizer,
        eod_token=args.eod_token,
        force=args.force,
        inspect_samples=args.inspect_samples,
    )

    # Load tokenizer
    module_logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Determine EOD token
    if args.eod_token == "pad":
        eod_token_id = tokenizer.pad_token_id
        module_logger.info(f"Using pad_token as EOD: id={eod_token_id}")
    else:
        eod_token_id = tokenizer.eos_token_id
        module_logger.info(f"Using eos_token as EOD: id={eod_token_id}")

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)

    # Tokenize each split
    total_samples = 0
    for split_name in args.splits:
        num_samples = tokenize_split(
            split_name=split_name,
            splits_dir=splits_dir,
            output_dir=output_dir,
            tokenizer=tokenizer,
            context_length=args.context_length,
            eod_token_id=eod_token_id,
            force=args.force,
        )
        total_samples += num_samples

    module_logger.info(f"\nTotal samples across all splits: {total_samples}")

    # Inspect samples if requested
    if args.inspect_samples:
        for split_name in args.splits:
            inspect_tokenized_samples(output_dir, split_name, tokenizer)


if __name__ == "__main__":
    main()
