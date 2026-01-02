#!/usr/bin/env python3
"""
Inspect random samples from downloaded datasets.

Usage (from repo root):
    python -m examples.agentic_llm_pretraining.datasets.inspect_samples --data-dir ../data
    python -m examples.agentic_llm_pretraining.datasets.inspect_samples --data-dir ../data --datasets fineweb-edu wikipedia --num-samples 5
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from rich.pretty import pretty_repr

from mlpug.mlpug_logging import get_logger, use_fancy_colors


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


def load_jsonl_samples(file_path: Path, num_samples: int, seed: int) -> List[Dict[str, Any]]:
    """
    Load random samples from a JSONL file.

    :param file_path: Path to JSONL file.
    :param num_samples: Number of samples to return.
    :param seed: Random seed for reproducibility.

    :return: List of sample dicts.
    """
    _l = module_logger

    # First pass: count lines
    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    if total_lines == 0:
        return []

    # Select random line indices
    random.seed(seed)
    num_to_sample = min(num_samples, total_lines)
    # set() for O(1) lookup when iterating
    selected_indices = set(random.sample(range(total_lines), num_to_sample))

    # Second pass: read selected lines
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in selected_indices:
                samples.append(json.loads(line))

    return samples


def inspect_dataset(file_path: Path, num_samples: int, seed: int, max_string: int) -> None:
    """
    Inspect samples from a single dataset file.

    :param file_path: Path to JSONL file.
    :param num_samples: Number of samples to display.
    :param seed: Random seed.
    :param max_string: Maximum string length before truncation.
    """
    _l = module_logger

    dataset_name = file_path.stem
    _l.info(f"\n{'='*60}")
    _l.info(f"Dataset: {dataset_name}")
    _l.info(f"{'='*60}")

    if not file_path.exists():
        _l.warning(f"File not found: {file_path}")
        return

    samples = load_jsonl_samples(file_path, num_samples, seed)

    if not samples:
        _l.warning(f"No samples found in {file_path}")
        return

    _l.info(f"Showing {len(samples)} random samples:")

    for i, sample in enumerate(samples, 1):
        _l.info(f"\n--- Sample {i} ---")
        _l.info(pretty_repr(sample, max_string=max_string))


def main():
    _l = module_logger

    parser = argparse.ArgumentParser(description="Inspect random samples from datasets")
    parser.add_argument(
        "--data-dir",
        default="../data",
        help="Directory containing downloaded JSONL files (default: ../data)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to inspect (default: all .jsonl files in data-dir)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples to show per dataset (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)"
    )
    parser.add_argument(
        "--max-string",
        type=int,
        default=500,
        help="Maximum string length before truncation (default: 500)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        _l.error(f"Data directory not found: {data_dir}")
        return

    if args.datasets:
        files = [data_dir / f"{name}.jsonl" for name in args.datasets]
    else:
        files = sorted(data_dir.glob("*.jsonl"))

    if not files:
        _l.warning(f"No JSONL files found in {data_dir}")
        return

    _l.info(f"Inspecting {len(files)} datasets from {data_dir}")

    for file_path in files:
        inspect_dataset(file_path, args.num_samples, args.seed, args.max_string)


if __name__ == "__main__":
    main()
