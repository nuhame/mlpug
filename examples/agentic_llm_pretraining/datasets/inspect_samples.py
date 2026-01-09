#!/usr/bin/env python3
"""
Inspect random samples from downloaded datasets.

Usage (from repo root):
    python -m examples.agentic_llm_pretraining.datasets.inspect_samples \
        --data-dir ../data/agentic_llm_pretraining \
        --metadata examples/agentic_llm_pretraining/datasets/inspect_metadata.json

    python -m examples.agentic_llm_pretraining.datasets.inspect_samples \
        --data-dir ../data/agentic_llm_pretraining \
        --metadata examples/agentic_llm_pretraining/datasets/inspect_metadata.json \
        --datasets fineweb-edu wikipedia --num-samples 5
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


def inspect_dataset(
    file_path: Path,
    num_samples: int,
    seed: int = 42,
    max_string: int = 1024,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Inspect samples from a single dataset file.

    :param file_path: Path to JSONL file.
    :param num_samples: Number of samples to display.
    :param seed: Random seed.
    :param max_string: Maximum string length before truncation.
    :param metadata: Optional metadata dict with description, primary_purpose, etc.
    """
    _l = module_logger

    dataset_name = file_path.stem
    _l.info(f"\n{'='*60}")
    _l.info(f"Dataset: {dataset_name}")
    if metadata:
        purpose = metadata.get("primary_purpose", "unknown")
        description = metadata.get("description", "")
        also_provides = metadata.get("also_provides", [])
        _l.info(f"Purpose: {purpose}")
        if also_provides:
            _l.info(f"Also provides: {', '.join(also_provides)}")
        if description:
            _l.info(f"Description: {description}")
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
        for key, value in sample.items():
            if isinstance(value, str):
                text = value
                if max_string and len(text) > max_string:
                    text = text[:max_string] + f"\n... [truncated, {len(text) - max_string} chars remaining]"
                _l.info(f"=== {key} ===")
                _l.info(text)
                _l.info(f"=== end {key} ===")
            else:
                _l.info(f"{key}: {pretty_repr(value, max_string=max_string)}")


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load metadata from JSON file."""
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Remove comment fields
    return {k: v for k, v in metadata.items() if not k.startswith("_")}


def main():
    _l = module_logger

    parser = argparse.ArgumentParser(description="Inspect random samples from datasets")
    parser.add_argument(
        "--data-dir",
        default="../data",
        help="Directory containing downloaded JSONL files (default: ../data)"
    )
    parser.add_argument(
        "--metadata",
        help="Path to metadata JSON file (for dataset descriptions)"
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
        default=1024,
        help="Maximum string length before truncation (default: 1024)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        _l.error(f"Data directory not found: {data_dir}")
        return

    # Load metadata if provided
    metadata = {}
    if args.metadata:
        metadata = load_metadata(Path(args.metadata))
        _l.info(f"Loaded metadata for {len(metadata)} datasets")

    if args.datasets:
        files = [data_dir / f"{name}.jsonl" for name in args.datasets]
    else:
        files = sorted(data_dir.glob("*.jsonl"))

    if not files:
        _l.warning(f"No JSONL files found in {data_dir}")
        return

    _l.info(f"Inspecting {len(files)} datasets from {data_dir}")

    for file_path in files:
        dataset_name = file_path.stem
        dataset_metadata = metadata.get(dataset_name)
        inspect_dataset(file_path, args.num_samples, args.seed, args.max_string, dataset_metadata)


if __name__ == "__main__":
    main()
