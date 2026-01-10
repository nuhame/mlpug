#!/usr/bin/env python3
"""
Create train/val/test splits from transformed datasets with true global shuffling.

Uses Data Forager for O(1) random access to samples:
1. Index all JSONL files (one-time, creates byte-offset index)
2. Create globally shuffled indices
3. Split indices into train/val/test
4. Write each split in shuffled order via random access

Memory: O(num_samples) for indices (~24 bytes/sample via Data Forager).

Usage (from repo root):
    python -m examples.agentic_llm_pretraining.datasets.create_splits \
        --transforms-dir ../data/agentic_llm_pretraining/full-08012026/transforms \
        --output-dir ../data/agentic_llm_pretraining/full-08012026/splits

    # Custom split ratios
    python -m examples.agentic_llm_pretraining.datasets.create_splits \
        --transforms-dir ../data/agentic_llm_pretraining/full-08012026/transforms \
        --output-dir ../data/agentic_llm_pretraining/full-08012026/splits \
        --train-ratio 0.8 \
        --val-ratio 0.1 \
        --test-ratio 0.1

    # Limit total samples for testing
    python -m examples.agentic_llm_pretraining.datasets.create_splits \
        --transforms-dir ../data/agentic_llm_pretraining/inspection/transforms \
        --output-dir ../data/agentic_llm_pretraining/inspection/splits \
        --num-samples 1000

    # Force re-indexing (if JSONL files changed)
    python -m examples.agentic_llm_pretraining.datasets.create_splits \
        --transforms-dir ../data/agentic_llm_pretraining/full-08012026/transforms \
        --output-dir ../data/agentic_llm_pretraining/full-08012026/splits \
        --force-reindex
"""

import argparse
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from data_forager.datasets.jsonl import JsonlDataset
from data_forager.index_stores.fs_based import IndexStore
from data_forager.indexers.jsonl_indexer import create_default_jsonl_indexer

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from mlpug.utils.git_logging import log_git_state


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


@dataclass
class SplitStats:
    """Statistics for split creation."""

    train_total: int = 0
    val_total: int = 0
    test_total: int = 0
    train_per_source: dict[str, int] = field(default_factory=dict)
    val_per_source: dict[str, int] = field(default_factory=dict)
    test_per_source: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.train_total + self.val_total + self.test_total

    def add_train(self, source: str) -> None:
        self.train_total += 1
        self.train_per_source[source] = self.train_per_source.get(source, 0) + 1

    def add_val(self, source: str) -> None:
        self.val_total += 1
        self.val_per_source[source] = self.val_per_source.get(source, 0) + 1

    def add_test(self, source: str) -> None:
        self.test_total += 1
        self.test_per_source[source] = self.test_per_source.get(source, 0) + 1


def describe_config(
    transforms_dir: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    num_samples: int | None,
    force_reindex: bool,
) -> None:
    """Log script configuration."""
    module_logger.info("Configuration:")
    module_logger.info(f"  transforms_dir: {transforms_dir}")
    module_logger.info(f"  output_dir: {output_dir}")
    module_logger.info(f"  train_ratio: {train_ratio}")
    module_logger.info(f"  val_ratio: {val_ratio}")
    module_logger.info(f"  test_ratio: {test_ratio}")
    module_logger.info(f"  seed: {seed}")
    module_logger.info(f"  num_samples: {num_samples}")
    module_logger.info(f"  force_reindex: {force_reindex}")


def build_or_load_index(
    transforms_dir: Path,
    force_reindex: bool = False,
    logger=None,
) -> JsonlDataset:
    """
    Build Data Forager index if needed, then load dataset.

    :param transforms_dir: Directory containing JSONL files.
    :param force_reindex: Force rebuilding the index.
    :param logger: Optional logger.

    :return: JsonlDataset with random access to all samples.
    """
    if logger is None:
        logger = module_logger

    index_store = IndexStore(str(transforms_dir))
    index_exists = index_store.exists()

    if force_reindex and index_exists:
        logger.info("Clearing existing index...")
        index_store.clear()
        index_exists = False

    if not index_exists:
        logger.info("Building Data Forager index...")
        indexer = create_default_jsonl_indexer(str(transforms_dir))
        indexer()
        logger.info("Index built successfully")
    else:
        logger.info("Using existing Data Forager index")

    logger.info("Loading dataset with random access...")
    dataset = JsonlDataset.create_from_index_on_filesystem(str(transforms_dir))
    logger.info(f"Loaded {len(dataset):,} samples")

    return dataset


def write_split(
    dataset: JsonlDataset,
    indices: list[int],
    output_path: Path,
    stats: SplitStats,
    add_to_stats: Callable[[str], None],
    split_name: str,
    logger=None,
) -> None:
    """
    Write samples to a split file in shuffled order.

    :param dataset: JsonlDataset for random access.
    :param indices: Sample indices to write (already shuffled).
    :param output_path: Path to output JSONL file.
    :param stats: SplitStats to update.
    :param add_to_stats: Method to call (stats.add_train, etc.)
    :param split_name: Name for logging (train/val/test).
    :param logger: Optional logger.
    """
    if logger is None:
        logger = module_logger

    logger.info(f"Writing {len(indices):,} samples to {split_name}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            source = sample.get("source", "unknown")
            add_to_stats(source)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            # Progress logging every 100k samples
            if (i + 1) % 100_000 == 0:
                logger.info(f"  {split_name}: {i + 1:,} / {len(indices):,}")

    logger.info(f"Wrote {len(indices):,} samples to {output_path}")


def print_summary(stats: SplitStats, logger=None) -> None:
    """Print summary of split statistics."""
    if logger is None:
        logger = module_logger

    total = stats.total
    if total == 0:
        logger.warning("No samples processed!")
        return

    logger.info("")
    logger.info("=" * 70)
    logger.info("Split Summary")
    logger.info("=" * 70)
    logger.info(f"  Total samples: {total:,}")
    logger.info(f"  Train: {stats.train_total:,} ({stats.train_total/total*100:.1f}%)")
    logger.info(f"  Val:   {stats.val_total:,} ({stats.val_total/total*100:.1f}%)")
    logger.info(f"  Test:  {stats.test_total:,} ({stats.test_total/total*100:.1f}%)")
    logger.info("-" * 70)

    # Show per-source breakdown
    all_sources = sorted(set(
        list(stats.train_per_source.keys()) +
        list(stats.val_per_source.keys()) +
        list(stats.test_per_source.keys())
    ))

    logger.info("Per-source breakdown:")
    logger.info(f"  {'Source':<30} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    logger.info(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for source in all_sources:
        train_n = stats.train_per_source.get(source, 0)
        val_n = stats.val_per_source.get(source, 0)
        test_n = stats.test_per_source.get(source, 0)
        source_total = train_n + val_n + test_n
        logger.info(f"  {source:<30} {train_n:>10,} {val_n:>10,} {test_n:>10,} {source_total:>10,}")

    logger.info("=" * 70)


def main():
    log_git_state()

    parser = argparse.ArgumentParser(
        description="Create train/val/test splits from transformed datasets"
    )
    parser.add_argument(
        "--transforms-dir",
        required=True,
        help="Directory containing transformed JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write split files (train.jsonl, val.jsonl, test.jsonl)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of data for training (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for test (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit total samples (default: all)",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force rebuilding the Data Forager index",
    )
    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio:.6f} "
            f"(train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})"
        )

    config = vars(args)
    describe_config(**config)

    transforms_dir = Path(args.transforms_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build or load Data Forager index
    dataset = build_or_load_index(
        transforms_dir,
        force_reindex=args.force_reindex,
        logger=module_logger,
    )

    # Shuffle ALL indices globally first
    total_available = len(dataset)
    module_logger.info(f"Shuffling {total_available:,} indices with seed={args.seed}...")
    random.seed(args.seed)
    indices = list(range(total_available))
    random.shuffle(indices)

    # Limit samples AFTER shuffling (so we get a random sample from all sources)
    if args.num_samples is not None:
        total_samples = min(total_available, args.num_samples)
        module_logger.info(f"Limiting to {total_samples:,} samples (from {total_available:,} total)")
        indices = indices[:total_samples]
    else:
        total_samples = total_available

    # Calculate split boundaries
    train_end = int(total_samples * args.train_ratio)
    val_end = train_end + int(total_samples * args.val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    module_logger.info(
        f"Split sizes: train={len(train_indices):,}, "
        f"val={len(val_indices):,}, test={len(test_indices):,}"
    )

    # Write splits
    stats = SplitStats()

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    write_split(
        dataset=dataset,
        indices=train_indices,
        output_path=train_path,
        stats=stats,
        add_to_stats=stats.add_train,
        split_name="train",
        logger=module_logger,
    )

    write_split(
        dataset=dataset,
        indices=val_indices,
        output_path=val_path,
        stats=stats,
        add_to_stats=stats.add_val,
        split_name="val",
        logger=module_logger,
    )

    write_split(
        dataset=dataset,
        indices=test_indices,
        output_path=test_path,
        stats=stats,
        add_to_stats=stats.add_test,
        split_name="test",
        logger=module_logger,
    )

    print_summary(stats)


if __name__ == "__main__":
    main()
