#!/usr/bin/env python3
"""
Verify train/val/test split files for correctness.

Checks:
1. Physical line counts match expected totals
2. All samples have valid JSON structure with required fields
3. No duplicate (source, index) keys within each split
4. Mutual exclusivity: no overlaps between train/val/test

Usage (from repo root):
    python -m examples.agentic_llm_pretraining.datasets.test_splits \
        --splits-dir ../data/agentic_llm_pretraining/inspection/splits

    # With expected counts (fails if mismatch)
    python -m examples.agentic_llm_pretraining.datasets.test_splits \
        --splits-dir ../data/agentic_llm_pretraining/full-08012026/splits \
        --expected-total 3793850
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from mlpug.utils.git_logging import log_git_state


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


@dataclass
class SplitData:
    """Data collected from a split file."""

    name: str
    path: Path
    keys: list = field(default_factory=list)
    unique_keys: set = field(default_factory=set)  # Set after loading, for reuse
    sources: dict = field(default_factory=dict)  # Non-unique sample count per source
    parse_errors: list = field(default_factory=list)


def describe_config(
    splits_dir: str,
    expected_total: int | None,
) -> None:
    """Log script configuration."""
    module_logger.info("Configuration:")
    module_logger.info(f"  splits_dir: {splits_dir}")
    module_logger.info(f"  expected_total: {expected_total}")


def load_split(filepath: Path, name: str, logger=None) -> SplitData:
    """
    Load a split file and collect all (source, index) keys.

    :param filepath: Path to JSONL split file.
    :param name: Name of the split (train/val/test).
    :param logger: Optional logger.

    :return: SplitData with all keys and source counts.
    """
    if logger is None:
        logger = module_logger

    data = SplitData(name=name, path=filepath)

    logger.info(f"Loading {name}...")

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                data.parse_errors.append((line_num, str(e)))
                continue

            if "source" not in sample or "index" not in sample:
                data.parse_errors.append((line_num, "missing 'source' or 'index' field"))
                continue

            source = sample["source"]
            index = sample["index"]
            key = (source, index)

            data.keys.append(key)
            data.sources[source] = data.sources.get(source, 0) + 1

            if len(data.keys) % 500_000 == 0:
                logger.info(f"  {name}: {len(data.keys):,} samples processed...")

    # Convert to set once for reuse in uniqueness and overlap checks
    data.unique_keys = set(data.keys)

    logger.info(f"  {name}: {len(data.keys):,} samples loaded")

    return data


def check_internal_uniqueness(data: SplitData, logger=None) -> int:
    """
    Check if all keys within a split are unique.

    Compares list length to set length - if they differ, duplicates exist.

    :param data: SplitData to check (must have unique_keys already set).
    :param logger: Optional logger.

    :return: Number of duplicates found.
    """
    if logger is None:
        logger = module_logger

    list_count = len(data.keys)
    unique_count = len(data.unique_keys)
    duplicate_count = list_count - unique_count

    if duplicate_count > 0:
        logger.error(f"  {data.name}: {duplicate_count:,} duplicates found!")
    else:
        logger.info(f"  {data.name}: all {list_count:,} keys unique ✓")

    return duplicate_count


def check_overlap(
    data_a: SplitData,
    data_b: SplitData,
    logger=None,
) -> int:
    """
    Check for overlapping keys between two splits.

    :param data_a: First split data (must have unique_keys already set).
    :param data_b: Second split data (must have unique_keys already set).
    :param logger: Optional logger.

    :return: Number of overlapping keys.
    """
    if logger is None:
        logger = module_logger

    overlap = data_a.unique_keys & data_b.unique_keys

    if overlap:
        logger.error(f"  {data_a.name} ∩ {data_b.name} = {len(overlap):,} samples!")
        examples = list(overlap)[:5]
        for key in examples:
            logger.error(f"    - {key}")
        if len(overlap) > 5:
            logger.error(f"    ... and {len(overlap) - 5:,} more")
    else:
        logger.info(f"  {data_a.name} ∩ {data_b.name} = ∅ ✓")

    return len(overlap)


def print_summary(
    train: SplitData,
    val: SplitData,
    test: SplitData,
    expected_total: int | None,
    logger=None,
) -> bool:
    """
    Print summary of source distribution.

    :param train: Train split data.
    :param val: Val split data.
    :param test: Test split data.
    :param expected_total: Expected total samples (optional).
    :param logger: Optional logger.

    :return: True if expected_total matches (or not specified).
    """
    if logger is None:
        logger = module_logger

    total = len(train.keys) + len(val.keys) + len(test.keys)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Split Verification Summary")
    logger.info("=" * 70)

    logger.info("Sample counts:")
    logger.info(f"  train: {len(train.keys):,}")
    logger.info(f"  val:   {len(val.keys):,}")
    logger.info(f"  test:  {len(test.keys):,}")
    logger.info(f"  total: {total:,}")

    total_matches = True
    if expected_total is not None:
        if total == expected_total:
            logger.info(f"  expected: {expected_total:,} ✓")
        else:
            logger.error(f"  expected: {expected_total:,} ✗ (diff: {total - expected_total:+,})")
            total_matches = False

    logger.info("-" * 70)
    logger.info("Source distribution:")
    all_sources = sorted(set(
        list(train.sources.keys()) +
        list(val.sources.keys()) +
        list(test.sources.keys())
    ))
    logger.info(f"  {'Source':<30} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    logger.info(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for source in all_sources:
        train_n = train.sources.get(source, 0)
        val_n = val.sources.get(source, 0)
        test_n = test.sources.get(source, 0)
        source_total = train_n + val_n + test_n
        logger.info(f"  {source:<30} {train_n:>10,} {val_n:>10,} {test_n:>10,} {source_total:>10,}")

    logger.info("=" * 70)

    return total_matches


def main():
    log_git_state()

    parser = argparse.ArgumentParser(
        description="Verify train/val/test split files for correctness"
    )
    parser.add_argument(
        "--splits-dir",
        required=True,
        help="Directory containing train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=None,
        help="Expected total number of samples (optional, fails if mismatch)",
    )
    args = parser.parse_args()

    config = vars(args)
    describe_config(**config)

    splits_dir = Path(args.splits_dir)

    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"
    test_path = splits_dir / "test.jsonl"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            module_logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")

    # Step 1: Load all keys from each split
    module_logger.info("")
    module_logger.info("Step 1: Loading split files...")
    train = load_split(train_path, "train", module_logger)
    val = load_split(val_path, "val", module_logger)
    test = load_split(test_path, "test", module_logger)

    all_passed = True

    # Check for parse errors
    total_parse_errors = len(train.parse_errors) + len(val.parse_errors) + len(test.parse_errors)
    if total_parse_errors > 0:
        module_logger.error(f"Parse errors found: {total_parse_errors}")
        for data in [train, val, test]:
            for line_num, error in data.parse_errors[:3]:
                module_logger.error(f"  {data.name} line {line_num}: {error}")
        all_passed = False

    # Step 2: Check internal uniqueness
    module_logger.info("")
    module_logger.info("Step 2: Checking internal uniqueness...")
    train_dups = check_internal_uniqueness(train, module_logger)
    val_dups = check_internal_uniqueness(val, module_logger)
    test_dups = check_internal_uniqueness(test, module_logger)

    if train_dups + val_dups + test_dups > 0:
        all_passed = False

    # Step 3: Check for overlaps between splits
    module_logger.info("")
    module_logger.info("Step 3: Checking for overlaps between splits...")
    train_val_overlap = check_overlap(train, val, module_logger)
    train_test_overlap = check_overlap(train, test, module_logger)
    val_test_overlap = check_overlap(val, test, module_logger)

    if train_val_overlap + train_test_overlap + val_test_overlap > 0:
        all_passed = False

    # Print summary
    total_matches = print_summary(train, val, test, args.expected_total, module_logger)
    if not total_matches:
        all_passed = False

    if all_passed:
        module_logger.info("")
        module_logger.info("All checks passed! ✓")
    else:
        module_logger.error("")
        module_logger.error("Some checks FAILED! ✗")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
