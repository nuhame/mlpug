"""
Common utilities for dataset transformation pipelines.

Shared between v1 and v2 dataset pipelines.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from mlpug.mlpug_logging import get_logger, use_fancy_colors


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


# Valid chat roles for Qwen3 format
VALID_CHAT_ROLES = ("user", "assistant", "tool")


@dataclass
class TransformStats:
    """Statistics from transform operation."""
    total: int
    success: int
    failed: int

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        return self.failed / self.total if self.total > 0 else 0.0


# Type alias for preprocess functions
# Returns dict of template fields, or None if sample is invalid
PreprocessFunc = Callable[[dict, int, str, logging.Logger], Optional[dict]]


def load_metadata(metadata_path: str) -> dict[str, any]:
    """
    Load metadata from JSON file, excluding comment fields.

    :param metadata_path: Path to metadata JSON file.

    :return: Dict mapping dataset names to their configurations.
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {k: v for k, v in metadata.items() if not k.startswith("_")}


def load_raw_samples(
    data_dir: Path,
    dataset_name: str,
    num_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> list[dict]:
    """
    Load raw samples from JSONL file.

    :param data_dir: Directory containing raw JSONL files.
    :param dataset_name: Name of dataset (used as filename).
    :param num_samples: Optional limit on number of samples to load.
    :param logger: Optional logger.

    :return: List of sample dicts.
    """
    if logger is None:
        logger = module_logger

    input_path = data_dir / f"{dataset_name}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {input_path}")

    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if num_samples is not None and len(samples) >= num_samples:
                break
            samples.append(json.loads(line))

    logger.info(f"{dataset_name}: loaded {len(samples)} raw samples from {input_path}")
    return samples


def print_summary(
    results: dict[str, TransformStats],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Print summary of all transform operations.

    :param results: Dict mapping dataset names to their TransformStats.
    :param logger: Optional logger.
    """
    if logger is None:
        logger = module_logger

    logger.info("")
    logger.info("=" * 50)
    logger.info("Transform Summary")
    logger.info("=" * 50)

    total_success = 0
    total_failed = 0
    total_samples = 0

    for name, stats in results.items():
        pct = stats.success_rate * 100
        logger.info(f"  {name}: {stats.success}/{stats.total} success ({pct:.1f}%)")
        total_success += stats.success
        total_failed += stats.failed
        total_samples += stats.total

    logger.info("-" * 50)
    total_pct = (total_success / total_samples * 100) if total_samples > 0 else 0
    logger.info(f"  Total: {total_success}/{total_samples} success ({total_pct:.1f}%)")
    logger.info(f"  Failed: {total_failed}")
    logger.info("=" * 50)


def extract_dialogstudio_messages(
    log: list,
    index: int,
    dataset_name: str,
    logger: logging.Logger,
) -> Optional[list[dict]]:
    """
    Extract messages from DialogStudio log format.

    Converts turns with "user utterance" and "system response" fields
    to message dicts with "user" and "assistant" keys.

    :param log: List of turn dicts with "user utterance" and "system response".
    :param index: Sample index for logging.
    :param dataset_name: Dataset name for logging.
    :param logger: Logger instance.

    :return: List of message dicts, or None if invalid.
    """
    if not log:
        logger.warning(f"{dataset_name}[{index}]: empty log field")
        return None

    messages = []
    for turn in log:
        user_utterance = turn.get("user utterance", "")
        system_response = turn.get("system response", "")

        if user_utterance:
            messages.append({"user": user_utterance})
        if system_response:
            messages.append({"assistant": system_response})

    if not messages:
        logger.warning(f"{dataset_name}[{index}]: no valid turns in log")
        return None

    return messages


def describe_config(
    metadata: str,
    data_dir: str,
    output_dir: str,
    datasets: list[str] | None,
    num_samples: int | None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log script configuration for transform scripts.

    :param metadata: Path to metadata JSON file.
    :param data_dir: Directory containing raw JSONL files.
    :param output_dir: Directory to write transformed output.
    :param datasets: List of specific datasets to transform, or None for all.
    :param num_samples: Limit on samples per dataset, or None for all.
    :param logger: Optional logger.
    """
    if logger is None:
        logger = module_logger

    logger.info("Configuration:")
    logger.info(f"  metadata: {metadata}")
    logger.info(f"  data_dir: {data_dir}")
    logger.info(f"  output_dir: {output_dir}")
    logger.info(f"  datasets: {datasets}")
    logger.info(f"  num_samples: {num_samples}")


def describe_dataset_config(
    name: str,
    config: dict[str, Any],
    template: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log per-dataset configuration from metadata.

    :param name: Dataset name.
    :param config: Dataset configuration dict from metadata.
    :param template: Optional template instance (v2 only).
    :param logger: Optional logger.
    """
    if logger is None:
        logger = module_logger

    from rich.pretty import pretty_repr

    if template is not None:
        logger.info(f"  [{name}] template: {type(template).__name__}")
    logger.info(f"  [{name}] config: {pretty_repr(config)}")
