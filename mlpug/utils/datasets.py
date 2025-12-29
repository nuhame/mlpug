"""
Dataset acquisition utilities for HuggingFace datasets.

Functions for downloading, filtering, and subsampling datasets to JSONL format.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any

from datasets import load_dataset

from mlpug.mlpug_logging import get_logger, use_fancy_colors


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


@dataclass
class AcquisitionStats:
    """Statistics from dataset acquisition."""
    seen: int
    kept: int
    filtered_out: int

    @property
    def keep_rate(self) -> float:
        return self.kept / self.seen if self.seen > 0 else 0.0

    @property
    def filter_rate(self) -> float:
        return self.filtered_out / self.seen if self.seen > 0 else 0.0


def download_subsample_hf(
    path: str,
    output_path: str,
    filter_func: Callable[[Dict], bool] | None = None,
    filter_config: Dict[str, Any] | None = None,
    target_count: int | None = None,
    seed: int = 42,
    **acquisition_config
) -> AcquisitionStats:
    """
    Download a HuggingFace dataset, optionally filter and subsample, save as JSONL.

    Use this for small to medium datasets that fit in memory.

    :param path: HuggingFace dataset path (e.g., "openai/gsm8k")
    :param output_path: Path to output JSONL file
    :param filter_func: Optional filter function: filter_func(sample, **filter_config) -> bool.
        Returns True to keep sample, False to discard.
    :param filter_config: Keyword arguments passed to filter_func.
    :param target_count: Number of samples to keep after filtering. If None, keep all.
    :param seed: Random seed for reproducibility.
    :param acquisition_config: Additional kwargs passed to load_dataset
        (e.g., name="main", split="train").

    :return: AcquisitionStats with seen, kept, and filtered_out counts.
    """
    random.seed(seed)
    filter_config = filter_config or {}

    # Load dataset
    dataset = load_dataset(path, **acquisition_config)

    seen = len(dataset)
    filtered_out = 0

    # Apply filter if provided
    if filter_func is not None:
        dataset = dataset.filter(lambda sample: filter_func(sample, **filter_config))
        filtered_out = seen - len(dataset)

    # Shuffle and subsample if target_count specified
    if target_count is not None and len(dataset) > target_count:
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(target_count))

    # Save to JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warned_fields: set[str] = set()
    kept = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            serializable = _to_serializable(sample, warned_fields)
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
            kept += 1

    stats = AcquisitionStats(seen=seen, kept=kept, filtered_out=filtered_out)
    module_logger.info(f"Acquisition complete: {stats}")
    return stats


def stream_subsample_hf(
    path: str,
    output_path: str,
    keep_probability: float,
    filter_func: Callable[[Dict, ...], bool] | None = None,
    filter_config: Dict[str, Any] | None = None,
    seed: int = 42,
    log_interval: int = 100_000,
    **acquisition_config
) -> AcquisitionStats:
    """
    Stream a large HuggingFace dataset with probability sampling, save as JSONL.

    Use this for very large datasets that don't fit in memory. Streams through
    the entire dataset once, making random keep/discard decisions per sample.

    :param path: HuggingFace dataset path (e.g., "HuggingFaceFW/fineweb-edu")
    :param output_path: Path to output JSONL file
    :param keep_probability: Probability of keeping each sample (after filtering).
        For example, if dataset has ~10B tokens and you want ~100M, use 0.01.
    :param filter_func: Optional filter function: filter_func(sample, **filter_config) -> bool.
        Returns True to keep sample, False to discard. Applied before probability sampling.
    :param filter_config: Keyword arguments passed to filter_func.
    :param seed: Random seed for reproducibility.
    :param log_interval: Log progress every N samples.
    :param acquisition_config: Additional kwargs passed to load_dataset
        (e.g., name="sample-10BT", split="train"). Note: streaming=True is added automatically.

    :return: AcquisitionStats with seen, kept, and filtered_out counts.
    """
    random.seed(seed)
    filter_config = filter_config or {}

    # Load dataset in streaming mode
    acquisition_config["streaming"] = True
    dataset = load_dataset(path, **acquisition_config)

    # Prepare output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warned_fields: set[str] = set()
    seen = 0
    kept = 0
    filtered_out = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            seen += 1

            # Apply filter first
            if filter_func is not None and not filter_func(sample, **filter_config):
                filtered_out += 1
                continue

            # Probability sampling
            if random.random() <= keep_probability:
                serializable = _to_serializable(sample, warned_fields)
                f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
                kept += 1

            # Log progress
            if seen % log_interval == 0:
                module_logger.info(
                    f"Progress: seen {seen:,}, kept {kept:,}, "
                    f"filtered {filtered_out:,} ({kept/seen*100:.2f}% kept)"
                )

    stats = AcquisitionStats(seen=seen, kept=kept, filtered_out=filtered_out)
    module_logger.info(f"Acquisition complete: {stats}")
    return stats


def _to_serializable(sample: Dict, warned_fields: set[str]) -> Dict:
    """
    Convert a sample dict to JSON-serializable types.

    :param sample: Sample dict from HuggingFace dataset.
    :param warned_fields: Set of field paths already warned about (modified in place).

    :return: Dict with only JSON-serializable values.
    """
    result = {}
    for key, value in sample.items():
        serialized = _serialize_value(value, key, warned_fields)
        if serialized is not None:
            result[key] = serialized
    return result


def _serialize_value(value: Any, field_path: str, warned_fields: set[str]) -> Any:
    """
    Serialize a single value, warning once per field path if non-serializable.

    :param value: Value to serialize.
    :param field_path: Dot-notation path to this field (e.g., "data.items[].name").
    :param warned_fields: Set of field paths already warned about (modified in place).

    :return: Serialized value, or None if not serializable.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value

    if isinstance(value, (list, tuple)):
        return [
            _serialize_value(v, f"{field_path}[]", warned_fields)
            for v in value
            if _serialize_value(v, f"{field_path}[]", warned_fields) is not None
        ]

    if isinstance(value, dict):
        return {
            k: _serialize_value(v, f"{field_path}.{k}", warned_fields)
            for k, v in value.items()
            if _serialize_value(v, f"{field_path}.{k}", warned_fields) is not None
        }

    # Non-serializable type
    if field_path not in warned_fields:
        module_logger.warning(f"Skipping non-serializable field '{field_path}': {type(value).__name__}")
        warned_fields.add(field_path)

    return None
