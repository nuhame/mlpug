"""
Dataset acquisition utilities for HuggingFace datasets.

Functions for downloading, filtering, and subsampling datasets to JSONL format.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, Optional

from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download

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


# Type alias for dataset loader functions - must return Dataset for .filter(), .shuffle(), .select()
LoadDatasetFunc = Callable[..., Dataset]


def download_subsample_hf(
    output_path: str,
    load_dataset_func: Optional[LoadDatasetFunc] = None,
    filter_func: Optional[Callable[[Dict], bool]] = None,
    filter_config: Optional[Dict[str, Any]] = None,
    target_count: Optional[int] = None,
    seed: int = 42,
    **acquisition_config
) -> AcquisitionStats:
    """
    Download a dataset, optionally filter and subsample, save as JSONL.

    Use this for small to medium datasets that fit in memory.

    :param output_path: Path to output JSONL file.
    :param load_dataset_func: Optional custom loader function. If None, uses HuggingFace
        `load_dataset`. Must accept **acquisition_config and return a `datasets.Dataset`
        (required for .filter(), .shuffle(), .select() methods).
    :param filter_func: Optional filter function: filter_func(sample, **filter_config) -> bool.
        Returns True to keep sample, False to discard.
    :param filter_config: Keyword arguments passed to filter_func.
    :param target_count: Number of samples to keep after filtering. If None, keep all.
    :param seed: Random seed for reproducibility.
    :param acquisition_config: Kwargs passed to load_dataset_func. For HuggingFace datasets,
        include path (e.g., "openai/gsm8k"), name, split, etc.

    :return: AcquisitionStats with seen, kept, and filtered_out counts.
    """
    random.seed(seed)
    filter_config = filter_config or {}

    # Load dataset using provided loader or default to HuggingFace
    loader = load_dataset_func if load_dataset_func is not None else load_dataset
    dataset = loader(**acquisition_config)

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
    output_path: str,
    keep_probability: float,
    load_dataset_func: Optional[LoadDatasetFunc] = None,
    target_count: Optional[int] = None,
    filter_func: Optional[Callable[[Dict, ...], bool]] = None,
    filter_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    log_interval: int = 100_000,
    **acquisition_config
) -> AcquisitionStats:
    """
    Stream a large dataset with probability sampling, save as JSONL.

    Use this for very large datasets that don't fit in memory. Streams through
    the dataset making random keep/discard decisions per sample.

    :param output_path: Path to output JSONL file.
    :param keep_probability: Probability of keeping each sample (after filtering).
        For example, if dataset has ~10B tokens and you want ~100M, use 0.01.
    :param load_dataset_func: Optional custom loader function. If None, uses HuggingFace
        `load_dataset`. Must accept **acquisition_config and return an iterable.
    :param target_count: Stop after keeping this many samples. If None, stream
        through the entire dataset.
    :param filter_func: Optional filter function: filter_func(sample, **filter_config) -> bool.
        Returns True to keep sample, False to discard. Applied before probability sampling.
    :param filter_config: Keyword arguments passed to filter_func.
    :param seed: Random seed for reproducibility.
    :param log_interval: Log progress every N samples.
    :param acquisition_config: Kwargs passed to load_dataset_func. For HuggingFace datasets,
        include path (e.g., "HuggingFaceFW/fineweb-edu"), name, split, etc.
        Note: streaming=True is added automatically for HuggingFace datasets.

    :return: AcquisitionStats with seen, kept, and filtered_out counts.
    """
    random.seed(seed)
    filter_config = filter_config or {}

    # Load dataset in streaming mode
    loader = load_dataset_func if load_dataset_func is not None else load_dataset
    if loader is load_dataset:
        acquisition_config["streaming"] = True
    dataset = loader(**acquisition_config)

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

                # Early exit if target reached
                if target_count is not None and kept >= target_count:
                    break

            # Log progress
            if seen % log_interval == 0:
                module_logger.info(
                    f"Progress: seen {seen:,}, kept {kept:,}, "
                    f"filtered {filtered_out:,} ({kept/seen*100:.2f}% kept)"
                )

    stats = AcquisitionStats(seen=seen, kept=kept, filtered_out=filtered_out)
    module_logger.info(f"Acquisition complete: {stats}")
    return stats


def load_with_hf_builder_script(
    path: str,
    name: str,
    split: str,
    script_name: str,
    **kwargs  # Ignore extra kwargs for compatibility
) -> Dataset:
    """
    Load a dataset using its HuggingFace GeneratorBasedBuilder script directly.

    Bypasses datasets 4.x restrictions by downloading the loading script from
    HuggingFace, importing it dynamically, and using the GeneratorBasedBuilder
    pattern to load the data.

    WARNING: This function executes arbitrary Python code from the dataset repository.
    Only use this with datasets from sources you trust (e.g., official HuggingFace
    datasets, well-known organizations like Salesforce, NVIDIA, etc.). The datasets
    library removed `trust_remote_code=True` support in version 4.x specifically to
    prevent execution of untrusted code.

    :param path: HuggingFace repo ID (e.g., "Salesforce/dialogstudio").
    :param name: Dataset subset/config name (e.g., "SODA", "MULTIWOZ2_2").
    :param split: Data split ("train", "validation", "test").
    :param script_name: Name of the builder script file (e.g., "dialogstudio.py").

    :return: Dataset object with the loaded samples.
    """
    import sys
    import importlib.util
    from datasets import GeneratorBasedBuilder

    source = f"{path}/{script_name}"
    module_logger.warning(
        f"Executing builder script from {path}. "
        f"This runs arbitrary code - only use with trusted sources."
    )
    module_name = script_name.replace(".py", "")

    # Download the loading script
    module_logger.info(f"Downloading loading script: {source}")
    try:
        script_path = hf_hub_download(
            repo_id=path,
            filename=script_name,
            repo_type="dataset",
        )
    except Exception as e:
        raise Exception(f"[{source}] Failed to download loading script") from e

    # Load the module dynamically
    module_logger.info(f"Loading module from {script_path}")
    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Register before exec to handle self-references
        spec.loader.exec_module(module)
    except Exception as e:
        raise Exception(f"[{source}] Failed to load module") from e

    # Find the builder class (subclass of GeneratorBasedBuilder)
    builder_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and
            issubclass(attr, GeneratorBasedBuilder) and
            attr is not GeneratorBasedBuilder):
            builder_class = attr
            break

    if builder_class is None:
        raise Exception(f"[{source}] No GeneratorBasedBuilder subclass found in module")

    # Instantiate the builder
    module_logger.info(f"Instantiating builder {builder_class.__name__} with config '{name}'")
    try:
        builder = builder_class(config_name=name)
    except Exception as e:
        raise Exception(f"[{source}] Failed to instantiate builder with config '{name}'") from e

    # Download and prepare the data
    base_url = f"https://huggingface.co/datasets/{path}/resolve/main/"
    module_logger.info(f"Downloading data from {base_url}")
    try:
        builder.download_and_prepare(base_path=base_url)
    except Exception as e:
        raise Exception(f"[{source}] Failed to download and prepare data") from e

    # Return the dataset for the requested split
    module_logger.info(f"Loading split '{split}'")
    try:
        return builder.as_dataset(split=split)
    except Exception as e:
        raise Exception(f"[{source}] Failed to load split '{split}'") from e


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
