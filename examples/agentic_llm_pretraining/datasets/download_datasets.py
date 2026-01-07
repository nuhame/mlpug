#!/usr/bin/env python3
"""
Download datasets from metadata JSON file.

Usage (from repo root):
    python -m examples.agentic_llm_pretraining.datasets.download_datasets --metadata examples/agentic_llm_pretraining/datasets/inspect_metadata.json
    python -m examples.agentic_llm_pretraining.datasets.download_datasets --metadata examples/agentic_llm_pretraining/datasets/inspect_metadata.json --output ../data/inspect
"""

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, Any

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from mlpug.utils.datasets import download_subsample_hf, stream_subsample_hf, load_with_hf_builder_script


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


ACQUISITION_METHODS: Dict[str, Callable[..., Any]] = {
    "download_subsample_hf": download_subsample_hf,
    "stream_subsample_hf": stream_subsample_hf,
}


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata from JSON file."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Remove comment fields
    return {k: v for k, v in metadata.items() if not k.startswith("_")}


def download_dataset(name: str, config: Dict[str, Any], output_dir: Path) -> None:
    """
    Download a single dataset based on its configuration.

    :param name: Dataset name (used for output filename).
    :param config: Dataset configuration from metadata.
    :param output_dir: Directory to save the output file.
    """
    method_name = config.get("acquisition_method")
    if method_name is None:
        raise ValueError(f"Dataset '{name}' missing 'acquisition_method'")

    acquisition_func = ACQUISITION_METHODS.get(method_name)
    if acquisition_func is None:
        raise ValueError(f"Unknown acquisition method: {method_name}")

    kwargs = config.get("acquisition_config", {}).copy()

    # Check for custom loader (for datasets with builder scripts incompatible with datasets 4.x)
    if config.get("use_custom_loader", False):
        kwargs["load_dataset_func"] = load_with_hf_builder_script

    # Add filter config if present
    filter_func_name = config.get("filter_func")
    filter_config = config.get("filter_config")
    if filter_func_name is not None:
        # Resolve filter_func from string to actual function
        from . import filters
        filter_func = getattr(filters, filter_func_name, None)
        if filter_func is None:
            raise ValueError(f"Unknown filter function: {filter_func_name}")
        kwargs["filter_func"] = filter_func
    if filter_config is not None:
        kwargs["filter_config"] = filter_config

    output_path = output_dir / f"{name}.jsonl"

    # Skip if already downloaded
    if output_path.exists():
        module_logger.info(f"Skipping '{name}': {output_path} already exists")
        return

    module_logger.info(f"Downloading '{name}' to {output_path}")

    stats = acquisition_func(str(output_path), **kwargs)
    module_logger.info(f"Completed '{name}': {stats}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets from metadata JSON")
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata JSON file"
    )
    parser.add_argument(
        "--output",
        default="../data",
        help="Output directory for downloaded datasets (default: ../data)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to download (default: all)"
    )
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_download = args.datasets if args.datasets else list(metadata.keys())

    module_logger.info(f"Downloading {len(datasets_to_download)} datasets to {output_dir}")

    failed = []
    for name in datasets_to_download:
        if name not in metadata:
            module_logger.warning(f"Dataset '{name}' not found in metadata, skipping")
            continue

        try:
            download_dataset(name, metadata[name], output_dir)
        except Exception as e:
            module_logger.error(f"Failed to download '{name}': {e}")
            failed.append(name)

    if failed:
        module_logger.warning(f"Failed datasets ({len(failed)}): {', '.join(failed)}")
    else:
        module_logger.info("All datasets downloaded successfully")


if __name__ == "__main__":
    main()
