#!/usr/bin/env python3
"""
V1: Transform datasets from raw JSONL to training-ready format.

Output format: {"source": "...", "index": N, "text": "..."}

Usage (from repo root):
    python -m examples.agentic_llm_pretraining.datasets.v1.transform_datasets \
        --metadata examples/agentic_llm_pretraining/datasets/v1/inspect_metadata.json

    python -m examples.agentic_llm_pretraining.datasets.v1.transform_datasets \
        --metadata examples/agentic_llm_pretraining/datasets/v1/inspect_metadata.json \
        --datasets gsm8k soda \
        --num-samples 100 \
        --output-dir ../data/transforms
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from mlpug.utils.git_logging import log_git_state

from examples.agentic_llm_pretraining.datasets.common import (
    TransformStats,
    load_metadata,
    load_raw_samples,
    print_summary,
    describe_config,
    describe_dataset_config,
)
from examples.agentic_llm_pretraining.datasets import preprocessing

from examples.agentic_llm_pretraining.datasets.v1.dataset_templates import TEMPLATES
from examples.agentic_llm_pretraining.datasets.v1.transform_functions import transform


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


# Default template for datasets without explicit config
DEFAULT_TEMPLATE = "{text}"


def transform_dataset(
    dataset_name: str,
    config: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    num_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> TransformStats:
    """
    Transform a single dataset and write to output file.

    :param dataset_name: Name of the dataset.
    :param config: Dataset configuration from metadata.
    :param data_dir: Directory containing raw JSONL files.
    :param output_dir: Directory to write transformed output.
    :param num_samples: Optional limit on samples to process.
    :param logger: Optional logger.

    :return: TransformStats with success/failure counts.
    """
    if logger is None:
        logger = module_logger

    # Load raw samples
    samples = load_raw_samples(data_dir, dataset_name, num_samples, logger)

    # Resolve preprocess function from preprocessing module
    preprocess_func_name = config.get("preprocess_func")
    if preprocess_func_name:
        preprocess_fn = getattr(preprocessing, preprocess_func_name, None)
        if preprocess_fn is None:
            raise ValueError(
                f"{dataset_name}: preprocess_func '{preprocess_func_name}' not found in preprocessing module"
            )
        logger.info(f"{dataset_name}: using preprocess_func '{preprocess_func_name}'")
    else:
        # Try lookup by dataset name convention: preprocess_{dataset_name with - replaced by _}
        default_func_name = f"preprocess_{dataset_name.replace('-', '_')}"
        preprocess_fn = getattr(preprocessing, default_func_name, None)
        if preprocess_fn:
            logger.info(
                f"{dataset_name}: no preprocess_func in metadata, "
                f"using default '{default_func_name}'"
            )
        else:
            logger.info(
                f"{dataset_name}: no preprocess_func found, using default passthrough"
            )
            preprocess_fn = None

    # Resolve template
    template = TEMPLATES.get(dataset_name, DEFAULT_TEMPLATE)

    # Transform samples
    transformed_texts, stats = transform(
        samples=samples,
        dataset_name=dataset_name,
        preprocess_fn=preprocess_fn,
        template=template,
        logger=logger,
    )

    # Write output with source and index for traceability
    output_path = output_dir / f"{dataset_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, text in enumerate(transformed_texts):
            record = {
                "source": dataset_name,
                "index": idx,
                "text": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        f"{dataset_name}: wrote {stats.success} samples to {output_path} "
        f"({stats.success}/{stats.total} success, {stats.failed} failed)"
    )

    return stats


def main():
    log_git_state()

    parser = argparse.ArgumentParser(
        description="Transform datasets from raw JSONL to training format (v1: text output)"
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata JSON file",
    )
    parser.add_argument(
        "--data-dir",
        default="../data/agentic_llm_pretraining",
        help="Directory containing raw JSONL files (default: ../data/agentic_llm_pretraining)",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/transforms",
        help="Directory to write transformed output (default: ../data/transforms)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to transform (default: all in metadata)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit number of samples per dataset (default: all)",
    )
    args = parser.parse_args()

    config = vars(args)
    describe_config(**config)

    metadata = load_metadata(args.metadata)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_transform = args.datasets if args.datasets else list(metadata.keys())

    module_logger.info(f"Transforming {len(datasets_to_transform)} datasets")

    results: dict[str, TransformStats] = {}
    failed_datasets: list[str] = []

    for name in datasets_to_transform:
        if name not in metadata:
            module_logger.warning(f"Dataset '{name}' not found in metadata, skipping")
            continue

        dataset_metadata = metadata[name]
        describe_dataset_config(name, dataset_metadata)

        try:
            stats = transform_dataset(
                dataset_name=name,
                config=dataset_metadata,
                data_dir=data_dir,
                output_dir=output_dir,
                num_samples=args.num_samples,
                logger=module_logger,
            )
            results[name] = stats
        except FileNotFoundError as e:
            module_logger.warning(f"{name}: {e}, skipping")
            failed_datasets.append(name)
        except Exception as e:
            module_logger.error(f"{name}: transform failed with {type(e).__name__}: {e}")
            failed_datasets.append(name)

    print_summary(results)

    if failed_datasets:
        module_logger.warning(f"Failed datasets ({len(failed_datasets)}): {', '.join(failed_datasets)}")


if __name__ == "__main__":
    main()
