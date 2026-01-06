#!/usr/bin/env python3
"""
Transform datasets from raw JSONL to training-ready format.

Usage (from repo root):
    python -m examples.agentic_llm_pretraining.datasets.transform_datasets \
        --metadata examples/agentic_llm_pretraining/datasets/inspect_metadata.json

    python -m examples.agentic_llm_pretraining.datasets.transform_datasets \
        --metadata examples/agentic_llm_pretraining/datasets/inspect_metadata.json \
        --datasets gsm8k soda \
        --num-samples 100 \
        --output-dir ../data/transforms
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

from mlpug.mlpug_logging import get_logger, use_fancy_colors

from .transform_functions import transform, TransformStats
from .dataset_templates import PREPROCESS_FUNCTIONS, TEMPLATES


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


# Default template for datasets without explicit config
DEFAULT_TEMPLATE = "{text}"


def load_metadata(metadata_path: str) -> dict[str, Any]:
    """Load metadata from JSON file, excluding comment fields."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return {k: v for k, v in metadata.items() if not k.startswith("_")}


def load_raw_samples(
    data_dir: Path,
    dataset_name: str,
    num_samples: Optional[int] = None,
    logger=None,
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


def transform_dataset(
    dataset_name: str,
    config: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    num_samples: Optional[int] = None,
    logger=None,
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

    # Resolve transform function
    transform_func_name = config.get("transform_func")
    if transform_func_name:
        preprocess_fn = PREPROCESS_FUNCTIONS.get(transform_func_name)
        if preprocess_fn is None:
            raise ValueError(
                f"{dataset_name}: transform_func '{transform_func_name}' not found in registry"
            )
        logger.info(f"{dataset_name}: using transform_func '{transform_func_name}'")
    else:
        # Try lookup by dataset name
        preprocess_fn = PREPROCESS_FUNCTIONS.get(dataset_name)
        if preprocess_fn:
            logger.info(
                f"{dataset_name}: no transform_func in metadata, "
                f"using registry lookup by name"
            )
        else:
            logger.info(
                f"{dataset_name}: no transform_func found, using default passthrough"
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


def print_summary(results: dict[str, TransformStats], logger=None):
    """Print summary of all transform operations."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Transform datasets from raw JSONL to training format"
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

    metadata = load_metadata(args.metadata)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_transform = args.datasets if args.datasets else list(metadata.keys())

    module_logger.info(f"Transforming {len(datasets_to_transform)} datasets")
    module_logger.info(f"  Data dir: {data_dir}")
    module_logger.info(f"  Output dir: {output_dir}")
    if args.num_samples:
        module_logger.info(f"  Samples per dataset: {args.num_samples}")

    results: dict[str, TransformStats] = {}
    failed_datasets: list[str] = []

    for name in datasets_to_transform:
        if name not in metadata:
            module_logger.warning(f"Dataset '{name}' not found in metadata, skipping")
            continue

        try:
            stats = transform_dataset(
                dataset_name=name,
                config=metadata[name],
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
