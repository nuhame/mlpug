#!/usr/bin/env python3
"""
V2: Transform datasets from raw JSONL to structured parts format.

Output format: {"parts": [{"type": "...", "text": "..."}, ...]}

This format enables loss masking during tokenization - prompts/system messages
can be masked out so the model only trains on responses.

Usage (from repo root):
    python -m examples.agentic_llm_pretraining.datasets.v2.transform_datasets \
        --metadata examples/agentic_llm_pretraining/datasets/v1/training_metadata.json

    python -m examples.agentic_llm_pretraining.datasets.v2.transform_datasets \
        --metadata examples/agentic_llm_pretraining/datasets/v1/training_metadata.json \
        --datasets gsm8k soda \
        --num-samples 100 \
        --output-dir ../data/transforms-v2
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
    describe_config as describe_config_base,
    describe_dataset_config,
)
from examples.agentic_llm_pretraining.datasets import preprocessing as v1_preprocessing

from examples.agentic_llm_pretraining.datasets.v2 import preprocessing as v2_preprocessing
from examples.agentic_llm_pretraining.datasets.v2.parts_templates import (
    TEMPLATES,
    TemplateBase,
    TextTemplate,
    SplitTemplate,
    DialogueTemplate,
)
from examples.agentic_llm_pretraining.datasets.v2.transform_functions import (
    Part,
    transform_samples_to_parts,
)


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))

DEFAULT_MAX_MASKED_CHARS = 8000


def describe_config(
    max_masked_chars: int,
    **kwargs,
) -> None:
    """
    Log v2-specific configuration, then delegate to base describe_config.

    :param max_masked_chars: Max chars for any single masked part. 0 disables filtering.
    :param kwargs: Passed through to base describe_config.
    """
    describe_config_base(**kwargs)
    if max_masked_chars > 0:
        module_logger.info(f"  max_masked_chars: {max_masked_chars}")
    else:
        module_logger.info(f"  max_masked_chars: disabled")


def get_preprocess_func(
    dataset_name: str,
    config: dict[str, Any],
    template: TemplateBase,
    logger: logging.Logger,
):
    """
    Get the appropriate preprocess function for a dataset.

    For TextTemplate/SplitTemplate: Uses v1 preprocess functions that return dict.
    For DialogueTemplate: Uses v2 preprocess functions that return DialogueData.

    :param dataset_name: Name of the dataset.
    :param config: Dataset configuration from metadata.
    :param template: Template instance for this dataset.
    :param logger: Logger for messages.

    :return: Preprocess function.

    :raises ValueError: If required preprocess function is not found.
    """
    if isinstance(template, DialogueTemplate):
        # Dialogue datasets use v2 preprocessing functions
        preprocess_fn = v2_preprocessing.DIALOGUE_PREPROCESS_FUNCS.get(dataset_name)
        if preprocess_fn is None:
            raise ValueError(
                f"{dataset_name}: No v2 dialogue preprocess function found. "
                f"Expected entry in DIALOGUE_PREPROCESS_FUNCS."
            )
        logger.info(f"{dataset_name}: using v2 dialogue preprocess function")
        return preprocess_fn

    # TextTemplate and SplitTemplate use v1 preprocessing functions
    func_name = config.get("preprocess_func")
    if func_name:
        preprocess_fn = getattr(v1_preprocessing, func_name, None)
        if preprocess_fn is None:
            raise ValueError(
                f"{dataset_name}: preprocess_func '{func_name}' "
                f"not found in v1 preprocessing module"
            )
        logger.info(f"{dataset_name}: using v1 preprocess_func '{func_name}'")
        return preprocess_fn

    # Try convention: preprocess_{name}
    default_name = f"preprocess_{dataset_name.replace('-', '_')}"
    preprocess_fn = getattr(v1_preprocessing, default_name, None)
    if preprocess_fn:
        logger.info(f"{dataset_name}: using default v1 func '{default_name}'")
        return preprocess_fn

    raise ValueError(
        f"{dataset_name}: No preprocess function found. "
        f"Expected 'preprocess_func' in config or '{default_name}' in preprocessing module."
    )


def transform_dataset(
    dataset_name: str,
    config: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    num_samples: Optional[int] = None,
    max_masked_chars: int = 0,
    logger: Optional[logging.Logger] = None,
) -> TransformStats:
    """
    Transform a single dataset and write to output file in parts format.

    :param dataset_name: Name of the dataset.
    :param config: Dataset configuration from metadata.
    :param data_dir: Directory containing raw JSONL files.
    :param output_dir: Directory to write transformed output.
    :param num_samples: Optional limit on samples to process.
    :param max_masked_chars: Skip samples where any single masked part
        exceeds this character count. 0 disables filtering.
    :param logger: Optional logger.

    :return: TransformStats with success/failure counts.
    """
    if logger is None:
        logger = module_logger

    # Get template for this dataset
    template = TEMPLATES.get(dataset_name)
    if template is None:
        raise ValueError(
            f"{dataset_name}: No template found in TEMPLATES registry. "
            f"Add an entry to parts_templates.py."
        )

    describe_dataset_config(dataset_name, config, template)

    # Load raw samples
    samples = load_raw_samples(data_dir, dataset_name, num_samples, logger)

    # Get the appropriate preprocess function
    preprocess_fn = get_preprocess_func(dataset_name, config, template, logger)

    # Transform samples to parts
    parts_list, stats = transform_samples_to_parts(
        samples=samples,
        dataset_name=dataset_name,
        preprocess_fn=preprocess_fn,
        template=template,
        max_masked_chars=max_masked_chars,
        logger=logger,
    )

    # Write output in parts format
    output_path = output_dir / f"{dataset_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, parts in enumerate(parts_list):
            record = {
                "source": dataset_name,
                "index": idx,
                "parts": [p.to_dict() for p in parts],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    filtered_info = f", {stats.filtered} filtered" if stats.filtered > 0 else ""
    logger.info(
        f"{dataset_name}: wrote {stats.success} samples to {output_path} "
        f"({stats.success}/{stats.total} success, {stats.failed} failed{filtered_info})"
    )

    return stats


def main():
    log_git_state()

    parser = argparse.ArgumentParser(
        description="Transform datasets from raw JSONL to parts format (v2: for loss masking)"
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata JSON file (uses v1 metadata, templates come from v2)",
    )
    parser.add_argument(
        "--data-dir",
        default="../data/agentic_llm_pretraining",
        help="Directory containing raw JSONL files (default: ../data/agentic_llm_pretraining)",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/transforms-v2",
        help="Directory to write transformed output (default: ../data/transforms-v2)",
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
    parser.add_argument(
        "--max-masked-chars",
        type=int,
        default=DEFAULT_MAX_MASKED_CHARS,
        help=(
            f"Skip samples where any single masked part exceeds this character count "
            f"(default: {DEFAULT_MAX_MASKED_CHARS}). Set to 0 to disable."
        ),
    )
    args = parser.parse_args()

    describe_config(
        max_masked_chars=args.max_masked_chars,
        metadata=args.metadata,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        datasets=args.datasets,
        num_samples=args.num_samples,
    )

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

        try:
            stats = transform_dataset(
                dataset_name=name,
                config=dataset_metadata,
                data_dir=data_dir,
                output_dir=output_dir,
                num_samples=args.num_samples,
                max_masked_chars=args.max_masked_chars,
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
        module_logger.warning(
            f"Failed datasets ({len(failed_datasets)}): {', '.join(failed_datasets)}"
        )


if __name__ == "__main__":
    main()
