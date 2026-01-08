#!/usr/bin/env python3
"""
Analyze tokens per sample from transformed inspection data.

Uses Qwen3 tokenizer to get accurate token counts and calculates
sample allocations for training metadata.

Usage (from repo root):
    python -m examples.agentic_llm_pretraining.datasets.analyze_tokens \
        --transforms-dir ../data/agentic_llm_pretraining/inspection/transforms
"""

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from mlpug.utils.git_logging import log_git_state


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


# Known dataset sizes (verified from HuggingFace dataset pages)
# Cross-checked against acquisition_config.path in inspect_metadata.json
#
# Sources verified:
# - HuggingFaceFW/fineweb-edu (sample-10BT): 97.3M rows
# - wikimedia/wikipedia (20231101.en): 6.41M rows
# - nvidia/OpenMathInstruct-1: 4.95M rows (before is_correct filter)
# - nvidia/OpenMathInstruct-2: 14M rows
# - openai/gsm8k: 7,473 rows
# - qwedsacf/competition_math: 12,500 rows
# - HuggingFaceTB/cosmopedia (wikihow): 179K rows
# - HuggingFaceH4/stack-exchange-preferences: 10.8M rows
# - princeton-nlp/SWE-bench: 19K rows
# - deepmind/code_contests: 13,328 rows
# - Team-ACE/ToolACE: 11,300 rows
# - NousResearch/hermes-function-calling-v1: 11,578 rows
# - glaiveai/glaive-function-calling-v2: 112,960 rows
# - tasksource/ecqa: 7,600 rows
# - allenai/openbookqa: 4,957 rows
# - Salesforce/dialogstudio (from Dataset_Stats.csv):
#   - SODA: 1,191,582 rows
#   - MULTIWOZ2_2: 8,437 rows
#   - wizard_of_wikipedia: 18,430 rows
#   - ShareGPT: 96,394 rows
#   - Empathetic: 17,802 rows
#   - SAMSum: 14,732 rows
# - neural-bridge/rag-dataset-12000: 9,600 train rows
# - rungalileo/ragbench (hotpotqa): 2,700 rows
#
# Additional verified:
# - wikimedia/wikipedia (20231101.simple): 242K
# - community-datasets/generics_kb (generics_kb): 3.43M
#
# Estimated (HF page lacks split breakdown):
# - code_search_net (python): ~412K (2M total across 6 languages)
DATASET_SIZES = {
    "fineweb-edu": 97_300_000,  # HuggingFaceFW/fineweb-edu sample-10BT
    # fineweb-edu-long: filtered subset with >= 120K chars (~32K tokens)
    # Verified from inspection download: 500/895,253 pass = 0.056% pass rate
    # Full dataset: 97.3M * 0.056% ≈ 54,000 samples
    "fineweb-edu-long": 54_000,  # HuggingFaceFW/fineweb-edu sample-10BT (filtered)
    "wikipedia": 6_410_000,  # wikimedia/wikipedia 20231101.en
    "simple-wikipedia": 242_000,  # wikimedia/wikipedia 20231101.simple
    "cosmopedia-wikihow": 179_000,  # HuggingFaceTB/cosmopedia wikihow
    "generics-kb": 3_430_000,  # community-datasets/generics_kb generics_kb
    "codesearchnet": 412_178,  # code_search_net python (estimate)
    "gsm8k": 7_473,  # openai/gsm8k
    "math": 12_500,  # qwedsacf/competition_math
    "openmath-instruct-1": 4_950_000,  # nvidia/OpenMathInstruct-1 (before filter)
    "openmath-instruct-2": 14_000_000,  # nvidia/OpenMathInstruct-2
    "ecqa": 7_600,  # tasksource/ecqa
    "stackexchange": 10_807_695,  # HuggingFaceH4/stack-exchange-preferences
    "swe-bench": 19_000,  # princeton-nlp/SWE-bench
    "code-contests": 13_328,  # deepmind/code_contests
    "toolace": 11_300,  # Team-ACE/ToolACE
    "hermes-function-calling": 11_578,  # NousResearch/hermes-function-calling-v1
    "glaive-function-calling": 112_960,  # glaiveai/glaive-function-calling-v2
    "openbookqa": 4_957,  # allenai/openbookqa
    "soda": 1_191_582,  # Salesforce/dialogstudio SODA
    "multiwoz": 8_437,  # Salesforce/dialogstudio MULTIWOZ2_2
    "wizard-of-wikipedia": 18_430,  # Salesforce/dialogstudio wizard_of_wikipedia
    "sharegpt": 96_394,  # Salesforce/dialogstudio ShareGPT
    "empathetic-dialogues": 17_802,  # Salesforce/dialogstudio Empathetic
    "samsum": 14_732,  # Salesforce/dialogstudio SAMSum
    "rag-dataset-12000": 9_600,  # neural-bridge/rag-dataset-12000 train split
    "ragbench-hotpotqa": 2_700,  # rungalileo/ragbench hotpotqa
}

# Generic datasets (60% of corpus by default)
GENERIC_DATASETS = [
    "fineweb-edu", "fineweb-edu-long", "wikipedia", "simple-wikipedia",
    "cosmopedia-wikihow", "generics-kb", "codesearchnet"
]


def describe_config(
    transforms_dir: str,
    output: str | None,
    generic_fraction: float,
) -> None:
    """Log script configuration."""
    module_logger.info("Configuration:")
    module_logger.info(f"  transforms_dir: {transforms_dir}")
    module_logger.info(f"  output: {output}")
    module_logger.info(f"  generic_fraction: {generic_fraction}")


def analyze_dataset(
    jsonl_path: Path,
    tokenizer,
    logger=None,
) -> dict:
    """
    Analyze a single dataset's token statistics.

    :param jsonl_path: Path to transformed JSONL file.
    :param tokenizer: Tokenizer instance.
    :param logger: Optional logger.

    :return: Dict with token statistics.
    """
    if logger is None:
        logger = module_logger

    token_counts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            text = sample.get("text", "")
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_counts.append(len(tokens))

    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    return {
        "num_samples": len(token_counts),
        "avg_tokens": avg_tokens,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
    }


def analyze_all_datasets(
    transforms_dir: Path,
    tokenizer,
    logger=None,
) -> dict[str, dict]:
    """
    Analyze all datasets in transforms directory.

    :param transforms_dir: Directory containing transformed JSONL files.
    :param tokenizer: Tokenizer instance.
    :param logger: Optional logger.

    :return: Dict mapping dataset name to statistics.
    """
    if logger is None:
        logger = module_logger

    results = {}
    for jsonl_path in sorted(transforms_dir.glob("*.jsonl")):
        name = jsonl_path.stem
        stats = analyze_dataset(jsonl_path, tokenizer, logger)
        stats["dataset_size"] = DATASET_SIZES.get(name, "unknown")
        results[name] = stats
        logger.info(f"{name}: avg={stats['avg_tokens']:.0f} tokens/sample")

    return results


def allocate_category(
    dataset_names: list[str],
    results: dict[str, dict],
    token_budget: int,
    category_name: str,
    logger=None,
) -> dict[str, int]:
    """
    Allocate tokens within a category of datasets.

    Each dataset gets an equal share of the budget. Small datasets that can't
    fill their quota have their deficit reallocated to larger datasets.
    Reallocation is done iteratively until all "large" datasets can meet
    their target.

    :param dataset_names: Names of datasets in this category.
    :param results: Token statistics per dataset.
    :param token_budget: Total token budget for this category.
    :param category_name: Name of category for logging.
    :param logger: Optional logger.

    :return: Dict mapping dataset name to target sample count.

    :raises ValueError: If any dataset has unknown size.
    """
    if logger is None:
        logger = module_logger

    # Validate all datasets have known sizes and build dataset info
    dataset_info = {}  # name -> (size, available_tokens, avg_tokens)
    for name in dataset_names:
        if name not in results:
            raise ValueError(f"Dataset '{name}' not found in results")
        size = results[name]["dataset_size"]
        if not isinstance(size, int):
            raise ValueError(
                f"Dataset '{name}' has unknown size: {size}. "
                "Please add verified size to DATASET_SIZES."
            )
        avg_tokens = results[name]["avg_tokens"]
        available_tokens = size * avg_tokens
        dataset_info[name] = (size, available_tokens, avg_tokens)

    num_datasets = len(dataset_names)
    if num_datasets == 0:
        return {}

    initial_per_dataset = token_budget // num_datasets

    logger.info("")
    logger.info(f"## {category_name} ({num_datasets} datasets)")
    logger.info(f"  Budget: {token_budget/1e6:.1f}M tokens")
    logger.info(f"  Initial target per dataset: {initial_per_dataset/1e6:.1f}M tokens")

    # Iteratively identify small vs large datasets
    # After reallocation, some "large" datasets may become too small for the
    # new target, so we need to repeat until convergence
    small_datasets = set()  # names of datasets that use all samples
    remaining_datasets = set(dataset_names)  # names still to be allocated

    iteration = 0
    max_iterations = len(dataset_names)  # Safety limit

    while iteration < max_iterations:
        iteration += 1

        # Calculate current target for remaining (large) datasets
        small_total_tokens = sum(
            dataset_info[name][1] for name in small_datasets
        )
        remaining_budget = token_budget - small_total_tokens
        remaining_count = len(remaining_datasets)

        if remaining_count == 0:
            break

        target_per_remaining = remaining_budget / remaining_count

        # Find datasets that can't meet the target
        newly_small = set()
        for name in remaining_datasets:
            size, available_tokens, avg_tokens = dataset_info[name]
            if available_tokens < target_per_remaining:
                newly_small.add(name)
                logger.info(
                    f"  {name}: {available_tokens/1e6:.1f}M available < "
                    f"{target_per_remaining/1e6:.1f}M target [SMALL]"
                )

        if not newly_small:
            # All remaining datasets can meet the target - we're done
            break

        # Move newly small datasets
        small_datasets.update(newly_small)
        remaining_datasets -= newly_small

    # Calculate final target for large datasets
    small_total_tokens = sum(dataset_info[name][1] for name in small_datasets)
    remaining_budget = token_budget - small_total_tokens
    remaining_count = len(remaining_datasets)
    tokens_per_large = remaining_budget / remaining_count if remaining_count > 0 else 0

    logger.info("")
    logger.info(f"  Small datasets ({len(small_datasets)}): {small_total_tokens/1e6:.1f}M tokens")
    logger.info(f"  Remaining for {remaining_count} large: {remaining_budget/1e6:.1f}M tokens")
    logger.info(f"  Final target per large: {tokens_per_large/1e6:.1f}M tokens")

    # Build allocations
    allocations = {}

    for name in sorted(small_datasets):
        size, available, avg_tok = dataset_info[name]
        allocations[name] = size  # Use all
        logger.info(f"  {name}: {size:,} samples (all)")

    for name in sorted(remaining_datasets):
        size, available, avg_tok = dataset_info[name]
        target_samples = int(tokens_per_large / avg_tok)
        allocations[name] = target_samples
        logger.info(f"  {name}: {target_samples:,} of {size:,} samples")

    return allocations


def calculate_allocations(
    results: dict[str, dict],
    total_tokens: int = 1_700_000_000,
    generic_fraction: float = 0.5,
    logger=None,
) -> dict[str, int]:
    """
    Calculate sample allocations based on token budget.

    Applies per-dataset budgets to both generic and structured datasets,
    with reallocation from small datasets to large datasets within each category.

    :param results: Token statistics per dataset.
    :param total_tokens: Total token budget (default 1.7B).
    :param generic_fraction: Fraction for generic datasets (default 0.5).
    :param logger: Optional logger.

    :return: Dict mapping dataset name to target sample count.

    :raises ValueError: If any dataset has unknown size.
    """
    if logger is None:
        logger = module_logger

    generic_budget = int(total_tokens * generic_fraction)
    structured_budget = total_tokens - generic_budget

    logger.info("")
    logger.info("=" * 70)
    logger.info("TOKEN ALLOCATION ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Total budget: {total_tokens/1e9:.2f}B tokens")
    logger.info(f"Generic datasets ({generic_fraction*100:.0f}%): {generic_budget/1e6:.0f}M tokens")
    logger.info(f"Structured datasets ({(1-generic_fraction)*100:.0f}%): {structured_budget/1e6:.0f}M tokens")

    # Allocate generic datasets
    generic_names = [n for n in GENERIC_DATASETS if n in results]
    generic_allocations = allocate_category(
        generic_names, results, generic_budget, "GENERIC DATASETS", logger
    )

    # Allocate structured datasets
    structured_names = [n for n in results if n not in GENERIC_DATASETS]
    structured_allocations = allocate_category(
        structured_names, results, structured_budget, "STRUCTURED DATASETS", logger
    )

    # Combine allocations
    allocations = {}
    allocations.update(generic_allocations)
    allocations.update(structured_allocations)

    return allocations


def print_final_summary(
    allocations: dict[str, int],
    results: dict[str, dict],
    logger=None,
) -> None:
    """Print final allocation summary."""
    if logger is None:
        logger = module_logger

    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SAMPLE ALLOCATIONS")
    logger.info("=" * 70)

    total_tokens = 0
    logger.info("")
    logger.info("Generic datasets:")
    for name in GENERIC_DATASETS:
        if name in allocations:
            count = allocations[name]
            avg = results[name]["avg_tokens"]
            tokens = count * avg
            total_tokens += tokens
            logger.info(f"  {name}: {count:,} samples ({tokens/1e6:.1f}M tokens)")

    logger.info("")
    logger.info("Structured datasets:")
    for name in sorted(allocations.keys()):
        if name not in GENERIC_DATASETS:
            count = allocations[name]
            avg = results[name]["avg_tokens"]
            tokens = count * avg
            total_tokens += tokens
            logger.info(f"  {name}: {count:,} samples ({tokens/1e6:.1f}M tokens)")

    logger.info("")
    logger.info(f"TOTAL: {total_tokens/1e9:.2f}B tokens")


def main():
    log_git_state()

    parser = argparse.ArgumentParser(
        description="Analyze tokens per sample from transformed inspection data"
    )
    parser.add_argument(
        "--transforms-dir",
        default="../data/agentic_llm_pretraining/inspection/transforms",
        help="Directory containing transformed JSONL files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save token analysis JSON (default: transforms_dir/../token_analysis.json)",
    )
    parser.add_argument(
        "--generic-fraction",
        type=float,
        default=0.6,
        help="Fraction of token budget for generic datasets (default: 0.6)",
    )
    args = parser.parse_args()

    config = vars(args)
    describe_config(**config)

    transforms_dir = Path(args.transforms_dir)
    if not transforms_dir.exists():
        module_logger.error(f"Transforms directory not found: {transforms_dir}")
        return

    module_logger.info("")
    module_logger.info("Loading Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    module_logger.info("")
    module_logger.info("Analyzing transformed datasets...")
    results = analyze_all_datasets(transforms_dir, tokenizer, module_logger)

    allocations = calculate_allocations(
        results,
        generic_fraction=args.generic_fraction,
        logger=module_logger,
    )
    print_final_summary(allocations, results, module_logger)

    # Save results
    output_path = Path(args.output) if args.output else transforms_dir.parent / "token_analysis.json"
    output_data = {
        "token_stats": results,
        "allocations": allocations,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    module_logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
