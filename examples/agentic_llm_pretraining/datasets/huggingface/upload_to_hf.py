#!/usr/bin/env python3
"""
Upload the agentic LLM pretraining dataset to HuggingFace.

The repository must already exist on HuggingFace. Create it first at:
https://huggingface.co/new-dataset

Usage:
    python -m examples.agentic_llm_pretraining.datasets.huggingface.upload_to_hf \
        --splits-dir /data/agentic_llm_pretraining/full-08012026/splits \
        --repo-id visionscaper/agentic-llm-pretraining-1.7b
"""

import argparse
import os
from pathlib import Path

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from huggingface_hub import HfApi, repo_exists

use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


def describe_config(
    splits_dir: str,
    readme_path: str,
    repo_id: str,
) -> None:
    """Log script configuration."""
    module_logger.info("Configuration:")
    module_logger.info(f"  splits_dir: {splits_dir}")
    module_logger.info(f"  readme_path: {readme_path}")
    module_logger.info(f"  repo_id: {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload agentic LLM pretraining dataset to HuggingFace"
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        required=True,
        help="Directory containing train.jsonl, val.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--readme-path",
        type=str,
        default=None,
        help="Path to README.md (default: README.md in same directory as this script)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="visionscaper/agentic-llm-pretraining-1.7b",
        help="HuggingFace repository ID",
    )
    args = parser.parse_args()

    # Default README path - same directory as this script
    if args.readme_path is None:
        script_dir = Path(__file__).parent
        args.readme_path = str(script_dir / "README.md")

    describe_config(
        splits_dir=args.splits_dir,
        readme_path=args.readme_path,
        repo_id=args.repo_id,
    )

    splits_dir = Path(args.splits_dir)
    readme_path = Path(args.readme_path)

    # Verify files exist
    split_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    for filename in split_files:
        filepath = splits_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Split file not found: {filepath}")
        module_logger.info(f"Found: {filepath} ({filepath.stat().st_size / 1e9:.2f} GB)")

    if not readme_path.exists():
        raise FileNotFoundError(f"README not found: {readme_path}")
    module_logger.info(f"Found: {readme_path}")

    # Initialize HuggingFace API
    hf_api = HfApi()

    # Verify repository exists
    module_logger.info(f"Checking repository: {args.repo_id}")
    if not repo_exists(repo_id=args.repo_id, repo_type="dataset"):
        module_logger.error(f"Repository does not exist: {args.repo_id}")
        module_logger.error(f"Please create it first at: https://huggingface.co/new-dataset")
        raise RuntimeError(f"Repository not found: {args.repo_id}")
    module_logger.info(f"Repository found: https://huggingface.co/datasets/{args.repo_id}")

    # Upload README first
    module_logger.info("Uploading README.md...")
    hf_api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    module_logger.info("README.md uploaded")

    # Upload split files
    for filename in split_files:
        filepath = splits_dir / filename
        module_logger.info(f"Uploading {filename}...")
        hf_api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filename,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        module_logger.info(f"{filename} uploaded")

    module_logger.info("")
    module_logger.info("=" * 60)
    module_logger.info("Upload complete!")
    module_logger.info(f"Dataset URL: https://huggingface.co/datasets/{args.repo_id}")
    module_logger.info("=" * 60)


if __name__ == "__main__":
    main()
