#!/usr/bin/env python3
"""
Upload the agentic LLM pretraining dataset to HuggingFace.

Usage:
    python -m examples.agentic_llm_pretraining.datasets.huggingface.upload_to_hf \
        --splits-dir /data/agentic_llm_pretraining/full-08012026/splits \
        --repo-id visionscaper/agentic-llm-pretraining-1.7b \
        --private
"""

import argparse
import os
from pathlib import Path

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from huggingface_hub import HfApi, create_repo

use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


def describe_config(
    splits_dir: str,
    readme_path: str,
    repo_id: str,
    private: bool,
) -> None:
    """Log script configuration."""
    module_logger.info("Configuration:")
    module_logger.info(f"  splits_dir: {splits_dir}")
    module_logger.info(f"  readme_path: {readme_path}")
    module_logger.info(f"  repo_id: {repo_id}")
    module_logger.info(f"  private: {private}")


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
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create as private repository",
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
        private=args.private,
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

    # Create repository
    module_logger.info(f"Creating repository: {args.repo_id}")
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        module_logger.info(f"Repository ready: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        module_logger.error(f"Failed to create repository: {e}")
        raise

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
