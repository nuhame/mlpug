"""
Core evaluation functions using BFCL (Berkeley Function Calling Leaderboard).

This module provides functions for evaluating language model checkpoints
on function/tool calling benchmarks using the BFCL framework.

Requirements:
    pip install bfcl-eval[oss_eval_vllm]  # For vLLM backend
    # or
    pip install bfcl-eval[oss_eval_sglang]  # For SGLang backend

Usage (standalone):
    from examples.agentic_llm_pretraining.evaluation.bfcl.benchmarks import (
        evaluate_checkpoint,
        DEFAULT_TEST_CATEGORIES,
    )

    results = evaluate_checkpoint(
        checkpoint_path="/path/to/checkpoint.pt",
        model_name="Qwen/Qwen3-1.7B-Base",
        test_categories=DEFAULT_TEST_CATEGORIES,
    )

Note:
    BFCL uses vLLM or SGLang as the inference backend. The model must be
    converted to HuggingFace format before evaluation.
"""
from typing import Optional

import csv
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from basics.logging import get_logger

from examples.agentic_llm_pretraining.evaluation.checkpoint import (
    convert_checkpoint_to_hf,
)

module_logger = get_logger(os.path.basename(__file__))


# Basic function calling test categories (recommended for initial testing)
BASIC_TEST_CATEGORIES = [
    "simple_python",      # Basic function calls (Python)
    "simple_java",        # Basic function calls (Java)
    "simple_javascript",  # Basic function calls (JavaScript)
]

# Default test categories for comprehensive function calling evaluation
DEFAULT_TEST_CATEGORIES = BASIC_TEST_CATEGORIES + [
    "parallel",   # Concurrent function execution
    "multiple",   # Sequential function calls
]

# All scoring categories (comprehensive evaluation)
ALL_SCORING_CATEGORIES = "all_scoring"

# Mapping from BFCL test category names to CSV column names in data_non_live.csv
# BFCL v4 outputs results to CSV files instead of JSON
CATEGORY_TO_CSV_COLUMN = {
    "simple_python": "Python Simple AST",
    "simple_java": "Java Simple AST",
    "simple_javascript": "JavaScript Simple AST",
    "parallel": "Parallel AST",
    "multiple": "Multiple AST",
    "parallel_multiple": "Parallel Multiple AST",
}

# CSV file containing non-live benchmark results (simple_*, parallel, multiple)
BFCL_NON_LIVE_CSV = "data_non_live.csv"


def bfcl_generate(
    model_path: str,
    test_category: str | list[str],
    templates_model_name: str = "Qwen/Qwen3-1.7B",
    backend: str = "vllm",
    num_gpus: int = 1,
    gpu_memory_utilization: float = 0.9,
    output_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Generate responses for BFCL evaluation using a local model.

    This runs `bfcl generate` to create model responses for evaluation.

    :param model_path: Path to HuggingFace model directory containing weights.
    :param test_category: Test category or list of categories to evaluate.
        Use "all_scoring" for all scoring categories.
    :param templates_model_name: Model name whose prompt templates to use.
        Must be recognized by BFCL (e.g., "Qwen/Qwen3-1.7B").
        The actual weights come from model_path via --local-model-path.
    :param backend: Inference backend ("vllm" or "sglang").
    :param num_gpus: Number of GPUs to use.
    :param gpu_memory_utilization: GPU memory utilization (0.0-1.0).
    :param output_dir: Directory for BFCL output (default: temp directory).
    :param logger: Optional logger for status messages.

    :return: Path to the output directory containing results.
    """
    if logger is None:
        logger = module_logger

    # Handle test_category as string or list
    if isinstance(test_category, list):
        categories = test_category
    else:
        categories = [test_category]

    # Set up output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="bfcl_output_")

    logger.info(f"Running BFCL generate")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Templates model: {templates_model_name}")
    logger.info(f"Test categories: {categories}")
    logger.info(f"Backend: {backend}, GPUs: {num_gpus}")

    # Build command for each category
    for category in categories:
        cmd = [
            "bfcl", "generate",
            "--model", templates_model_name,  # Model whose templates to use
            "--test-category", category,
            "--backend", backend,
            "--num-gpus", str(num_gpus),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--local-model-path", model_path,  # Actual path to model weights
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        # Set BFCL_PROJECT_ROOT to our output directory
        env = os.environ.copy()
        env["BFCL_PROJECT_ROOT"] = output_dir

        try:
            subprocess.run(
                cmd,
                check=True,
                env=env,
            )
            logger.info(f"Generate completed for category: {category}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Generate failed for category {category}: {e}")
            raise

    return output_dir


def bfcl_evaluate(
    test_category: str | list[str],
    output_dir: str,
    templates_model_name: str = "Qwen/Qwen3-1.7B",
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Evaluate generated BFCL responses.

    This runs `bfcl evaluate` on previously generated responses.

    :param test_category: Test category or list of categories to evaluate.
    :param output_dir: Directory containing BFCL output from bfcl_generate().
    :param templates_model_name: Model name (must match bfcl_generate).
    :param logger: Optional logger for status messages.

    :return: Dictionary mapping category names to accuracy percentages.
    """
    if logger is None:
        logger = module_logger

    # Handle test_category as string or list
    if isinstance(test_category, list):
        categories = test_category
    else:
        categories = [test_category]

    logger.info(f"Running BFCL evaluate")
    logger.info(f"Templates model: {templates_model_name}")
    logger.info(f"Test categories: {categories}")

    # Set BFCL_PROJECT_ROOT to our output directory
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = output_dir

    # Run evaluate for each category
    for category in categories:
        cmd = [
            "bfcl", "evaluate",
            "--model", templates_model_name,  # Must match the model used in generate
            "--test-category", category,
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(
                cmd,
                check=True,
                env=env,
            )
            logger.info(f"Evaluate completed for category: {category}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluate failed for category {category}: {e}")
            raise

    # Parse results from CSV (BFCL v4 outputs CSV, not JSON)
    results = _parse_bfcl_csv_results(output_dir, categories, templates_model_name, logger)

    return results


def _parse_bfcl_csv_results(
    output_dir: str,
    categories: list[str],
    templates_model_name: str,
    logger: logging.Logger,
) -> dict:
    """
    Parse BFCL evaluation results from CSV files.

    BFCL v4 outputs results to CSV files in the score/ directory.
    This function reads data_non_live.csv and extracts accuracy for each category.

    :param output_dir: BFCL output directory containing score/ subdirectory.
    :param categories: List of category names to extract.
    :param templates_model_name: Model name used in BFCL (for row matching).
    :param logger: Logger for status messages.

    :return: Dictionary mapping category names to accuracy percentages.
    """
    results = {}

    # Read data_non_live.csv for simple_*, parallel, multiple categories
    csv_path = Path(output_dir) / "score" / BFCL_NON_LIVE_CSV

    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return results

    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            logger.warning(f"CSV file is empty: {csv_path}")
            return results

        # Find the row for our model
        # BFCL uses various model name formats in CSV:
        # - "Qwen/Qwen3-1.7B" (original)
        # - "Qwen_Qwen3-1.7B" (slash replaced with underscore)
        # - "Qwen3-1.7B (Prompt)" (model name without org, with suffix)
        model_name_variants = [
            templates_model_name,
            templates_model_name.replace("/", "_"),
        ]
        # Add variant without org prefix + " (Prompt)" suffix
        if "/" in templates_model_name:
            model_short = templates_model_name.split("/")[-1]
            model_name_variants.append(f"{model_short} (Prompt)")

        model_row = None
        for row in rows:
            model_col = row.get("Model", "")
            if model_col in model_name_variants:
                model_row = row
                break

        if model_row is None:
            logger.warning(
                f"Model '{templates_model_name}' not found in CSV. "
                f"Tried variants: {model_name_variants}. "
                f"Available models: {[r.get('Model', '') for r in rows]}"
            )
            return results

        # Extract accuracy for each requested category
        for category in categories:
            csv_column = CATEGORY_TO_CSV_COLUMN.get(category)

            if csv_column is None:
                logger.warning(f"No CSV column mapping for category: {category}")
                continue

            accuracy_str = model_row.get(csv_column, "")

            if accuracy_str:
                try:
                    # Convert percentage string to float
                    # Handle both "69.25" and "69.25%" formats
                    accuracy_clean = accuracy_str.rstrip('%').strip()
                    accuracy = float(accuracy_clean)
                    results[category] = {"accuracy": accuracy}
                    logger.info(f"{category}: {accuracy:.2f}%")
                except ValueError:
                    logger.warning(f"Could not parse accuracy for {category}: {accuracy_str}")
            else:
                logger.warning(f"No accuracy value for {category} in CSV")

    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")

    return results


def evaluate_checkpoint(
    checkpoint_path: Optional[str] = None,
    hf_model: Optional[str] = None,
    model_name: str = "Qwen/Qwen3-1.7B-Base",
    templates_model_name: str = "Qwen/Qwen3-1.7B",
    test_categories: str | list[str] | None = None,
    backend: str = "vllm",
    num_gpus: int = 1,
    gpu_memory_utilization: float = 0.9,
    output_path: Optional[str] = None,
    keep_temp_model: bool = False,
    temp_model_dir: Optional[str] = None,
    capture_samples_path: Optional[str] = None,
    num_samples_to_capture: int = 10,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Evaluate an MLPug checkpoint or HuggingFace model using BFCL.

    This is the main entry point for BFCL evaluation. Provide either:
    - checkpoint_path: Loads the checkpoint, converts to HF format, then evaluates
    - hf_model: Evaluates the HuggingFace model directly (no conversion needed)

    :param checkpoint_path: Path to the .pt checkpoint file (mutually exclusive with hf_model).
    :param hf_model: HuggingFace model name/path to evaluate directly (mutually exclusive with checkpoint_path).
    :param model_name: HuggingFace model name for architecture (only used with checkpoint_path).
    :param templates_model_name: Model name whose prompt templates to use.
        Must be recognized by BFCL (e.g., "Qwen/Qwen3-1.7B"). The actual weights
        come from the checkpoint or hf_model.
    :param test_categories: Test category or list of categories (default: DEFAULT_TEST_CATEGORIES).
    :param backend: Inference backend ("vllm" or "sglang").
    :param num_gpus: Number of GPUs to use.
    :param gpu_memory_utilization: GPU memory utilization (0.0-1.0).
    :param output_path: Path to save results JSON file.
    :param keep_temp_model: If True, don't delete temporary model directory (only used with checkpoint_path).
    :param temp_model_dir: Custom directory for temporary model (only used with checkpoint_path).
    :param capture_samples_path: If provided, capture sample prompts and responses to this file.
    :param num_samples_to_capture: Number of samples to capture per category (default: 10).
    :param logger: Optional logger.

    :return: Dictionary with evaluation results.
    """
    if logger is None:
        logger = module_logger

    if checkpoint_path is None and hf_model is None:
        raise ValueError("Must provide either checkpoint_path or hf_model")
    if checkpoint_path is not None and hf_model is not None:
        raise ValueError("Cannot provide both checkpoint_path and hf_model")

    if test_categories is None:
        test_categories = DEFAULT_TEST_CATEGORIES

    # Determine model path
    temp_dir = None
    cleanup_temp = False

    if hf_model is not None:
        # HuggingFace model: download to local directory
        # BFCL requires a local path for --local-model-path
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if temp_model_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="bfcl_model_")
            cleanup_temp = not keep_temp_model
        else:
            temp_dir = temp_model_dir
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading HuggingFace model {hf_model} to {temp_dir}")
        hf_model_instance = AutoModelForCausalLM.from_pretrained(hf_model)
        hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
        hf_model_instance.save_pretrained(temp_dir)
        hf_tokenizer.save_pretrained(temp_dir)
        del hf_model_instance  # Free memory
        logger.info(f"Model saved to {temp_dir}")
        model_path = temp_dir
    else:
        # Checkpoint: convert to HF format
        if temp_model_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="bfcl_model_")
            cleanup_temp = not keep_temp_model
        else:
            temp_dir = temp_model_dir
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

        model_path = convert_checkpoint_to_hf(
            checkpoint_path=checkpoint_path,
            output_dir=temp_dir,
            model_name=model_name,
            device="cpu",  # Use CPU to minimize memory
            logger=logger,
        )

    # Create temp directory for BFCL output
    bfcl_output_dir = tempfile.mkdtemp(prefix="bfcl_output_")

    try:
        # Generate responses
        bfcl_generate(
            model_path=model_path,
            test_category=test_categories,
            templates_model_name=templates_model_name,
            backend=backend,
            num_gpus=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            output_dir=bfcl_output_dir,
            logger=logger,
        )

        # Evaluate responses
        results = bfcl_evaluate(
            test_category=test_categories,
            output_dir=bfcl_output_dir,
            templates_model_name=templates_model_name,
            logger=logger,
        )

        # Capture sample responses before cleanup if requested
        if capture_samples_path:
            _capture_bfcl_samples(
                bfcl_output_dir,
                capture_samples_path,
                test_categories if isinstance(test_categories, list) else [test_categories],
                num_samples_to_capture,
                logger,
            )

        # Log summary
        _log_results_summary(results, logger)

        # Save results if output path specified
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results

    finally:
        # Cleanup temporary directories
        if cleanup_temp and temp_dir:
            logger.info(f"Cleaning up temporary model directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Always cleanup BFCL output directory
        shutil.rmtree(bfcl_output_dir, ignore_errors=True)


def _find_bfcl_test_data_dir() -> Path | None:
    """
    Find the BFCL test data directory in the installed bfcl_eval package.

    :return: Path to the data directory, or None if not found.
    """
    try:
        import bfcl_eval
        package_dir = Path(bfcl_eval.__file__).parent
        data_dir = package_dir / "data"
        if data_dir.exists():
            return data_dir
    except ImportError:
        pass
    return None


def _load_bfcl_test_data(category: str, logger: logging.Logger) -> dict:
    """
    Load BFCL test data for a category and build a lookup dict by sample ID.

    Loads both the test questions (from data/) and expected answers (from data/possible_answer/).

    :param category: Test category name (e.g., "simple_python").
    :param logger: Logger for status messages.

    :return: Dictionary mapping sample ID to test data (question, function, ground_truth).
    """
    test_data = {}
    data_dir = _find_bfcl_test_data_dir()

    if data_dir is None:
        logger.warning("BFCL test data directory not found")
        return test_data

    # Try both v4 and v3 naming for test questions
    for version in ["v4", "v3"]:
        test_file = data_dir / f"BFCL_{version}_{category}.json"
        if test_file.exists():
            logger.info(f"  Loading test data from: {test_file.name}")
            try:
                with open(test_file) as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            sample_id = sample.get("id")
                            if sample_id:
                                test_data[sample_id] = sample
                logger.info(f"  Loaded {len(test_data)} test samples")
                break
            except Exception as e:
                logger.warning(f"  Error loading test data: {e}")

    # Load expected answers from possible_answer/ directory
    for version in ["v4", "v3"]:
        answer_file = data_dir / "possible_answer" / f"BFCL_{version}_{category}.json"
        if answer_file.exists():
            logger.info(f"  Loading expected answers from: {answer_file.name}")
            try:
                with open(answer_file) as f:
                    for line in f:
                        if line.strip():
                            answer = json.loads(line)
                            sample_id = answer.get("id")
                            if sample_id and sample_id in test_data:
                                test_data[sample_id]["ground_truth"] = answer.get("ground_truth")
                logger.info(f"  Loaded expected answers")
                break
            except Exception as e:
                logger.warning(f"  Error loading expected answers: {e}")

    if not test_data:
        logger.warning(f"  No test data file found for category: {category}")

    return test_data


def _load_bfcl_score_data(
    output_dir: str,
    category: str,
    logger: logging.Logger,
) -> dict:
    """
    Load BFCL score data (failed samples) for a category.

    BFCL stores only failed samples in the score file. If a sample ID is not in the
    score file, it passed the evaluation.

    :param output_dir: BFCL output directory (BFCL_PROJECT_ROOT).
    :param category: Test category name (e.g., "simple_python").
    :param logger: Logger for status messages.

    :return: Dictionary mapping failed sample IDs to their error details.
    """
    failed_samples = {}
    score_dir = Path(output_dir) / "score"

    if not score_dir.exists():
        logger.warning(f"  Score directory not found: {score_dir}")
        return failed_samples

    # Search in model subdirectories and their subdirectories (non_live/, live/, etc.)
    search_dirs = []
    for model_dir in score_dir.iterdir():
        if model_dir.is_dir():
            search_dirs.append(model_dir)
            for subdir in model_dir.iterdir():
                if subdir.is_dir():
                    search_dirs.append(subdir)

    # Find score file
    score_file = None
    for search_dir in search_dirs:
        for version in ["v4", "v3"]:
            candidate = search_dir / f"BFCL_{version}_{category}_score.json"
            if candidate.exists():
                score_file = candidate
                break
        if score_file:
            break

    if score_file is None:
        logger.warning(f"  Score file not found for category: {category}")
        return failed_samples

    logger.info(f"  Loading score data from: {score_file.name}")

    try:
        with open(score_file) as f:
            content = f.read().strip()

        # Try JSONL first, fall back to JSON array
        entries = []
        try:
            for line in content.split('\n'):
                if line.strip():
                    entries.append(json.loads(line))
        except json.JSONDecodeError:
            data = json.loads(content)
            if isinstance(data, list):
                entries = data
            else:
                entries = [data]

        # First entry is the header with accuracy info
        # Subsequent entries are failed samples
        for entry in entries[1:]:  # Skip header
            sample_id = entry.get("id")
            if sample_id:
                failed_samples[sample_id] = entry

        logger.info(f"  Loaded {len(failed_samples)} failed samples from score file")

    except Exception as e:
        logger.warning(f"  Error loading score data: {e}")

    return failed_samples


def _format_bfcl_prompt(sample_id: str, question: list, functions: list) -> str:
    """
    Format the full BFCL prompt as presented to the model.

    Uses BFCL's formulate_system_prompt to reconstruct the exact prompt.

    :param sample_id: Sample ID (used to extract prompt format config).
    :param question: The question/prompt messages from test data.
    :param functions: The available functions from test data.

    :return: Formatted prompt string.
    """
    try:
        from bfcl_eval.model_handler.utils import (
            formulate_system_prompt,
            extract_prompt_format_from_id,
        )

        # Get the prompt format config from the sample ID
        prompt_format = extract_prompt_format_from_id(sample_id)

        # Generate the system prompt using BFCL's logic
        system_prompt = formulate_system_prompt(
            format_sensitivity_config=prompt_format,
            functions=functions,
        )

        # Build the full prompt
        formatted = "=== SYSTEM PROMPT ===\n\n"
        formatted += system_prompt
        formatted += "\n\n=== USER MESSAGE ===\n\n"

        # Format the user question
        if isinstance(question, list) and question:
            for turn in question:
                if isinstance(turn, list):
                    for msg in turn:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        if role != 'system':  # System prompt already shown
                            formatted += f"[{role}]: {content}\n\n"
                elif isinstance(turn, dict):
                    role = turn.get('role', 'unknown')
                    content = turn.get('content', '')
                    if role != 'system':
                        formatted += f"[{role}]: {content}\n\n"

        return formatted

    except ImportError as e:
        # Fall back to simple format if BFCL not available
        return f"[BFCL import error: {e}]\n\nQuestion: {question}\n\nFunctions: {json.dumps(functions, indent=2)}"


def _capture_bfcl_samples(
    output_dir: str,
    capture_path: str,
    categories: list[str],
    num_samples: int,
    logger: logging.Logger,
) -> None:
    """
    Capture sample prompts and model responses from BFCL output.

    For each sample, captures:
    1. Full prompt (system prompt + user message) as presented to the model
    2. Model response
    3. Expected output (ground truth)
    4. Assessment result (PASS/FAIL with error details if failed)

    BFCL stores model outputs in:
      result/MODEL_NAME/BFCL_v4_TEST_CATEGORY_result.json

    Reference: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard

    :param output_dir: BFCL output directory (BFCL_PROJECT_ROOT).
    :param capture_path: Path to save captured samples.
    :param categories: List of categories to capture.
    :param num_samples: Number of samples per category.
    :param logger: Logger for status messages.
    """
    logger.info(f"Capturing BFCL samples to {capture_path}")

    # BFCL stores model outputs in result/MODEL_NAME/
    result_dir = Path(output_dir) / "result"

    if not result_dir.exists():
        logger.warning(f"BFCL result directory not found: {result_dir}")
        logger.info(f"Contents of {output_dir}:")
        for item in Path(output_dir).iterdir():
            logger.info(f"  {item.name}")
        return

    with open(capture_path, 'w') as f:
        sample_count = 0

        # Iterate through model subdirectories
        for model_dir in result_dir.iterdir():
            if not model_dir.is_dir():
                continue

            logger.info(f"Found model directory: {model_dir.name}")

            # BFCL v4 uses subdirectories: non_live/, live/, multi_turn/, etc.
            # Check both direct files and subdirectories
            search_dirs = [model_dir]
            for subdir in model_dir.iterdir():
                if subdir.is_dir():
                    search_dirs.append(subdir)
                    logger.info(f"  Found subdirectory: {subdir.name}")

            # Find result files for each category
            for category in categories:
                result_file = None

                # Search all directories for the result file
                for search_dir in search_dirs:
                    # BFCL naming: BFCL_v3_{category}_result.json or BFCL_v4_{category}_result.json
                    for version in ["v4", "v3"]:
                        candidate = search_dir / f"BFCL_{version}_{category}_result.json"
                        if candidate.exists():
                            result_file = candidate
                            break
                    if result_file:
                        break

                    # Try alternative patterns
                    alt_patterns = [
                        f"*{category}*_result.json",
                        f"*{category}*.json",
                    ]
                    for pattern in alt_patterns:
                        matches = list(search_dir.glob(pattern))
                        if matches:
                            result_file = matches[0]
                            break
                    if result_file:
                        break

                if not result_file:
                    logger.info(f"  No result file for category: {category}")
                    logger.info(f"  Searched directories: {[str(d) for d in search_dirs]}")
                    continue

                logger.info(f"  Reading: {result_file.name}")

                # Load test data for this category to get prompts and expected outputs
                test_data = _load_bfcl_test_data(category, logger)

                # Load score data to determine pass/fail for each sample
                failed_samples = _load_bfcl_score_data(output_dir, category, logger)

                try:
                    with open(result_file) as rf:
                        content = rf.read().strip()

                    # BFCL v4 uses JSONL format (one JSON object per line)
                    # Try JSONL first, fall back to JSON array
                    samples = []
                    try:
                        # Try parsing as JSONL (one JSON object per line)
                        for line in content.split('\n'):
                            if line.strip():
                                samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Fall back to single JSON array
                        data = json.loads(content)
                        if isinstance(data, list):
                            samples = data
                        else:
                            samples = [data]

                    # Write samples to capture file
                    for i, sample in enumerate(samples[:num_samples]):
                        sample_count += 1
                        sample_id = sample.get('id', f'unknown_{i}')

                        f.write(f"{'#' * 80}\n")
                        f.write(f"# SAMPLE {sample_count}\n")
                        f.write(f"{'#' * 80}\n\n")
                        f.write(f"**Category:** {category}\n")
                        f.write(f"**Sample ID:** {sample_id}\n\n")

                        # Get test data for this sample (contains prompt and function defs)
                        test_sample = test_data.get(sample_id, {})

                        # 1. FULL PROMPT (as presented to model)
                        question = test_sample.get('question', [])
                        functions = test_sample.get('function', [])

                        f.write(f"## 1. PROMPT (as presented to model)\n\n")
                        if question or functions:
                            full_prompt = _format_bfcl_prompt(sample_id, question, functions)
                            f.write(f"```\n{full_prompt}\n```\n\n")
                        else:
                            f.write("*[No prompt data available]*\n\n")

                        # 2. MODEL RESPONSE
                        response = sample.get('result', sample.get('output', sample.get('response', 'N/A')))
                        f.write(f"## 2. MODEL RESPONSE\n\n")
                        f.write(f"```\n{response}\n```\n\n")

                        # 3. EXPECTED OUTPUT (ground truth)
                        ground_truth = test_sample.get('ground_truth', [])
                        f.write(f"## 3. EXPECTED OUTPUT (acceptable function calls)\n\n")
                        if ground_truth:
                            for gt in ground_truth:
                                f.write(f"```\n{gt}\n```\n")
                        else:
                            f.write("*[No expected output available]*\n")
                        f.write("\n")

                        # 4. ASSESSMENT RESULT
                        f.write(f"## 4. ASSESSMENT RESULT\n\n")
                        if sample_id in failed_samples:
                            error_info = failed_samples[sample_id]
                            error_type = error_info.get('error_type', 'unknown')
                            error_details = error_info.get('error', [])
                            f.write(f"**FAIL**\n\n")
                            f.write(f"- Error type: `{error_type}`\n")
                            if error_details:
                                f.write(f"- Error details:\n")
                                for err in error_details:
                                    f.write(f"  - {err}\n")
                        else:
                            f.write(f"**PASS**\n")
                        f.write("\n")

                        f.write("=" * 80 + "\n\n")

                except Exception as e:
                    logger.warning(f"Error reading {result_file}: {e}")

        logger.info(f"Captured {sample_count} samples to {capture_path}")


def _log_results_summary(results: dict, logger: logging.Logger) -> None:
    """Log a summary of BFCL evaluation results."""
    if not results:
        return

    logger.info("=" * 60)
    logger.info("BFCL Evaluation Results Summary")
    logger.info("=" * 60)

    for category, category_results in results.items():
        if isinstance(category_results, dict):
            # Try to find accuracy metric
            accuracy = category_results.get('accuracy', category_results.get('ast_accuracy'))
            if accuracy is not None:
                logger.info(f"  {category}: {accuracy:.4f}")
            else:
                logger.info(f"  {category}: {category_results}")
        else:
            logger.info(f"  {category}: {category_results}")

    logger.info("=" * 60)


def get_results_summary(results: dict) -> dict:
    """
    Extract a summary of key metrics from BFCL evaluation results.

    :param results: Full results dictionary from BFCL evaluation.

    :return: Dictionary mapping categories to their primary accuracy metric.
    """
    summary = {}

    for category, category_results in results.items():
        if isinstance(category_results, dict):
            accuracy = category_results.get('accuracy', category_results.get('ast_accuracy'))
            if accuracy is not None:
                summary[category] = {
                    'metric': 'accuracy',
                    'value': accuracy,
                }

    return summary
