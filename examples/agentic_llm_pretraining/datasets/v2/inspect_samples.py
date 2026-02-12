#!/usr/bin/env python3
"""
Inspect v2 transformed samples with loss mask visualization.

Shows each part with its type and whether it will be masked (excluded from loss)
or trained on during NTP.

Loss mask rules:
- MASKED (loss=0): system, prompt - model sees but doesn't train on
- TRAINED (loss=1): response, thinking, text - model trains on these

Usage (from repo root):
    python -m examples.agentic_llm_pretraining.datasets.v2.inspect_samples \
        --data-dir ../data/transforms-v2

    python -m examples.agentic_llm_pretraining.datasets.v2.inspect_samples \
        --data-dir ../data/transforms-v2 \
        --datasets gsm8k soda \
        --num-samples 3
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.markup import escape as escape_markup
from rich.panel import Panel
from rich.text import Text

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from mlpug.utils.git_logging import log_git_state


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))

# Part types and their loss mask status
# True = masked (excluded from loss), False = trained on
PART_MASK_RULES = {
    "system": True,    # System prompts - masked
    "prompt": True,    # User prompts - masked
    "response": False, # Assistant responses - trained
    "thinking": False, # Thinking/reasoning - trained
    "text": False,     # Plain text (no masking) - trained
}

# Colors for visualization
MASK_COLORS = {
    True: "dim",       # Masked parts shown dim
    False: "green",    # Trained parts shown green
}

TYPE_LABELS = {
    "system": "[SYSTEM]",
    "prompt": "[PROMPT]",
    "response": "[RESPONSE]",
    "thinking": "[THINKING]",
    "text": "[TEXT]",
}


def load_jsonl_samples(
    file_path: Path,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Load random samples from a v2 transformed JSONL file.

    :param file_path: Path to JSONL file.
    :param num_samples: Number of samples to return.
    :param seed: Random seed for reproducibility.

    :return: List of sample dicts with 'parts' field.
    """
    # First pass: count lines
    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    if total_lines == 0:
        return []

    # Select random line indices
    random.seed(seed)
    num_to_sample = min(num_samples, total_lines)
    selected_indices = set(random.sample(range(total_lines), num_to_sample))

    # Second pass: read selected lines
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in selected_indices:
                samples.append(json.loads(line))

    return samples


def _truncate_text(
    text: str,
    max_length: int,
    tail_length: int,
    context_length: int = 80,
) -> str:
    """
    Truncate long text, showing start, end, and context around </think> boundary.

    For texts containing <think>...</think>, shows additional context around the
    </think> tag so the transition from reasoning to response is visible.

    :param text: Full text to truncate.
    :param max_length: Characters to show from the start.
    :param tail_length: Characters to show from the end.
    :param context_length: Characters to show before/after </think>.

    :return: Truncated text with markers.
    """
    if len(text) <= max_length + tail_length:
        return text

    think_close_pos = text.find("</think>")

    # No </think> or it falls within the already-visible start/end regions
    if (
        think_close_pos == -1
        or think_close_pos < max_length
        or think_close_pos + len("</think>") > len(text) - tail_length
    ):
        truncated_chars = len(text) - max_length - tail_length
        return (
            text[:max_length] +
            f"\n... [truncated, {truncated_chars} chars] ...\n" +
            text[-tail_length:]
        )

    # Show: start ... context around </think> ... end
    think_tag_end = think_close_pos + len("</think>")
    ctx_before = max(max_length, think_close_pos - context_length)
    ctx_after = min(len(text) - tail_length, think_tag_end + context_length)

    gap1 = ctx_before - max_length
    gap2 = (len(text) - tail_length) - ctx_after

    parts = [text[:max_length]]

    if gap1 > 0:
        parts.append(f"\n... [truncated, {gap1} chars] ...\n")

    parts.append(text[ctx_before:ctx_after])

    if gap2 > 0:
        parts.append(f"\n... [truncated, {gap2} chars] ...\n")

    parts.append(text[-tail_length:])

    return "".join(parts)


def format_part_for_display(
    part: dict[str, str],
    max_length: int = 500,
    tail_length: int = 80,
) -> tuple[str, str, bool]:
    """
    Format a single part for display.

    :param part: Part dict with 'type' and 'text' fields.
    :param max_length: Maximum text length for the start before truncation.
    :param tail_length: Number of characters to show from the end when truncated.

    :return: Tuple of (type_label, display_text, is_masked).
    """
    part_type = part.get("type", "unknown")
    text = part.get("text", "")

    type_label = TYPE_LABELS.get(part_type, f"[{part_type.upper()}]")
    is_masked = PART_MASK_RULES.get(part_type, False)

    text = _truncate_text(text, max_length, tail_length)

    return type_label, text, is_masked


def display_sample(
    sample: dict[str, Any],
    sample_num: int,
    console: Console,
    max_part_length: int = 500,
) -> None:
    """
    Display a single v2 sample with mask visualization.

    :param sample: Sample dict with 'source', 'index', 'parts'.
    :param sample_num: Display number for this sample.
    :param console: Rich console for output.
    :param max_part_length: Maximum length per part before truncation.
    """
    source = sample.get("source", "unknown")
    index = sample.get("index", "?")
    parts = sample.get("parts", [])

    console.print(f"\n[bold]--- Sample {sample_num} (source: {source}, index: {index}) ---[/bold]")

    if not parts:
        console.print("[yellow]No parts in sample[/yellow]")
        return

    # Display legend
    console.print("[dim]Legend: [/dim][dim]MASKED (dim)[/dim] | [green]TRAINED (green)[/green]")
    console.print()

    for i, part in enumerate(parts):
        type_label, text, is_masked = format_part_for_display(part, max_part_length)
        color = MASK_COLORS[is_masked]
        mask_status = "MASKED" if is_masked else "TRAINED"

        # Create styled text
        header = Text()
        header.append(f"Part {i+1}: ", style="bold")
        header.append(type_label, style=f"bold {color}")
        header.append(f" ({mask_status})", style=color)

        console.print(header)
        # Escape markup to prevent Rich from interpreting [...] as tags
        console.print(Panel(escape_markup(text), style=color, expand=False))


def inspect_dataset(
    file_path: Path,
    num_samples: int,
    seed: int,
    max_part_length: int,
    console: Console,
) -> None:
    """
    Inspect samples from a single v2 transformed dataset.

    :param file_path: Path to transformed JSONL file.
    :param num_samples: Number of samples to display.
    :param seed: Random seed.
    :param max_part_length: Maximum length per part before truncation.
    :param console: Rich console for output.
    """
    dataset_name = file_path.stem

    console.print(f"\n[bold blue]{'='*60}[/bold blue]")
    console.print(f"[bold blue]Dataset: {dataset_name}[/bold blue]")
    console.print(f"[bold blue]{'='*60}[/bold blue]")

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        return

    samples = load_jsonl_samples(file_path, num_samples, seed)

    if not samples:
        console.print(f"[yellow]No samples found in {file_path}[/yellow]")
        return

    console.print(f"Showing {len(samples)} random samples:")

    for i, sample in enumerate(samples, 1):
        display_sample(sample, i, console, max_part_length)


def print_mask_summary(console: Console) -> None:
    """Print summary of loss mask rules."""
    console.print("\n[bold]Loss Mask Rules:[/bold]")
    console.print("  [dim]MASKED (excluded from loss):[/dim]")
    console.print("    - system: System prompts/instructions")
    console.print("    - prompt: User inputs, questions, context")
    console.print("  [green]TRAINED (included in loss):[/green]")
    console.print("    - response: Assistant outputs")
    console.print("    - thinking: Reasoning traces")
    console.print("    - text: Plain text (full document training)")


def main():
    log_git_state()

    parser = argparse.ArgumentParser(
        description="Inspect v2 transformed samples with loss mask visualization"
    )
    parser.add_argument(
        "--data-dir",
        default="../data/transforms-v2",
        help="Directory containing v2 transformed JSONL files (default: ../data/transforms-v2)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to inspect (default: all .jsonl files in data-dir)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of samples to show per dataset (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--max-part-length",
        type=int,
        default=500,
        help="Maximum text length per part before truncation (default: 500)",
    )
    parser.add_argument(
        "--show-rules",
        action="store_true",
        help="Show loss mask rules summary and exit",
    )
    args = parser.parse_args()

    console = Console()

    if args.show_rules:
        print_mask_summary(console)
        return

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        console.print(f"[red]Data directory not found: {data_dir}[/red]")
        return

    if args.datasets:
        files = [data_dir / f"{name}.jsonl" for name in args.datasets]
    else:
        files = sorted(data_dir.glob("*.jsonl"))

    if not files:
        console.print(f"[yellow]No JSONL files found in {data_dir}[/yellow]")
        return

    # Show mask rules at the start
    print_mask_summary(console)

    console.print(f"\n[bold]Inspecting {len(files)} datasets from {data_dir}[/bold]")

    for file_path in files:
        inspect_dataset(
            file_path,
            args.num_samples,
            args.seed,
            args.max_part_length,
            console,
        )


if __name__ == "__main__":
    main()
