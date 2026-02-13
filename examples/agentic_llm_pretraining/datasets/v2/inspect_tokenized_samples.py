#!/usr/bin/env python3
"""
Inspect tokenized samples with token-by-token loss mask alignment.

Shows decoded tokens aligned with their loss mask values around mask
transitions (boundaries between masked and trained regions). This verifies
that prompt/response boundaries survive tokenization correctly.

Display format around each transition:
    Tokens:  the  answer  is  ·  <|im_start|>  system  \n  You
    Mask:     0     0     0   0       1           1      1    1

Loss mask values:
- 1 = masked (excluded from loss): system prompts, user prompts
- 0 = trained: responses, thinking, text

Usage (from repo root):
    # Inspect 3 train samples, 8 tokens of context around boundaries
    python -m examples.agentic_llm_pretraining.datasets.v2.inspect_tokenized_samples \
        --tokenized-dir ../data/agentic_llm_pretraining/v2/tokenized

    # Inspect specific split with more context
    python -m examples.agentic_llm_pretraining.datasets.v2.inspect_tokenized_samples \
        --tokenized-dir ../data/agentic_llm_pretraining/v2/tokenized \
        --split val --num-samples 5 --context 12

    # Inspect specific sample indices
    python -m examples.agentic_llm_pretraining.datasets.v2.inspect_tokenized_samples \
        --tokenized-dir ../data/agentic_llm_pretraining/v2/tokenized \
        --sample-indices 0 100 500
"""

import argparse
import os
import random

import numpy as np
from transformers import AutoTokenizer

from mlpug.mlpug_logging import get_logger, use_fancy_colors
from mlpug.utils.git_logging import log_git_state

from data_forager.datasets.tokens_with_aux import TokensWithAuxDataset


use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))


# Default tokenizer for Qwen3
DEFAULT_TOKENIZER = "Qwen/Qwen3-1.7B"


def describe_config(
    tokenized_dir: str,
    split: str,
    num_samples: int,
    sample_indices: list[int] | None,
    context: int,
    tokenizer: str,
) -> None:
    """Log script configuration."""
    module_logger.info("Configuration:")
    module_logger.info(f"  tokenized_dir: {tokenized_dir}")
    module_logger.info(f"  split: {split}")
    module_logger.info(f"  num_samples: {num_samples}")
    module_logger.info(f"  sample_indices: {sample_indices}")
    module_logger.info(f"  context: {context}")
    module_logger.info(f"  tokenizer: {tokenizer}")


def escape_token(text: str) -> str:
    """
    Escape a decoded token for display.

    Makes whitespace and special characters visible:
    - Space → ·
    - Newline → \\n
    - Tab → \\t
    - Carriage return → \\r
    - Empty string → <empty>

    :param text: Decoded token text.

    :return: Escaped string suitable for display.
    """
    if not text:
        return "<empty>"

    text = text.replace("\r", "\\r")
    text = text.replace("\n", "\\n")
    text = text.replace("\t", "\\t")
    text = text.replace(" ", "·")

    return text


def find_transitions(loss_mask: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Find positions where the loss mask changes value.

    :param loss_mask: Array of mask values (0 or 1).

    :return: List of (position, old_value, new_value) tuples.
    """
    transitions = []
    for i in range(1, len(loss_mask)):
        if loss_mask[i] != loss_mask[i - 1]:
            transitions.append((i, int(loss_mask[i - 1]), int(loss_mask[i])))

    return transitions


def render_boundary(
    tokens: np.ndarray,
    loss_mask: np.ndarray,
    transition_pos: int,
    old_val: int,
    new_val: int,
    context: int,
    tokenizer,
) -> str:
    """
    Render a mask transition with surrounding token context.

    Shows decoded tokens on one line and centered mask values on the line
    below, for `context` tokens before and after the transition point.

    :param tokens: Full token array for the sample.
    :param loss_mask: Full loss mask array for the sample.
    :param transition_pos: Position where mask changes.
    :param old_val: Mask value before transition.
    :param new_val: Mask value after transition.
    :param context: Number of tokens to show before/after.
    :param tokenizer: HuggingFace tokenizer for decoding.

    :return: Formatted string showing the boundary.
    """
    start = max(0, transition_pos - context)
    end = min(len(tokens), transition_pos + context)

    # Decode each token individually
    escaped_tokens = []
    mask_values = []
    for i in range(start, end):
        decoded = tokenizer.decode([int(tokens[i])], skip_special_tokens=False)
        escaped = escape_token(decoded)
        escaped_tokens.append(escaped)
        mask_values.append(str(int(loss_mask[i])))

    # Calculate column widths — each column is as wide as the wider of
    # the escaped token or the mask value (always 1 char), plus 1 space padding
    col_widths = [max(len(tok), 1) for tok in escaped_tokens]

    # Build token line and mask line with aligned columns
    token_parts = []
    mask_parts = []
    for tok, mask_val, width in zip(escaped_tokens, mask_values, col_widths):
        token_parts.append(tok.ljust(width))
        mask_parts.append(mask_val.center(width))

    separator = "  "
    token_line = separator.join(token_parts)
    mask_line = separator.join(mask_parts)

    # Direction label
    if old_val == 0 and new_val == 1:
        direction = "trained → masked"
    else:
        direction = "masked → trained"

    header = f"--- Transition at pos {transition_pos}: {direction} ({old_val}→{new_val}) ---"

    return f"{header}\n  Tokens:  {token_line}\n  Mask:    {mask_line}"


def inspect_sample(
    sample_idx: int,
    dataset: TokensWithAuxDataset,
    tokenizer,
    context: int,
) -> None:
    """
    Inspect a single tokenized sample, showing all mask transitions.

    :param sample_idx: Index of the sample in the dataset.
    :param dataset: TokensWithAuxDataset instance.
    :param tokenizer: HuggingFace tokenizer for decoding.
    :param context: Tokens of context around each transition.
    """
    sample = dataset[sample_idx]
    tokens = sample["tokens"]
    loss_mask = sample["loss_mask"]

    masked_count = int(loss_mask.sum())
    trained_count = len(loss_mask) - masked_count
    mask_ratio = masked_count / len(loss_mask) * 100

    module_logger.info("")
    module_logger.info("=" * 70)
    module_logger.info(
        f"Sample {sample_idx}  |  "
        f"{len(tokens)} tokens  |  "
        f"masked: {masked_count} ({mask_ratio:.1f}%)  |  "
        f"trained: {trained_count} ({100 - mask_ratio:.1f}%)"
    )
    module_logger.info("=" * 70)

    transitions = find_transitions(loss_mask)

    if not transitions:
        # No transitions — entire sample has uniform mask
        val = int(loss_mask[0])
        label = "masked" if val == 1 else "trained"
        module_logger.info(f"  No transitions — entire sample is {label} (mask={val})")

        # Show first few tokens as context
        preview_len = min(context * 2, len(tokens))
        escaped = []
        for i in range(preview_len):
            decoded = tokenizer.decode([int(tokens[i])], skip_special_tokens=False)
            escaped.append(escape_token(decoded))

        module_logger.info(f"  First {preview_len} tokens: {' '.join(escaped)}")
        return

    module_logger.info(f"  {len(transitions)} transition(s)")

    # Merge overlapping boundary windows
    boundaries = []
    for pos, old_val, new_val in transitions:
        window_start = max(0, pos - context)
        window_end = min(len(tokens), pos + context)

        if boundaries and window_start <= boundaries[-1][1]:
            # Merge with previous window
            prev = boundaries[-1]
            boundaries[-1] = (
                prev[0],
                window_end,
                prev[2] + [(pos, old_val, new_val)],
            )
        else:
            boundaries.append((window_start, window_end, [(pos, old_val, new_val)]))

    for window_start, window_end, window_transitions in boundaries:
        # If multiple transitions in one merged window, render the full window
        # with all transitions marked
        if len(window_transitions) == 1:
            pos, old_val, new_val = window_transitions[0]
            rendered = render_boundary(
                tokens, loss_mask, pos, old_val, new_val, context, tokenizer
            )
            module_logger.info(f"\n{rendered}")
        else:
            # Multiple transitions close together — render the full merged window
            positions = [t[0] for t in window_transitions]
            labels = ", ".join(
                f"pos {p}: {ov}→{nv}" for p, ov, nv in window_transitions
            )
            header = f"--- Merged transitions: {labels} ---"

            escaped_tokens = []
            mask_values = []
            for i in range(window_start, window_end):
                decoded = tokenizer.decode(
                    [int(tokens[i])], skip_special_tokens=False
                )
                escaped_tokens.append(escape_token(decoded))
                mask_values.append(str(int(loss_mask[i])))

            col_widths = [max(len(tok), 1) for tok in escaped_tokens]
            separator = "  "
            token_line = separator.join(
                tok.ljust(w) for tok, w in zip(escaped_tokens, col_widths)
            )
            mask_line = separator.join(
                mv.center(w) for mv, w in zip(mask_values, col_widths)
            )

            module_logger.info(
                f"\n{header}\n  Tokens:  {token_line}\n  Mask:    {mask_line}"
            )


def main():
    log_git_state()

    parser = argparse.ArgumentParser(
        description="Inspect tokenized samples with token-by-token loss mask alignment"
    )
    parser.add_argument(
        "--tokenized-dir",
        required=True,
        help="Directory containing tokenized splits (train/, val/, test/)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to inspect (default: train)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of random samples to inspect (default: 3)",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample indices to inspect (overrides --num-samples)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=8,
        help="Tokens of context before/after each mask transition (default: 8)",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"HuggingFace tokenizer for decoding (default: {DEFAULT_TOKENIZER})",
    )
    args = parser.parse_args()

    describe_config(
        tokenized_dir=args.tokenized_dir,
        split=args.split,
        num_samples=args.num_samples,
        sample_indices=args.sample_indices,
        context=args.context,
        tokenizer=args.tokenizer,
    )

    # Load tokenizer
    module_logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load dataset
    split_dir = os.path.join(args.tokenized_dir, args.split)
    module_logger.info(f"Loading tokenized dataset from: {split_dir}")
    dataset = TokensWithAuxDataset.create_from_index_on_filesystem(split_dir)
    module_logger.info(f"Dataset has {len(dataset)} samples")

    # Determine which samples to inspect
    if args.sample_indices is not None:
        indices = args.sample_indices
    else:
        if args.num_samples >= len(dataset):
            indices = list(range(len(dataset)))
        else:
            indices = sorted(random.sample(range(len(dataset)), args.num_samples))

    module_logger.info(f"Inspecting samples: {indices}")

    # Inspect each sample
    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            module_logger.warning(f"Sample index {idx} out of range, skipping")
            continue
        inspect_sample(idx, dataset, tokenizer, args.context)

    module_logger.info("")
    module_logger.info("Done.")


if __name__ == "__main__":
    main()
