"""
Interactive chat script for model checkpoints and HuggingFace models.

Supports configurable inference parameters via CLI flags and/or YAML config files.
YAML provides defaults; CLI flags override YAML values.

Usage (checkpoint):
    python -m examples.agentic_llm_pretraining.inference.chat \
        --checkpoint /path/to/checkpoint.pt

Usage (HuggingFace model):
    python -m examples.agentic_llm_pretraining.inference.chat \
        --hf-model Qwen/Qwen3-1.7B-Base

Usage (with YAML config):
    python -m examples.agentic_llm_pretraining.inference.chat \
        --checkpoint /path/to/checkpoint.pt --config creative.yaml

Usage (deterministic / greedy):
    python -m examples.agentic_llm_pretraining.inference.chat \
        --checkpoint /path/to/checkpoint.pt --temperature 0

In-chat commands:
    /quit or /exit  — Exit the chat
    /clear          — Clear conversation history
    /system <text>  — Set a new system prompt
    /config         — Show current generation config
"""
from typing import Optional

import argparse
import logging
import os
import sys

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer

from basics.logging import get_logger

import mlpug.pytorch as mlp
from mlpug.utils.git_logging import log_git_state

from examples.agentic_llm_pretraining.checkpoint import load_model_from_checkpoint

mlp.logging.use_fancy_colors()
module_logger = get_logger(os.path.basename(__file__))

DEFAULT_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"

# Hardcoded defaults for generation parameters.
# These are used when neither CLI nor YAML provides a value.
GENERATION_DEFAULTS = {
    "temperature": 0,
    "top_k": 50,
    "top_p": 0.9,
    "max_response_length": 512,
    "repetition_penalty": 1.0,
}


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for chat script."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with a model checkpoint or HuggingFace model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "In-chat commands:\n"
            "  /quit or /exit  — Exit the chat\n"
            "  /clear          — Clear conversation history\n"
            "  /system <text>  — Set a new system prompt\n"
            "  /config         — Show current generation config\n"
        ),
    )

    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to MLPug model checkpoint (.pt file)",
    )
    model_group.add_argument(
        "--hf-model",
        type=str,
        help="HuggingFace model name or path (e.g., Qwen/Qwen3-1.7B-Base)",
    )

    # Model configuration (only used with --checkpoint)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "HuggingFace model name for architecture "
            f"(only used with --checkpoint, default: {DEFAULT_MODEL_NAME})"
        ),
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file with generation parameters",
    )

    # Generation parameters — defaults are None so we can distinguish
    # "user explicitly set" from "use YAML/hardcoded default"
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (0 = greedy/deterministic, default: 0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling (default: 0.9)",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty (1.0 = no penalty, default: 1.0)",
    )

    # System prompt
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to use for the conversation",
    )

    # Device/dtype
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Model dtype (default: bfloat16)",
    )

    return parser


def load_yaml_config(config_path: str, logger: logging.Logger) -> dict:
    """
    Load generation config from a YAML file.

    :param config_path: Path to the YAML config file.
    :param logger: Logger instance.

    :return: Dict of config values.
    """
    logger.info(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(config).__name__}")

    # Normalize keys: YAML uses underscores, CLI uses hyphens
    normalized = {}
    for key, value in config.items():
        normalized[key.replace("-", "_")] = value

    return normalized


def resolve_config(args: argparse.Namespace, logger: logging.Logger) -> dict:
    """
    Merge CLI args, YAML config, and hardcoded defaults.

    Priority: CLI (explicit) > YAML > hardcoded defaults.

    :param args: Parsed CLI arguments.
    :param logger: Logger instance.

    :return: Resolved config dict.
    """
    # Load YAML config if provided
    yaml_config = {}
    if args.config:
        yaml_config = load_yaml_config(args.config, logger)

    # The generation parameters that participate in the merge
    merge_keys = [
        "temperature", "top_k", "top_p", "max_response_length",
        "repetition_penalty", "system_prompt", "device", "dtype",
    ]

    resolved = {}
    for key in merge_keys:
        cli_value = getattr(args, key, None)
        yaml_value = yaml_config.get(key)
        default_value = GENERATION_DEFAULTS.get(key)

        if cli_value is not None:
            resolved[key] = cli_value
        elif yaml_value is not None:
            resolved[key] = yaml_value
        elif default_value is not None:
            resolved[key] = default_value
        # else: stays unset (None) — e.g., system_prompt

    # Apply non-generation defaults that don't participate in YAML merge
    if resolved.get("device") is None:
        if torch.cuda.is_available():
            resolved["device"] = "cuda"
        elif torch.backends.mps.is_available():
            resolved["device"] = "mps"
        else:
            resolved["device"] = "cpu"

    if resolved.get("dtype") is None:
        resolved["dtype"] = "bfloat16"

    return resolved


def build_generate_kwargs(config: dict) -> dict:
    """
    Build kwargs dict for model.generate() from resolved config.

    :param config: Resolved config dict.

    :return: kwargs for model.generate().
    """
    temperature = config["temperature"]
    is_greedy = (temperature == 0)

    kwargs = {
        "max_new_tokens": config["max_response_length"],
        "repetition_penalty": config["repetition_penalty"],
    }

    if is_greedy:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        kwargs["top_k"] = config["top_k"]
        kwargs["top_p"] = config["top_p"]

    return kwargs


def load_model(
    args: argparse.Namespace,
    device: str,
    dtype: str,
    logger: logging.Logger,
) -> tuple:
    """
    Load model and tokenizer from checkpoint or HuggingFace model name.

    :param args: Parsed CLI arguments.
    :param device: Device to load to.
    :param dtype: Model dtype string.
    :param logger: Logger instance.

    :return: Tuple of (model, tokenizer).
    """
    torch_dtype = getattr(torch, dtype, torch.bfloat16)

    if args.checkpoint:
        model_name = args.model_name or DEFAULT_MODEL_NAME
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        logger.info(f"Architecture: {model_name}")

        model, tokenizer = load_model_from_checkpoint(
            checkpoint_path=args.checkpoint,
            model_name=model_name,
            device=device,
            logger=logger,
        )
        model = model.to(dtype=torch_dtype)

        # Load generation config from the base model (checkpoint doesn't include it)
        model.generation_config = GenerationConfig.from_pretrained(
            model_name, trust_remote_code=True,
        )
    else:
        hf_model = args.hf_model
        logger.info(f"Loading HuggingFace model: {hf_model}")

        model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            dtype=torch_dtype,
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)

    return model, tokenizer


def describe_config(
    checkpoint: str | None,
    hf_model: str | None,
    model_name: str | None,
    config_file: str | None,
    generation_config: dict,
    logger: logging.Logger | None = None,
) -> None:
    """Log script configuration."""
    if logger is None:
        logger = module_logger

    logger.info("Configuration:")
    logger.info(f"  checkpoint: {checkpoint}")
    logger.info(f"  hf_model: {hf_model}")
    logger.info(f"  model_name: {model_name}")
    logger.info(f"  config_file: {config_file}")
    for key, value in generation_config.items():
        logger.info(f"  {key}: {value}")


def get_stop_token_ids(
    model,
    tokenizer,
    logger: logging.Logger,
) -> list[int]:
    """
    Build stop token IDs from model's generation config and tokenizer.

    Base models may not include chat turn-boundary tokens (e.g., <|im_end|>)
    in their generation config. This function collects stop tokens from the
    generation config and adds any end-of-turn special tokens from the tokenizer.

    :param model: The loaded model (with generation_config).
    :param tokenizer: The tokenizer.
    :param logger: Logger instance.

    :return: List of token IDs to use as eos_token_id.
    """
    gen_eos = model.generation_config.eos_token_id
    if isinstance(gen_eos, int):
        stop_ids = {gen_eos}
    elif isinstance(gen_eos, list):
        stop_ids = set(gen_eos)
    else:
        stop_ids = {tokenizer.eos_token_id}

    # Check tokenizer's additional special tokens for end-of-turn markers
    # Common patterns: <|im_end|> (ChatML), <|eot_id|> (Llama), <end_of_turn> (Gemma)
    for token in tokenizer.additional_special_tokens:
        if "end" in token.lower() or "eot" in token.lower():
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id not in stop_ids:
                stop_ids.add(token_id)
                logger.info(f"Added stop token from tokenizer: {token} ({token_id})")

    result = sorted(stop_ids)
    logger.info(f"Stop token IDs: {result}")
    return result


def format_config_display(config: dict) -> str:
    """Format config dict for display in chat."""
    lines = ["Current generation config:"]
    for key, value in sorted(config.items()):
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def chat_loop(
    model,
    tokenizer,
    generate_kwargs: dict,
    config: dict,
    logger: logging.Logger,
) -> None:
    """
    Run the interactive chat loop.

    :param model: The loaded language model.
    :param tokenizer: The tokenizer.
    :param generate_kwargs: kwargs for model.generate().
    :param config: Resolved config dict (for /config display).
    :param logger: Logger instance.
    """
    messages = []
    system_prompt = config.get("system_prompt")
    device = next(model.parameters()).device
    stop_token_ids = get_stop_token_ids(model, tokenizer, logger)

    if system_prompt:
        print(f"\nSystem prompt: {system_prompt}")

    print("\nChat started. Type /quit to exit, /clear to reset, /config to show settings.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/quit", "/exit"):
            print("Exiting.")
            break

        if user_input.lower() == "/clear":
            messages.clear()
            print("Conversation history cleared.\n")
            continue

        if user_input.lower() == "/config":
            print(format_config_display(config))
            print()
            continue

        if user_input.lower().startswith("/system "):
            system_prompt = user_input[8:].strip()
            config["system_prompt"] = system_prompt
            messages.clear()
            print(f"System prompt set. Conversation history cleared.\n")
            continue

        # Build message list
        if system_prompt and (not messages or messages[0]["role"] != "system"):
            messages.insert(0, {"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_input})

        # Tokenize with chat template
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        # Generate with streaming
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        print("Assistant: ", end="", flush=True)

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
                streamer=streamer,
                **generate_kwargs,
            )

        # Extract only the generated tokens (skip the input)
        generated_ids = output_ids[0, input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response})

        print()  # Blank line after response


def main() -> None:
    """Main entry point for interactive chat."""
    log_git_state()

    parser = create_arg_parser()
    args = parser.parse_args()

    # Resolve config: CLI > YAML > defaults
    config = resolve_config(args, module_logger)

    describe_config(
        checkpoint=args.checkpoint,
        hf_model=args.hf_model,
        model_name=args.model_name,
        config_file=args.config,
        generation_config=config,
        logger=module_logger,
    )

    # Load model
    model, tokenizer = load_model(
        args=args,
        device=config["device"],
        dtype=config["dtype"],
        logger=module_logger,
    )

    # Build generation kwargs
    generate_kwargs = build_generate_kwargs(config)

    module_logger.info(f"Generation kwargs: {generate_kwargs}")

    # Start chat
    chat_loop(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs=generate_kwargs,
        config=config,
        logger=module_logger,
    )


if __name__ == "__main__":
    main()
