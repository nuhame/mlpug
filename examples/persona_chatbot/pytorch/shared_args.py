import logging

from examples.persona_chatbot.shared_args import create_arg_parser as create_base_arg_parser
from examples.persona_chatbot.shared_args import describe_config as describe_base_config


def create_arg_parser(description="Finetune GPT2 as persona aware chatbot using PyTorch"):
    parser = create_base_arg_parser(description=description)

    parser.add_argument(
        '--num-dataloader-workers',
        type=int, required=False, default=2,
        help='Number of dataloader workers.')

    return parser


def describe_config(
    num_dataloader_workers: int,
    logger: logging.Logger | None = None,
    **kwargs,
) -> None:
    """
    Log PyTorch persona chatbot training configuration.

    :param num_dataloader_workers: Number of dataloader workers.
    :param logger: Logger to use.
    :param kwargs: Additional arguments passed to base describe_config.
    """
    describe_base_config(logger=logger, **kwargs)

    logger.info(f"  num_dataloader_workers: {num_dataloader_workers}")
