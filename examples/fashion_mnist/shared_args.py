import logging

from examples.shared_args import create_arg_parser as create_base_arg_parser
from examples.shared_args import describe_config as describe_base_config


def create_arg_parser(description="Train on Fashion MNIST dataset using MLPug"):
    parser = create_base_arg_parser(description=description)

    parser.add_argument(
        '--hidden_size',
        type=int, required=False, default=128,
        help='Model hidden size')

    return parser


def describe_config(
    hidden_size: int,
    logger: logging.Logger | None = None,
    **kwargs,
) -> None:
    """
    Log Fashion MNIST training configuration.

    :param hidden_size: Model hidden size.
    :param logger: Logger to use.
    :param kwargs: Additional arguments passed to base describe_config.
    """
    describe_base_config(logger=logger, **kwargs)

    logger.info(f"  hidden_size: {hidden_size}")
