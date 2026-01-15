from examples.shared_args import create_arg_parser as create_base_arg_parser
# TODO: Update to use describe_config instead of describe_args
# This requires updating train.py as well
from examples.shared_args import describe_config as describe_base_config


def create_arg_parser(description="Train on Fashion MNIST dataset using MLPug"):
    parser = create_base_arg_parser(description=description)

    parser.add_argument(
        '--hidden_size',
        type=int, required=False, default=128,
        help='Model hidden size')

    return parser


# Legacy function - TODO: migrate to describe_config pattern
def describe_args(args, logger):
    # Convert args to kwargs for describe_config
    config = vars(args).copy()
    # Handle no_loss_scaling → convert but describe_config now expects no_loss_scaling
    describe_base_config(logger=logger, **config)

    logger.info(f"  hidden_size: {args.hidden_size}")
