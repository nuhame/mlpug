from mlpug.examples.shared_args import create_arg_parser as create_base_arg_parser
from mlpug.examples.shared_args import describe_args as describe_base_args


def create_arg_parser(description="Train on Fashion MNIST dataset using MLPug"):
    parser = create_base_arg_parser(description=description)

    parser.add_argument(
        '--hidden_size',
        type=int, required=False, default=128,
        help='Model hidden size')

    return parser


def describe_args(args, logger):
    describe_base_args(args, logger)

    logger.info(f"Model hidden size: {args.hidden_size}")
