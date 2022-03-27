import argparse


def create_arg_parser(parser=None, description="Train model using MLPug"):
    if parser is None:
        parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--experiment-name',
        type=str, required=False, default='training-test',
        help='Experiment name')

    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Set to distribute training over multiple computing devices')

    parser.add_argument(
        '--num_devices',
        type=int, required=False, default=-1,
        help='Number of computing devices to use in distributed mode. '
             'Default is all available devices')

    parser.add_argument(
        '--batch_size',
        type=int, required=False, default=64,
        help='Batch size (per process/replica)')

    parser.add_argument(
        '--learning_rate',
        type=float, required=False, default=1e-3,
        help='Learning rate')

    parser.add_argument(
        '--num_epochs',
        type=int, required=False, default=10,
        help='Number of epochs to train')

    parser.add_argument(
        '--progress_log_period',
        type=int, required=False, default=200,
        help='Period in global (batch) iterations before we log the training progress again.')

    parser.add_argument(
        '--seed',
        type=int, required=False, default=0,
        help='Random seed to use to ensure same random split at each restart')

    return parser


def describe_args(args, logger):
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Progress log period: {args.progress_log_period}")
    logger.info(f"Num. training epochs: {args.num_epochs}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Distributed: {args.distributed}")

    num_devices_str = args.num_devices if args.num_devices > 0 else "Use all available"
    logger.info(f"Number of computing devices: {num_devices_str}")
