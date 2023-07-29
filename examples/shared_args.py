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
        '--num-devices',
        type=int, required=False, default=-1,
        help='Number of computing devices to use in distributed mode. '
             'Default is all available devices')

    parser.add_argument(
        '--force-on-cpu',
        action='store_true',
        help='When flag is set, the training process will run on CPU only')

    parser.add_argument(
        '--no-graph-compilation',
        action='store_true',
        help='When flag is set, forward and backward computation graphs will NOT be compiled (i.e. eager mode')

    parser.add_argument(
        '--batch-size',
        type=int, required=False, default=16,
        help='Batch size (per process/replica)')

    parser.add_argument(
        '--batch-chunk-size',
        type=int, required=False, default=None,
        help='Batch chunk size for gradient accumulation')

    parser.add_argument(
        '--learning-rate',
        type=float, required=False, default=1e-3,
        help='Learning rate')

    parser.add_argument(
        '--num-epochs',
        type=int, required=False, default=10,
        help='Number of epochs to train')

    parser.add_argument(
        '--progress-log-period',
        type=int, required=False, default=200,
        help='Period in global (batch) iterations before we log the training progress again.')

    parser.add_argument(
        '--seed',
        type=int, required=False, default=0,
        help='Random seed to use to ensure same random split at each restart')

    parser.add_argument(
        '--remote-debug-ip',
        type=str, required=False, default=None,
        help='Allows to provide <ip>:<port> for remote debugging using PyCharm')

    return parser


def describe_args(args, logger):
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Batch chunk size (for gradient accumulation): {args.batch_chunk_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Progress log period: {args.progress_log_period}")
    logger.info(f"Num. training epochs: {args.num_epochs}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Distributed: {args.distributed}")
    logger.info(f"No graph compilation (eager mode): {args.no_graph_compilation}")

    num_devices_str = args.num_devices if args.num_devices > 0 else "Use all available"
    logger.info(f"Number of computing devices: {num_devices_str}")
    logger.info(f"Force on CPU: {args.force_on_cpu}")

    logger.info(f"Remote debug with PyCharm at: {args.remote_debug_ip}")

