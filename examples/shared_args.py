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
        type=int, required=False, default=None,
        help='Number of computing devices to use in distributed mode. '
             'Default is all available devices')

    parser.add_argument(
        '--force-on-cpu',
        action='store_true',
        help='When flag is set, the training process will run on CPU only')

    parser.add_argument(
        '--eager-mode',
        action='store_true',
        help='When flag is set, forward and backward computation graphs will NOT be compiled (i.e. eager mode)')

    parser.add_argument(
        '--use-mixed-precision',
        action='store_true',
        help='When flag is set, mixed precision will be applied during training')

    parser.add_argument(
        '--batch-size',
        type=int, required=False, default=16,
        help='Effective batch size for optimization (per process/replica)')

    parser.add_argument(
        '--micro-batch-size',
        type=int, required=False, default=None,
        help='Micro-batch size - what fits in memory. When set, enables gradient '
             'accumulation: accumulation_steps = batch_size / micro_batch_size')

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
    """
    Log parsed arguments (legacy function for compatibility).

    :param args: Parsed argparse namespace.
    :param logger: Logger to use.
    """
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Micro-batch size (for gradient accumulation): {args.micro_batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Progress log period: {args.progress_log_period}")
    logger.info(f"Num. training epochs: {args.num_epochs}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Distributed: {args.distributed}")
    logger.info(f"Eager mode (no graph compilation): {args.eager_mode}")
    logger.info(f"Use mixed precision: {args.use_mixed_precision}")

    num_devices_str = args.num_devices if args.num_devices is not None and args.num_devices > 0 else "Use all available"
    logger.info(f"Number of computing devices: {num_devices_str}")
    logger.info(f"Force on CPU: {args.force_on_cpu}")

    logger.info(f"Remote debug with PyCharm at: {args.remote_debug_ip}")


def describe_config(
    experiment_name: str,
    batch_size: int,
    micro_batch_size: int | None,
    learning_rate: float,
    num_epochs: int,
    seed: int,
    progress_log_period: int,
    eager_mode: bool,
    use_mixed_precision: bool,
    force_on_cpu: bool,
    logger,
    **kwargs,
) -> None:
    """
    Log training configuration.

    This function logs the common training arguments. Task-specific describe_config
    functions should call this and then log their additional arguments.

    :param experiment_name: Experiment name.
    :param batch_size: Effective batch size per device.
    :param micro_batch_size: Micro-batch size for gradient accumulation.
    :param learning_rate: Learning rate.
    :param num_epochs: Number of training epochs.
    :param seed: Random seed.
    :param progress_log_period: Logging frequency in batch steps.
    :param eager_mode: Whether to disable torch.compile.
    :param use_mixed_precision: Whether to use AMP.
    :param force_on_cpu: Whether to force CPU training.
    :param logger: Logger to use.
    :param kwargs: Additional arguments (absorbed, not logged).
    """
    logger.info("Configuration:")
    logger.info(f"  experiment_name: {experiment_name}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  micro_batch_size: {micro_batch_size}")
    logger.info(f"  learning_rate: {learning_rate}")
    logger.info(f"  num_epochs: {num_epochs}")
    logger.info(f"  seed: {seed}")
    logger.info(f"  progress_log_period: {progress_log_period}")
    logger.info(f"  eager_mode: {eager_mode}")
    logger.info(f"  use_mixed_precision: {use_mixed_precision}")
    logger.info(f"  force_on_cpu: {force_on_cpu}")

