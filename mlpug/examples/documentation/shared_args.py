import argparse


def base_argument_set(description='Train model'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--batch-size',
        type=int, required=False, default=64,
        help='Batch size (per process/replica)')

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

    return parser
