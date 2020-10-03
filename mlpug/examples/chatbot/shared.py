import argparse


def create_argument_parser(description='Train reference chatbot on processed sentence pair data'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--experiment-name',
        type=str, required=True,
        help='Name of experiment, used as file name base for checkpoints')

    parser.add_argument(
        '--dataset-path',
        type=str, required=True,
        help='Path to dataset files')

    parser.add_argument(
        '--base-dataset-filename',
        type=str, required=True,
        help='Shared filename postfix for training and validation set')

    parser.add_argument(
        '--training-checkpoint',
        type=str, required=False, default=None,
        help='Path to training checkpoint file')

    parser.add_argument('--float16',
                        action='store_true',
                        required=False,
                        help='Provide this flag to use Mixed Precision Training')

    parser.add_argument(
        '--num-gpus',
        type=int, required=False, default=None,
        help='The number of GPUs to use in one process (DataParallel will be used)')

    parser.add_argument(
        '--attention-model',
        type=str, required=False, default='dot',
        help='Attention model type: "dot", "general" or "concat"')

    parser.add_argument(
        '--embedding-size',
        type=int, required=False, default=256,
        help='Input embedding size')

    parser.add_argument(
        '--encoder-state-size',
        type=int, required=False, default=1000,
        help='Encoder state size')

    parser.add_argument(
        '--num-encoder-layers',
        type=int, required=False, default=2,
        help='Encoder state size')

    parser.add_argument(
        '--num-decoder-layers',
        type=int, required=False, default=2,
        help='Decoder state size')

    parser.add_argument(
        '--dropout',
        type=float, required=False, default=0.2,
        help='Drop out rate')

    parser.add_argument(
        '--batch-size',
        type=int, required=False, default=32,
        help='Batch size')

    parser.add_argument(
        '--gradient-clipping',
        type=float, required=False, default=50.0,
        help='Gradient clipping threshold')

    parser.add_argument(
        '--teacher-forcing-ratio',
        type=float, required=False, default=1.0,
        help='How often teacher forcing should be applied, compared to not applying it')

    parser.add_argument(
        '--learning-rate',
        type=float, required=False, default=0.0001,
        help='Base learning rate')

    parser.add_argument(
        '--decoder-learning-rate-ratio',
        type=float, required=False, default=5.0,
        help='Decoder LR ratio, such that the decoder LR is ratio * base LR')

    parser.add_argument(
        '--num-epochs',
        type=int, required=False, default=50,
        help='Number of epochs to train')

    parser.add_argument(
        '--progress-logging-period',
        type=int, required=False, default=200,
        help='Period in global (batch) iterations before we log the training progress again.')

    parser.add_argument(
        '--seed',
        type=int, required=False, default=0,
        help='Random seed to use to ensure same random split at each restart')

    return parser
