import argparse

parser = argparse.ArgumentParser(description='Train reference chatbot on processed sentence pair data')

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

