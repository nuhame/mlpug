import os
import argparse

import time

import random

import pickle

import numpy as np

from basics.logging import get_logger

logger = get_logger(os.path.basename(__file__))

parser = argparse.ArgumentParser(description='Split the given dataset in to multiple subsets, as defined by '
                                             '--split and --set-names')

parser.add_argument(
    '--conversations-dataset',
    type=str, required=True,
    help='path to the conversations dataset')

parser.add_argument(
    '--split',
    type=float, nargs='+', required=False, default=[0.7, 0.3],
    help='Names of sets to split the original set in to')

parser.add_argument(
    '--set-names',
    type=str, nargs='+', required=False, default=['training', 'validation'],
    help='Names of sets to split the original set in to')


args = parser.parse_args()


def load_conversations_data(conversations_dataset_file):
    logger.info(f'\nLoading conversations dataset file : {conversations_dataset_file}')

    with open(conversations_dataset_file, 'rb') as f:
        data = pickle.load(f)
        return data


def save_conversations_data(conversations_dataset_file, pairs, original_metadata, voc=None):
    logger.info(f'\nSaving conversations dataset file : {conversations_dataset_file}')
    logger.info(f'\nData set size : {len(pairs)}')

    with open(conversations_dataset_file, 'wb') as f:

        conversation_data = {
            'pairs': pairs,
            'args': args,
            'script': os.path.basename(__file__),
            'original_metadata': original_metadata
        }

        if voc is not None:
            conversation_data['voc'] = voc

        pickle.dump(conversation_data, f)


def print_samples(pairs):
    # Print some conversations to validate
    for pair in pairs[:10]:
        logger.info(f"\n{pair[0]}\n{pair[1]}\n")


split = np.array(args.split)
logger.info(f'split : {split}')
if split.sum() != 1.0:
    logger.error("The split ratios should sum up to one")
    exit(-1)

set_names = args.set_names
logger.info(f'set_names : {set_names}')
if len(split) != len(set_names):
    logger.error("The number or set names given should be equal to the number of split ratios given")
    exit(-2)

conversations_dataset_file = args.conversations_dataset
data = load_conversations_data(conversations_dataset_file)

pairs = data['pairs']
voc = data['voc'] if 'voc' in data else None

original_dataset_metadata = data.copy()
del original_dataset_metadata['pairs']

if voc is not None:
    del original_dataset_metadata['voc']

# TODO : This code should be DRYed up
num_conversations = len(pairs)
logger.info(f'Number of conversations in dataset : {num_conversations}')
indices = list(range(num_conversations))
random.shuffle(indices)

set_lengths = list(map(lambda r: int(r*num_conversations), split))
rest = num_conversations - sum(set_lengths)
set_lengths[-1] += rest

if sum(set_lengths) != num_conversations:
    logger.error("Unexpected : length of subsets is not equal to given dataset")
    exit(-3)

data_path, base_filename = os.path.split(conversations_dataset_file)

current_time = int(time.time())

start_idx = 0
for name, l in zip(set_names, set_lengths):
    logger.info(f'Creating subset {name} of size {l} ...')
    dataset_filename = f'{name}-{current_time}-{base_filename}'
    dataset_path = os.path.join(data_path, dataset_filename)

    indices_subset = indices[start_idx:start_idx+l]

    conversations_subset = [pairs[i] for i in indices_subset]
    print_samples(conversations_subset)

    save_conversations_data(dataset_path, conversations_subset, original_dataset_metadata, voc)

    start_idx += l
