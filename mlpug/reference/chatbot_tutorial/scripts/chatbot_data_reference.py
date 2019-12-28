#
# This script contains the code, extracted from the chatbot_reference.py script, to parse and process the
# Cornell Movie Dialogs Corpus (provided through commandline argument `--corpus-path`).
#
# The resulting sentence pairs and vocabulary is stored in the given `--output-file`
#

from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import pickle

from mlpug.reference.chatbot_tutorial.conversation_parsing import \
    printLines, \
    process_corpus_and_extract_sentence_pairs, \
    loadPrepareData, \
    trimRareWords

# import pydevd
# pydevd.settrace('192.168.178.8', port=57491, stdoutToServer=True, stderrToServer=True)

parser = argparse.ArgumentParser(description='Reference implementation to convert the Cornell Movie Dialog Corpus data '
                                             'in to a data structure we can use.')

parser.add_argument(
    '--corpus-path',
    type=str, required=True,
    help='path to the corpus data')

parser.add_argument(
    '--output-file',
    type=str, required=True,
    help='path to output file')

parser.add_argument(
    '--max-sequence-length',
    type=int, default=10,
    help='Max sequence length')

parser.add_argument(
    '--min-word-count',
    type=int, default=3,
    help='Minimum vocabulary word count')

args = parser.parse_args()


##################################################
#
# [START] Setup
#
##################################################

# ######## Conversations dataset parsing #########
corpus_name = "cornell-movie-dialogs-corpus"
corpus_path = args.corpus_path

sentence_pair_csv_delimiter = '\t'
raw_sentence_pair_filename = 'formatted_movie_lines.txt'

output_file_path = args.output_file

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = args.max_sequence_length  # 50  # 10  # Maximum sentence length to consider

MIN_COUNT = args.min_word_count  # Minimum word count threshold for trimming
##################################################

##################################################
#
# [END] Setup
#
##################################################

# # Load & Preprocess Data
printLines(os.path.join(corpus_path, "movie_lines.txt"))

# ## Extract sentence pairs and save to disk.

# Define path to new file
sentence_pairs_path = os.path.join(corpus_path, raw_sentence_pair_filename)
process_corpus_and_extract_sentence_pairs(corpus_path, sentence_pairs_path, sentence_pair_csv_delimiter)

# Print a sample of lines
print("\nSample lines from file:")
printLines(sentence_pairs_path)

# ## Filter out long sentences and create vocabulary

# Load/Assemble voc and pairs
voc, pairs = loadPrepareData(corpus_name,
                             sentence_pairs_path,
                             MAX_LENGTH,
                             PAD_token, SOS_token, EOS_token)

# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

# ## Load sentence pairs and remove pairs with rare words

# Trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)

with open(args.output_file, 'wb') as f:
    pickle.dump({
        'pairs': pairs,
        'voc': voc,
        'args': args
    }, f)
