
# The is a condensed version of the original PyTorch chatbot tutorial. All the code has been put in separate files as
# much as possible.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import random
import os

from mlpug.examples.chatbot.conversation_parsing import \
    printLines, \
    process_corpus_and_extract_sentence_pairs, \
    loadPrepareData, \
    trimRareWords

from mlpug.examples.chatbot.pytorch.original_chatbot_tutorial.model_data_generation import batch2TrainData

from mlpug.examples.chatbot.pytorch.original_chatbot_tutorial.seq2seq import EncoderRNN, LuongAttnDecoderRNN

from mlpug.examples.chatbot.pytorch.original_chatbot_tutorial.training import Seq2SeqTrainModel, trainIters

from mlpug.examples.chatbot.pytorch.original_chatbot_tutorial.evaluation import GreedySearchDecoder, evaluateInput

# import pydevd
# pydevd.settrace('192.168.178.8', port=57491, stdoutToServer=True, stderrToServer=True)

USE_CUDA = torch.cuda.is_available()

print(f"Use CUDA : {USE_CUDA}")

device = torch.device("cuda" if USE_CUDA else "cpu")

##################################################
#
# [START] Setup
#
##################################################

# ######## Conversations dataset parsing #########
# Note : FS : Updated to use my own copy already downloaded a long time ago.
corpus_name = "cornell-movie-dialogs-corpus"
corpus_path = os.path.join("../data", corpus_name)

sentence_pair_csv_delimiter = '\t'
sentence_pair_filename = 'formatted_movie_lines.txt'

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 40  # 50  # 10  # Maximum sentence length to consider

MIN_COUNT = 3  # Minimum word count threshold for trimming
##################################################

# ############ Model configuration ###############
model_name = 'cb_model'
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'
encoder_state_size = 1920  # 1600  # 500
embedding_size = 256  # encoder_state_size
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
##################################################

# ########### Training/optimization ##############
model_checkpoint_dir = os.path.join("checkpoints")
checkpoint_iter_to_load = -1  # 6500

USE_MIXED_PRECSION = False
MIXED_PRECISION_OPT_LEVEL = 'O1'

# num_gpus = 1 if USE_CUDA else -1
num_gpus = torch.cuda.device_count() if USE_CUDA else -1

batch_size = 12  # 16  # 64
if num_gpus > 1:
    print(f"Parallelizing to {num_gpus} ...")

    batch_size = num_gpus * batch_size

clip = 50.0
teacher_forcing_ratio = 1.0

# 0.0003 is average between original LR for encoder and decoder
learning_rate = 0.0003  # 0.0001
decoder_learning_ratio = 5.0  # TODO : In this version, not used
n_iteration = 92000  # 4000  # 64000
print_every = 100
save_every = 12000

##################################################

##################################################
#
# [END] Setup
#
##################################################

if USE_MIXED_PRECSION:
    from apex import amp

# # Load & Preprocess Data

printLines(os.path.join(corpus_path, "movie_lines.txt"))

# ## Extract sentence pairs and save to disk.

# Define path to new file
sentence_pairs_path = os.path.join(corpus_path, sentence_pair_filename)

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

# # Data for models | Batching

# Example for validation
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)], PAD_token, EOS_token)
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)

# # Build Model

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
if checkpoint_iter_to_load >= 0:
    loadFilename = os.path.join(model_checkpoint_dir, model_name, corpus_name,
                                '{}-{}-{}-{}'.format(embedding_size,
                                                     encoder_state_size,
                                                     encoder_n_layers,
                                                     decoder_n_layers),
                                '{}_checkpoint.tar'.format(checkpoint_iter_to_load))


# Load model if a loadFilename is provided
checkpoint = None
if loadFilename:
    print(f'Loading checkpoint {loadFilename} ...')

    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

    embedding_sd = checkpoint['embedding']
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    optimizer_sd = checkpoint['opt']

    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, embedding_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(encoder_state_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, encoder_state_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

train_model = Seq2SeqTrainModel(encoder, decoder)
# Use appropriate device
train_model = train_model.to(device)

print('Models built and ready to go!')

# # Train Model

# Ensure dropout layers are in train mode
train_model.train()

# Initialize optimizers
# TODO : what we miss now is seperate decoder and encoder learning rate (based on decoder_learning_ratio)
print('Building optimizer ...')
optimizer = optim.Adam(train_model.parameters(), lr=learning_rate)

if USE_MIXED_PRECSION:
    keep_batchnorm_fp32 = None
    if MIXED_PRECISION_OPT_LEVEL == 'O2' or MIXED_PRECISION_OPT_LEVEL == 'O3':
        keep_batchnorm_fp32 = True

    model, optimizer = amp.initialize(
       train_model, optimizer, opt_level=MIXED_PRECISION_OPT_LEVEL,
       keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")

if num_gpus > 1:
    print("We are using", num_gpus, "GPUs!")
    train_model = nn.DataParallel(train_model, dim=1)

if loadFilename:
    optimizer.load_state_dict(optimizer_sd)

if num_gpus <= 1:
    # If you have cuda, configure cuda to call
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

# Run training iterations
print("Starting Training!")

trainIters(model_name, voc, pairs, PAD_token, EOS_token, SOS_token, train_model, optimizer,
           model_checkpoint_dir, n_iteration, batch_size, teacher_forcing_ratio, USE_MIXED_PRECSION, clip,
           print_every, save_every, corpus_name, checkpoint, device)

# # Evaluate Model

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder, SOS_token, device)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc, MAX_LENGTH, EOS_token, device)
