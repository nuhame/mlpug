import os
import argparse
import pickle

import torch
import torch.nn as nn

from mlpug.reference.chatbot_tutorial.evaluation import GreedySearchDecoder, evaluateInput

from mlpug.reference.chatbot_tutorial.seq2seq import EncoderRNN, LuongAttnDecoderRNN

from basics.logging import get_logger

# import pydevd
# pydevd.settrace('192.168.178.7', port=57491, stdoutToServer=True, stderrToServer=True)

logger_name = os.path.basename(__file__)
logger = get_logger(logger_name)

parser = argparse.ArgumentParser(description='Evaluate trained reference chat bot')

parser.add_argument(
    '--model-checkpoint',
    type=str, help='Path to model checkpoint file')

parser.add_argument(
    '--vocabulary-file',
    type=str, help='Path to file containing the vocabulary')

parser.add_argument(
    '--token-selection-method',
    type=str, help='Selection method for token from output distribution : \'max\' or \'random-sample\'. '
                   'In case of random-sample, you can also set --sampling-temp')
parser.add_argument(
    '--sampling-temp',
    type=float, required=False, default=0.2,
    help='Random sampling temperature, used when --token-selection-method \'random-sample\' is selected.')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

logger.info(f"Using CUDA? {use_cuda}")

# We need to store the setup with the model checkpoint
##################################################
#
# [START] Setup
#
##################################################

# ############ vocabulary #############
# Default word tokens


PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

##################################################

# ############ Model configuration ###############
# attn_model = 'dot'
attn_model = 'general'
# attn_model = 'concat'
encoder_state_size = 1000  # 1920  # 1600  # 500
embedding_size = 256  # encoder_state_size  # 256  # encoder_state_size
encoder_n_layers = 2
decoder_n_layers = 2
##################################################

# ################# Evaluation ###################
max_seq_length = 40
##################################################

##################################################
#
# [END] Setup
#
##################################################


def load_vocabulary(vocabulary_file):
    logger.info(f'Loading vocabulary file : {vocabulary_file}')

    with open(vocabulary_file, 'rb') as f:
        data = pickle.load(f)
        return data['voc']


def random_sample(p):
    return sample(p, args.sampling_temp)


def sample(p, temperature=1.0):
    """
    TODO : make it work for batch_size > 0

    :param p: shape (1, 1, num_tokens)
    :param temperature:
    :return:
    """
    p = torch.pow(p, 1.0 / temperature)
    p_sum = torch.sum(p, dim=2)
    p = p / p_sum[:, :, None]

    token_idx = torch.multinomial(p.view(-1), 1)
    token_p = torch.gather(p, 2, token_idx.view(1, -1, 1))

    token_idx = token_idx.view(1, -1)
    token_p = token_p.view(1, -1)

    return token_p, token_idx


voc = load_vocabulary(args.vocabulary_file)

logger.info(f'Building model ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, embedding_size)

# Initialize encoder & decoder models
encoder = EncoderRNN(encoder_state_size, embedding, encoder_n_layers, dropout=0)
decoder = LuongAttnDecoderRNN(attn_model, embedding, encoder_state_size, voc.num_words, decoder_n_layers, dropout=0)

logger.info(f'Loading model checkpoint ...')
checkpoint = torch.load(args.model_checkpoint)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.to(device)
decoder.to(device)

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

select_token_func = None
logger.info(f'Token selection method : {args.token_selection_method}')
if args.token_selection_method == 'max':
    pass
elif args.token_selection_method == 'random-sample':
    logger.info(f'Sampling temperature : {args.sampling_temp}')
    select_token_func = random_sample

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder, SOS_token, device, select_token_func)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, voc, max_seq_length, EOS_token, device)
