import torch
import torch.nn as nn

from mlpug.examples.chatbot.conversation_parsing import normalizeString
from mlpug.examples.chatbot.conversation_dataset import indexesFromSentence


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, SOS_token, device, select_token_func=None):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.SOS_token = SOS_token

        self.device = device

        self.select_token_func = select_token_func
        if not self.select_token_func:
            self.select_token_func = lambda output: torch.max(output, dim=2)

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        # decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_n_layers = self.decoder.n_layers
        decoder_hidden = 0.5 * (encoder_hidden[:decoder_n_layers, 0, :, :] + encoder_hidden[:decoder_n_layers, 1, :, :])

        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * self.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = self.select_token_func(decoder_output)

            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length, EOS_token, device):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence, EOS_token)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Batch dimension should be second (in order for portioning over multiple GPUs)
    # TODO : that means unsqueeze(1)
    lengths = lengths.unsqueeze(0)

    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc, max_length, EOS_token, device):
    input_sentence = ''
    while True:
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == '*q*' or input_sentence == '*quit*':
                break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, max_length, EOS_token, device)

            # output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]

            # Format and print response sentence
            response = []
            for word in output_words:
                if word == 'EOS':
                    break
                response.append(word)

            print('Bot:', ' '.join(response))

        except KeyError:
            print("Error: Encountered unknown word.")
