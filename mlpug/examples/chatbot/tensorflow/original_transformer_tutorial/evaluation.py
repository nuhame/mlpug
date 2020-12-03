# Note : Optimized for chatbot usecase

import tensorflow as tf

import matplotlib.pyplot as plt

from mlpug.examples.chatbot.tensorflow.original_transformer_tutorial.training import create_masks


class Evaluator:
    # Note : only deterministic output

    def __init__(self, tokenizer, transformer, max_length):

        self.tokenizer = tokenizer
        self.transformer = transformer
        self.max_length = max_length

        self.start_token = [self.tokenizer.vocab_size]
        self.end_token = [self.tokenizer.vocab_size + 1]

    def __call__(self, inp_sentence):

        inp_sentence = self.start_token + self.tokenizer.encode(inp_sentence) + self.end_token
        encoder_input = tf.expand_dims(inp_sentence, 0)

        decoder_input = [self.tokenizer.vocab_size]
        output = tf.expand_dims(decoder_input, 0)

        for i in range(self.max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.transformer(encoder_input,
                                                              output,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.tokenizer.vocab_size + 1:
                return tf.squeeze(output, axis=0), attention_weights

            # concatenate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights


class AttentionWeightPlotter:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def __call__(self, attention, sentence, result, layer):
        fig = plt.figure(figsize=(16, 8))

        sentence = self.tokenizer.encode(sentence)

        attention = tf.squeeze(attention[layer], axis=0)

        for head in range(attention.shape[0]):
            ax = fig.add_subplot(2, 4, head + 1)

            # plot the attention weights
            ax.matshow(attention[head][:-1, :], cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(sentence) + 2))
            ax.set_yticks(range(len(result)))

            ax.set_ylim(len(result) - 1.5, -0.5)

            ax.set_xticklabels(
                ['<start>'] + [self.tokenizer.decode([i]) for i in sentence] + ['<end>'],
                fontdict=fontdict, rotation=90)

            ax.set_yticklabels([self.tokenizer.decode([i]) for i in result
                                if i < self.tokenizer.vocab_size],
                               fontdict=fontdict)

            ax.set_xlabel('Head {}'.format(head + 1))

        plt.tight_layout()
        plt.show()


class Chatbot:
    # Note : only deterministic output (depends on evaluator)

    def __init__(self, evaluator):

        self.evaluator = evaluator
        self.tokenizer = self.evaluator.tokenizer

        self.attention_weights_plotter = AttentionWeightPlotter(self.tokenizer)

    def __call__(self, sentence, layer_attention_to_plot_idx=None):
        result, attention_weights = self.evaluator(sentence)

        predicted_sentence = self.tokenizer.decode([i for i in result
                                                    if i < self.tokenizer.vocab_size])

        print('User input: {}'.format(sentence))
        print('Bot output: {}'.format(predicted_sentence))

        if layer_attention_to_plot_idx is not None:
            self.attention_weights_plotter(attention_weights,
                                           sentence,
                                           result,
                                           layer_attention_to_plot_idx)
