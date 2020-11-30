import time

import os
import sys

import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

from mlpug.examples.chatbot.conversation_dataset import load_sentence_pair_data

from mlpug.examples.chatbot.tensorflow.original_transformer_tutorial.model_data_generation import \
    filter_max_length, \
    create_chatbot_tf_encode_func

from mlpug.examples.chatbot.tensorflow.original_transformer_tutorial.training import \
    CustomSchedule, \
    create_masks, \
    loss_function

from mlpug.examples.chatbot.tensorflow.original_transformer_tutorial.transformer import Transformer

# ############# SETUP ###############
REMOTE_DEBUG = False

experiment_name = "large-transformer-30112020-lr-1e_4"

data_set_path = "./mlpug/examples/chatbot/data/"

training_set_file = "training-1606433382-cmdc-sentence-pairs-with-voc-max-len-40-min-word-occurance-3-26112020.pickle"
validation_set_file = "validation-1606433382-cmdc-sentence-pairs-with-voc-max-len-40-min-word-occurance-3-26112020.pickle"

num_layers = 12
d_model = 768
dff = 3072
num_heads = 12

dropout_rate = 0.1

log_interval = 20

MAX_LENGTH = 40

BUFFER_SIZE = 20000
BATCH_SIZE = 64

EPOCHS = 200
#####################################

if REMOTE_DEBUG:
    import pydevd
    pydevd.settrace('192.168.178.85', port=57491, stdoutToServer=True, stderrToServer=True)


# ########### SETUP DATA ############

def create_dataset_generator(pairs):

    def generator():
        for pair in pairs:
            yield tuple(pair)

    return generator

train_examples, _unused_ = load_sentence_pair_data(os.path.join(data_set_path, training_set_file))
val_examples, _unused_ = load_sentence_pair_data(os.path.join(data_set_path, validation_set_file))

all_training_sentences = []
for pair in train_examples:
    all_training_sentences += pair

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    all_training_sentences, target_vocab_size=2**13)

vocab_size = tokenizer.vocab_size + 2

tf_encode = create_chatbot_tf_encode_func(tokenizer)

train_dataset = tf.data.Dataset.from_generator(
    create_dataset_generator(train_examples),
    (tf.string, tf.string),
    (tf.TensorShape([]), tf.TensorShape([])))

train_dataset = train_dataset.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = tf.data.Dataset.from_generator(
    create_dataset_generator(val_examples),
    (tf.string, tf.string),
    (tf.TensorShape([]), tf.TensorShape([])))

val_dataset = val_dataset.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

#####################################


# ######## SETUP OPTIMIZER ##########
# learning_rate = CustomSchedule(d_model)

learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#####################################

# ########## SETUP MODEL ############
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          vocab_size, vocab_size,
                          pe_input=vocab_size,
                          pe_target=vocab_size,
                          rate=dropout_rate)

checkpoint_path = os.path.join("../trained-models/", experiment_name)
#####################################


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


def evaluate(inp_sentence):
    start_token = [tokenizer.vocab_size]
    end_token = [tokenizer.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer.encode(sentence)

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
            ['<start>'] + [tokenizer.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer.decode([i]) for i in result
                            if i < tokenizer.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = tokenizer.decode([i for i in result
                                           if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % log_interval == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            sys.stdout.flush()

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
