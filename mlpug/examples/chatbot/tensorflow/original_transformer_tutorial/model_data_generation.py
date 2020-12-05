import tensorflow as tf

def create_translation_tf_encode_func(tokenizer_pt, tokenizer_en):

    def encode(lang1, lang2):
        lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

        lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang2.numpy()) + [tokenizer_en.vocab_size + 1]

        return lang1, lang2

    def tf_encode(pt, en):
        result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    return tf_encode


def create_chatbot_tf_encode_func(tokenizer):

    sos_token_idx = tokenizer.vocab_size
    eos_token_idx = tokenizer.vocab_size+1

    def encode(input_sentence, output_sentence):
        input_sentence = [sos_token_idx] + tokenizer.encode(input_sentence.numpy()) + [eos_token_idx]
        output_sentence = [sos_token_idx] + tokenizer.encode(output_sentence.numpy()) + [eos_token_idx]

        return input_sentence, output_sentence

    def tf_encode(input_sentence, output_sentence):
        input_sentence,  output_sentence = tf.py_function(encode,
                                                          [input_sentence, output_sentence],
                                                          [tf.int64, tf.int64])
        input_sentence.set_shape([None])
        output_sentence.set_shape([None])

        return input_sentence, output_sentence

    return tf_encode


def create_length_filter_func(max_length):

    def filter_max_length(x, y):
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

    return filter_max_length
