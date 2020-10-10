import tensorflow as tf

MAX_LENGTH = 40


def create_tf_encode_func(tokenizer_pt, tokenizer_en):

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


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)
