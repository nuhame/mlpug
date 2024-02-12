import numpy as np

import tensorflow as tf

from examples.persona_chatbot.tensorflow.collation import MultipleChoiceCollator

samples = [
    [
        (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
            3,
            0
        ),
        (
            np.array([7, 8]),
            np.array([9, 10]),
            np.array([11, 12]),
            2,
            1
        ),
    ],
    [
        (
            np.array([11, 12, 13, 14, 15]),
            np.array([16, 17, 18, 19, 20]),
            np.array([21, 22, 23, 24, 25]),
            5,
            1
        ),
        (
            np.array([7]),
            np.array([9]),
            np.array([10]),
            1,
            0
        ),
    ]
]


choice_collator = MultipleChoiceCollator(
    pad_token_idx=0,
    max_sequence_length=10)


def sample_generator():
    for sample in samples:
        yield choice_collator(sample)


dataset = tf.data.Dataset\
            .from_generator(
                sample_generator,
                output_types=(tf.int64, tf.int64, tf.int64, tf.int64, tf.int8))

for sample in dataset:
    print(sample)

