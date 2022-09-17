from typing import List

import numpy as np
import tensorflow as tf

from mlpug.base import Base

from examples.chatbot.datasets.conversations import ConversationSample


class CollatedSampleGenerator:

    def __init__(self, multiple_choice_samples, choice_collator):
        self._multiple_choice_samples = multiple_choice_samples
        self._choice_collator = choice_collator

    def __call__(self):
        for sample in self._multiple_choice_samples:
            yield self._choice_collator(sample)


class MultipleChoiceCollator(Base):

    def __init__(self, pad_token_idx, max_sequence_length=None, ignore_label_idx=-100, name=None):
        super().__init__(pybase_logger_name=name)

        self._pad_token_idx = pad_token_idx
        self._ignore_label_idx = ignore_label_idx
        self._max_sequence_length = max_sequence_length

    def __call__(self, sample_choices: List[ConversationSample]):
        """

        :param sample_choices:

                List with num_choices conversation samples:
                (
                    input_ids,        # 0: input_ids
                    token_type_ids,   # 1: token_type_ids
                    token_label_ids,  # 2: labels
                    last_token_idx,   # 3: mc_token_ids
                    reply_class       # 4: 0 = not real reply, 1 = real reply. ==> mc_labels
                )

        :return:
        """

        num_choices = len(sample_choices)

        max_seq_len = self._max_sequence_length
        if max_seq_len is None:
            max_seq_len = max([len(sample_choices[choice_idx][0]) for choice_idx in range(num_choices)])

        mc_input_ids = self._pad_token_idx*np.ones([num_choices, max_seq_len], dtype=np.int64)
        mc_token_type_ids = self._pad_token_idx*np.ones([num_choices, max_seq_len], dtype=np.int64)
        mc_token_labels_ids = self._ignore_label_idx*np.ones([num_choices, max_seq_len], dtype=np.int64)

        mc_last_token_idx = np.zeros([num_choices], dtype=np.int64)

        for c_idx, choice in enumerate(sample_choices):
            input_ids, token_type_ids, token_label_ids, last_token_idx, _ = choice

            mc_input_ids[c_idx, :len(input_ids)] = np.array(input_ids)

            mc_token_type_ids[c_idx, :len(token_type_ids)] = np.array(token_type_ids)
            mc_token_labels_ids[c_idx, :len(token_label_ids)] = np.array(token_label_ids)

            mc_last_token_idx[c_idx] = last_token_idx

        reply_class_idx = np.array([choice[4] for choice in sample_choices].index(1), dtype=np.int8)

        return tf.convert_to_tensor(mc_input_ids), \
            tf.convert_to_tensor(mc_token_type_ids), \
            tf.convert_to_tensor(mc_token_labels_ids), \
            tf.convert_to_tensor(mc_last_token_idx), \
            tf.convert_to_tensor(reply_class_idx)
