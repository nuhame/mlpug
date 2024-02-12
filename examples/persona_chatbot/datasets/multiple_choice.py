from typing import Optional, Callable, List, Dict
from collections.abc import Sequence
import os

import random

from basics.logging_utils import log_exception
from basics.logging import get_logger

from mlpug.base import Base

from examples.persona_chatbot.datasets.conversations import ConversationSample


module_logger = get_logger(os.path.basename(__file__))


try:
    from tqdm import tqdm
except Exception as e:
    log_exception(module_logger, "Please `pip install tqdm`", e)


def max_sequence_length_in(conversation_choices):
    """

    :param conversation_choices: List of Tuples:
        (
            input_ids,        # input_ids
            token_type_ids,   # token_type_ids
            token_label_ids,  # labels
            last_token_idx,   # mc_token_ids
            reply_class       # 0 = not real reply, 1 = real reply. ==> mc_labels
        )

    :return:
    """
    # First item in conversation_choice tuple are the input_ids
    return max([len(conversation_choice[0]) for conversation_choice in conversation_choices])


class MultipleChoiceGenerator(Base, Sequence):

    def __init__(self,
                 persona_chat_dataset: List[Dict],
                 sample_factory: Callable,
                 max_num_samples: Optional[int] = None,
                 num_choices: Optional[int] = None,
                 shuffle: bool = True,
                 name: Optional[str] = None):
        """
        A MultipleChoiceGenerator can generate samples of multiple conversation choices for any sample_idx.

        Usage:

        mc_generator = MultipleChoiceGenerator(persona_chat_dataset, sample_factory)
        mc_generator.initialize()

        multiple_conversation_choices = mc_factory[10]
        # depending on the sample factory:
        # input_ids of third conversation choice
        input_ids = multiple_conversation_choices[2][0]

        :param persona_chat_dataset: dataset structured like
                                     https://huggingface.co/datasets/bavard/personachat_truecased

        :param sample_factory: Callable:
                               sample_factory(personality: List[str],
                                              chat_history: List[str],
                                              candidate_reply: str,
                                              is_real_reply: bool) -> model input sample data

        :param max_num_samples: Optional int, when not provided all samples will be used
        :param num_choices: Optional int, when not provided all reply candidates are used

        :param shuffle: Boolean. If true the dataset will be randomly shuffled

        :param name: Optional sample generator name
        """
        super().__init__(pybase_logger_name=name)

        self._persona_chat_dataset = persona_chat_dataset
        self._sample_factory = sample_factory

        self._max_num_samples = max_num_samples
        self._num_choices = num_choices

        self._shuffle = shuffle

        self._conversation_metadata = None

    def initialize(self):
        self._log.info("Generating chat sample metadata ...")
        self._generate_conversation_metadata()

        if self._shuffle:
            random.shuffle(self._conversation_metadata)

        if self._max_num_samples is not None:
            num_samples = len(self._conversation_metadata)
            max_num_samples = min(num_samples, self._max_num_samples)
            self._log.info(f"Reducing dataset to {max_num_samples} samples, original size: {num_samples}")

            self._conversation_metadata = self._conversation_metadata[:max_num_samples]

    def __len__(self):
        return len(self._conversation_metadata)

    def __getitem__(self, item) -> List[ConversationSample]:
        conversation_metadata = self._conversation_metadata[item]

        num_choices = conversation_metadata['num_choices']

        multiple_conversation_choices = [self._get_conversation_data(conversation_metadata, choice_idx)
                                         for choice_idx in range(num_choices)]

        multiple_conversation_choices = [self._sample_factory(*conv_data)
                                         for conv_data in multiple_conversation_choices]

        return multiple_conversation_choices

    def _generate_conversation_metadata(self):
        # TODO: add persona permutations

        self._conversation_metadata = []
        for chat_idx, chat_data in enumerate(self._persona_chat_dataset):
            num_candidates = len(chat_data["candidates"])

            num_choices = num_candidates
            if self._num_choices is not None:
                num_choices = min(self._num_choices, num_candidates)
                if num_choices < self._num_choices:
                    # This should not happen (and is not expected to using the persona dataset)
                    self._log.warning(f"Conversation has less than {self._num_choices} candidates, "
                                      f"using {num_choices} available candidates.")

            self._conversation_metadata += [{
                'chat_idx': chat_idx,
                'num_choices': num_choices,
                # Use the last `num_choices` choices
                'candidate_offset': num_candidates - num_choices
            }]

    def _get_conversation_data(self, conversation_metadata, choice_idx):
        chat_data = self._persona_chat_dataset[conversation_metadata['chat_idx']]

        num_choices = conversation_metadata['num_choices']
        offset = conversation_metadata['candidate_offset']

        is_real_reply = choice_idx == num_choices - 1

        return (chat_data['personality'],
                chat_data['history'],
                chat_data['candidates'][choice_idx+offset],  # candidate reply
                is_real_reply)
