from typing import Optional, Callable, List, Dict

import random

from mlpug.base import Base


class ChatSampleGenerator(Base):

    def __init__(self,
                 persona_chat_dataset: List[Dict],
                 sample_factory: Callable,
                 shuffle: bool = True,
                 name: Optional[str] = None):
        """
        Usage:

        sample_generator = ChatSampleGenerator(persona_chat_dataset, sample_factory)

        for model_input_sample in sample_generator:
            # Do stuff


        :param persona_chat_dataset: dataset structured like
                                     https://huggingface.co/datasets/bavard/personachat_truecased

        :param sample_factory: Callable:
                               sample_factory(personality: List[str],
                                              chat_history: List[str],
                                              candidate_reply: str,
                                              is_real_reply: bool) -> model input sample data

        :param shuffle: Boolean. If true the chat samples will be randomly shuffled

        :param name: Optional sample generator name
        """
        super().__init__(pybase_logger_name=name)

        self._persona_chat_dataset = persona_chat_dataset
        self._sample_factory = sample_factory

        self._shuffle = shuffle

        self._sample_metadata = None
        self._sample_metadata_iter = None

    def initialize(self):
        self._log.info("Generating chat sample metadata ...")
        self._generate_sample_metadata()

        if self._shuffle:
            random.shuffle(self._sample_metadata)

    def __iter__(self):
        self._sample_metadata_iter = iter(self._sample_metadata)

        return self

    def __next__(self):
        sample_metadata = next(self._sample_metadata_iter)
        
        sample_data = self._get_sample_data(sample_metadata)

        return self._sample_factory(*sample_data)

    def _generate_sample_metadata(self):
        self._sample_metadata = []

        for chat_idx, chat_data in enumerate(self._persona_chat_dataset):
            num_candidates = chat_data["candidates"]

            self._sample_metadata += [{
                'chat_idx': chat_idx,
                'candidate_idx': candidate_idx,
                'is_real_continuation': candidate_idx == num_candidates-1
            } for candidate_idx in range(num_candidates)]

    def _get_sample_data(self, sample_metadata):
        chat_data = self._persona_chat_dataset[sample_metadata['chat_idx']]

        return (chat_data['personality'],
                chat_data['history'],
                chat_data['candidates'][sample_metadata['candidate_idx']],  # candidate reply
                chat_data['is_real_reply'])
