from typing import Optional, Callable, Tuple, List, Dict
import os
import random

import pickle

from pathlib import Path

from functools import partial

import multiprocessing as mp

from basics.logging_utils import log_exception
from basics.logging import get_logger

from mlpug.base import Base


module_logger = get_logger(os.path.basename(__file__))

try:
    from tqdm import tqdm
except Exception as e:
    log_exception(module_logger, "Please `pip install tqdm`", e)


global _MULTIPLE_CHOICE_DATASET


def init_worker(dataset):
    """
    Used by DistributedSampleGenerator.

    The dataset is assumed to be a MultipleConversationChoicesDataset instance.
    Setting the dataset to the global _MULTIPLE_CHOICE_DATASET allows the dataset to be used
    without copying it to a worker process every time a sample generation task is sent to it.

    :param dataset:
    :return:
    """
    global _MULTIPLE_CHOICE_DATASET
    _MULTIPLE_CHOICE_DATASET = dataset


def generate_sample(idx):
    """
    Used by DistributedSampleGenerator.
    Calling _MULTIPLE_CHOICE_DATASET[idx] triggers the dataset to build the multiple choice conversation sample.

    :param idx:
    :return:
    """
    return _MULTIPLE_CHOICE_DATASET[idx]


class DistributedSampleGenerator(Base):

    def __init__(self, num_workers=None, log_progress=True, name=None):
        """

        Speeding up pre-generation of all samples by distributing the work over the available CPU cores.
        The generated samples will be cached. If cached samples are already available the generation will be skipped.

        :param num_workers:
        :param log_progress:
        :param name:
        """
        super().__init__(pybase_logger_name=name)

        if num_workers is None:
            num_workers = max(mp.cpu_count()-1, 1)

        self._num_workers = num_workers
        self._log_progress = log_progress

        self._log.info(f"Using {num_workers} workers to generate multiple choice conversation samples.")

        self._cache_path = os.path.join(Path.home(), '.cache/mlpug/')

        Path(self._cache_path).mkdir(exist_ok=True)

        self._log.info(f"Using cache path: {self._cache_path}")

    def __call__(self, multiple_choice_dataset, dataset_name=None, force_generate=False):
        cached_samples_path = Path(os.path.join(
            self._cache_path,
            f"multiple_choice_samples-{dataset_name}.pickle"))

        cached_samples_found = cached_samples_path.is_file()

        if not cached_samples_found or force_generate:
            multiple_choice_samples = self._generate_samples(multiple_choice_dataset, dataset_name)

            self._log.info(f"Caching generated {dataset_name} multiple choice samples to:\n{cached_samples_path}\n")
            with open(cached_samples_path, 'wb') as f:
                pickle.dump(multiple_choice_samples, f)

            return multiple_choice_samples

        self._log.info(f"Loading generated {dataset_name} multiple choice samples from:\n{cached_samples_path}\n")
        with open(cached_samples_path, 'rb') as f:
            return pickle.load(f)

    def _generate_samples(self, multiple_choice_dataset, dataset_name):
        num_samples = len(multiple_choice_dataset)

        chunksize = max(int(num_samples / (self._num_workers * 40)), 1)

        pool = mp.Pool(self._num_workers,
                       initializer=init_worker,
                       initargs=(multiple_choice_dataset,))

        with pool as p:
            sample_iter = p.imap(generate_sample, range(num_samples), chunksize=chunksize)

            if self._log_progress:
                sample_iter = tqdm(sample_iter, desc=f"Generating {dataset_name} samples", total=num_samples)

            return [sample for sample in sample_iter]


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


class MultipleConversationChoicesDataset(Base):

    def __init__(self,
                 persona_chat_dataset: List[Dict],
                 sample_factory: Callable,
                 max_num_samples: Optional[int] = None,
                 num_choices: Optional[int] = None,
                 shuffle: bool = True,
                 name: Optional[str] = None):
        """
        Usage:

        mc_dataset = MultipleConversationChoicesDataset(persona_chat_dataset, sample_factory)

        multiple_conversation_choices = mc_dataset[10]
        # depending on the sample factory:
        # input_ids of third conversation choice
        input_ids = multiple_conversation_choices[0][2]

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

    def __getitem__(self, item):
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
