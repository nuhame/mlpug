from typing import Optional, Callable, Collection, List, Dict
import os
from functools import partial

import io
import pickle

import random

from pathlib import Path

import multiprocessing as mp


from basics.logging_utils import log_exception
from basics.logging import get_logger

from mlpug.base import Base


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


def generate_sample_worker_func(idx, sample_generator):
    """
    Used by DistributedDatasetGenerator.

    Calling sample_generator[idx] triggers the generator to build the requested multiple choice conversation sample.

    :param idx:
    :param sample_generator:

    :return:
    """
    return sample_generator[idx]


class DistributedDatasetGenerator(Base):

    def __init__(self, num_workers=None, name=None, disable_logging=False):
        """
        # TODO: refactor to a more generic tool

        Generates dataset by distributing the sample generation over the available CPU core, speeding up the process.
        The generated dataset will be cached. If a cached dataset is already available the generation will be skipped.

        :param num_workers:
        :param log_progress:
        :param name:
        :param disable_logging:

        """
        super().__init__(pybase_logger_name=name, disable_logging=disable_logging)

        if num_workers is None:
            num_workers = max(mp.cpu_count()-1, 1)

        self._num_workers = num_workers
        self._log.info(f"Using {num_workers} workers to generate multiple choice conversation samples.")

        self._cache_path = os.path.join(Path.home(), '.cache/mlpug/')
        Path(self._cache_path).mkdir(exist_ok=True)

        self._log.info(f"Using cache path: {self._cache_path}")

    def __call__(self, sample_generator: Collection, dataset_name=None, force_generate=False, no_caching=False):
        cached_samples_path = Path(os.path.join(
            self._cache_path,
            f"multiple_choice_samples-{dataset_name}.pickle"))

        cached_samples_found = cached_samples_path.is_file()
        if not cached_samples_found or force_generate:
            multiple_choice_samples = self._generate_samples(sample_generator, dataset_name)

            if not no_caching:
                self._write_samples_to_cache(multiple_choice_samples, cached_samples_path, dataset_name)

            return multiple_choice_samples

        return self._read_samples_from_cache(cached_samples_path, dataset_name)

    def _write_samples_to_cache(self, multiple_choice_samples, cached_samples_path, dataset_name):
        self._log.info(f"Caching generated {dataset_name} multiple choice samples to:\n{cached_samples_path}\n")
        with io.open(cached_samples_path, 'wb') as f:
            pickler = pickle.Pickler(f)

            if not self.logging_disabled:
                num_samples = len(multiple_choice_samples)

                multiple_choice_samples = tqdm(
                    multiple_choice_samples,
                    desc=f"Writing {dataset_name} samples",
                    total=num_samples)

            for sample in multiple_choice_samples:
                pickler.dump(sample)

    def _read_samples_from_cache(self, cached_samples_path, dataset_name):
        self._log.info(f"Loading generated {dataset_name} multiple choice samples from:\n{cached_samples_path}\n")

        with io.open(cached_samples_path, 'rb') as p:
            unpickler = pickle.Unpickler(p)

            def yield_samples():
                while p.peek(1):
                    yield unpickler.load()

            sample_reader = yield_samples()
            if not self.logging_disabled:
                sample_reader = tqdm(sample_reader, desc=f"Reading {dataset_name} samples")

            samples = [sample for sample in sample_reader]

        return samples

    def _generate_samples(self, sample_generator, dataset_name):
        num_samples = len(sample_generator)

        chunksize = max(int(num_samples / (self._num_workers * 20)), 1)

        generate_sample = partial(generate_sample_worker_func, sample_generator=sample_generator)

        with mp.Pool(self._num_workers) as p:
            sample_iter = p.imap(generate_sample, range(num_samples), chunksize=chunksize)

            if not self.logging_disabled:
                sample_iter = tqdm(sample_iter, desc=f"Generating {dataset_name} samples", total=num_samples)

            samples = [sample for sample in sample_iter]

        return samples


class MCSampleGenerator(Base):

    def __init__(self,
                 persona_chat_dataset: List[Dict],
                 sample_factory: Callable,
                 max_num_samples: Optional[int] = None,
                 num_choices: Optional[int] = None,
                 shuffle: bool = True,
                 name: Optional[str] = None):
        """
        Multiple Choice (MC) sample generator, acts as a collection where a sample for some sample_idx is
        generated on the fly.

        Usage:

        mc_generator = MCSampleGenerator(persona_chat_dataset, sample_factory)
        mc_generator.initialize()

        multiple_conversation_choices = mc_generator[10]
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
