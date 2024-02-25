import io
import os
import pickle
from functools import cached_property
from pathlib import Path
from typing import Optional, Sequence, List, Dict

from mlpug.base import Base

from examples.persona_chatbot.datasets.conversations import ConversationSample
from examples.persona_chatbot.datasets.multiple_choice import MultipleChoiceGenerator
from examples.persona_chatbot.datasets.distributed import DistributedSampleGenerator

from basics.logging_utils import log_exception
from basics.logging import get_logger


module_logger = get_logger(os.path.basename(__file__))


try:
    from tqdm import tqdm
except Exception as e:
    log_exception(module_logger, "Please `pip install tqdm`", e)

try:
    from datasets import load_dataset
except Exception as e:
    log_exception(module_logger, "Please `pip install datasets`", e)


class DatasetManager(Base):

    def __init__(self,
                 sample_factory,
                 persona_dataset_name="bavard/personachat_truecased",
                 distributed_generation: bool = True,
                 distributed_generation_config: dict = None,
                 disable_logging=False,
                 name=None):

        super(DatasetManager, self).__init__(pybase_logger_name=name, disable_logging=disable_logging)

        self._sample_factory = sample_factory

        self._persona_dataset_name = persona_dataset_name

        if distributed_generation and distributed_generation_config is None:
            # Use defaults
            distributed_generation_config = {}

        self._distributed_generation = distributed_generation
        self._distributed_generation_config = distributed_generation_config

        self._cache_path = os.path.join(Path.home(), '.cache/mlpug/')
        Path(self._cache_path).mkdir(exist_ok=True)

        self._persona_dataset = None

        self._orig_num_tokens = None
        self._num_special_tokens = None

    def get_dataset_for(self,
                        dataset_name,
                        max_num_samples=None,
                        num_choices_per_sample=None,
                        force_generate=False,
                        no_caching=False):

        config = {
            "persona_dataset_name": self._persona_dataset_name,
            "dataset_name": dataset_name,
            "max_num_samples": max_num_samples,
            "num_choices_per_sample": num_choices_per_sample
        }

        cached_samples_config_path = self._cached_samples_config_path(**config)
        try:
            with open(cached_samples_config_path, 'rb') as config_file:
                cached_config = pickle.load(config_file)
        except FileNotFoundError:
            cached_config = None

        if cached_config != config or force_generate:
            multiple_choice_samples = self._generate_samples_for(**config)

            if not no_caching:
                self._write_samples_to_cache(multiple_choice_samples, config)

            return multiple_choice_samples

        return self._read_samples_from_cache(config)

    @cached_property
    def persona_dataset(self):
        self._log.info(f"Loading persona dataset {self._persona_dataset_name} ...")
        return load_dataset(self._persona_dataset_name)

    def _cached_samples_config_path(self,
                                    dataset_name: str,
                                    max_num_samples: Optional[int] = None,
                                    num_choices_per_sample: Optional[int] = None,
                                    **kwargs) -> Path:
        filename = self._samples_filename(dataset_name, max_num_samples, num_choices_per_sample, "config")
        return Path(os.path.join(self._cache_path, filename))

    def _cached_samples_path(self,
                             dataset_name: str,
                             max_num_samples: Optional[int],
                             num_choices_per_sample: Optional[int],
                             **kwargs) -> Path:
        filename = self._samples_filename(dataset_name, max_num_samples, num_choices_per_sample)
        return Path(os.path.join(self._cache_path, filename))

    def _samples_filename(self,
                          dataset_name: str,
                          max_num_samples: Optional[int] = None,
                          num_choices_per_sample: Optional[int] = None,
                          postfix: Optional[str] = None,
                          **kwargs) -> str:
        filename = f"multiple_choice_samples-{dataset_name}"
        if max_num_samples is not None:
            filename += f"-{max_num_samples}"
        if num_choices_per_sample is not None:
            filename += f"-{num_choices_per_sample}"
        if postfix is not None:
            filename += f"-{postfix}"

        filename += ".pickle"

        return filename

    def _generate_samples_for(self,
                              dataset_name: str,
                              max_num_samples: Optional[int] = None,
                              num_choices_per_sample: Optional[int] = None,
                              **kwargs) -> List[List[ConversationSample]]:

        sample_generator = MultipleChoiceGenerator(self.persona_dataset[dataset_name],
                                                   self._sample_factory,
                                                   max_num_samples=max_num_samples,
                                                   num_choices=num_choices_per_sample,
                                                   name=f"SampleGenerator[{dataset_name}]")
        sample_generator.initialize()

        return self._generate_samples(sample_generator, dataset_name)

    def _write_samples_to_cache(self,
                                multiple_choice_samples: List[List[ConversationSample]],
                                config: dict):
        """
        Write multiple-choice conversation samples to cache

        :param multiple_choice_samples:
        :param config:

        :return:
        """

        dataset_name = config["dataset_name"]

        cached_samples_path = self._cached_samples_path(**config)

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

        cached_samples_config_path = self._cached_samples_config_path(**config)
        with open(cached_samples_config_path, 'wb') as config_file:
            pickle.dump(config, config_file)

    def _read_samples_from_cache(self, config: Dict) -> List[List[ConversationSample]]:
        """
        Read multiple-choice conversation samples to cache

        :param dataset_name:

        :return:
        """
        dataset_name = config["dataset_name"]
        cached_samples_path = self._cached_samples_path(**config)

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

    def _generate_samples(self,
                          sample_generator: Sequence[List[ConversationSample]],
                          dataset_name: str) -> List[List[ConversationSample]]:
        """
        Generate multiple-choice conversation samples

        :param sample_generator:
        :param dataset_name:
        :return:
        """

        self._log.info(f"Generating multiple choice conversation samples for {dataset_name} set ...")

        if self._distributed_generation:
            sample_generator = DistributedSampleGenerator(
                sample_generator,
                name=f"DistributedSampleGenerator[{dataset_name}]",
                **self._distributed_generation_config)

        if not self.logging_disabled:
            num_samples = len(sample_generator)

            sample_generator = tqdm(
                sample_generator,
                desc=f"Generating {dataset_name} samples",
                total=num_samples)

        return [sample for sample in sample_generator]
