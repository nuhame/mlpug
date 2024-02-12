from typing import Protocol, List
from mlpug.base import Base


class HFTokenizerInterface(Protocol):

    def encode(self, text: str) -> List[int]:
        """Encode text in to token IDs"""


class HFTokenizer(Base):

    def __init__(self, hf_tokenizer: HFTokenizerInterface, name=None):
        super().__init__(pybase_logger_name=name)

        self._hf_tokenizer = hf_tokenizer

    def __call__(self, text: str):
        return self._hf_tokenizer.encode(text)
