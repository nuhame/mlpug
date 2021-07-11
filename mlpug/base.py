import logging
from basics.base import Base as PyBase


class Base(PyBase):

    def __init__(self, *args, disable_logging=False, **kwargs):
        # print(f"mlpug.Base(*args={args}, kwargs={kwargs})")
        super().__init__(*args, **kwargs)

        self._logging_disabled = None
        self._set_logging_disabled(disable_logging)

    @property
    def logging_disabled(self):
        return self._logging_disabled

    def _set_logging_disabled(self, disable):
        self._log.setLevel(logging.WARN if disable else logging.DEBUG)
        self._logging_disabled = disable
