from basics.base import Base as PyBase


class Base(PyBase):

    def __init__(self, *args, disable_logging=False, **kwargs):
        super().__init__(*args, **kwargs)

        self._log.disabled = disable_logging
        self._logging_disabled = disable_logging

    @property
    def logging_disabled(self):
        return self._logging_disabled
