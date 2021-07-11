import os

DEBUG_MULTI_PROCESSING = os.environ['DEBUG_MULTI_PROCESSING'] if 'DEBUG_MULTI_PROCESSING' in os.environ else False


class MultiProcessingMixin:

    def __init__(self, *args,
                 is_distributed=False,
                 is_primary=True,
                 process_index=0,
                 disable_logging=None,
                 **kwargs):
        # print(f"mlpug.MultiProcessingMixin(args={args}, kwargs={kwargs})")

        self._is_distributed = is_distributed
        self._is_primary = is_primary
        self._process_index = process_index

        if is_distributed and bool(DEBUG_MULTI_PROCESSING) is True:
            disable_logging = False

        if disable_logging is None:
            disable_logging = is_distributed and not self._is_primary

        super().__init__(*args, disable_logging=disable_logging, **kwargs)

    @property
    def is_distributed(self):
        return self._is_distributed

    @property
    def is_primary(self):
        return self._is_primary

    @property
    def process_index(self):
        return self._process_index

    def _pybase_get_logger_name(self):
        if self.is_distributed:
            return f"[Worker {self.process_index}] {super()._pybase_get_logger_name()}"

        return super()._pybase_get_logger_name()
