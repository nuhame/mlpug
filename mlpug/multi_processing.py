import abc
import os

from typing import Any, Optional

from mlpug.base import Base

from mlpug.mlpug_logging import get_logger

DEBUG_MULTI_PROCESSING = os.environ['DEBUG_MULTI_PROCESSING'] if 'DEBUG_MULTI_PROCESSING' in os.environ else False


class MultiProcessingContextBase(Base, metaclass=abc.ABCMeta):

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(pybase_logger_name=name, **kwargs)

    @abc.abstractmethod
    def is_distributed(self) -> bool:
        raise NotImplemented("Please implement this method in your child class")

    @abc.abstractmethod
    def is_primary(self) -> bool:
        raise NotImplemented("Please implement this method in your child class")

    @abc.abstractmethod
    def device_rank(self) -> int:
        raise NotImplemented("Please implement this method in your child class")

    @abc.abstractmethod
    def world_size(self) -> int:
        raise NotImplemented("Please implement this method in your child class")


class MultiProcessingManager:

    _log = get_logger("MultiProcessingManager")

    _context = None

    @classmethod
    def set_context(cls, context: MultiProcessingContextBase):
        cls._context = context

        if bool(DEBUG_MULTI_PROCESSING) is True:
            cls._log.info(f"Using multi-processing context {str(context)}.")

    @classmethod
    def get_context(cls) -> MultiProcessingContextBase:
        return cls._context


class MultiProcessingMixin:

    def __init__(self,
                 *args: Any,
                 is_distributed: Optional[bool] = None,
                 is_primary: Optional[bool] = None,
                 device_rank: Optional[int] = None,
                 disable_logging: Optional[bool] = None,
                 **kwargs: Any):

        mp_context = MultiProcessingManager.get_context()

        if is_distributed is None:
            is_distributed = mp_context.is_distributed()

        if device_rank is None:
            device_rank = mp_context.device_rank() if is_distributed else 0

        if is_primary is None:
            is_primary = (not is_distributed) or mp_context.is_primary()

        if is_distributed and bool(DEBUG_MULTI_PROCESSING) is True:
            disable_logging = False

        if disable_logging is None:
            disable_logging = is_distributed and not is_primary

        self._is_distributed = is_distributed
        self._device_rank = device_rank
        self._is_primary = is_primary

        super().__init__(*args, disable_logging=disable_logging, **kwargs)

    @property
    def is_distributed(self) -> bool:
        return self._is_distributed

    @property
    def is_primary(self) -> bool:
        return self._is_primary

    @property
    def device_rank(self) -> int:
        return self._device_rank

    def _pybase_get_logger_name(self) -> str:
        if self.is_distributed:
            return f"[Device {self.device_rank}] {super()._pybase_get_logger_name()}"

        return super()._pybase_get_logger_name()
