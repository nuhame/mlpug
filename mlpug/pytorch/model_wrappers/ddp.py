from typing import Optional, Dict

import torch

from basics.base import Base

from torch.nn.parallel import DistributedDataParallel as DDP

from mlpug.trainers import ModelWrapperFunc


class DDPModelWrapper(Base, ModelWrapperFunc):

    def __init__(self, rank: int, device: Optional[torch.device] = None):
        super().__init__()

        self._rank = rank
        self._device = device

    def __call__(
        self,
        model: torch.nn.Module,
        eager_mode: bool = False,
        compile_kwargs: Optional[Dict] = None,
        **other_kwargs
    ) -> torch.nn.Module:
        if compile_kwargs is None:
            compile_kwargs = {}

        if not eager_mode:
            self._log.debug(f"Compiling model using config: {compile_kwargs}")
            model = torch.compile(model, **compile_kwargs)

        if self._device is not None:
            device_type = self._device.type
        else:
            self._log.info("No device specified, assuming CUDA device.")
            device_type = 'cuda'

        device_ids = [self._rank] if device_type == 'cuda' else None

        return DDP(model, device_ids=device_ids)
