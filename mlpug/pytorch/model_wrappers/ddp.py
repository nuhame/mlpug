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
        **ddp_kwargs,
    ) -> torch.nn.Module:
        """
        Wrap model with DDP, optionally compiling first.

        :param model: The model to wrap.
        :param eager_mode: If True, skip torch.compile.
        :param compile_kwargs: Arguments passed to torch.compile().
        :param ddp_kwargs: Additional arguments passed to DDP constructor.
            Useful options include:
            - gradient_as_bucket_view=True: Reduces memory by avoiding
                gradient copy to communication buckets.
            - bucket_cap_mb: Size of gradient buckets (default 25MB).
            - static_graph=True: Enables optimizations for static models.

        :return: The wrapped model.
        """
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

        if ddp_kwargs:
            self._log.debug(f"DDP kwargs: {ddp_kwargs}")

        return DDP(model, device_ids=device_ids, **ddp_kwargs)
