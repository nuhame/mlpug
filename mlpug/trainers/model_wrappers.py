from typing import Protocol


class ModelWrapperFunc(Protocol):

    def __call__(self, model: object, **trainer_kwargs) -> object:
        """
        Wraps the model in a specific way. This model wrapper function can be used by the trainer to
        enable certain capabilities of model training and evaluation, such as using Distributed Data Parallel training
        or Fully Sharded Data Parallel training.

        E.g. for PyTorch to use a model with DDP this could be implemented by a DDPModelWrapper:
        apply_ddp = DDPModelWrapper(rank, device)
        model = apply_ddp(model, **trainer_kwargs)

        :param: model: the model to wrap
        :param: trainer_kwargs: additional keyword arguments to pass to the wrapped model.
            You can assume that trainer constructor keyword arguments are injected.

        :returns Wrapped model
        """
        ...

