import torch

from mlpug.trainers.callbacks.callback import Callback


class EmptyCudaCache(Callback):

    def __init__(self, on_batch_training_complete=True, on_epoch_complete=True, name="EmptyCudaCache"):
        super(EmptyCudaCache, self).__init__(name=name)

        self._on_batch_training_complete = on_batch_training_complete
        self._on_epoch_complete = on_epoch_complete

        self._log.debug(f"Will empty CUDA cache on batch training complete : {self._on_batch_training_complete}")
        self._log.debug(f"Will empty CUDA cache on epoch complete : {self._on_epoch_complete}")

    def on_batch_training_completed(self, training_batch, logs):
        if not self._on_batch_training_complete:
            return True

        torch.cuda.empty_cache()

        return True

    def on_epoch_completed(self, logs):
        if not self._on_epoch_complete:
            return True

        torch.cuda.empty_cache()

        return True
