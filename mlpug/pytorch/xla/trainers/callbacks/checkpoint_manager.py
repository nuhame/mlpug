from mlpug.pytorch.trainers.callbacks.checkpoint_manager import CheckpointManagerMixin

from mlpug.pytorch.xla.multi_processing import MultiProcessingMixin

class CheckpointManager(MultiProcessingMixin, CheckpointManagerMixin):
    pass
