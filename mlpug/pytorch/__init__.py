from . import trainers
from .trainers import callbacks

from . import evaluation

import mlpug.mlpug_logging as logging

from mlpug.multi_processing import MultiProcessingManager
from mlpug.pytorch.multi_processing import PyTorchDistributedContext

if not isinstance(MultiProcessingManager.get_context(), PyTorchDistributedContext):
    MultiProcessingManager.set_context(PyTorchDistributedContext())

