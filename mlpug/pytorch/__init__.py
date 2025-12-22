from . import trainers
from .trainers import callbacks

# TODO only export limited set of relevant classes and functions
from . import evaluation

from . import scheduler_funcs

import mlpug.mlpug_logging as logging
from mlpug.base import Base

from mlpug.multi_processing import MultiProcessingManager
from mlpug.pytorch.multi_processing import MultiProcessingMixin, PyTorchDistributedContext

if not isinstance(MultiProcessingManager.get_context(), PyTorchDistributedContext):
    MultiProcessingManager.set_context(PyTorchDistributedContext())

