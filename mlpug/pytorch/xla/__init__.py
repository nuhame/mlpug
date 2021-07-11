from . import trainers
from .trainers import callbacks

from . import evaluation

import mlpug.mlpug_logging as logging

from mlpug.multi_processing import MultiProcessingManager
from mlpug.pytorch.xla.multi_processing import XLADistributedContext

if not isinstance(MultiProcessingManager.get_context(), XLADistributedContext):
    MultiProcessingManager.set_context(XLADistributedContext())
