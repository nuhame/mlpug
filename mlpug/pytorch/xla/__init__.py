from . import trainers
from .trainers import callbacks

# TODO only export limited set of relevant classes and functions
from . import evaluation

from . import scheduler_funcs

import mlpug.mlpug_logging as logging
from mlpug.base import Base

from mlpug.multi_processing import MultiProcessingManager
from mlpug.pytorch.xla.multi_processing import MultiProcessingMixin, XLADistributedContext

if not isinstance(MultiProcessingManager.get_context(), XLADistributedContext):
    MultiProcessingManager.set_context(XLADistributedContext())
