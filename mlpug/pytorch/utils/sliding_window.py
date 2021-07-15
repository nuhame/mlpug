from mlpug.utils.sliding_window import SlidingWindow as SlidingWindowBase

from mlpug.multi_processing import MultiProcessingMixin


class SlidingWindow(MultiProcessingMixin, SlidingWindowBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
