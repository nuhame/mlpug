from basics.base import Base
from mlpug.mlpug_exceptions import StateInvalidException, InvalidParametersException


class SlidingWindow(Base):

    def __init__(self, length=None, init_window_values=None, state=None, **kwargs):
        """

        Can raise StateInvalidException or InvalidParametersException

        :param length:
        :type length:
        :param init_window_values:
        :type init_window_values:
        :param state:
        :type state:
        :param kwargs:
        :type kwargs:
        """
        super().__init__(**kwargs)

        if state:
            try:
                length = state["length"]
                init_window_values = state["window"]
            except Exception as e:
                raise StateInvalidException() from e

        if not length:
            raise InvalidParametersException("No length value given, unable to instantiate SlidingWindow object.")

        if length % 2 != 1:
            self._log.warn("Given window length is even, centre will not be exact")

        self.length = length
        self.center_index = self.length // 2

        if hasattr(init_window_values, "__getitem__"):
            self._log.debug("Using given initial values to fill the window as much as possible.")

            l = len(init_window_values)
            l = l if l < self.length else self.length

            self.window = init_window_values[-l:][:]
        else:
            self.window = list()

    def get_state(self):
        return {
            "window": self.window,
            "length": self.length
        }

    def flush(self):
        self.window.clear()

    def center(self):
        if not self.is_filled():
            self._log.error("Window not completely filled yet, unable to get center")
            return None

        return self.window[self.center_index]

    def last(self):
        return self.window[-1]

    def first(self):
        return self.window[0]

    def is_filled(self):
        return len(self.window) >= self.length

    def is_empty(self):
        return len(self.window) == 0

    def slide(self, value):
        if self.is_filled():
            self.window = self.window[1:]

        self.window.append(value)
