from basics.base import Base
from mlpug.mlpug_exceptions import StateInvalidException, InvalidParametersException


class SlidingWindow(Base):

    def __init__(self, length=None, init_window_values=None, name=None, state=None, **kwargs):
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
        if state:
            try:
                length = state["length"]
                init_window_values = state["window"]
                name = state["name"] if "name" in state else None
            except Exception as e:
                raise StateInvalidException() from e

        logger_name = self.__class__.__name__
        if name is not None:
            logger_name = f"{logger_name}[{name}]"

        super().__init__(pybase_logger_name=logger_name, **kwargs)

        self.name = name

        if not length:
            raise InvalidParametersException("No length value given, unable to instantiate SlidingWindow object.")

        self._log.debug(f"Window length : {length}")

        self._centre_not_exact_warning_given = False

        self.length = length
        self.center_index = self.length // 2

        if hasattr(init_window_values, "__getitem__"):
            self._log.debug("Using given initial values to fill the window as much as possible.")

            iwv_len = len(init_window_values)

            if iwv_len < self.length:
                self._log.debug(f"Given initial window values list is shorter than window: {iwv_len}")

                if iwv_len < 1:
                    self._log.warn("Unexpected : The given initial window values list is empty, continuing anyway ...")
            elif iwv_len > self.length:
                iwv_len = self.length
                self._log.debug(f"Given initial window values list is longer than window, "
                                f"only using the last {iwv_len} values to fill the whole window")
            else:
                self._log.debug(f"Given initial window values list matches the window length perfectly")

            self.window = init_window_values[-iwv_len:][:]
        else:
            self.window = list()

    def get_state(self):
        return {
            "window": self.window,
            "length": self.length,
            "name": self.name
        }

    def flush(self):
        self.window.clear()

    def center(self):

        if not self.is_filled():
            self._log.error("Window not completely filled yet, unable to get center")
            return None

        if self.length % 2 != 1 and not self._centre_not_exact_warning_given:
            self._log.warn("Given window length is even, centre is not exact")
            self._centre_not_exact_warning_given = True

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
