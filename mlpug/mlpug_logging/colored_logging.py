import logging

from basics.logging import LOGGING_FORMAT

from mlpug.mlpug_logging.base import FG_COLOR_CODE, BG_COLOR_CODE, FormatterBase

WHITE = 15
PINK = 13
MINT = 14
GRAY = 248


class FancyColorFormatter(FormatterBase):
    LEVEL_BG_COLOR_MAP = {
        'WARNING': MINT,
        'ERROR': PINK
    }

    LEVEL_TEXT_COLOR_MAP = {
      'WARNING': WHITE,
      'DEBUG': GRAY,
      'CRITICAL': WHITE,
      'ERROR': WHITE
    }

    MSG_COLOR_MAP = {
      'DEBUG': GRAY
    }

    def format(self, record):
        level = record.levelname

        # Align length of level string
        level_aligned = level + ' '*max(8-len(level), 0)

        level_fmt = self._set_bg_color(level, level_aligned, self.LEVEL_BG_COLOR_MAP)
        level_fmt = self._set_fg_color(level, level_fmt, self.LEVEL_TEXT_COLOR_MAP)
        if level_aligned != level_fmt:
            level_fmt = self.finalize_fmt(level_fmt)

        record.levelname = level_fmt

        name = record.name
        name_fmt = self._set_fg_color(level, name, self.MSG_COLOR_MAP)
        if name != name_fmt:
            name_fmt = self.finalize_fmt(name_fmt)

        record.name = name_fmt

        func_name = record.funcName
        func_name_fmt = self._set_fg_color(level, func_name, self.MSG_COLOR_MAP)
        if func_name != func_name_fmt:
            func_name_fmt = self.finalize_fmt(func_name_fmt)

        record.funcName = func_name_fmt

        msg = record.msg
        msg_fmt = self._set_fg_color(level, msg, self.MSG_COLOR_MAP)
        if msg != msg_fmt:
            msg_fmt = self.finalize_fmt(msg_fmt)

        record.msg = msg_fmt

        return super().format(record)

    def _set_bg_color(self, level, s, color_map):
        return self.apply_fmt(BG_COLOR_CODE, level, s, color_map)

    def _set_fg_color(self, level, s, color_map):
        return self.apply_fmt(FG_COLOR_CODE, level, s, color_map)


def use_fancy_colors(log_level=logging.DEBUG):
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    try:
        handler = root_logger.handlers[0]
        handler.setFormatter(FancyColorFormatter(LOGGING_FORMAT))
    except IndexError:
        print("ERROR : Unable to use fancy colors for logging, use logging.basicConfig or logging.config first")
