import logging

ESCAPE = '\033'
FG_COLOR_CODE = '[38;5;%dm'
BG_COLOR_CODE = '[48;5;%dm'
RESET_CODE = '[0;00m'


class FormatterBase(logging.Formatter):

    @staticmethod
    def apply_fmt(format_code, level, s, color_map):
        color = color_map[level] if level in color_map else None
        if color is None:
            return s

        return ESCAPE + (format_code % color) + s

    @staticmethod
    def finalize_fmt(s):
        return s + ESCAPE + RESET_CODE
