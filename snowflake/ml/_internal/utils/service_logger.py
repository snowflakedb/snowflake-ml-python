import enum
import logging
import sys


class LogColor(enum.Enum):
    GREY = "\x1b[38;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    YELLOW = "\x1b[33;20m"
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"


class CustomFormatter(logging.Formatter):

    reset = "\x1b[0m"
    log_format = "%(name)s [%(asctime)s] [%(levelname)s] %(message)s"

    def __init__(self, info_color: LogColor) -> None:
        super().__init__()
        self.level_colors = {
            logging.DEBUG: LogColor.GREY.value,
            logging.INFO: info_color.value,
            logging.WARNING: LogColor.YELLOW.value,
            logging.ERROR: LogColor.RED.value,
            logging.CRITICAL: LogColor.BOLD_RED.value,
        }

    def format(self, record: logging.LogRecord) -> str:
        # default to DEBUG color
        fmt = self.level_colors.get(record.levelno, self.level_colors[logging.DEBUG]) + self.log_format + self.reset
        formatter = logging.Formatter(fmt)

        # split the log message by lines and format each line individually
        original_message = record.getMessage()
        message_lines = original_message.splitlines()
        formatted_lines = [
            formatter.format(
                logging.LogRecord(
                    name=record.name,
                    level=record.levelno,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg=line,
                    args=None,
                    exc_info=None,
                )
            )
            for line in message_lines
        ]

        return "\n".join(formatted_lines)


def get_logger(logger_name: str, info_color: LogColor) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(CustomFormatter(info_color))
    logger.addHandler(handler)
    return logger
