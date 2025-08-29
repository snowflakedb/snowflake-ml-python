import enum
import logging
import os
import sys
import tempfile
import time
import uuid
from typing import Optional

import platformdirs


class ProgressBarAwareConsoleHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """A logging handler that adapts to different progress bar systems to avoid interfering with display."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)

            # Check if tqdm progress bars are active - use tqdm.write() to avoid interference
            try:
                import tqdm

                if hasattr(tqdm.tqdm, "_instances") and tqdm.tqdm._instances:
                    tqdm.tqdm.write(msg, file=self.stream)
                    return
            except (ImportError, AttributeError):
                pass

            # Fallback to regular stream writing (works for all contexts including Streamlit)
            self.stream.write(msg + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


# Module-level logger for operational messages that should appear on console
stdout_handler = ProgressBarAwareConsoleHandler(sys.stdout)
stdout_handler.setFormatter(logging.Formatter("%(message)s"))

console_logger = logging.getLogger(__name__)
console_logger.addHandler(stdout_handler)
console_logger.setLevel(logging.INFO)
console_logger.propagate = False


class LogColor(enum.Enum):
    GREY = "\x1b[38;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    YELLOW = "\x1b[33;20m"
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"
    ORANGE = "\x1b[38;5;214m"
    BOLD_ORANGE = "\x1b[38;5;214;1m"
    PURPLE = "\x1b[35;20m"
    BOLD_PURPLE = "\x1b[35;1m"


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


def _test_writability(directory: str) -> bool:
    """Test if a directory is writable by creating and removing a test file."""
    try:
        os.makedirs(directory, exist_ok=True)
        test_file = os.path.join(directory, f".write_test_{uuid.uuid4().hex[:8]}")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except OSError:
        return False


def _try_log_location(log_dir: str, operation_id: str) -> Optional[str]:
    """Try to create a log file in the given directory if it's writable."""
    if _test_writability(log_dir):
        return os.path.join(log_dir, f"{operation_id}.log")
    return None


def _get_log_file_path(operation_id: str) -> Optional[str]:
    """Get platform-independent log file path. Returns None if no writable location found."""
    # Try locations in order of preference
    locations = [
        # Primary: User log directory
        platformdirs.user_log_dir("snowflake-ml", "Snowflake"),
        # Fallback 1: System temp directory
        os.path.join(tempfile.gettempdir(), "snowflake-ml-logs"),
        # Fallback 2: Current working directory
        ".",
    ]

    for location in locations:
        log_file_path = _try_log_location(location, operation_id)
        if log_file_path:
            return log_file_path

    # No writable location found
    return None


def _get_or_create_parent_logger(operation_id: str) -> logging.Logger:
    """Get or create a parent logger with FileHandler for the operation."""
    parent_logger_name = f"snowflake_ml_operation_{operation_id}"
    parent_logger = logging.getLogger(parent_logger_name)
    parent_logger.setLevel(logging.DEBUG)
    parent_logger.propagate = False

    if not parent_logger.handlers:
        log_file_path = _get_log_file_path(operation_id)

        if log_file_path:
            try:
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setFormatter(logging.Formatter("%(name)s [%(asctime)s] [%(levelname)s] %(message)s"))
                parent_logger.addHandler(file_handler)

                console_logger.info(f"create_service logs saved to: {log_file_path}")
            except OSError as e:
                console_logger.warning(f"Could not create log file at {log_file_path}: {e}.")
        else:
            # No writable location found, use console-only logging
            console_logger.warning("No writable location found for create_service log file.")

        if logging.getLogger().level > logging.INFO:
            console_logger.info(
                "To see logs in console, set log level to INFO: logging.getLogger().setLevel(logging.INFO)"
            )

    return parent_logger


def get_logger(logger_name: str, info_color: LogColor, operation_id: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    root_logger = logging.getLogger()

    # If operation_id provided, set up parent logger with file handler
    if operation_id:
        parent_logger = _get_or_create_parent_logger(operation_id)
        logger.parent = parent_logger
        logger.propagate = True

        if root_logger.level <= logging.INFO:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(CustomFormatter(info_color))
            logger.addHandler(handler)
    else:
        # No operation_id - add console handler only if user wants verbose logging
        if root_logger.level <= logging.INFO and not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(CustomFormatter(info_color))
            logger.addHandler(handler)

    return logger


def get_operation_id() -> str:
    """Generate a unique operation ID."""
    return f"model_deploy_{uuid.uuid4().hex[:8]}_{int(time.time())}"


def get_log_file_location(operation_id: str) -> Optional[str]:
    """Get the log file path for an operation ID. Returns None if no writable location available."""
    return _get_log_file_path(operation_id)
