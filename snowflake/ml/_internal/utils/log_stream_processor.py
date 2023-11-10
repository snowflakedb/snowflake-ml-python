import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LogStreamProcessor:
    def __init__(self) -> None:
        self.last_line_seen = 0

    def process_new_logs(self, job_logs: Optional[str], *, log_level: int = logging.INFO) -> None:
        if not job_logs:
            return
        log_entries = job_logs.split("\n")
        start_index = self.last_line_seen
        log_length = len(log_entries)
        for i in range(start_index, log_length):
            log_entry = log_entries[i]
            if log_level == logging.DEBUG:
                logger.debug(log_entry)
            elif log_level == logging.INFO:
                logger.info(log_entry)
            elif log_level == logging.WARNING:
                logger.warning(log_entry)
            elif log_level == logging.ERROR:
                logger.error(log_entry)
            elif log_level == logging.CRITICAL:
                logger.critical(log_entry)

        self.last_line_seen = log_length
