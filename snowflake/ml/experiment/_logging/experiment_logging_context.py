import contextlib
from dataclasses import dataclass

from snowflake.ml._internal.utils import tee
from snowflake.ml.experiment._logging.experiment_logger import ExperimentLogger


@dataclass(frozen=True)
class ExperimentLoggingContext:
    stdout_logger: ExperimentLogger
    stderr_logger: ExperimentLogger
    stdout_ctx: contextlib.redirect_stdout[tee.OutputTee]
    stderr_ctx: contextlib.redirect_stderr[tee.OutputTee]
