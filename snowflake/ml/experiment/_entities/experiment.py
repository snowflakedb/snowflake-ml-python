import enum

from snowflake.ml._internal.utils import sql_identifier


class ExperimentType(enum.Enum):
    AI_FUNCTION_EVALUATION = "AI_FUNCTION_EVALUATION"
    AI_FUNCTION_OPTIMIZATION = "AI_FUNCTION_OPTIMIZATION"


class Experiment:
    def __init__(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
    ) -> None:
        self.name = experiment_name
