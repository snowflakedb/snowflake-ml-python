from snowflake.ml._internal.utils import sql_identifier


class Experiment:
    def __init__(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
    ) -> None:
        self.name = experiment_name
