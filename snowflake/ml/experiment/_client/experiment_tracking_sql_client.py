from typing import Optional

from snowflake.ml._internal.utils import query_result_checker, sql_identifier
from snowflake.ml.model._client.sql import _base
from snowflake.ml.utils import sql_client
from snowflake.snowpark import row, session


class ExperimentTrackingSQLClient(_base._BaseSQLClient):

    RUN_NAME_COL_NAME = "name"
    RUN_METADATA_COL_NAME = "metadata"

    def __init__(
        self,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
    ) -> None:
        """Snowflake SQL Client to manage experiment tracking.

        Args:
            session: Active snowpark session.
            database_name: Name of the Database where experiment tracking resources are provisioned.
            schema_name: Name of the Schema where experiment tracking resources are provisioned.
        """
        super().__init__(session, database_name=database_name, schema_name=schema_name)

    def create_experiment(
        self,
        experiment_name: sql_identifier.SqlIdentifier,
        creation_mode: sql_client.CreationMode,
    ) -> None:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        if_not_exists_sql = "IF NOT EXISTS" if creation_mode.if_not_exists else ""
        query_result_checker.SqlResultValidator(
            self._session, f"CREATE EXPERIMENT {if_not_exists_sql} {experiment_fqn}"
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def drop_experiment(self, *, experiment_name: sql_identifier.SqlIdentifier) -> None:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        query_result_checker.SqlResultValidator(self._session, f"DROP EXPERIMENT {experiment_fqn}").has_dimensions(
            expected_rows=1, expected_cols=1
        ).validate()

    def add_run(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
        live: bool = True,
    ) -> None:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        query_result_checker.SqlResultValidator(
            self._session, f"ALTER EXPERIMENT {experiment_fqn} ADD {'LIVE' if live else ''} RUN {run_name}"
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def commit_run(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
    ) -> None:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        query_result_checker.SqlResultValidator(
            self._session, f"ALTER EXPERIMENT {experiment_fqn} COMMIT RUN {run_name}"
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def drop_run(
        self, *, experiment_name: sql_identifier.SqlIdentifier, run_name: sql_identifier.SqlIdentifier
    ) -> None:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        query_result_checker.SqlResultValidator(
            self._session, f"ALTER EXPERIMENT {experiment_fqn} DROP RUN {run_name}"
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def modify_run(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
        run_metadata: str,
    ) -> None:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        query_result_checker.SqlResultValidator(
            self._session,
            f"ALTER EXPERIMENT {experiment_fqn} MODIFY RUN {run_name} SET METADATA=$${run_metadata}$$",
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def show_runs_in_experiment(
        self, *, experiment_name: sql_identifier.SqlIdentifier, like: Optional[str] = None
    ) -> list[row.Row]:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        like_clause = f"LIKE '{like}'" if like else ""
        return query_result_checker.SqlResultValidator(
            self._session, f"SHOW RUNS {like_clause} IN EXPERIMENT {experiment_fqn}"
        ).validate()
