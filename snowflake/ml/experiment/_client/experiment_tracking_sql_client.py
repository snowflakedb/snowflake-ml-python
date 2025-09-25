from typing import Optional

from snowflake.ml._internal.utils import query_result_checker, sql_identifier
from snowflake.ml.experiment._client import artifact
from snowflake.ml.model._client.sql import _base
from snowflake.ml.utils import sql_client
from snowflake.snowpark import file_operation, row, session


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

    def modify_run_add_metrics(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
        metrics: str,
    ) -> None:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        query_result_checker.SqlResultValidator(
            self._session,
            f"ALTER EXPERIMENT {experiment_fqn} MODIFY RUN {run_name} ADD METRICS=$${metrics}$$",
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def modify_run_add_params(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
        params: str,
    ) -> None:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        query_result_checker.SqlResultValidator(
            self._session,
            f"ALTER EXPERIMENT {experiment_fqn} MODIFY RUN {run_name} ADD PARAMETERS=$${params}$$",
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def put_artifact(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
        artifact_path: str,
        file_path: str,
        auto_compress: bool = False,
    ) -> file_operation.PutResult:
        return self._session.file.put(
            local_file_name=file_path,
            stage_location=self._build_snow_uri(experiment_name, run_name, artifact_path),
            overwrite=True,
            auto_compress=auto_compress,
        )[0]

    def list_artifacts(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
        artifact_path: str,
    ) -> list[artifact.ArtifactInfo]:
        results = (
            query_result_checker.SqlResultValidator(
                self._session, f"LIST {self._build_snow_uri(experiment_name, run_name, artifact_path)}"
            )
            .has_dimensions(expected_cols=4)
            .validate()
        )
        return [
            artifact.ArtifactInfo(
                name=str(result.name).removeprefix(f"/versions/{run_name}/"),
                size=result.size,
                md5=result.md5,
                last_modified=result.last_modified,
            )
            for result in results
        ]

    def get_artifact(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
        artifact_path: str,
        target_path: str,
    ) -> file_operation.GetResult:
        return self._session.file.get(
            stage_location=self._build_snow_uri(experiment_name, run_name, artifact_path),
            target_directory=target_path,
        )[0]

    def show_runs_in_experiment(
        self, *, experiment_name: sql_identifier.SqlIdentifier, like: Optional[str] = None
    ) -> list[row.Row]:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        like_clause = f"LIKE '{like}'" if like else ""
        return query_result_checker.SqlResultValidator(
            self._session, f"SHOW RUNS {like_clause} IN EXPERIMENT {experiment_fqn}"
        ).validate()

    def _build_snow_uri(
        self, experiment_name: sql_identifier.SqlIdentifier, run_name: sql_identifier.SqlIdentifier, artifact_path: str
    ) -> str:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        uri = f"snow://experiment/{experiment_fqn}/versions/{run_name}"
        if artifact_path:
            uri += f"/{artifact_path}"
        return uri
