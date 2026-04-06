from typing import Any, Optional

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import query_result_checker, sql_identifier
from snowflake.ml.experiment._client import artifact
from snowflake.ml.model._client.sql import _base
from snowflake.ml.utils import sql_client
from snowflake.snowpark import dataframe, file_operation, row, session, types

RUN_NAME_COL_NAME = "name"
RUN_METADATA_COL_NAME = "metadata"


def pivot_run_attributes(
    sp_session: session.Session,
    rows: list[row.Row],
    run_names: list[str],
    cast_value: Optional[type] = None,
) -> dataframe.DataFrame:
    """Pivot SHOW RUN METRICS/PARAMETERS rows into one row per run.

    Each row is expected to have "run_name", "name", and "value" keys.
    Returns a Snowpark DataFrame with a "run_name" column and one column per distinct
    attribute name (sorted alphabetically). Runs present in ``run_names`` but absent
    from ``rows`` appear with all-NULL attribute columns.

    Args:
        sp_session: Snowpark session used to create the result DataFrame.
        rows: Collected rows from a SHOW RUN METRICS/PARAMETERS command.
        run_names: Exhaustive list of run names (from SHOW RUNS) to include in the output.
        cast_value: If provided, cast each value with this callable (e.g. ``float`` for metrics).
                    Values that fail to cast are set to ``None``.

    Returns:
        A Snowpark DataFrame with a "run_name" column and one column per distinct attribute name.
    """
    pivoted: dict[str, dict[str, Any]] = {name: {} for name in run_names}
    attr_names: set[str] = set()
    for r in rows:
        d = r.as_dict()
        value = d["value"]
        if cast_value is not None:
            try:
                value = cast_value(value)
            except (ValueError, TypeError):
                value = None
        pivoted.setdefault(d["run_name"], {})[d["name"]] = value
        attr_names.add(d["name"])

    col_order = sorted(attr_names)
    data = [[run_name] + [attrs.get(c) for c in col_order] for run_name, attrs in pivoted.items()]
    value_type = types.DoubleType() if cast_value is float else types.StringType()
    schema = types.StructType(
        [types.StructField('"run_name"', types.StringType())]
        + [types.StructField(f'"{c}"', value_type) for c in col_order]
    )
    return sp_session.create_dataframe(data, schema=schema)


class ExperimentTrackingSQLClient(_base._BaseSQLClient):
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

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
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

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
    def get_experiment_id(
        self,
        experiment_name: sql_identifier.SqlIdentifier,
    ) -> int:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)

        result = (
            query_result_checker.SqlResultValidator(
                self._session,
                f"CALL SYSTEM$RESOLVE_EXPERIMENT_ID('{experiment_fqn}')",
            )
            .has_dimensions(expected_rows=1, expected_cols=1)
            .validate()
        )
        return int(result[0][0])

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
    def drop_experiment(
        self,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        experiment_name: sql_identifier.SqlIdentifier,
    ) -> None:
        experiment_fqn = self.fully_qualified_object_name(database_name, schema_name, experiment_name)
        query_result_checker.SqlResultValidator(self._session, f"DROP EXPERIMENT {experiment_fqn}").has_dimensions(
            expected_rows=1, expected_cols=1
        ).validate()

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
    def add_run(self, *, experiment_name: sql_identifier.SqlIdentifier, run_name: sql_identifier.SqlIdentifier) -> None:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        query_result_checker.SqlResultValidator(
            self._session, f"ALTER EXPERIMENT {experiment_fqn} ADD RUN {run_name}"
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
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

    def get_run_id(
        self,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
    ) -> int:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)

        result = (
            query_result_checker.SqlResultValidator(
                self._session,
                f"CALL SYSTEM$RESOLVE_EXPERIMENT_RUN_ID('{experiment_fqn}', '{run_name}')",
            )
            .has_dimensions(expected_rows=1, expected_cols=1)
            .validate()
        )
        return int(result[0][0])

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
    def drop_run(
        self, *, experiment_name: sql_identifier.SqlIdentifier, run_name: sql_identifier.SqlIdentifier
    ) -> None:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        query_result_checker.SqlResultValidator(
            self._session, f"ALTER EXPERIMENT {experiment_fqn} DROP RUN {run_name}"
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
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

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
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

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
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

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
    def list_artifacts(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: sql_identifier.SqlIdentifier,
        artifact_path: str,
    ) -> list[artifact.ArtifactInfo]:
        results = self._session.sql(f"LIST {self._build_snow_uri(experiment_name, run_name, artifact_path)}").collect()
        return [
            artifact.ArtifactInfo(
                name=str(result.name).removeprefix(f"/versions/{run_name}/"),
                size=result.size,
                md5=result.md5,
                last_modified=result.last_modified,
            )
            for result in results
        ]

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
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

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
    def show_runs_in_experiment(
        self, *, experiment_name: sql_identifier.SqlIdentifier, like: Optional[str] = None
    ) -> list[row.Row]:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        like_clause = f"LIKE '{like}'" if like else ""
        return query_result_checker.SqlResultValidator(
            self._session, f"SHOW RUNS {like_clause} IN EXPERIMENT {experiment_fqn}"
        ).validate()

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
    def show_run_metrics_in_experiment(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: Optional[sql_identifier.SqlIdentifier] = None,
        like: Optional[str] = None,
    ) -> list[row.Row]:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        run_name_clause = f"RUN {run_name}" if run_name else ""
        like_clause = f"LIKE '{like}'" if like else ""
        return query_result_checker.SqlResultValidator(
            self._session, f"SHOW RUN METRICS {like_clause} IN EXPERIMENT {experiment_fqn} {run_name_clause}"
        ).validate()

    @telemetry.send_api_usage_telemetry(project=telemetry.TelemetryProject.EXPERIMENT_TRACKING.value)
    def show_run_parameters_in_experiment(
        self,
        *,
        experiment_name: sql_identifier.SqlIdentifier,
        run_name: Optional[sql_identifier.SqlIdentifier] = None,
        like: Optional[str] = None,
    ) -> list[row.Row]:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        run_name_clause = f"RUN {run_name}" if run_name else ""
        like_clause = f"LIKE '{like}'" if like else ""
        return query_result_checker.SqlResultValidator(
            self._session, f"SHOW RUN PARAMETERS {like_clause} IN EXPERIMENT {experiment_fqn} {run_name_clause}"
        ).validate()

    def _build_snow_uri(
        self, experiment_name: sql_identifier.SqlIdentifier, run_name: sql_identifier.SqlIdentifier, artifact_path: str
    ) -> str:
        experiment_fqn = self.fully_qualified_object_name(self._database_name, self._schema_name, experiment_name)
        uri = f"snow://experiment/{experiment_fqn}/versions/{run_name}"
        if artifact_path:
            uri += f"/{artifact_path}"
        return uri
