import contextlib
import dataclasses
import enum
import logging
import textwrap
from typing import Any, Generator, Optional

from snowflake import snowpark
from snowflake.ml._internal.utils import (
    identifier,
    query_result_checker,
    sql_identifier,
)
from snowflake.ml.model._client.sql import _base
from snowflake.ml.model._model_composer.model_method import constants
from snowflake.snowpark import dataframe, functions as F, row, types as spt
from snowflake.snowpark._internal import utils as snowpark_utils

logger = logging.getLogger(__name__)

# Using this token instead of '?' to avoid escaping issues
# After quotes are escaped, we replace this token with '|| ? ||'
QMARK_RESERVED_TOKEN = "<QMARK_RESERVED_TOKEN>"
QMARK_PARAMETER_TOKEN = "'|| ? ||'"


class ServiceStatus(enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    DONE = "DONE"
    SUSPENDING = "SUSPENDING"
    SUSPENDED = "SUSPENDED"
    DELETING = "DELETING"
    DELETED = "DELETED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class InstanceStatus(enum.Enum):
    PENDING = "PENDING"
    READY = "READY"
    FAILED = "FAILED"
    TERMINATING = "TERMINATING"
    SUCCEEDED = "SUCCEEDED"


class ContainerStatus(enum.Enum):
    PENDING = "PENDING"
    READY = "READY"
    DONE = "DONE"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass
class ServiceStatusInfo:
    """
    Class containing information about service container status.
    Reference: https://docs.snowflake.com/en/sql-reference/sql/show-service-containers-in-service
    """

    service_status: ServiceStatus
    instance_id: Optional[int] = None
    instance_status: Optional[InstanceStatus] = None
    container_status: Optional[ContainerStatus] = None
    message: Optional[str] = None


class ServiceSQLClient(_base._BaseSQLClient):
    MODEL_INFERENCE_SERVICE_ENDPOINT_NAME_COL_NAME = "name"
    MODEL_INFERENCE_SERVICE_ENDPOINT_INGRESS_URL_COL_NAME = "ingress_url"
    MODEL_INFERENCE_SERVICE_ENDPOINT_PRIVATELINK_INGRESS_URL_COL_NAME = "privatelink_ingress_url"
    SERVICE_STATUS = "service_status"
    INSTANCE_ID = "instance_id"
    INSTANCE_STATUS = "instance_status"
    CONTAINER_STATUS = "status"
    MESSAGE = "message"

    @contextlib.contextmanager
    def _qmark_paramstyle(self) -> Generator[None, None, None]:
        """Context manager that temporarily changes paramstyle to qmark and restores original value on exit."""
        if not hasattr(self._session, "_options"):
            yield
        else:
            original_paramstyle = self._session._options["paramstyle"]
            try:
                self._session._options["paramstyle"] = "qmark"
                yield
            finally:
                self._session._options["paramstyle"] = original_paramstyle

    def deploy_model(
        self,
        *,
        stage_path: Optional[str] = None,
        model_deployment_spec_yaml_str: Optional[str] = None,
        model_deployment_spec_file_rel_path: Optional[str] = None,
        query_params: Optional[list[Any]] = None,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> tuple[str, snowpark.AsyncJob]:
        assert model_deployment_spec_yaml_str or model_deployment_spec_file_rel_path
        if model_deployment_spec_yaml_str:
            model_deployment_spec_yaml_str = snowpark_utils.escape_single_quotes(
                model_deployment_spec_yaml_str
            )  # type: ignore[no-untyped-call]
            model_deployment_spec_yaml_str = model_deployment_spec_yaml_str.replace(  # type: ignore[union-attr]
                QMARK_RESERVED_TOKEN, QMARK_PARAMETER_TOKEN
            )
            logger.info(f"Deploying model with spec={model_deployment_spec_yaml_str}")
            sql_str = f"CALL SYSTEM$DEPLOY_MODEL('{model_deployment_spec_yaml_str}')"
        else:
            sql_str = f"CALL SYSTEM$DEPLOY_MODEL('@{stage_path}/{model_deployment_spec_file_rel_path}')"
        with self._qmark_paramstyle():
            async_job = self._session.sql(
                sql_str,
                params=query_params if query_params else None,
            ).collect(block=False, statement_params=statement_params)
        assert isinstance(async_job, snowpark.AsyncJob)
        return async_job.query_id, async_job

    def invoke_function_method(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        method_name: sql_identifier.SqlIdentifier,
        input_df: dataframe.DataFrame,
        input_args: list[sql_identifier.SqlIdentifier],
        returns: list[tuple[str, spt.DataType, sql_identifier.SqlIdentifier]],
        statement_params: Optional[dict[str, Any]] = None,
    ) -> dataframe.DataFrame:
        with_statements = []
        actual_database_name = database_name or self._database_name
        actual_schema_name = schema_name or self._schema_name

        if len(input_df.queries["queries"]) == 1 and len(input_df.queries["post_actions"]) == 0:
            INTERMEDIATE_TABLE_NAME = ServiceSQLClient.get_tmp_name_with_prefix("SNOWPARK_ML_MODEL_INFERENCE_INPUT")
            with_statements.append(f"{INTERMEDIATE_TABLE_NAME} AS ({input_df.queries['queries'][0]})")
        else:
            tmp_table_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE)
            INTERMEDIATE_TABLE_NAME = identifier.get_schema_level_object_identifier(
                actual_database_name.identifier(),
                actual_schema_name.identifier(),
                tmp_table_name,
            )
            input_df.write.save_as_table(
                table_name=INTERMEDIATE_TABLE_NAME,
                mode="errorifexists",
                table_type="temporary",
                statement_params=statement_params,
            )

        INTERMEDIATE_OBJ_NAME = ServiceSQLClient.get_tmp_name_with_prefix("TMP_RESULT")

        with_sql = f"WITH {','.join(with_statements)}" if with_statements else ""
        args_sql_list = []
        for input_arg_value in input_args:
            args_sql_list.append(input_arg_value)
        args_sql = ", ".join(args_sql_list)

        wide_input = len(input_args) > constants.SNOWPARK_UDF_INPUT_COL_LIMIT
        if wide_input:
            input_args_sql = ", ".join(f"'{arg}', {arg.identifier()}" for arg in input_args)
            args_sql = f"object_construct_keep_null({input_args_sql})"

        fully_qualified_service_name = self.fully_qualified_object_name(
            actual_database_name, actual_schema_name, service_name
        )
        fully_qualified_function_name = f"{fully_qualified_service_name}!{method_name.identifier()}"

        sql = textwrap.dedent(
            f"""{with_sql}
                SELECT *,
                    {fully_qualified_function_name}({args_sql}) AS {INTERMEDIATE_OBJ_NAME}
                FROM {INTERMEDIATE_TABLE_NAME}"""
        )

        output_df = self._session.sql(sql)

        # Prepare the output
        output_cols = []
        output_names = []

        for output_name, output_type, output_col_name in returns:
            output_cols.append(F.col(INTERMEDIATE_OBJ_NAME)[output_name].astype(output_type))
            output_names.append(output_col_name)

        output_df = output_df.with_columns(
            col_names=output_names,
            values=output_cols,
        ).drop(INTERMEDIATE_OBJ_NAME)

        if statement_params:
            output_df._statement_params = statement_params  # type: ignore[assignment]

        return output_df

    def get_service_logs(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        instance_id: str = "0",
        container_name: str,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> str:
        system_func = "SYSTEM$GET_SERVICE_LOGS"
        rows = (
            query_result_checker.SqlResultValidator(
                self._session,
                (
                    f"CALL {system_func}("
                    f"'{self.fully_qualified_object_name(database_name, schema_name, service_name)}', '{instance_id}', "
                    f"'{container_name}')"
                ),
                statement_params=statement_params,
            )
            .has_dimensions(expected_rows=1, expected_cols=1)
            .validate()
        )
        return str(rows[0][system_func])

    def get_service_container_statuses(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        include_message: bool = False,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[ServiceStatusInfo]:
        fully_qualified_object_name = self.fully_qualified_object_name(database_name, schema_name, service_name)
        query = f"SHOW SERVICE CONTAINERS IN SERVICE {fully_qualified_object_name}"
        rows = self._session.sql(query).collect(statement_params=statement_params)
        statuses = []
        for r in rows:
            instance_status, container_status = None, None
            if r[ServiceSQLClient.INSTANCE_STATUS] is not None:
                instance_status = InstanceStatus(r[ServiceSQLClient.INSTANCE_STATUS])
            if r[ServiceSQLClient.CONTAINER_STATUS] is not None:
                container_status = ContainerStatus(r[ServiceSQLClient.CONTAINER_STATUS])
            statuses.append(
                ServiceStatusInfo(
                    service_status=ServiceStatus(r[ServiceSQLClient.SERVICE_STATUS]),
                    instance_id=r[ServiceSQLClient.INSTANCE_ID],
                    instance_status=instance_status,
                    container_status=container_status,
                    message=r[ServiceSQLClient.MESSAGE] if include_message else None,
                )
            )
        return statuses

    def drop_service(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            f"DROP SERVICE {self.fully_qualified_object_name(database_name, schema_name, service_name)}",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def show_endpoints(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[row.Row]:
        fully_qualified_service_name = self.fully_qualified_object_name(database_name, schema_name, service_name)
        res = (
            query_result_checker.SqlResultValidator(
                self._session,
                (f"SHOW ENDPOINTS IN SERVICE {fully_qualified_service_name}"),
                statement_params=statement_params,
            )
            .has_column(ServiceSQLClient.MODEL_INFERENCE_SERVICE_ENDPOINT_NAME_COL_NAME, allow_empty=True)
            .has_column(ServiceSQLClient.MODEL_INFERENCE_SERVICE_ENDPOINT_INGRESS_URL_COL_NAME, allow_empty=True)
        )

        return res.validate()
