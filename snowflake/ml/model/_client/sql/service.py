import enum
import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from snowflake import snowpark
from snowflake.ml._internal.utils import (
    identifier,
    query_result_checker,
    sql_identifier,
)
from snowflake.ml.model._client.sql import _base
from snowflake.snowpark import dataframe, functions as F, types as spt
from snowflake.snowpark._internal import utils as snowpark_utils


class ServiceStatus(enum.Enum):
    UNKNOWN = "UNKNOWN"  # status is unknown because we have not received enough data from K8s yet.
    PENDING = "PENDING"  # resource set is being created, can't be used yet
    READY = "READY"  # resource set has been deployed.
    DELETING = "DELETING"  # resource set is being deleted
    FAILED = "FAILED"  # resource set has failed and cannot be used anymore
    DONE = "DONE"  # resource set has finished running
    NOT_FOUND = "NOT_FOUND"  # not found or deleted
    INTERNAL_ERROR = "INTERNAL_ERROR"  # there was an internal service error.


class ServiceSQLClient(_base._BaseSQLClient):
    def build_model_container(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        compute_pool_name: sql_identifier.SqlIdentifier,
        image_repo_database_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_schema_name: Optional[sql_identifier.SqlIdentifier],
        image_repo_name: sql_identifier.SqlIdentifier,
        gpu: Optional[str],
        force_rebuild: bool,
        external_access_integration: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        actual_image_repo_database = image_repo_database_name or self._database_name
        actual_image_repo_schema = image_repo_schema_name or self._schema_name
        actual_model_database = database_name or self._database_name
        actual_model_schema = schema_name or self._schema_name
        fq_model_name = self.fully_qualified_object_name(actual_model_database, actual_model_schema, model_name)
        fq_image_repo_name = identifier.get_schema_level_object_identifier(
            actual_image_repo_database.identifier(),
            actual_image_repo_schema.identifier(),
            image_repo_name.identifier(),
        )
        is_gpu_str = "TRUE" if gpu else "FALSE"
        force_rebuild_str = "TRUE" if force_rebuild else "FALSE"
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"CALL SYSTEM$BUILD_MODEL_CONTAINER('{fq_model_name}', '{version_name}', '{compute_pool_name}',"
                f" '{fq_image_repo_name}', '{is_gpu_str}', '{force_rebuild_str}', '', '{external_access_integration}')"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def deploy_model(
        self,
        *,
        stage_path: str,
        model_deployment_spec_file_rel_path: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, snowpark.AsyncJob]:
        async_job = self._session.sql(
            f"CALL SYSTEM$DEPLOY_MODEL('@{stage_path}/{model_deployment_spec_file_rel_path}')"
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
        input_args: List[sql_identifier.SqlIdentifier],
        returns: List[Tuple[str, spt.DataType, sql_identifier.SqlIdentifier]],
        statement_params: Optional[Dict[str, Any]] = None,
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

        function_name = identifier.concat_names([service_name.identifier(), "_", method_name.identifier()])
        fully_qualified_function_name = identifier.get_schema_level_object_identifier(
            actual_database_name.identifier(),
            actual_schema_name.identifier(),
            function_name,
        )

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
        statement_params: Optional[Dict[str, Any]] = None,
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

    def get_service_status(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        include_message: bool = False,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ServiceStatus, Optional[str]]:
        system_func = "SYSTEM$GET_SERVICE_STATUS"
        rows = (
            query_result_checker.SqlResultValidator(
                self._session,
                f"CALL {system_func}('{self.fully_qualified_object_name(database_name, schema_name, service_name)}')",
                statement_params=statement_params,
            )
            .has_dimensions(expected_rows=1, expected_cols=1)
            .validate()
        )
        metadata = json.loads(rows[0][system_func])[0]
        if metadata and metadata["status"]:
            service_status = ServiceStatus(metadata["status"])
            message = metadata["message"] if include_message else None
            return service_status, message
        return ServiceStatus.UNKNOWN, None

    def drop_service(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        service_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            f"DROP SERVICE {self.fully_qualified_object_name(database_name, schema_name, service_name)}",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()
