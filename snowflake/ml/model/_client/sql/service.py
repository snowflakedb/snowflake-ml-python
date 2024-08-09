import textwrap
from typing import Any, Dict, List, Optional, Tuple

from snowflake.ml._internal.utils import (
    identifier,
    query_result_checker,
    sql_identifier,
)
from snowflake.ml.model._client.sql import _base
from snowflake.snowpark import dataframe, functions as F, types as spt
from snowflake.snowpark._internal import utils as snowpark_utils


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
        fq_model_name = self.fully_qualified_object_name(database_name, schema_name, model_name)
        fq_image_repo_name = "/" + "/".join(
            [
                actual_image_repo_database.identifier(),
                actual_image_repo_schema.identifier(),
                image_repo_name.identifier(),
            ]
        )
        is_gpu = gpu is not None
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"CALL SYSTEM$BUILD_MODEL_CONTAINER('{fq_model_name}', '{version_name}', '{compute_pool_name}',"
                f" '{fq_image_repo_name}', '{is_gpu}', '{force_rebuild}', '', '{external_access_integration}')"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def deploy_model(
        self,
        *,
        stage_path: str,
        model_deployment_spec_file_rel_path: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            f"CALL SYSTEM$DEPLOY_MODEL('@{stage_path}/{model_deployment_spec_file_rel_path}')",
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

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
        if len(input_df.queries["queries"]) == 1 and len(input_df.queries["post_actions"]) == 0:
            INTERMEDIATE_TABLE_NAME = "SNOWPARK_ML_MODEL_INFERENCE_INPUT"
            with_statements.append(f"{INTERMEDIATE_TABLE_NAME} AS ({input_df.queries['queries'][0]})")
        else:
            actual_database_name = database_name or self._database_name
            actual_schema_name = schema_name or self._schema_name
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

        INTERMEDIATE_OBJ_NAME = "TMP_RESULT"

        with_sql = f"WITH {','.join(with_statements)}" if with_statements else ""
        args_sql_list = []
        for input_arg_value in input_args:
            args_sql_list.append(input_arg_value)
        args_sql = ", ".join(args_sql_list)

        sql = textwrap.dedent(
            f"""{with_sql}
                SELECT *,
                    {service_name.identifier()}_{method_name.identifier()}({args_sql}) AS {INTERMEDIATE_OBJ_NAME}
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
