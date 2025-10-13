import json
import pathlib
import textwrap
from typing import Any, Optional
from urllib.parse import ParseResult

from snowflake.ml._internal.utils import (
    identifier,
    query_result_checker,
    sql_identifier,
)
from snowflake.ml.model._client.sql import _base
from snowflake.ml.model._model_composer.model_method import constants
from snowflake.snowpark import dataframe, functions as F, row, types as spt
from snowflake.snowpark._internal import utils as snowpark_utils


def _normalize_url_for_sql(url: str) -> str:
    if url.startswith("'") and url.endswith("'"):
        url = url[1:-1]
    url = url.replace("'", "\\'")
    return f"'{url}'"


class ModelVersionSQLClient(_base._BaseSQLClient):
    FUNCTION_NAME_COL_NAME = "name"
    FUNCTION_RETURN_TYPE_COL_NAME = "return_type"

    def create_from_stage(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        stage_path: str,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"CREATE MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
                f" WITH VERSION {version_name.identifier()} FROM {stage_path}"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def create_from_model_version(
        self,
        *,
        source_database_name: Optional[sql_identifier.SqlIdentifier],
        source_schema_name: Optional[sql_identifier.SqlIdentifier],
        source_model_name: sql_identifier.SqlIdentifier,
        source_version_name: sql_identifier.SqlIdentifier,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        fq_source_model_name = self.fully_qualified_object_name(
            source_database_name, source_schema_name, source_model_name
        )
        fq_model_name = self.fully_qualified_object_name(database_name, schema_name, model_name)
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"CREATE MODEL {fq_model_name} WITH VERSION {version_name} FROM MODEL {fq_source_model_name}"
                f" VERSION {source_version_name}"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def create_live_version(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        sql = (
            f"CREATE MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
            f" WITH LIVE VERSION {version_name.identifier()}"
        )
        query_result_checker.SqlResultValidator(
            self._session,
            sql,
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def add_live_version(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        sql = (
            f"ALTER MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
            f" ADD LIVE VERSION {version_name.identifier()}"
        )
        query_result_checker.SqlResultValidator(
            self._session,
            sql,
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def commit_version(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        sql = (
            f"ALTER MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
            f" COMMIT VERSION {version_name.identifier()}"
        )

        query_result_checker.SqlResultValidator(
            self._session,
            sql,
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    # TODO(SNOW-987381): Merge with above when we have `create or alter module m [with] version v1 ...`
    def add_version_from_stage(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        stage_path: str,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"ALTER MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
                f" ADD VERSION {version_name.identifier()} FROM {stage_path}"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def add_version_from_model_version(
        self,
        *,
        source_database_name: Optional[sql_identifier.SqlIdentifier],
        source_schema_name: Optional[sql_identifier.SqlIdentifier],
        source_model_name: sql_identifier.SqlIdentifier,
        source_version_name: sql_identifier.SqlIdentifier,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        fq_source_model_name = self.fully_qualified_object_name(
            source_database_name, source_schema_name, source_model_name
        )
        fq_model_name = self.fully_qualified_object_name(database_name, schema_name, model_name)
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"ALTER MODEL {fq_model_name} ADD VERSION {version_name} FROM MODEL {fq_source_model_name}"
                f" VERSION {source_version_name}"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def set_default_version(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"ALTER MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)} "
                f"SET DEFAULT_VERSION = {version_name.identifier()}"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def set_alias(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        alias_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"ALTER MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)} "
                f"VERSION {version_name.identifier()} SET ALIAS = {alias_name.identifier()}"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def unset_alias(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_or_alias_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"ALTER MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)} "
                f"VERSION {version_or_alias_name.identifier()} UNSET ALIAS"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def list_file(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        file_path: pathlib.PurePosixPath,
        is_dir: bool = False,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[row.Row]:
        # Workaround for snowURL bug.
        trailing_slash = "/" if is_dir else ""

        stage_location = (
            pathlib.PurePosixPath(
                self.fully_qualified_object_name(database_name, schema_name, model_name),
                "versions",
                version_name.resolved(),
                file_path,
            ).as_posix()
            + trailing_slash
        )
        stage_location_url = ParseResult(
            scheme="snow", netloc="model", path=stage_location, params="", query="", fragment=""
        ).geturl()

        return (
            query_result_checker.SqlResultValidator(
                self._session,
                f"List {_normalize_url_for_sql(stage_location_url)}",
                statement_params=statement_params,
            )
            .has_column("name", allow_empty=True)
            .validate()
        )

    def get_file(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        file_path: pathlib.PurePosixPath,
        target_path: pathlib.Path,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> pathlib.Path:
        stage_location = pathlib.PurePosixPath(
            self.fully_qualified_object_name(database_name, schema_name, model_name),
            "versions",
            version_name.resolved(),
            file_path,
        ).as_posix()
        stage_location_url = ParseResult(
            scheme="snow", netloc="model", path=stage_location, params="", query="", fragment=""
        ).geturl()
        local_location = target_path.resolve().as_posix()
        local_location_url = f"file://{local_location}"

        if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            options = {"parallel": 10}
            cursor = self._session._conn._cursor
            cursor._download(stage_location_url, str(target_path), options)
            cursor.fetchall()
        else:
            query_result_checker.SqlResultValidator(
                self._session,
                f"GET {_normalize_url_for_sql(stage_location_url)} {_normalize_url_for_sql(local_location_url)}",
                statement_params=statement_params,
            ).has_dimensions(expected_rows=1).validate()
        return target_path / file_path.name

    def show_functions(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> list[row.Row]:
        res = query_result_checker.SqlResultValidator(
            self._session,
            (
                f"SHOW FUNCTIONS IN MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
                f" VERSION {version_name.identifier()}"
            ),
            statement_params=statement_params,
        ).has_column(ModelVersionSQLClient.FUNCTION_NAME_COL_NAME, allow_empty=True)

        return res.validate()

    def set_comment(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        comment: str,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"ALTER MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)} "
                f"MODIFY VERSION {version_name.identifier()} SET COMMENT=$${comment}$$"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def invoke_function_method(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        method_name: sql_identifier.SqlIdentifier,
        input_df: dataframe.DataFrame,
        input_args: list[sql_identifier.SqlIdentifier],
        returns: list[tuple[str, spt.DataType, sql_identifier.SqlIdentifier]],
        statement_params: Optional[dict[str, Any]] = None,
    ) -> dataframe.DataFrame:
        with_statements = []
        if len(input_df.queries["queries"]) == 1 and len(input_df.queries["post_actions"]) == 0:
            INTERMEDIATE_TABLE_NAME = ModelVersionSQLClient.get_tmp_name_with_prefix(
                "SNOWPARK_ML_MODEL_INFERENCE_INPUT"
            )
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

        INTERMEDIATE_OBJ_NAME = ModelVersionSQLClient.get_tmp_name_with_prefix("TMP_RESULT")

        module_version_alias = ModelVersionSQLClient.get_tmp_name_with_prefix("MODEL_VERSION_ALIAS")
        with_statements.append(
            f"{module_version_alias} AS "
            f"MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
            f" VERSION {version_name.identifier()}"
        )

        args_sql_list = []
        for input_arg_value in input_args:
            args_sql_list.append(input_arg_value)

        args_sql = ", ".join(args_sql_list)

        wide_input = len(input_args) > constants.SNOWPARK_UDF_INPUT_COL_LIMIT
        if wide_input:
            input_args_sql = ", ".join(f"'{arg}', {arg.identifier()}" for arg in input_args)
            args_sql = f"object_construct_keep_null({input_args_sql})"

        sql = textwrap.dedent(
            f"""WITH {','.join(with_statements)}
                SELECT *,
                    {module_version_alias}!{method_name.identifier()}({args_sql}) AS {INTERMEDIATE_OBJ_NAME}
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

    def invoke_table_function_method(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        method_name: sql_identifier.SqlIdentifier,
        input_df: dataframe.DataFrame,
        input_args: list[sql_identifier.SqlIdentifier],
        returns: list[tuple[str, spt.DataType, sql_identifier.SqlIdentifier]],
        partition_column: Optional[sql_identifier.SqlIdentifier],
        statement_params: Optional[dict[str, Any]] = None,
        is_partitioned: bool = True,
        explain_case_sensitive: bool = False,
    ) -> dataframe.DataFrame:
        with_statements = []
        if len(input_df.queries["queries"]) == 1 and len(input_df.queries["post_actions"]) == 0:
            INTERMEDIATE_TABLE_NAME = (
                f"SNOWPARK_ML_MODEL_INFERENCE_INPUT_{snowpark_utils.generate_random_alphanumeric().upper()}"
            )
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

        module_version_alias = f"MODEL_VERSION_ALIAS_{snowpark_utils.generate_random_alphanumeric().upper()}"
        with_statements.append(
            f"{module_version_alias} AS "
            f"MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
            f" VERSION {version_name.identifier()}"
        )

        partition_by = partition_column.identifier() if partition_column is not None else "1"

        args_sql_list = []
        for input_arg_value in input_args:
            args_sql_list.append(input_arg_value)

        args_sql = ", ".join(args_sql_list)

        wide_input = len(input_args) > constants.SNOWPARK_UDF_INPUT_COL_LIMIT
        if wide_input:
            input_args_sql = ", ".join(f"'{arg}', {arg.identifier()}" for arg in input_args)
            args_sql = f"object_construct_keep_null({input_args_sql})"

        sql = textwrap.dedent(
            f"""WITH {','.join(with_statements)}
                SELECT *,
                FROM {INTERMEDIATE_TABLE_NAME},
                    TABLE({module_version_alias}!{method_name.identifier()}({args_sql}))"""
        )

        if is_partitioned or partition_column is not None:
            sql = textwrap.dedent(
                f"""WITH {','.join(with_statements)}
                    SELECT *,
                    FROM {INTERMEDIATE_TABLE_NAME},
                        TABLE({module_version_alias}!{method_name.identifier()}({args_sql})
                        OVER (PARTITION BY {partition_by}))"""
            )

        output_df = self._session.sql(sql)

        # Prepare the output
        output_cols = []
        output_names = []
        cols_to_drop = []

        for output_name, output_type, output_col_name in returns:
            case_sensitive = "explain" in method_name.resolved().lower() and explain_case_sensitive
            output_identifier = sql_identifier.SqlIdentifier(output_name, case_sensitive=case_sensitive).identifier()
            if output_identifier != output_col_name:
                cols_to_drop.append(output_identifier)
            output_cols.append(F.col(output_identifier).astype(output_type))
            output_names.append(output_col_name)

        if partition_column is not None:
            output_cols.append(F.col(partition_column.identifier()))
            output_names.append(partition_column)

        output_df = output_df.with_columns(
            col_names=output_names,
            values=output_cols,
        )
        if statement_params:
            output_df._statement_params = statement_params  # type: ignore[assignment]

        if cols_to_drop:
            output_df = output_df.drop(cols_to_drop)

        return output_df

    def set_metadata(
        self,
        metadata_dict: dict[str, Any],
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        json_metadata = json.dumps(metadata_dict)
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"ALTER MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
                f" MODIFY VERSION {version_name.identifier()} SET METADATA=$${json_metadata}$$"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()

    def drop_version(
        self,
        *,
        database_name: Optional[sql_identifier.SqlIdentifier],
        schema_name: Optional[sql_identifier.SqlIdentifier],
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        query_result_checker.SqlResultValidator(
            self._session,
            (
                f"ALTER MODEL {self.fully_qualified_object_name(database_name, schema_name, model_name)}"
                f" DROP VERSION {version_name.identifier()}"
            ),
            statement_params=statement_params,
        ).has_dimensions(expected_rows=1, expected_cols=1).validate()
