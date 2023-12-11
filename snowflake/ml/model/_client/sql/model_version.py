import pathlib
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import ParseResult

from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.snowpark import dataframe, functions as F, session, types as spt
from snowflake.snowpark._internal import utils as snowpark_utils


def _normalize_url_for_sql(url: str) -> str:
    if url.startswith("'") and url.endswith("'"):
        url = url[1:-1]
    url = url.replace("'", "\\'")
    return f"'{url}'"


class ModelVersionSQLClient:
    def __init__(
        self,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
    ) -> None:
        self._session = session
        self._database_name = database_name
        self._schema_name = schema_name

    def fully_qualified_model_name(self, model_name: sql_identifier.SqlIdentifier) -> str:
        return identifier.get_schema_level_object_identifier(
            self._database_name.identifier(), self._schema_name.identifier(), model_name.identifier()
        )

    def create_from_stage(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        stage_path: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._version_name = version_name
        self._session.sql(
            f"CREATE MODEL {self.fully_qualified_model_name(model_name)} WITH VERSION {version_name.identifier()}"
            f" FROM {stage_path}"
        ).collect(statement_params=statement_params)

    # TODO(SNOW-987381): Merge with above when we have `create or alter module m [with] version v1 ...`
    def add_version_from_stage(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        stage_path: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._version_name = version_name
        self._session.sql(
            f"ALTER MODEL {self.fully_qualified_model_name(model_name)} ADD VERSION {version_name.identifier()}"
            f" FROM {stage_path}"
        ).collect(statement_params=statement_params)

    def get_file(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        file_path: pathlib.PurePosixPath,
        target_path: pathlib.Path,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> pathlib.Path:
        stage_location = pathlib.PurePosixPath(
            self.fully_qualified_model_name(model_name), "versions", version_name.resolved(), file_path
        ).as_posix()
        stage_location_url = ParseResult(
            scheme="snow", netloc="model", path=stage_location, params="", query="", fragment=""
        ).geturl()
        local_location = target_path.absolute().as_posix()
        local_location_url = ParseResult(
            scheme="file", netloc="", path=local_location, params="", query="", fragment=""
        ).geturl()

        self._session.sql(
            f"GET {_normalize_url_for_sql(stage_location_url)} {_normalize_url_for_sql(local_location_url)}"
        ).collect(statement_params=statement_params)
        return target_path / file_path.name

    def invoke_method(
        self,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        method_name: sql_identifier.SqlIdentifier,
        input_df: dataframe.DataFrame,
        input_args: List[sql_identifier.SqlIdentifier],
        returns: List[Tuple[str, spt.DataType, sql_identifier.SqlIdentifier]],
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> dataframe.DataFrame:
        tmp_table_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.TABLE)
        INTERMEDIATE_TABLE_NAME = identifier.get_schema_level_object_identifier(
            self._database_name.identifier(),
            self._schema_name.identifier(),
            tmp_table_name,
        )
        input_df.write.save_as_table(  # type: ignore[call-overload]
            table_name=INTERMEDIATE_TABLE_NAME,
            mode="errorifexists",
            table_type="temporary",
            statement_params=statement_params,
        )

        INTERMEDIATE_OBJ_NAME = "TMP_RESULT"

        module_version_alias = "MODEL_VERSION_ALIAS"
        model_version_alias_sql = (
            f"WITH {module_version_alias} AS "
            f"MODEL {self.fully_qualified_model_name(model_name)} VERSION {version_name.identifier()}"
        )

        args_sql_list = []
        for input_arg_value in input_args:
            args_sql_list.append(input_arg_value)

        args_sql = ", ".join(args_sql_list)

        sql = textwrap.dedent(
            f"""{model_version_alias_sql}
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
