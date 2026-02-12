import copy
import os
import pathlib
from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.sql import model_version as model_version_sql
from snowflake.ml.model._model_composer.model_method import constants
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import DataFrame, Row, Session, functions as F, types as spt
from snowflake.snowpark._internal import utils as snowpark_utils


class ModelVersionSQLTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)

    def test_create_from_stage(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully created.")], collect_statement_params=m_statement_params
        )
        stage_path = '@TEMP."test".MODEL/V1'
        self.m_session.add_mock_sql(
            f"""CREATE MODEL TEMP."test".MODEL WITH VERSION V1 FROM {stage_path}""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).create_from_stage(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V1"),
            stage_path=stage_path,
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(
            f"""CREATE MODEL TEMP."test".MODEL WITH VERSION V1 FROM {stage_path}""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).create_from_stage(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V1"),
            stage_path=stage_path,
            statement_params=m_statement_params,
        )

    def test_create_from_model_version(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully created.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql(
            (
                'CREATE MODEL TEMP."test".MODEL WITH VERSION V1'
                ' FROM MODEL SOURCE_TEMP."source_test".SOURCE_MODEL VERSION SOURCE_VERSION'
            ),
            m_df,
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).create_from_model_version(
            source_database_name=sql_identifier.SqlIdentifier("SOURCE_TEMP"),
            source_schema_name=sql_identifier.SqlIdentifier("source_test", case_sensitive=True),
            source_model_name=sql_identifier.SqlIdentifier("SOURCE_MODEL"),
            source_version_name=sql_identifier.SqlIdentifier("SOURCE_VERSION"),
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V1"),
            statement_params=m_statement_params,
        )

    def test_add_version_from_model_version(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully created.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql(
            (
                'ALTER MODEL TEMP."test".MODEL ADD VERSION V1'
                ' FROM MODEL SOURCE_TEMP."source_test".SOURCE_MODEL VERSION SOURCE_VERSION'
            ),
            m_df,
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).add_version_from_model_version(
            source_database_name=sql_identifier.SqlIdentifier("SOURCE_TEMP"),
            source_schema_name=sql_identifier.SqlIdentifier("source_test", case_sensitive=True),
            source_model_name=sql_identifier.SqlIdentifier("SOURCE_MODEL"),
            source_version_name=sql_identifier.SqlIdentifier("SOURCE_VERSION"),
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V1"),
            statement_params=m_statement_params,
        )

    def test_add_version_from_stage(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully altered.")], collect_statement_params=m_statement_params
        )
        stage_path = '@TEMP."test".MODEL/V2'
        self.m_session.add_mock_sql(
            f"""ALTER MODEL TEMP."test".MODEL ADD VERSION V2 FROM {stage_path}""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).add_version_from_stage(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V2"),
            stage_path=stage_path,
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(
            f"""ALTER MODEL TEMP."test".MODEL ADD VERSION V2 FROM {stage_path}""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).add_version_from_stage(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V2"),
            stage_path=stage_path,
            statement_params=m_statement_params,
        )

    def test_set_default_version(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully altered.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL SET DEFAULT_VERSION = V2""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).set_default_version(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V2"),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL SET DEFAULT_VERSION = V2""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).set_default_version(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V2"),
            statement_params=m_statement_params,
        )

    def test_set_comment(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(collect_result=[Row("")], collect_statement_params=m_statement_params)
        comment = "This is my comment"
        self.m_session.add_mock_sql(
            f"""ALTER MODEL TEMP."test".MODEL MODIFY VERSION "v1" SET COMMENT=$${comment}$$""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).set_comment(
            comment=comment,
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(
            f"""ALTER MODEL TEMP."test".MODEL MODIFY VERSION "v1" SET COMMENT=$${comment}$$""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).set_comment(
            comment=comment,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )

    def test_list_file(self) -> None:
        m_statement_params = {"test": "1"}
        m_res = [Row(name="versions/v1/model.yaml", size=419, md5="1234", last_modified="")]
        m_df = mock_data_frame.MockDataFrame(
            collect_result=m_res,
            collect_statement_params=m_statement_params,
        )
        self.m_session.add_mock_sql(
            """LIST 'snow://model/TEMP."test".MODEL/versions/v1/model.yaml'""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        res = model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).list_file(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            file_path=pathlib.PurePosixPath("model.yaml"),
            statement_params=m_statement_params,
        )
        self.assertEqual(res, m_res)

        self.m_session.add_mock_sql(
            """LIST 'snow://model/TEMP."test".MODEL/versions/v1/model.yaml'""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        res = model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).list_file(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            file_path=pathlib.PurePosixPath("model.yaml"),
            statement_params=m_statement_params,
        )
        self.assertEqual(res, m_res)

    def test_list_file_curdir_dir(self) -> None:
        m_statement_params = {"test": "1"}
        m_res = [Row(name="versions/v1/MANIFEST.yml", size=419, md5="1234", last_modified="")]
        m_df = mock_data_frame.MockDataFrame(
            collect_result=m_res,
            collect_statement_params=m_statement_params,
        )
        self.m_session.add_mock_sql("""LIST 'snow://model/TEMP."test".MODEL/versions/v1/'""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        res = model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).list_file(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            file_path=pathlib.PurePosixPath(os.curdir),
            is_dir=True,
            statement_params=m_statement_params,
        )
        self.assertEqual(res, m_res)

    def test_get_file(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row(file="946964364/MANIFEST.yml", size=419, status="DOWNLOADED", message="")],
            collect_statement_params=m_statement_params,
        )
        path = pathlib.Path("/tmp").resolve().as_posix()
        self.m_session.add_mock_sql(
            f"""GET 'snow://model/TEMP."test".MODEL/versions/v1/model.yaml' 'file://{path}'""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        res = model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).get_file(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            file_path=pathlib.PurePosixPath("model.yaml"),
            target_path=pathlib.Path("/tmp"),
            statement_params=m_statement_params,
        )
        self.assertEqual(res, pathlib.Path("/tmp/model.yaml"))

        self.m_session.add_mock_sql(
            f"""GET 'snow://model/TEMP."test".MODEL/versions/v1/model.yaml' 'file://{path}'""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        res = model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).get_file(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            file_path=pathlib.PurePosixPath("model.yaml"),
            target_path=pathlib.Path("/tmp"),
            statement_params=m_statement_params,
        )

    def test_invoke_function_method(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """
            SELECT *,
                MODEL(TEMP."test".MODEL, V1)!PREDICT(COL1, COL2) AS TMP_RESULT_ABCDEF0123
            FROM TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT_ABCDEF0123")
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.add_query("queries", "query_1")
        m_df.add_query("queries", "query_2")
        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"
        ):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_function_method(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                statement_params=m_statement_params,
            )
            mock_save_as_table.assert_called_once_with(
                table_name='TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123',
                mode="errorifexists",
                table_type="temporary",
                statement_params=m_statement_params,
            )

    def test_invoke_function_method_fully_qualified(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """
            SELECT *,
                MODEL(TEMP."test".MODEL, V1)!PREDICT(COL1, COL2) AS TMP_RESULT_ABCDEF0123
            FROM TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT_ABCDEF0123")
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.add_query("queries", "query_1")
        m_df.add_query("queries", "query_2")
        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"
        ):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("foo"),
                schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
            ).invoke_function_method(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                statement_params=m_statement_params,
            )
            mock_save_as_table.assert_called_once_with(
                table_name='TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123',
                mode="errorifexists",
                table_type="temporary",
                statement_params=m_statement_params,
            )

    def test_invoke_function_method_1(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """
            SELECT *,
                MODEL(TEMP."test".MODEL, V1)!PREDICT(COL1, COL2) AS TMP_RESULT_ABCDEF0123
            FROM TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT_ABCDEF0123")
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.add_query("queries", "query_1")
        m_df.add_query("queries", "query_2")
        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"
        ):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_function_method(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                statement_params=m_statement_params,
            )
            mock_save_as_table.assert_called_once_with(
                table_name='TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123',
                mode="errorifexists",
                table_type="temporary",
                statement_params=m_statement_params,
            )

    def test_invoke_function_method_2(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """WITH SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123 AS (query_1)
            SELECT *,
                MODEL(TEMP."test".MODEL, V1)!PREDICT(COL1, COL2) AS TMP_RESULT_ABCDEF0123
            FROM SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT_ABCDEF0123")
        c_session = cast(Session, self.m_session)
        m_df.add_query("queries", "query_1")
        with mock.patch.object(snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_function_method(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                statement_params=m_statement_params,
            )

    def test_invoke_function_method_with_parameters(self) -> None:
        """Test invoke_function_method with optional parameters."""
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """WITH SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123 AS (query_1)
            SELECT *,
                MODEL(TEMP."test".MODEL, V1)!PREDICT(COL1, COL2, 0.7, 50)
                    AS TMP_RESULT_ABCDEF0123
            FROM SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT_ABCDEF0123")
        c_session = cast(Session, self.m_session)
        m_df.add_query("queries", "query_1")
        with mock.patch.object(snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_function_method(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                statement_params=m_statement_params,
                params=[
                    (sql_identifier.SqlIdentifier("TEMPERATURE"), 0.7),
                    (sql_identifier.SqlIdentifier("TOP_K"), 50),
                ],
            )

    def test_invoke_function_method_with_string_parameter(self) -> None:
        """Test invoke_function_method with string parameter value."""
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """WITH SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123 AS (query_1)
            SELECT *,
                MODEL(TEMP."test".MODEL, V1)!PREDICT(COL1, 'gpt-4')
                    AS TMP_RESULT_ABCDEF0123
            FROM SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT_ABCDEF0123")
        c_session = cast(Session, self.m_session)
        m_df.add_query("queries", "query_1")
        with mock.patch.object(snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_function_method(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                statement_params=m_statement_params,
                params=[
                    (sql_identifier.SqlIdentifier("MODEL_NAME"), "gpt-4"),
                ],
            )

    def test_invoke_table_function_method_no_partition_col(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """
            SELECT *,
            FROM TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123,
                 TABLE(MODEL(TEMP."test".MODEL, V1)!EXPLAIN(COL1, COL2))
            """,
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")])
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.add_query("queries", "query_1")
        m_df.add_query("queries", "query_2")
        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"
        ):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_table_function_method(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("EXPLAIN"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                partition_column=None,
                statement_params=m_statement_params,
                is_partitioned=False,
            )
            mock_save_as_table.assert_called_once_with(
                table_name='TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123',
                mode="errorifexists",
                table_type="temporary",
                statement_params=m_statement_params,
            )

    def test_invoke_table_function_method_partition_col(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        partition_column = "partition_col"
        self.m_session.add_mock_sql(
            f"""
            SELECT *,
            FROM TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123,
                 TABLE(MODEL(TEMP."test".MODEL, V1)!PREDICT_TABLE(COL1, COL2) OVER (PARTITION BY {partition_column}))
            """,
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")])
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.add_query("queries", "query_1")
        m_df.add_query("queries", "query_2")
        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"
        ):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_table_function_method(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT_TABLE"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                partition_column=sql_identifier.SqlIdentifier(partition_column),
                statement_params=m_statement_params,
            )
            mock_save_as_table.assert_called_once_with(
                table_name='TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123',
                mode="errorifexists",
                table_type="temporary",
                statement_params=m_statement_params,
            )

    def test_invoke_table_function_method_partition_col_fully_qualified(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        partition_column = "partition_col"
        self.m_session.add_mock_sql(
            f"""
            SELECT *,
            FROM TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123,
                 TABLE(MODEL(TEMP."test".MODEL, V1)!PREDICT_TABLE(COL1, COL2) OVER (PARTITION BY {partition_column}))
            """,
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")])
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.add_query("queries", "query_1")
        m_df.add_query("queries", "query_2")
        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"
        ):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("foo"),
                schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
            ).invoke_table_function_method(
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT_TABLE"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                partition_column=sql_identifier.SqlIdentifier(partition_column),
                statement_params=m_statement_params,
            )
            mock_save_as_table.assert_called_once_with(
                table_name='TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123',
                mode="errorifexists",
                table_type="temporary",
                statement_params=m_statement_params,
            )

    def test_invoke_table_function_method_with_parameters(self) -> None:
        """Test invoke_table_function_method with optional parameters."""
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        partition_column = "partition_col"
        self.m_session.add_mock_sql(
            f"""
            SELECT *,
            FROM TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123,
                 TABLE(MODEL(TEMP."test".MODEL, V1)!PREDICT_TABLE(COL1, COL2, 0.5, 40)
                     OVER (PARTITION BY {partition_column}))
            """,
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")])
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.add_query("queries", "query_1")
        m_df.add_query("queries", "query_2")
        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"
        ):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_table_function_method(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT_TABLE"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                partition_column=sql_identifier.SqlIdentifier(partition_column),
                statement_params=m_statement_params,
                params=[
                    (sql_identifier.SqlIdentifier("TEMPERATURE"), 0.5),
                    (sql_identifier.SqlIdentifier("TOP_K"), 40),
                ],
            )
            mock_save_as_table.assert_called_once_with(
                table_name='TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123',
                mode="errorifexists",
                table_type="temporary",
                statement_params=m_statement_params,
            )

    def test_invoke_table_function_method_with_string_parameter(self) -> None:
        """Test invoke_table_function_method with string parameter value."""
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """WITH SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123 AS (query_1)
            SELECT *,
            FROM SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123,
                 TABLE(MODEL(TEMP."test".MODEL, V1)!PREDICT_TABLE(COL1, 'gpt-4') OVER (PARTITION BY 1))
            """,
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")])
        c_session = cast(Session, self.m_session)
        m_df.add_query("queries", "query_1")
        with mock.patch.object(snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"):
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_table_function_method(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT_TABLE"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                partition_column=None,
                statement_params=m_statement_params,
                params=[
                    (sql_identifier.SqlIdentifier("MODEL_NAME"), "gpt-4"),
                ],
            )

    def test_invoke_function_method_wide_input_with_params(self) -> None:
        """Test invoke_function_method with wide input format including params."""
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """WITH SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123 AS (query_1)
            SELECT *,
                MODEL(TEMP."test".MODEL, V1)!PREDICT(object_construct_keep_null("""
            """'COL1', COL1, 'COL2', COL2, 'TEMPERATURE', 0.7, 'TOP_K', 50))
                    AS TMP_RESULT_ABCDEF0123
            FROM SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT_ABCDEF0123")
        c_session = cast(Session, self.m_session)
        m_df.add_query("queries", "query_1")
        with mock.patch.object(snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"):
            with mock.patch.object(constants, "SNOWPARK_UDF_INPUT_COL_LIMIT", 2):
                model_version_sql.ModelVersionSQLClient(
                    c_session,
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                ).invoke_function_method(
                    database_name=None,
                    schema_name=None,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    method_name=sql_identifier.SqlIdentifier("PREDICT"),
                    input_df=cast(DataFrame, m_df),
                    input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                    returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                    statement_params=m_statement_params,
                    params=[
                        (sql_identifier.SqlIdentifier("TEMPERATURE"), 0.7),
                        (sql_identifier.SqlIdentifier("TOP_K"), 50),
                    ],
                )

    def test_invoke_table_function_method_wide_input_with_params(self) -> None:
        """Test invoke_table_function_method with wide input format including params."""
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        partition_column = "partition_col"
        self.m_session.add_mock_sql(
            f"""
            SELECT *,
            FROM TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123,
                 TABLE(MODEL(TEMP."test".MODEL, V1)!PREDICT_TABLE(object_construct_keep_null("""
            f"""'COL1', COL1, 'COL2', COL2, 'TEMPERATURE', 0.5))
                     OVER (PARTITION BY {partition_column}))
            """,
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")])
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.add_query("queries", "query_1")
        m_df.add_query("queries", "query_2")
        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"
        ):
            with mock.patch.object(constants, "SNOWPARK_UDF_INPUT_COL_LIMIT", 2):
                model_version_sql.ModelVersionSQLClient(
                    c_session,
                    database_name=sql_identifier.SqlIdentifier("TEMP"),
                    schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
                ).invoke_table_function_method(
                    database_name=None,
                    schema_name=None,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("V1"),
                    method_name=sql_identifier.SqlIdentifier("PREDICT_TABLE"),
                    input_df=cast(DataFrame, m_df),
                    input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                    returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                    partition_column=sql_identifier.SqlIdentifier(partition_column),
                    statement_params=m_statement_params,
                    params=[
                        (sql_identifier.SqlIdentifier("TEMPERATURE"), 0.5),
                    ],
                )
                mock_save_as_table.assert_called_once_with(
                    table_name='TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123',
                    mode="errorifexists",
                    table_type="temporary",
                    statement_params=m_statement_params,
                )

    def test_set_metadata(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(collect_result=[Row("")], collect_statement_params=m_statement_params)
        metadata = {"metrics": {"a": 1, "c": "This is my comment"}, "other": 2.0}
        self.m_session.add_mock_sql(
            """ALTER MODEL TEMP."test".MODEL MODIFY VERSION "v1"
            SET METADATA=$${"metrics": {"a": 1, "c": "This is my comment"}, "other": 2.0}$$""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).set_metadata(
            metadata,
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(
            """ALTER MODEL TEMP."test".MODEL MODIFY VERSION "v1"
            SET METADATA=$${"metrics": {"a": 1, "c": "This is my comment"}, "other": 2.0}$$""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).set_metadata(
            metadata,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )

    def test_drop_version(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully altered.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL DROP VERSION V2""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).drop_version(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V2"),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL DROP VERSION V2""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).drop_version(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V2"),
            statement_params=m_statement_params,
        )

    def test_show_functions(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row(name="foo")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""SHOW FUNCTIONS IN MODEL TEMP."test".MODEL VERSION "v1" """, copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_functions(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql("""SHOW FUNCTIONS IN MODEL TEMP."test".MODEL VERSION "v1" """, copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).show_functions(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )

    def test_set_alias(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(collect_result=[Row("")], collect_statement_params=m_statement_params)
        alias = "ally"
        self.m_session.add_mock_sql(
            """ALTER MODEL TEMP."test".MODEL VERSION "v1" SET ALIAS=ALLY""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).set_alias(
            alias_name=sql_identifier.SqlIdentifier(alias),
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(
            """ALTER MODEL TEMP."test".MODEL VERSION "v1" SET ALIAS=ALLY""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).set_alias(
            alias_name=sql_identifier.SqlIdentifier(alias),
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )

    def test_unset_alias(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(collect_result=[Row("")], collect_statement_params=m_statement_params)
        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL VERSION V1 UNSET ALIAS""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).unset_alias(
            version_or_alias_name=sql_identifier.SqlIdentifier("v1"),
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL VERSION ALLY UNSET ALIAS""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        alias = "ally"
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).unset_alias(
            version_or_alias_name=sql_identifier.SqlIdentifier(alias),
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )


if __name__ == "__main__":
    absltest.main()
