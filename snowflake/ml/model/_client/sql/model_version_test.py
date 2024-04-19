import pathlib
from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.sql import model_version as model_version_sql
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
        self.m_session.add_mock_sql(f"""CREATE MODEL TEMP."test".MODEL WITH VERSION V1 FROM {stage_path}""", m_df)
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).create_from_stage(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V1"),
            stage_path=stage_path,
            statement_params=m_statement_params,
        )

    def test_add_version_from_stage(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully altered.")], collect_statement_params=m_statement_params
        )
        stage_path = '@TEMP."test".MODEL/V2'
        self.m_session.add_mock_sql(f"""ALTER MODEL TEMP."test".MODEL ADD VERSION V2 FROM {stage_path}""", m_df)
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).add_version_from_stage(
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
        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL SET DEFAULT_VERSION = V2""", m_df)
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).set_default_version(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V2"),
            statement_params=m_statement_params,
        )

    def test_set_comment(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(collect_result=[Row("")], collect_statement_params=m_statement_params)
        comment = "This is my comment"
        self.m_session.add_mock_sql(
            f"""ALTER MODEL TEMP."test".MODEL MODIFY VERSION "v1" SET COMMENT=$${comment}$$""", m_df
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).set_comment(
            comment=comment,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )

    def test_get_file(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row(file="946964364/MANIFEST.yml", size=419, status="DOWNLOADED", message="")],
            collect_statement_params=m_statement_params,
        )
        path = pathlib.Path("/tmp").resolve().as_posix()
        self.m_session.add_mock_sql(
            f"""GET 'snow://model/TEMP."test".MODEL/versions/v1/model.yaml' 'file://{path}'""", m_df
        )
        c_session = cast(Session, self.m_session)
        res = model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).get_file(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            file_path=pathlib.PurePosixPath("model.yaml"),
            target_path=pathlib.Path("/tmp"),
            statement_params=m_statement_params,
        )
        self.assertEqual(res, pathlib.Path("/tmp/model.yaml"))

    def test_invoke_method(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """WITH MODEL_VERSION_ALIAS AS MODEL TEMP."test".MODEL VERSION V1
            SELECT *,
                MODEL_VERSION_ALIAS!PREDICT(COL1, COL2) AS TMP_RESULT
            FROM TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT")
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.__setattr__("queries", {"queries": ["query_1", "query_2"], "post_actions": []})
        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_TABLE_ABCDEF0123"
        ) as mock_random_name_for_temp_object:
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_method(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                statement_params=m_statement_params,
            )
            mock_random_name_for_temp_object.assert_called_once_with(snowpark_utils.TempObjectType.TABLE)
            mock_save_as_table.assert_called_once_with(
                table_name='TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123',
                mode="errorifexists",
                table_type="temporary",
                statement_params=m_statement_params,
            )

    def test_invoke_method_1(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """WITH MODEL_VERSION_ALIAS AS MODEL TEMP."test".MODEL VERSION V1
            SELECT *,
                MODEL_VERSION_ALIAS!PREDICT(COL1, COL2) AS TMP_RESULT
            FROM TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT")
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.__setattr__("queries", {"queries": ["query_1"], "post_actions": ["query_2"]})
        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_TABLE_ABCDEF0123"
        ) as mock_random_name_for_temp_object:
            model_version_sql.ModelVersionSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_method(
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("V1"),
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                statement_params=m_statement_params,
            )
            mock_random_name_for_temp_object.assert_called_once_with(snowpark_utils.TempObjectType.TABLE)
            mock_save_as_table.assert_called_once_with(
                table_name='TEMP."test".SNOWPARK_TEMP_TABLE_ABCDEF0123',
                mode="errorifexists",
                table_type="temporary",
                statement_params=m_statement_params,
            )

    def test_invoke_method_2(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()
        self.m_session.add_mock_sql(
            """WITH SNOWPARK_ML_MODEL_INFERENCE_INPUT AS (query_1),
            MODEL_VERSION_ALIAS AS MODEL TEMP."test".MODEL VERSION V1
            SELECT *,
                MODEL_VERSION_ALIAS!PREDICT(COL1, COL2) AS TMP_RESULT
            FROM SNOWPARK_ML_MODEL_INFERENCE_INPUT""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT")
        c_session = cast(Session, self.m_session)
        m_df.__setattr__("queries", {"queries": ["query_1"], "post_actions": []})
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).invoke_method(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V1"),
            method_name=sql_identifier.SqlIdentifier("PREDICT"),
            input_df=cast(DataFrame, m_df),
            input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
            returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
            statement_params=m_statement_params,
        )

    def test_set_metadata(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(collect_result=[Row("")], collect_statement_params=m_statement_params)
        metadata = {"metrics": {"a": 1, "c": "This is my comment"}, "other": 2.0}
        self.m_session.add_mock_sql(
            """ALTER MODEL TEMP."test".MODEL MODIFY VERSION "v1"
            SET METADATA=$${"metrics": {"a": 1, "c": "This is my comment"}, "other": 2.0}$$""",
            m_df,
        )
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).set_metadata(
            metadata,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )

    def test_drop_version(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully altered.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL DROP VERSION V2""", m_df)
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).drop_version(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V2"),
            statement_params=m_statement_params,
        )

    def test_show_functions(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row(name="foo")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""SHOW FUNCTIONS IN MODEL TEMP."test".MODEL VERSION "v1" """, m_df)
        c_session = cast(Session, self.m_session)
        model_version_sql.ModelVersionSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_functions(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )


if __name__ == "__main__":
    absltest.main()
