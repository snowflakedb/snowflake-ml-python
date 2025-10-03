import copy
from typing import cast

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.sql import stage as stage_sql
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session


class StageSQLTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)

    def test_create_tmp_stage(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Stage MODEL successfully created.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""CREATE SCOPED TEMPORARY STAGE TEMP."test".MODEL""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        stage_sql.StageSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).create_tmp_stage(
            database_name=None,
            schema_name=None,
            stage_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql("""CREATE SCOPED TEMPORARY STAGE TEMP."test".MODEL""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        stage_sql.StageSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).create_tmp_stage(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            stage_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )

    def test_list_stage_with_files(self) -> None:
        """Test list_stage when stage contains files."""
        stage_location = "@test_stage/"
        mock_files = [
            Row(name="file1.txt", size=1024, md5="abc123", last_modified="2023-01-01 12:00:00"),
            Row(name="file2.txt", size=2048, md5="def456", last_modified="2023-01-01 12:30:00"),
            Row(name="subdir/file3.txt", size=512, md5="ghi789", last_modified="2023-01-01 13:00:00"),
        ]
        m_df = mock_data_frame.MockDataFrame(collect_result=mock_files)
        self.m_session.add_mock_sql(f"LIST {stage_location}", copy.deepcopy(m_df))

        c_session = cast(Session, self.m_session)
        stage_client = stage_sql.StageSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        )

        result = stage_client.list_stage(stage_location)
        self.assertEqual(result, mock_files)
        self.assertEqual(len(result), 3)

    def test_list_stage_sql_exception(self) -> None:
        """Test list_stage when SQL LIST command fails."""
        stage_location = "@nonexistent_stage/"

        self.m_session.add_mock_sql(
            f"LIST {stage_location}",
            mock_data_frame.MockDataFrame(collect_result=Exception("Stage '@nonexistent_stage/' does not exist")),
        )

        c_session = cast(Session, self.m_session)
        stage_client = stage_sql.StageSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        )

        with self.assertRaises(RuntimeError) as cm:
            stage_client.list_stage(stage_location)

        self.assertIn("Failed to check stage location '@nonexistent_stage/'", str(cm.exception))
        self.assertIn("Stage '@nonexistent_stage/' does not exist", str(cm.exception))


if __name__ == "__main__":
    absltest.main()
