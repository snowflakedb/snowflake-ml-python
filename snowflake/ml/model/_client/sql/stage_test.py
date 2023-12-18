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
        self.m_session.add_mock_sql("""CREATE TEMPORARY STAGE TEMP."test".MODEL""", m_df)
        c_session = cast(Session, self.m_session)
        stage_sql.StageSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).create_tmp_stage(
            stage_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )


if __name__ == "__main__":
    absltest.main()
