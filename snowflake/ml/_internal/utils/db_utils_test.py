from typing import cast

from absl.testing import absltest

from snowflake.ml._internal.utils import db_utils, sql_identifier
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session


class DbUtilsTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.test_db_name = sql_identifier.SqlIdentifier("SNOWML_OBSERVABILITY")
        self.test_schema_name = sql_identifier.SqlIdentifier("METADATA")

        self.session = cast(Session, self.m_session)

    def test_warehouse_exists(self) -> None:
        test_wh_name = sql_identifier.SqlIdentifier("test_wh")
        self.m_session.add_mock_sql(
            query=f"""SHOW WAREHOUSES LIKE '{test_wh_name}'""",
            result=mock_data_frame.MockDataFrame([Row(name=test_wh_name)]),
        )
        self.assertTrue(db_utils.db_object_exists(self.session, db_utils.SnowflakeDbObjectType.WAREHOUSE, test_wh_name))
        self.m_session.finalize()

    def test_warehouse_not_exists(self) -> None:
        test_wh_name = sql_identifier.SqlIdentifier("test_wh")
        self.m_session.add_mock_sql(
            query=f"""SHOW WAREHOUSES LIKE '{test_wh_name}'""",
            result=mock_data_frame.MockDataFrame([]),
        )
        self.assertFalse(
            db_utils.db_object_exists(self.session, db_utils.SnowflakeDbObjectType.WAREHOUSE, test_wh_name)
        )
        self.m_session.finalize()

    def test_table_exists(self) -> None:
        test_tbl_name = sql_identifier.SqlIdentifier("test_tbl")
        test_db = sql_identifier.SqlIdentifier("test_db")
        test_schema = sql_identifier.SqlIdentifier("test_schema")
        self.m_session.add_mock_sql(
            query=f"""SHOW TABLES LIKE '{test_tbl_name}' IN {test_db}.{test_schema}""",
            result=mock_data_frame.MockDataFrame([Row(name=test_tbl_name)]),
        )
        self.assertTrue(
            db_utils.db_object_exists(
                self.session,
                db_utils.SnowflakeDbObjectType.TABLE,
                test_tbl_name,
                database_name=test_db,
                schema_name=test_schema,
            )
        )
        self.m_session.finalize()

    def test_table_not_exists(self) -> None:
        test_tbl_name = sql_identifier.SqlIdentifier("test_tbl")
        test_db = sql_identifier.SqlIdentifier("test_db")
        test_schema = sql_identifier.SqlIdentifier("test_schema")
        self.m_session.add_mock_sql(
            query=f"""SHOW TABLES LIKE '{test_tbl_name}' IN {test_db}.{test_schema}""",
            result=mock_data_frame.MockDataFrame([]),
        )
        self.assertFalse(
            db_utils.db_object_exists(
                self.session,
                db_utils.SnowflakeDbObjectType.TABLE,
                test_tbl_name,
                database_name=test_db,
                schema_name=test_schema,
            )
        )
        self.m_session.finalize()


if __name__ == "__main__":
    absltest.main()
