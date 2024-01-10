from typing import cast

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.sql import tag as tag_sql
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session


class ModuleTagSQLTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)

    def test_set_tag_on_model(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Tag MYTAG successfully set.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql(
            """ALTER MODEL TEMP."test".MODEL SET TAG DB."schema".MYTAG = $$tag content$$""", m_df
        )
        c_session = cast(Session, self.m_session)
        tag_sql.ModuleTagSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).set_tag_on_model(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            tag_database_name=sql_identifier.SqlIdentifier("DB"),
            tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
            tag_name=sql_identifier.SqlIdentifier("MYTAG"),
            tag_value="tag content",
            statement_params=m_statement_params,
        )

    def test_unset_tag_on_model(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Tag MYTAG successfully unset.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL UNSET TAG DB."schema".MYTAG""", m_df)
        c_session = cast(Session, self.m_session)
        tag_sql.ModuleTagSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).unset_tag_on_model(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            tag_database_name=sql_identifier.SqlIdentifier("DB"),
            tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
            tag_name=sql_identifier.SqlIdentifier("MYTAG"),
            statement_params=m_statement_params,
        )

    def test_get_tag_value(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row(TAG_VALUE="tag content")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql(
            """SELECT SYSTEM$GET_TAG($$DB."schema".MYTAG$$, $$TEMP."test".MODEL$$, 'MODULE') AS TAG_VALUE""", m_df
        )
        c_session = cast(Session, self.m_session)
        res = tag_sql.ModuleTagSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).get_tag_value(
            module_name=sql_identifier.SqlIdentifier("MODEL"),
            tag_database_name=sql_identifier.SqlIdentifier("DB"),
            tag_schema_name=sql_identifier.SqlIdentifier("schema", case_sensitive=True),
            tag_name=sql_identifier.SqlIdentifier("MYTAG"),
            statement_params=m_statement_params,
        )
        self.assertEqual(res, Row(TAG_VALUE="tag content"))

    def test_list_tags(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row(TAG_DATABASE="DB", TAG_SCHEMA="schema", TAG_NAME="MYTAG", TAG_VALUE="tag content")],
            collect_statement_params=m_statement_params,
        )
        self.m_session.add_mock_sql(
            """SELECT TAG_DATABASE, TAG_SCHEMA, TAG_NAME, TAG_VALUE
FROM TABLE(TEMP.INFORMATION_SCHEMA.TAG_REFERENCES($$TEMP."test".MODEL$$, 'MODULE'))""",
            m_df,
        )
        c_session = cast(Session, self.m_session)
        res = tag_sql.ModuleTagSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).get_tag_list(
            module_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )
        self.assertListEqual(
            res, [Row(TAG_DATABASE="DB", TAG_SCHEMA="schema", TAG_NAME="MYTAG", TAG_VALUE="tag content")]
        )


if __name__ == "__main__":
    absltest.main()
