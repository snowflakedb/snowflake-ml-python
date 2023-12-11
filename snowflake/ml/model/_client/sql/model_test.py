from typing import cast

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.sql import model as model_sql
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session


class ModelSQLTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)

    def test_show_models_1(self) -> None:
        m_statement_params = {"test": "1"}
        m_df_final = mock_data_frame.MockDataFrame(
            collect_result=[
                Row(
                    create_on="06/01",
                    name="MODEL",
                    comment="This is a comment",
                    model_name="MODEL",
                    database_name="TEMP",
                    schema_name="test",
                ),
                Row(
                    create_on="06/01",
                    name="Model",
                    comment="This is a comment",
                    model_name="MODEL",
                    database_name="TEMP",
                    schema_name="test",
                ),
            ],
            collect_statement_params=m_statement_params,
        )
        self.m_session.add_mock_sql("""SHOW MODELS IN SCHEMA TEMP."test" """, m_df_final)
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_models(
            statement_params=m_statement_params,
        )

    def test_show_models_2(self) -> None:
        m_statement_params = {"test": "1"}
        m_df_final = mock_data_frame.MockDataFrame(
            collect_result=[
                Row(
                    create_on="06/01",
                    name="Model",
                    comment="This is a comment",
                    model_name="MODEL",
                    database_name="TEMP",
                    schema_name="test",
                ),
            ],
            collect_statement_params=m_statement_params,
        )
        self.m_session.add_mock_sql("""SHOW MODELS LIKE 'Model' IN SCHEMA TEMP."test" """, m_df_final)
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_models(
            model_name=sql_identifier.SqlIdentifier("Model", case_sensitive=True),
            statement_params=m_statement_params,
        )

    def test_show_versions_1(self) -> None:
        m_statement_params = {"test": "1"}
        m_df_final = mock_data_frame.MockDataFrame(
            collect_result=[
                Row(
                    create_on="06/01",
                    name="v1",
                    comment="This is a comment",
                    model_name="MODEL",
                    is_default_version=True,
                ),
                Row(
                    create_on="06/01",
                    name="V1",
                    comment="This is a comment",
                    model_name="MODEL",
                    is_default_version=False,
                ),
            ],
            collect_statement_params=m_statement_params,
        )
        self.m_session.add_mock_sql("""SHOW VERSIONS IN MODEL TEMP."test".MODEL""", m_df_final)
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_versions(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )

    def test_show_versions_2(self) -> None:
        m_statement_params = {"test": "1"}
        m_df_final = mock_data_frame.MockDataFrame(
            collect_result=[
                Row(
                    create_on="06/01",
                    name="v1",
                    comment="This is a comment",
                    model_name="MODEL",
                    is_default_version=True,
                ),
            ],
            collect_statement_params=m_statement_params,
        )
        self.m_session.add_mock_sql("""SHOW VERSIONS LIKE 'v1' IN MODEL TEMP."test".MODEL""", m_df_final)
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_versions(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )


if __name__ == "__main__":
    absltest.main()
