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
                    default_version_name="V1",
                ),
                Row(
                    create_on="06/01",
                    name="Model",
                    comment="This is a comment",
                    model_name="MODEL",
                    database_name="TEMP",
                    schema_name="test",
                    default_version_name="v1",
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
                    default_version_name="V1",
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
                    metadata="{}",
                    user_data="{}",
                    is_default_version=True,
                ),
                Row(
                    create_on="06/01",
                    name="V1",
                    comment="This is a comment",
                    model_name="MODEL",
                    metadata="{}",
                    user_data="{}",
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
                    metadata="{}",
                    user_data="{}",
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

    def test_show_versions_3(self) -> None:
        m_statement_params = {"test": "1"}
        m_df_final = mock_data_frame.MockDataFrame(
            collect_result=[
                Row(
                    create_on="06/01",
                    name="v1",
                    comment="This is a comment",
                    model_name="MODEL",
                    metadata="{}",
                    user_data="{}",
                    is_default_version=True,
                    model_spec="{}",
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
            check_model_details=True,
            statement_params=m_statement_params,
        )

    def test_set_comment_for_model(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(collect_result=[Row("")], collect_statement_params=m_statement_params)
        comment = "This is my comment"
        self.m_session.add_mock_sql(f"""COMMENT ON MODEL TEMP."test".MODEL IS $${comment}$$""", m_df)
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).set_comment(
            model_name=sql_identifier.SqlIdentifier("MODEL"), comment=comment, statement_params=m_statement_params
        )

    def test_drop_model(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully dropped.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""DROP MODEL TEMP."test".MODEL""", m_df)
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).drop_model(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )

    def test_rename(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully dropped.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL RENAME TO TEMP."test".MODEL2""", m_df)
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).rename(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            new_model_db=None,
            new_model_schema=None,
            new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
            statement_params=m_statement_params,
        )

    def test_rename_fully_qualified_name(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully dropped.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL RENAME TO TEMP2."test2".MODEL2""", m_df)
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).rename(
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            new_model_db=sql_identifier.SqlIdentifier("TEMP2"),
            new_model_schema=sql_identifier.SqlIdentifier("test2", case_sensitive=True),
            new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
            statement_params=m_statement_params,
        )


if __name__ == "__main__":
    absltest.main()
