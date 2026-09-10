import copy
from typing import cast
from unittest import mock

from absl.testing import absltest

from snowflake import connector
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.sql import model as model_sql
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import Row, Session


def _fake_retry(**kwargs):  # type: ignore[no-untyped-def]
    def decorator(fn):  # type: ignore[no-untyped-def]
        def wrapped(*args, **inner_kwargs):  # type: ignore[no-untyped-def]
            last_exc: BaseException | None = None
            for _ in range(int(kwargs.get("stop_max_attempt_number", 5))):
                try:
                    return fn(*args, **inner_kwargs)
                except Exception as exc:
                    last_exc = exc
                    if not kwargs["retry_on_exception"](exc):
                        raise
            assert last_exc is not None
            raise last_exc

        return wrapped

    return decorator


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
        self.m_session.add_mock_sql("""SHOW MODELS IN SCHEMA TEMP."test" """, copy.deepcopy(m_df_final))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_models(
            database_name=None,
            schema_name=None,
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql("""SHOW MODELS IN SCHEMA TEMP."test" """, copy.deepcopy(m_df_final))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).show_models(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
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
        self.m_session.add_mock_sql("""SHOW MODELS LIKE 'Model' IN SCHEMA TEMP."test" """, copy.deepcopy(m_df_final))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_models(
            database_name=None,
            schema_name=None,
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
        self.m_session.add_mock_sql("""SHOW VERSIONS IN MODEL TEMP."test".MODEL""", copy.deepcopy(m_df_final))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_versions(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql("""SHOW VERSIONS IN MODEL TEMP."test".MODEL""", copy.deepcopy(m_df_final))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).show_versions(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
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
        self.m_session.add_mock_sql("""SHOW VERSIONS LIKE 'v1' IN MODEL TEMP."test".MODEL""", copy.deepcopy(m_df_final))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_versions(
            database_name=None,
            schema_name=None,
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
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            check_model_details=True,
            statement_params=m_statement_params,
        )

    def test_show_versions_4(self) -> None:
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
                    runnable_in='["WAREHOUSE"]',
                ),
            ],
            collect_statement_params=m_statement_params,
        )
        self.m_session.add_mock_sql("""SHOW VERSIONS LIKE 'v1' IN MODEL TEMP."test".MODEL""", m_df_final)
        c_session = cast(Session, self.m_session)
        client = model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        )
        result = client.show_versions(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            statement_params=m_statement_params,
        )
        # Verify the runnable_in column is accessible via the constant
        self.assertIn(client.MODEL_VERSION_RUNNABLE_IN_COL_NAME, result[0])
        self.assertEqual(result[0][client.MODEL_VERSION_RUNNABLE_IN_COL_NAME], '["WAREHOUSE"]')

    def test_is_empty_version_result(self) -> None:
        self.assertFalse(model_sql._is_empty_version_result(ValueError("nope")))
        self.assertTrue(
            model_sql._is_empty_version_result(
                connector.DataError(
                    "Query Result did not match expected number of rows. Expected 1 rows, found: 0 rows."
                )
            )
        )
        self.assertFalse(
            model_sql._is_empty_version_result(
                connector.DataError(
                    "Query Result did not match expected number of rows. Expected 1 rows, found: 2 rows."
                )
            )
        )
        self.assertFalse(
            model_sql._is_empty_version_result(connector.DataError("Query Result did not have expected column name."))
        )

    def test_show_versions_retries_empty_result(self) -> None:
        m_statement_params = {"test": "1"}
        query = """SHOW VERSIONS LIKE 'v1' IN MODEL TEMP."test".MODEL"""
        empty_df = mock_data_frame.MockDataFrame()
        empty_df.add_collect_result([], m_statement_params)
        success_df = mock_data_frame.MockDataFrame(
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
        self.m_session.add_mock_sql(query, empty_df)
        self.m_session.add_mock_sql(query, copy.deepcopy(success_df))
        c_session = cast(Session, self.m_session)
        client = model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        )
        with mock.patch("retrying.retry", _fake_retry):
            result = client.show_versions(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=m_statement_params,
                retry=True,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "v1")

    def test_show_versions_empty_result_without_retry(self) -> None:
        m_statement_params = {"test": "1"}
        query = """SHOW VERSIONS LIKE 'v1' IN MODEL TEMP."test".MODEL"""
        empty_df = mock_data_frame.MockDataFrame()
        empty_df.add_collect_result([], m_statement_params)
        self.m_session.add_mock_sql(query, empty_df)
        c_session = cast(Session, self.m_session)
        client = model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        )
        with self.assertRaises(connector.DataError) as raised:
            client.show_versions(
                database_name=None,
                schema_name=None,
                model_name=sql_identifier.SqlIdentifier("MODEL"),
                version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                statement_params=m_statement_params,
            )
        self.assertIn("Expected 1 rows", str(raised.exception))
        self.assertIn("found: 0 rows", str(raised.exception))

    def test_show_versions_empty_result_exhausted_retries(self) -> None:
        m_statement_params = {"test": "1"}
        query = """SHOW VERSIONS LIKE 'v1' IN MODEL TEMP."test".MODEL"""
        for _ in range(5):
            empty_df = mock_data_frame.MockDataFrame()
            empty_df.add_collect_result([], m_statement_params)
            self.m_session.add_mock_sql(query, empty_df)
        c_session = cast(Session, self.m_session)
        client = model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        )
        with mock.patch("retrying.retry", _fake_retry):
            with self.assertRaises(connector.DataError) as raised:
                client.show_versions(
                    database_name=None,
                    schema_name=None,
                    model_name=sql_identifier.SqlIdentifier("MODEL"),
                    version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
                    statement_params=m_statement_params,
                    retry=True,
                )
        self.assertIn("Expected 1 rows", str(raised.exception))
        self.assertIn("found: 0 rows", str(raised.exception))

    def test_set_comment_for_model(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(collect_result=[Row("")], collect_statement_params=m_statement_params)
        comment = "This is my comment"
        self.m_session.add_mock_sql(f"""COMMENT ON MODEL TEMP."test".MODEL IS $${comment}$$""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).set_comment(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            comment=comment,
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(f"""COMMENT ON MODEL TEMP."test".MODEL IS $${comment}$$""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).set_comment(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            comment=comment,
            statement_params=m_statement_params,
        )

    def test_drop_model(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully dropped.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql("""DROP MODEL TEMP."test".MODEL""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).drop_model(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql("""DROP MODEL TEMP."test".MODEL""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).drop_model(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            statement_params=m_statement_params,
        )

    def test_rename(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Model MODEL successfully dropped.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql(
            """ALTER MODEL TEMP."test".MODEL RENAME TO TEMP."test".MODEL2""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).rename(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            new_model_db=None,
            new_model_schema=None,
            new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql("""ALTER MODEL TEMP."test".MODEL RENAME TO FOO."bar".MODEL2""", copy.deepcopy(m_df))
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).rename(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
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
        self.m_session.add_mock_sql(
            """ALTER MODEL TEMP."test".MODEL RENAME TO TEMP2."test2".MODEL2""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).rename(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            new_model_db=sql_identifier.SqlIdentifier("TEMP2"),
            new_model_schema=sql_identifier.SqlIdentifier("test2", case_sensitive=True),
            new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(
            """ALTER MODEL TEMP."test".MODEL RENAME TO TEMP2."test2".MODEL2""", copy.deepcopy(m_df)
        )
        c_session = cast(Session, self.m_session)
        model_sql.ModelSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).rename(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            new_model_db=sql_identifier.SqlIdentifier("TEMP2"),
            new_model_schema=sql_identifier.SqlIdentifier("test2", case_sensitive=True),
            new_model_name=sql_identifier.SqlIdentifier("MODEL2"),
            statement_params=m_statement_params,
        )


if __name__ == "__main__":
    absltest.main()
