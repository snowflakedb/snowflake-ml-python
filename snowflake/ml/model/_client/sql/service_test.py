import copy
import uuid
from typing import Any, cast
from unittest import mock

from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml._internal.utils import sql_identifier, string_matcher
from snowflake.ml.model._client.sql import service as service_sql
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import DataFrame, Row, Session, functions as F, types as spt
from snowflake.snowpark._internal import utils as snowpark_utils


class ServiceSQLTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)

    def test_build_model_container(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Image built successfully.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql(
            """
                CALL SYSTEM$BUILD_MODEL_CONTAINER('TEMP."test".MODEL', 'V1', '"my_pool"',
                'TEMP."test"."image_repo"', 'FALSE', 'FALSE', '','MY_EAI')""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)
        service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).build_model_container(
            database_name=None,
            schema_name=None,
            model_name=sql_identifier.SqlIdentifier("MODEL"),
            version_name=sql_identifier.SqlIdentifier("V1"),
            compute_pool_name=sql_identifier.SqlIdentifier("my_pool", case_sensitive=True),
            image_repo_database_name=None,
            image_repo_schema_name=None,
            image_repo_name=sql_identifier.SqlIdentifier("image_repo", case_sensitive=True),
            gpu=None,
            force_rebuild=False,
            external_access_integration=sql_identifier.SqlIdentifier("MY_EAI"),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(
            """
                CALL SYSTEM$BUILD_MODEL_CONTAINER('DB_1."sch_1"."model"', '"v1"', 'MY_POOL',
                '"db_2".SCH_2.IMAGE_REPO', 'TRUE', 'TRUE', '', '"my_eai"')""",
            copy.deepcopy(m_df),
        )
        service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).build_model_container(
            database_name=sql_identifier.SqlIdentifier("DB_1"),
            schema_name=sql_identifier.SqlIdentifier("sch_1", case_sensitive=True),
            model_name=sql_identifier.SqlIdentifier("model", case_sensitive=True),
            version_name=sql_identifier.SqlIdentifier("v1", case_sensitive=True),
            compute_pool_name=sql_identifier.SqlIdentifier("my_pool"),
            image_repo_database_name=sql_identifier.SqlIdentifier("db_2", case_sensitive=True),
            image_repo_schema_name=sql_identifier.SqlIdentifier("SCH_2"),
            image_repo_name=sql_identifier.SqlIdentifier("image_repo"),
            gpu="1",
            force_rebuild=True,
            external_access_integration=sql_identifier.SqlIdentifier("my_eai", case_sensitive=True),
            statement_params=m_statement_params,
        )

    def test_deploy_model(self) -> None:
        m_statement_params = {"test": "1"}
        m_async_job = mock.MagicMock(spec=snowpark.AsyncJob)
        m_async_job.query_id = uuid.uuid4()
        m_df = mock_data_frame.MockDataFrame(
            collect_block=False,
            collect_result=m_async_job,
            collect_statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(
            """CALL SYSTEM$DEPLOY_MODEL('@stage_path/model_deployment_spec_file_rel_path')""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)

        service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).deploy_model(
            stage_path="stage_path",
            model_deployment_spec_file_rel_path="model_deployment_spec_file_rel_path",
            statement_params=m_statement_params,
        )

    def test_deploy_model_inline_yaml(self) -> None:
        m_statement_params = {"test": "1"}
        m_async_job = mock.MagicMock(spec=snowpark.AsyncJob)
        m_async_job.query_id = uuid.uuid4()
        m_df = mock_data_frame.MockDataFrame(
            collect_block=False,
            collect_result=m_async_job,
            collect_statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(
            """CALL SYSTEM$DEPLOY_MODEL('mock_yaml_str')""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)

        service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).deploy_model(
            model_deployment_spec_yaml_str="mock_yaml_str",
            statement_params=m_statement_params,
        )

    def test_invoke_function_method(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame()

        self.m_session.add_mock_sql(
            """SELECT *,
                TEMP."test".SERVICE!PREDICT(COL1, COL2) AS TMP_RESULT_ABCDEF0123
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
            service_sql.ServiceSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_function_method(
                database_name=None,
                schema_name=None,
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
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
        # sqlparse library does not know ! syntax, hence use string matcher. We need to be careful about
        # upper/lower case of keywords in expected SQL statement.
        self.m_session.add_mock_sql(
            """SELECT *,
                FOO."bar"."service"!PREDICT(COL1, COL2) AS TMP_RESULT_ABCDEF0123
            FROM FOO."bar".SNOWPARK_TEMP_TABLE_ABCDEF0123""",
            m_df,
            matcher=string_matcher.StringMatcherIgnoreWhitespace,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT_ABCDEF0123")
        c_session = cast(Session, self.m_session)
        mock_writer = mock.MagicMock()
        m_df.__setattr__("write", mock_writer)
        m_df.add_query("queries", "query_1")
        m_df.add_query("queries", "query_2")

        with mock.patch.object(mock_writer, "save_as_table") as mock_save_as_table, mock.patch.object(
            snowpark_utils, "random_name_for_temp_object", return_value="SNOWPARK_TEMP_TABLE_ABCDEF0123"
        ) as mock_random_name_for_temp_object, mock.patch.object(
            snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"
        ):
            service_sql.ServiceSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_function_method(
                database_name=sql_identifier.SqlIdentifier("FOO"),
                schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
                service_name=sql_identifier.SqlIdentifier("service", case_sensitive=True),
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                statement_params=m_statement_params,
            )
            mock_random_name_for_temp_object.assert_called_once_with(snowpark_utils.TempObjectType.TABLE)
            mock_save_as_table.assert_called_once_with(
                table_name='FOO."bar".SNOWPARK_TEMP_TABLE_ABCDEF0123',
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
                TEMP."test".SERVICE!PREDICT(COL1, COL2) AS TMP_RESULT_ABCDEF0123
            FROM SNOWPARK_ML_MODEL_INFERENCE_INPUT_ABCDEF0123""",
            m_df,
        )
        m_df.add_mock_with_columns(["OUTPUT_1"], [F.col("OUTPUT_1")]).add_mock_drop("TMP_RESULT_ABCDEF0123")
        c_session = cast(Session, self.m_session)
        m_df.add_query("queries", "query_1")
        with mock.patch.object(snowpark_utils, "generate_random_alphanumeric", return_value="ABCDEF0123"):
            service_sql.ServiceSQLClient(
                c_session,
                database_name=sql_identifier.SqlIdentifier("TEMP"),
                schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            ).invoke_function_method(
                database_name=None,
                schema_name=None,
                service_name=sql_identifier.SqlIdentifier("SERVICE"),
                method_name=sql_identifier.SqlIdentifier("PREDICT"),
                input_df=cast(DataFrame, m_df),
                input_args=[sql_identifier.SqlIdentifier("COL1"), sql_identifier.SqlIdentifier("COL2")],
                returns=[("output_1", spt.IntegerType(), sql_identifier.SqlIdentifier("OUTPUT_1"))],
                statement_params=m_statement_params,
            )

    def test_get_service_logs(self) -> None:
        m_statement_params = {"test": "1"}
        row = Row("SYSTEM$GET_SERVICE_LOGS")
        m_res = "INFO: Test"
        m_df = mock_data_frame.MockDataFrame(collect_result=[row(m_res)], collect_statement_params=m_statement_params)

        self.m_session.add_mock_sql(
            """CALL SYSTEM$GET_SERVICE_LOGS('TEMP."test".MYSERVICE', '0', 'model-container')""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)

        res = service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).get_service_logs(
            database_name=None,
            schema_name=None,
            service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
            instance_id="0",
            container_name="model-container",
            statement_params=m_statement_params,
        )
        self.assertEqual(res, m_res)

    def test_get_service_container_statuses_include_message(self) -> None:
        m_statement_params = {"test": "1"}
        m_service_status = service_sql.ServiceStatus("RUNNING")
        m_message = "test message"
        Outcome = Row("service_status", "instance_id", "instance_status", "status", "message")
        rows = [
            Outcome(
                m_service_status,
                0,
                service_sql.InstanceStatus("READY"),
                service_sql.ContainerStatus("READY"),
                m_message,
            ),
            Outcome(
                m_service_status,
                1,
                service_sql.InstanceStatus("TERMINATING"),
                service_sql.ContainerStatus("UNKNOWN"),
                m_message,
            ),
        ]
        m_df = mock_data_frame.MockDataFrame(collect_result=rows, collect_statement_params=m_statement_params)
        self.m_session.add_mock_sql(
            """SHOW SERVICE CONTAINERS IN SERVICE TEMP."test".MYSERVICE""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)
        res = service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).get_service_container_statuses(
            database_name=None,
            schema_name=None,
            service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
            include_message=True,
            statement_params=m_statement_params,
        )
        m_res = [
            service_sql.ServiceStatusInfo(
                service_status=m_service_status,
                instance_id=0,
                instance_status=service_sql.InstanceStatus("READY"),
                container_status=service_sql.ContainerStatus("READY"),
                message=m_message,
            ),
            service_sql.ServiceStatusInfo(
                service_status=m_service_status,
                instance_id=1,
                instance_status=service_sql.InstanceStatus("TERMINATING"),
                container_status=service_sql.ContainerStatus("UNKNOWN"),
                message=m_message,
            ),
        ]
        self.assertEqual(res, m_res)

    def test_get_service_container_statuses_exclude_message(self) -> None:
        m_statement_params = {"test": "1"}
        m_service_status = service_sql.ServiceStatus("RUNNING")
        m_message = "test message"
        Outcome = Row("service_status", "instance_id", "instance_status", "status", "message")
        rows = [
            Outcome(
                m_service_status,
                0,
                service_sql.InstanceStatus("READY"),
                service_sql.ContainerStatus("READY"),
                m_message,
            ),
        ]
        m_df = mock_data_frame.MockDataFrame(collect_result=rows, collect_statement_params=m_statement_params)
        self.m_session.add_mock_sql(
            """SHOW SERVICE CONTAINERS IN SERVICE TEMP."test".MYSERVICE""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)
        res = service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).get_service_container_statuses(
            database_name=None,
            schema_name=None,
            service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
            include_message=False,
            statement_params=m_statement_params,
        )
        m_res = [
            service_sql.ServiceStatusInfo(
                service_status=m_service_status,
                instance_id=0,
                instance_status=service_sql.InstanceStatus("READY"),
                container_status=service_sql.ContainerStatus("READY"),
                message=None,
            ),
        ]
        self.assertEqual(res, m_res)

    def test_get_service_container_statuses_suspended_service(self) -> None:
        m_statement_params = {"test": "1"}
        m_service_status = service_sql.ServiceStatus("SUSPENDED")
        Outcome = Row("service_status", "instance_id", "instance_status", "status", "message")
        rows = [
            Outcome(m_service_status, None, None, None, None),
        ]
        m_df = mock_data_frame.MockDataFrame(collect_result=rows, collect_statement_params=m_statement_params)
        self.m_session.add_mock_sql(
            """SHOW SERVICE CONTAINERS IN SERVICE TEMP."test".MYSERVICE""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)
        res = service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).get_service_container_statuses(
            database_name=None,
            schema_name=None,
            service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
            include_message=True,
            statement_params=m_statement_params,
        )
        m_res = [
            service_sql.ServiceStatusInfo(
                service_status=m_service_status,
                instance_id=None,
                instance_status=None,
                container_status=None,
                message=None,
            )
        ]
        self.assertEqual(res, m_res)

    def test_get_service_container_statuses_no_status(self) -> None:
        m_statement_params = {"test": "1"}
        rows: list[Any] = []
        m_df = mock_data_frame.MockDataFrame(collect_result=rows)
        self.m_session.add_mock_sql(
            """SHOW SERVICE CONTAINERS IN SERVICE TEMP."test".MYSERVICE""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)
        res = service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).get_service_container_statuses(
            database_name=None,
            schema_name=None,
            service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
            include_message=False,
            statement_params=m_statement_params,
        )
        self.assertEqual(res, [])

    def test_drop_service(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row("Service MYSERVICE successfully dropped.")], collect_statement_params=m_statement_params
        )
        self.m_session.add_mock_sql(
            """DROP SERVICE TEMP."test".MYSERVICE""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)

        service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).drop_service(
            database_name=None,
            schema_name=None,
            service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
            statement_params=m_statement_params,
        )

    def test_show_endpoints(self) -> None:
        m_statement_params = {"test": "1"}
        m_df = mock_data_frame.MockDataFrame(
            collect_result=[Row(name="inference", ingress_url="foo.snowflakecomputing.app")],
            collect_statement_params=m_statement_params,
        )
        self.m_session.add_mock_sql(
            """SHOW ENDPOINTS IN SERVICE TEMP."test".MYSERVICE""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)

        service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
        ).show_endpoints(
            database_name=None,
            schema_name=None,
            service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
            statement_params=m_statement_params,
        )

        self.m_session.add_mock_sql(
            """SHOW ENDPOINTS IN SERVICE TEMP."test".MYSERVICE""",
            copy.deepcopy(m_df),
        )
        c_session = cast(Session, self.m_session)

        service_sql.ServiceSQLClient(
            c_session,
            database_name=sql_identifier.SqlIdentifier("foo"),
            schema_name=sql_identifier.SqlIdentifier("bar", case_sensitive=True),
        ).show_endpoints(
            database_name=sql_identifier.SqlIdentifier("TEMP"),
            schema_name=sql_identifier.SqlIdentifier("test", case_sensitive=True),
            service_name=sql_identifier.SqlIdentifier("MYSERVICE"),
            statement_params=m_statement_params,
        )


if __name__ == "__main__":
    absltest.main()
