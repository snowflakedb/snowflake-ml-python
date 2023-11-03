import json
from typing import cast

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake import snowpark
from snowflake.ml.model._deploy_client.utils import constants
from snowflake.ml.model._deploy_client.utils.snowservice_client import SnowServiceClient
from snowflake.ml.test_utils import exception_utils, mock_data_frame, mock_session
from snowflake.snowpark import session


class SnowServiceClientTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.client = SnowServiceClient(cast(session.Session, self.m_session))
        self.m_service_name = "mock_service_name"

    def test_create_or_replace_service(self) -> None:
        m_min_instances = 1
        m_max_instances = 2
        m_compute_pool = "mock_compute_pool"
        m_stage = "@mock_spec_stage"
        m_stage_path = "a/hello.yaml"
        m_spec_storgae_location = f"{m_stage}/{m_stage_path}"

        self.m_session.add_mock_sql(
            query="drop service if exists mock_service_name", result=mock_data_frame.MockDataFrame(collect_result=[])
        )

        self.m_session.add_mock_sql(
            query=f"""
             CREATE SERVICE {self.m_service_name}
                IN COMPUTE POOL {m_compute_pool}
                FROM {m_stage}
                SPEC = '{m_stage_path}'
                MIN_INSTANCES={m_min_instances}
                MAX_INSTANCES={m_max_instances}
            """,
            result=mock_data_frame.MockDataFrame(collect_result=[]),
        )

        self.client.create_or_replace_service(
            service_name=self.m_service_name,
            min_instances=m_min_instances,
            max_instances=m_max_instances,
            compute_pool=m_compute_pool,
            spec_stage_location=m_spec_storgae_location,
        )

    def test_create_job_successfully(self) -> None:
        with mock.patch.object(self.client, "get_resource_status", return_value=constants.ResourceStatus.DONE):
            m_compute_pool = "mock_compute_pool"
            m_stage = "@mock_spec_stage"
            m_stage_path = "a/hello.yaml"
            m_spec_storgae_location = f"{m_stage}/{m_stage_path}"
            expected_job_id = "abcd"
            self.m_session.add_mock_sql(
                query=f"""
                    EXECUTE SERVICE
                    IN COMPUTE POOL {m_compute_pool}
                    FROM {m_stage}
                    SPEC = '{m_stage_path}'
                """,
                result=mock_data_frame.MockDataFrame(collect_result=[]),
            )
            row = snowpark.Row(**{"QUERY_ID": expected_job_id})
            self.m_session.add_mock_sql(
                query="SELECT LAST_QUERY_ID() AS QUERY_ID",
                result=mock_data_frame.MockDataFrame(collect_result=[row]),
            )
            self.client.create_job(
                compute_pool=m_compute_pool,
                spec_stage_location=m_spec_storgae_location,
            )

    def test_create_job_failed(self) -> None:
        with self.assertLogs(level="ERROR") as cm:
            with mock.patch.object(self.client, "get_resource_status", return_value=constants.ResourceStatus.FAILED):
                with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
                    test_log = "Job fails because of xyz."
                    m_compute_pool = "mock_compute_pool"
                    m_stage = "@mock_spec_stage"
                    m_stage_path = "a/hello.yaml"
                    m_spec_storgae_location = f"{m_stage}/{m_stage_path}"
                    expected_job_id = "abcd"

                    self.m_session.add_mock_sql(
                        query=f"""
                            EXECUTE SERVICE
                            IN COMPUTE POOL {m_compute_pool}
                            FROM {m_stage}
                            SPEC = '{m_stage_path}'
                        """,
                        result=mock_data_frame.MockDataFrame(collect_result=[]),
                    )

                    row = snowpark.Row(**{"QUERY_ID": expected_job_id})
                    self.m_session.add_mock_sql(
                        query="SELECT LAST_QUERY_ID() AS QUERY_ID",
                        result=mock_data_frame.MockDataFrame(collect_result=[row]),
                    )

                    self.m_session.add_mock_sql(
                        query=f"CALL SYSTEM$GET_JOB_LOGS('{expected_job_id}', '{constants.KANIKO_CONTAINER_NAME}')",
                        result=mock_data_frame.MockDataFrame(
                            collect_result=[snowpark.Row(**{"SYSTEM$GET_JOB_LOGS": test_log})]
                        ),
                    )

                    self.client.create_job(
                        compute_pool=m_compute_pool,
                        spec_stage_location=m_spec_storgae_location,
                    )

                    self.assertTrue(cm.output, test_log)

    def test_create_service_function(self) -> None:
        m_service_func_name = "mock_service_func_name"
        m_service_name = "mock_service_name"
        m_endpoint_name = "mock_endpoint_name"
        m_path_at_endpoint = "mock_route"

        m_sql = f"""
        CREATE OR REPLACE FUNCTION {m_service_func_name}(input OBJECT)
            RETURNS OBJECT
            SERVICE={m_service_name}
            ENDPOINT={m_endpoint_name}
            AS '/{m_path_at_endpoint}'
        """

        self.m_session.add_mock_sql(
            query=m_sql,
            result=mock_data_frame.MockDataFrame(collect_result=[]),
        )

        self.client.create_or_replace_service_function(
            service_func_name=m_service_func_name,
            service_name=m_service_name,
            endpoint_name=m_endpoint_name,
            path_at_service_endpoint=m_path_at_endpoint,
        )

    def test_create_service_function_max_batch_rows(self) -> None:
        m_service_func_name = "mock_service_func_name"
        m_service_name = "mock_service_name"
        m_endpoint_name = "mock_endpoint_name"
        m_path_at_endpoint = "mock_route"
        m_max_batch_rows = 1

        m_sql = f"""
        CREATE OR REPLACE FUNCTION {m_service_func_name}(input OBJECT)
            RETURNS OBJECT
            SERVICE={m_service_name}
            ENDPOINT={m_endpoint_name}
            MAX_BATCH_ROWS={m_max_batch_rows}
            AS '/{m_path_at_endpoint}'
        """

        self.m_session.add_mock_sql(
            query=m_sql,
            result=mock_data_frame.MockDataFrame(collect_result=[]),
        )

        self.client.create_or_replace_service_function(
            service_func_name=m_service_func_name,
            service_name=m_service_name,
            endpoint_name=m_endpoint_name,
            path_at_service_endpoint=m_path_at_endpoint,
            max_batch_rows=m_max_batch_rows,
        )

    def test_get_service_status(self) -> None:
        row = snowpark.Row(
            **{
                "SYSTEM$GET_SERVICE_STATUS": json.dumps(
                    [
                        {
                            "status": "READY",
                            "message": "Running",
                            "containerName": "inference-server",
                            "instanceId": "0",
                            "serviceName": "SERVICE_DFC46DE9CEC441B2A3185266C11E79BA",
                            "image": "image",
                            "restartCount": 0,
                        }
                    ]
                )
            }
        )
        self.m_session.add_mock_sql(
            query="call system$GET_SERVICE_STATUS('mock_service_name');",
            result=mock_data_frame.MockDataFrame(collect_result=[row]),
        )

        self.assertEqual(
            self.client.get_resource_status(self.m_service_name, constants.ResourceType.SERVICE),
            constants.ResourceStatus("READY"),
        )

        row = snowpark.Row(
            **{
                "SYSTEM$GET_SERVICE_STATUS": json.dumps(
                    [
                        {
                            "status": "FAILED",
                            "message": "Running",
                            "containerName": "inference-server",
                            "instanceId": "0",
                            "serviceName": "SERVICE_DFC46DE9CEC441B2A3185266C11E79BA",
                            "image": "image",
                            "restartCount": 0,
                        }
                    ]
                )
            }
        )
        self.m_session.add_mock_sql(
            query="call system$GET_SERVICE_STATUS('mock_service_name');",
            result=mock_data_frame.MockDataFrame(collect_result=[row]),
        )

        self.assertEqual(
            self.client.get_resource_status(self.m_service_name, constants.ResourceType.SERVICE),
            constants.ResourceStatus("FAILED"),
        )

        row = snowpark.Row(
            **{
                "SYSTEM$GET_SERVICE_STATUS": json.dumps(
                    [
                        {
                            "status": "",
                            "message": "Running",
                            "containerName": "inference-server",
                            "instanceId": "0",
                            "serviceName": "SERVICE_DFC46DE9CEC441B2A3185266C11E79BA",
                            "image": "image",
                            "restartCount": 0,
                        }
                    ]
                )
            }
        )
        self.m_session.add_mock_sql(
            query="call system$GET_SERVICE_STATUS('mock_service_name');",
            result=mock_data_frame.MockDataFrame(collect_result=[row]),
        )
        self.assertEqual(self.client.get_resource_status(self.m_service_name, constants.ResourceType.SERVICE), None)

    def test_block_until_service_is_ready_happy_path(self) -> None:
        with mock.patch.object(self.client, "get_resource_status", return_value=constants.ResourceStatus("READY")):
            self.client.block_until_resource_is_ready(
                self.m_service_name, constants.ResourceType.SERVICE, max_retries=1, retry_interval_secs=1
            )

    def test_block_until_service_is_ready_timeout(self) -> None:
        test_log = "service fails because of xyz."
        self.m_session.add_mock_sql(
            query=f"CALL SYSTEM$GET_SERVICE_LOGS('{self.m_service_name}', '0',"
            f"'{constants.INFERENCE_SERVER_CONTAINER}')",
            result=mock_data_frame.MockDataFrame(
                collect_result=[snowpark.Row(**{"SYSTEM$GET_SERVICE_LOGS": test_log})]
            ),
        )
        self.m_session.add_mock_sql(
            query=f"DROP SERVICE IF EXISTS {self.m_service_name}",
            result=mock_data_frame.MockDataFrame(collect_result=[]),
        )

        with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
            with mock.patch.object(self.client, "get_resource_status", side_effect=[None, None, None, "READY"]):
                self.client.block_until_resource_is_ready(
                    self.m_service_name, constants.ResourceType.SERVICE, max_retries=1, retry_interval_secs=1
                )

    def test_block_until_service_is_ready_retries_and_ready(self) -> None:
        # Service becomes ready on 2nd retry.
        with mock.patch.object(
            self.client, "get_resource_status", side_effect=[None, constants.ResourceStatus("READY")]
        ):
            self.client.block_until_resource_is_ready(
                self.m_service_name, constants.ResourceType.SERVICE, max_retries=2, retry_interval_secs=1
            )

    def test_block_until_service_is_ready_retries_and_fail(self) -> None:
        test_log = "service fails because of abc."
        self.m_session.add_mock_sql(
            query=f"CALL SYSTEM$GET_SERVICE_LOGS('{self.m_service_name}', '0',"
            f"'{constants.INFERENCE_SERVER_CONTAINER}')",
            result=mock_data_frame.MockDataFrame(
                collect_result=[snowpark.Row(**{"SYSTEM$GET_SERVICE_LOGS": test_log})]
            ),
        )
        self.m_session.add_mock_sql(
            query=f"DROP SERVICE IF EXISTS {self.m_service_name}",
            result=mock_data_frame.MockDataFrame(collect_result=[]),
        )

        # Service show failure status on 2nd retry.
        with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
            with mock.patch.object(
                self.client, "get_resource_status", side_effect=[None, constants.ResourceStatus("FAILED")]
            ):
                self.client.block_until_resource_is_ready(
                    self.m_service_name, constants.ResourceType.SERVICE, max_retries=2, retry_interval_secs=1
                )


if __name__ == "__main__":
    absltest.main()
