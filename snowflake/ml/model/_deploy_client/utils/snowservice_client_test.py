import json
from typing import cast

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake import snowpark
from snowflake.ml.model._deploy_client.utils import constants
from snowflake.ml.model._deploy_client.utils.snowservice_client import SnowServiceClient
from snowflake.ml.test_utils import mock_data_frame, mock_session
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
        m_spec_storgae_location = "mock_spec_storage_location"

        self.m_session.add_mock_sql(
            query="drop service if exists mock_service_name", result=mock_data_frame.MockDataFrame(collect_result=[])
        )

        self.m_session.add_mock_sql(
            query="create service mock_service_name"
            " min_instances=1"
            " max_instances=2"
            " compute_pool=mock_compute_pool"
            " spec=@mock_spec_storage_location",
            result=mock_data_frame.MockDataFrame(collect_result=[]),
        )

        self.client.create_or_replace_service(
            service_name=self.m_service_name,
            min_instances=m_min_instances,
            max_instances=m_max_instances,
            compute_pool=m_compute_pool,
            spec_stage_location=m_spec_storgae_location,
        )

    def test_create_service_function(self) -> None:
        m_service_func_name = "mock_service_func_name"
        m_service_name = "mock_service_name"
        m_endpoint_name = "mock_endpoint_name"

        self.m_session.add_mock_sql(
            query="create or replace function mock_service_func_name(input OBJECT)"
            " returns OBJECT"
            " service=mock_service_name"
            " endpoint=mock_endpoint_name"
            " as '/predict'",
            result=mock_data_frame.MockDataFrame(collect_result=[]),
        )

        self.client.create_or_replace_service_function(
            service_func_name=m_service_func_name, service_name=m_service_name, endpoint_name=m_endpoint_name
        )

    def test_get_service_status(self) -> None:
        row = snowpark.Row(
            **{
                "SYSTEM$GET_SNOWSERVICE_STATUS": json.dumps(
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
            query="call system$get_snowservice_status('mock_service_name');",
            result=mock_data_frame.MockDataFrame(collect_result=[row]),
        )

        self.assertEqual(self.client.get_resource_status(self.m_service_name, constants.ResourceType.SERVICE), "READY")

        row = snowpark.Row(
            **{
                "SYSTEM$GET_SNOWSERVICE_STATUS": json.dumps(
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
            query="call system$get_snowservice_status('mock_service_name');",
            result=mock_data_frame.MockDataFrame(collect_result=[row]),
        )

        self.assertEqual(self.client.get_resource_status(self.m_service_name, constants.ResourceType.SERVICE), "FAILED")

        row = snowpark.Row(
            **{
                "SYSTEM$GET_SNOWSERVICE_STATUS": json.dumps(
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
            query="call system$get_snowservice_status('mock_service_name');",
            result=mock_data_frame.MockDataFrame(collect_result=[row]),
        )
        self.assertEqual(self.client.get_resource_status(self.m_service_name, constants.ResourceType.SERVICE), None)

    def test_block_until_service_is_ready_happy_path(self) -> None:
        with mock.patch.object(self.client, "get_resource_status", return_value="READY"):
            self.client.block_until_resource_is_ready(
                self.m_service_name, constants.ResourceType.SERVICE, max_retries=1, retry_interval_secs=1
            )

    def test_block_until_service_is_ready_timeout(self) -> None:
        with self.assertRaises(RuntimeError):
            with mock.patch.object(self.client, "get_resource_status", side_effect=[None, None, None, "READY"]):
                self.client.block_until_resource_is_ready(
                    self.m_service_name, constants.ResourceType.SERVICE, max_retries=1, retry_interval_secs=1
                )

    def test_block_until_service_is_ready_retries_and_ready(self) -> None:
        # Service becomes ready on 2nd retry.
        with mock.patch.object(self.client, "get_resource_status", side_effect=[None, "READY"]):
            self.client.block_until_resource_is_ready(
                self.m_service_name, constants.ResourceType.SERVICE, max_retries=2, retry_interval_secs=1
            )

    def test_block_until_service_is_ready_retries_and_fail(self) -> None:
        # Service show failure status on 2nd retry.
        with self.assertRaises(RuntimeError):
            with mock.patch.object(self.client, "get_resource_status", side_effect=[None, "FAILED"]):
                self.client.block_until_resource_is_ready(
                    self.m_service_name, constants.ResourceType.SERVICE, max_retries=2, retry_interval_secs=1
                )
