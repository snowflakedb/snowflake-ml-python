import datetime
from typing import Any, Dict, cast
from unittest import mock

from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import spcs_attribution_utils
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import session


class SpcsAttributionUtilsTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._m_session = mock_session.MockSession(conn=None, test_case=self)
        self._fully_qualified_service_name = "db.schema.my_service"
        self._m_compute_pool_name = "my_pool"
        self._service_created_on = datetime.datetime.strptime(
            "2023-11-16 13:01:00.062 -0800", spcs_attribution_utils._DATETIME_FORMAT
        )

        mock_service_detail = self._get_mock_service_details()
        self._m_session.add_mock_sql(
            query=f"DESC SERVICE {self._fully_qualified_service_name}",
            result=mock_data_frame.MockDataFrame(collect_result=[snowpark.Row(**mock_service_detail)]),
        )

        mock_compute_pool_detail = self._get_mock_compute_pool_details()
        self._m_session.add_mock_sql(
            query=f"DESC COMPUTE POOL {self._m_compute_pool_name}",
            result=mock_data_frame.MockDataFrame(collect_result=[snowpark.Row(**mock_compute_pool_detail)]),
        )

    def _get_mock_service_details(self) -> Dict[str, Any]:
        return {
            "name": "my_service",
            "database_name": "my_db",
            "schema_name": "my_schema",
            "owner": "Engineer",
            "compute_pool": self._m_compute_pool_name,
            "spec": "--- spec:",
            "dns_name": "service-dummy.my-schema.my-db.snowflakecomputing.internal",
            "public_endpoints": {"predict": "dummy.snowflakecomputing.app"},
            "min_instances": 1,
            "max_instances": 1,
            "created_on": self._service_created_on,
            "updated_on": "2023-11-16 13:01:00.595 -0800",
            "comment": None,
        }

    def _get_mock_compute_pool_details(self) -> Dict[str, Any]:
        return {
            "name": self._m_compute_pool_name,
            "state": "Active",
            "min_nodes": 1,
            "max_nodes": 1,
            "instance_family": "STANDARD_2",
            "num_services": 1,
            "num_jobs": 2,
            "active_nodes": 1,
            "idle_nodes": 1,
            "created_on": "2023-09-21 09:17:39.627 -0700",
            "resumed_on": "2023-09-21 09:17:39.628 -0700",
            "updated_on": "2023-11-27 15:08:55.725 -0800",
            "owner": "ACCOUNTADMIN",
            "comment": None,
        }

    def test_record_service_start(self) -> None:
        with mock.patch.object(spcs_attribution_utils, "_send_service_telemetry", return_value=None) as m_telemetry:
            with self.assertLogs(level="INFO") as cm:
                spcs_attribution_utils.record_service_start(
                    cast(session.Session, self._m_session), self._fully_qualified_service_name
                )

                assert len(cm.output) == 1, "there should only be 1 log"
                log = cm.output[0]

                service_details = self._get_mock_service_details()
                compute_pool_details = self._get_mock_compute_pool_details()

                self.assertEqual(
                    log,
                    f"INFO:snowflake.ml._internal.utils.spcs_attribution_utils:Service "
                    f"{self._fully_qualified_service_name} created with compute pool {self._m_compute_pool_name}.",
                )
                m_telemetry.assert_called_once_with(
                    fully_qualified_name=self._fully_qualified_service_name,
                    compute_pool_name=self._m_compute_pool_name,
                    service_details=service_details,
                    compute_pool_details=compute_pool_details,
                    kwargs={telemetry.TelemetryField.KEY_CUSTOM_TAGS.value: spcs_attribution_utils._SERVICE_START},
                )

    def test_record_service_end(self) -> None:
        current_datetime = self._service_created_on + datetime.timedelta(days=2, hours=1, minutes=30, seconds=20)
        expected_duration = 178220  # 2 days 1 hour 30 minutes and 20 seconds.

        with mock.patch(
            "snowflake.ml._internal.utils.spcs_attribution_utils._get_current_time"
        ) as mock_datetime_now, mock.patch.object(
            spcs_attribution_utils, "_send_service_telemetry", return_value=None
        ) as m_telemetry:
            with self.assertLogs(level="INFO") as cm:
                mock_datetime_now.return_value = current_datetime

                spcs_attribution_utils.record_service_end(
                    cast(session.Session, self._m_session), self._fully_qualified_service_name
                )
                assert len(cm.output) == 1, "there should only be 1 log"
                log = cm.output[0]

                service_details = self._get_mock_service_details()
                compute_pool_details = self._get_mock_compute_pool_details()

                self.assertEqual(
                    log,
                    f"INFO:snowflake.ml._internal.utils.spcs_attribution_utils:Service "
                    f"{self._fully_qualified_service_name} deleted from compute pool {self._m_compute_pool_name}",
                )

                m_telemetry.assert_called_once_with(
                    fully_qualified_name=self._fully_qualified_service_name,
                    compute_pool_name=self._m_compute_pool_name,
                    service_details=service_details,
                    compute_pool_details=compute_pool_details,
                    duration_in_seconds=expected_duration,
                    kwargs={telemetry.TelemetryField.KEY_CUSTOM_TAGS.value: spcs_attribution_utils._SERVICE_END},
                )


if __name__ == "__main__":
    absltest.main()
