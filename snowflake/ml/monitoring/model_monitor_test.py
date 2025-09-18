from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.monitoring import model_monitor
from snowflake.ml.test_utils import mock_session


class ModelMonitorInstanceTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.test_db_name = sql_identifier.SqlIdentifier("SNOWML_OBSERVABILITY")
        self.test_schema_name = sql_identifier.SqlIdentifier("METADATA")

        self.test_monitor_name = sql_identifier.SqlIdentifier("TEST")
        self.monitor_sql_client = mock.MagicMock(name="sql_client")

        self.model_monitor = model_monitor.ModelMonitor._ref(
            model_monitor_client=self.monitor_sql_client,
            name=self.test_monitor_name,
        )

    def test_suspend(self) -> None:
        with mock.patch.object(self.model_monitor._model_monitor_client, "suspend_monitor") as mock_suspend:
            self.model_monitor.suspend()
            mock_suspend.assert_called_once_with(self.test_monitor_name, statement_params=mock.ANY)

    def test_resume(self) -> None:
        with mock.patch.object(self.model_monitor._model_monitor_client, "resume_monitor") as mock_resume:
            self.model_monitor.resume()
            mock_resume.assert_called_once_with(self.test_monitor_name, statement_params=mock.ANY)

    def test_add_segment_column(self) -> None:
        test_segment_column = "CUSTOMER_SEGMENT"
        with mock.patch.object(self.model_monitor._model_monitor_client, "add_segment_column") as mock_add_segment:
            self.model_monitor.add_segment_column(test_segment_column)
            # Verify it was called with the monitor name, a SqlIdentifier for the segment column, and statement params
            mock_add_segment.assert_called_once()
            call_args = mock_add_segment.call_args
            self.assertEqual(call_args[0][0], self.test_monitor_name)  # monitor_name
            self.assertEqual(call_args[0][1].identifier(), test_segment_column)  # segment_column as SqlIdentifier
            self.assertIsNotNone(call_args[1]["statement_params"])  # statement_params

    def test_drop_segment_column(self) -> None:
        test_segment_column = "CUSTOMER_SEGMENT"
        with mock.patch.object(self.model_monitor._model_monitor_client, "drop_segment_column") as mock_drop_segment:
            self.model_monitor.drop_segment_column(test_segment_column)
            # Verify it was called with the monitor name, a SqlIdentifier for the segment column, and statement params
            mock_drop_segment.assert_called_once()
            call_args = mock_drop_segment.call_args
            self.assertEqual(call_args[0][0], self.test_monitor_name)  # monitor_name
            self.assertEqual(call_args[0][1].identifier(), test_segment_column)  # segment_column as SqlIdentifier
            self.assertIsNotNone(call_args[1]["statement_params"])  # statement_params

    def test_add_custom_metric_column(self) -> None:
        test_custom_metric_column = "CUSTOM_METRIC"
        with mock.patch.object(
            self.model_monitor._model_monitor_client, "add_custom_metric_column"
        ) as mock_add_custom_metric:
            self.model_monitor.add_custom_metric_column(test_custom_metric_column)
            mock_add_custom_metric.assert_called_once_with(
                self.test_monitor_name, test_custom_metric_column, statement_params=mock.ANY
            )
            call_args = mock_add_custom_metric.call_args
            self.assertEqual(call_args[0][0], self.test_monitor_name)
            self.assertEqual(call_args[0][1].identifier(), test_custom_metric_column)
            self.assertIsNotNone(call_args[1]["statement_params"])

    def test_drop_custom_metric_column(self) -> None:
        test_custom_metric_column = "CUSTOM_METRIC"
        with mock.patch.object(
            self.model_monitor._model_monitor_client, "drop_custom_metric_column"
        ) as mock_drop_custom_metric:
            self.model_monitor.drop_custom_metric_column(test_custom_metric_column)
            mock_drop_custom_metric.assert_called_once_with(
                self.test_monitor_name, test_custom_metric_column, statement_params=mock.ANY
            )
            call_args = mock_drop_custom_metric.call_args
            self.assertEqual(call_args[0][0], self.test_monitor_name)
            self.assertEqual(call_args[0][1].identifier(), test_custom_metric_column)
            self.assertIsNotNone(call_args[1]["statement_params"])


if __name__ == "__main__":
    absltest.main()
