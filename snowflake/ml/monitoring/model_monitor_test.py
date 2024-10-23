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
        with self.assertRaisesRegex(
            NotImplementedError,
            "Model Monitoring is not implemented in python yet.",
        ):
            self.model_monitor.suspend()

    def test_resume(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            "Model Monitoring is not implemented in python yet.",
        ):
            self.model_monitor.resume()


if __name__ == "__main__":
    absltest.main()
