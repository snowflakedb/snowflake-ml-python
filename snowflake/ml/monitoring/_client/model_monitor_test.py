from typing import cast
from unittest import mock

import pandas as pd
from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.monitoring._client import model_monitor
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import DataFrame, Row


class ModelMonitorInstanceTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.test_db_name = sql_identifier.SqlIdentifier("SNOWML_OBSERVABILITY")
        self.test_schema_name = sql_identifier.SqlIdentifier("METADATA")

        self.test_monitor_name = sql_identifier.SqlIdentifier("TEST")
        self.test_model_version_name = sql_identifier.SqlIdentifier("TEST_MODEL_VERSION")
        self.test_model_name = sql_identifier.SqlIdentifier("TEST_MODEL")
        self.test_fq_model_name = f"{self.test_db_name}.{self.test_schema_name}.{self.test_model_name}"
        self.test_prediction_column_name = sql_identifier.SqlIdentifier("PREDICTION")
        self.test_label_column_name = sql_identifier.SqlIdentifier("LABEL")
        self.monitor_sql_client = mock.MagicMock(name="sql_client")

        self.model_monitor = model_monitor.ModelMonitor._ref(
            model_monitor_client=self.monitor_sql_client,
            name=self.test_monitor_name,
            fully_qualified_model_name=self.test_fq_model_name,
            version_name=self.test_model_version_name,
            function_name=sql_identifier.SqlIdentifier("predict"),
            prediction_columns=[sql_identifier.SqlIdentifier(self.test_prediction_column_name)],
            label_columns=[sql_identifier.SqlIdentifier(self.test_label_column_name)],
        )

    def test_set_baseline(self) -> None:
        baseline_df = mock_data_frame.MockDataFrame(
            [
                Row(
                    ID=1,
                    TIMESTAMP=1,
                    PREDICTION=0.5,
                    LABEL=1,
                ),
                Row(
                    ID=2,
                    TIMESTAMP=2,
                    PREDICTION=0.6,
                    LABEL=0,
                ),
            ],
            columns=[
                "ID",
                "TIMESTAMP",
                "PREDICTION",
                "LABEL",
            ],
        )
        with mock.patch.object(self.monitor_sql_client, "materialize_baseline_dataframe") as mock_materialize:
            self.model_monitor.set_baseline(cast(DataFrame, baseline_df))
            mock_materialize.assert_called_once_with(
                baseline_df, self.test_fq_model_name, self.test_model_version_name, statement_params=mock.ANY
            )

    def test_set_baseline_pandas_df(self) -> None:
        # Initialize a test pandas dataframe
        pandas_baseline_df = pd.DataFrame(
            {
                "ID": [1, 2],
                "TIMESTAMP": [1, 2],
                "PREDICTION": [0.5, 0.6],
                "LABEL": [1, 0],
            }
        )
        snowflake_baseline_df = mock_data_frame.MockDataFrame(
            [
                Row(
                    ID=1,
                    TIMESTAMP=1,
                    PREDICTION=0.5,
                    LABEL=1,
                ),
                Row(
                    ID=2,
                    TIMESTAMP=2,
                    PREDICTION=0.6,
                    LABEL=0,
                ),
            ],
            columns=[
                "ID",
                "TIMESTAMP",
                "PREDICTION",
                "LABEL",
            ],
        )

        with mock.patch.object(
            self.monitor_sql_client, "materialize_baseline_dataframe"
        ) as mock_materialize, mock.patch.object(self.monitor_sql_client._sql_client, "_session"), mock.patch.object(
            self.monitor_sql_client._sql_client._session, "create_dataframe", return_value=snowflake_baseline_df
        ) as mock_create_df:
            self.model_monitor.set_baseline(pandas_baseline_df)
            mock_materialize.assert_called_once_with(
                snowflake_baseline_df, self.test_fq_model_name, self.test_model_version_name, statement_params=mock.ANY
            )
            mock_create_df.assert_called_once_with(pandas_baseline_df)

    def test_set_baseline_missing_columns(self) -> None:
        baseline_df = mock_data_frame.MockDataFrame(
            [
                Row(
                    ID=1,
                    TIMESTAMP=1,
                    PREDICTION=0.5,
                    LABEL=1,
                ),
                Row(
                    ID=2,
                    TIMESTAMP=2,
                    PREDICTION=0.6,
                    LABEL=0,
                ),
            ],
            columns=[
                "ID",
                "TIMESTAMP",
                "LABEL",
            ],
        )

        expected_msg = "Specified prediction columns were not found in the baseline dataframe. Columns provided were: "
        with self.assertRaisesRegex(ValueError, expected_msg):
            self.model_monitor.set_baseline(cast(DataFrame, baseline_df))

    def test_suspend(self) -> None:
        with mock.patch.object(
            self.model_monitor._model_monitor_client, "suspend_monitor_dynamic_tables"
        ) as mock_suspend:
            self.model_monitor.suspend()
            mock_suspend.assert_called_once_with(
                model_name=self.test_model_name, version_name=self.test_model_version_name, statement_params=mock.ANY
            )

    def test_resume(self) -> None:
        with mock.patch.object(
            self.model_monitor._model_monitor_client, "resume_monitor_dynamic_tables"
        ) as mock_suspend:
            self.model_monitor.resume()
            mock_suspend.assert_called_once_with(
                model_name=self.test_model_name, version_name=self.test_model_version_name, statement_params=mock.ANY
            )


if __name__ == "__main__":
    absltest.main()
