from typing import Optional, cast

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.experiment._client import experiment_tracking_sql_client as sql_client
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.ml.utils import sql_client as sql_client_utils
from snowflake.snowpark import row, session


class ExperimentTrackingSQLClientTest(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self, check_call_sequence_completion=True)
        self.database_name = sql_identifier.SqlIdentifier("TEST_DB")
        self.schema_name = sql_identifier.SqlIdentifier("TEST_SCHEMA")
        self.c_session = cast(session.Session, self.m_session)

        self.client = sql_client.ExperimentTrackingSQLClient(
            self.c_session,
            database_name=self.database_name,
            schema_name=self.schema_name,
        )

    def tearDown(self) -> None:
        self.m_session.finalize()  # Check that all expected operations were executed

    def _create_mock_df(self, result: Optional[list[row.Row]] = None) -> mock_data_frame.MockDataFrame:
        if result is None:
            result = [row.Row("")]
        return mock_data_frame.MockDataFrame(collect_result=result)

    def test_create_experiment(self) -> None:
        # Test creating an experiment
        experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")

        # Test without if_not_exists flag
        self.m_session.add_mock_sql("CREATE EXPERIMENT  TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT", self._create_mock_df())

        self.client.create_experiment(experiment_name, creation_mode=sql_client_utils.CreationMode(if_not_exists=False))

        # Test with if_not_exists flag
        self.m_session.add_mock_sql(
            "CREATE EXPERIMENT IF NOT EXISTS TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT", self._create_mock_df()
        )

        self.client.create_experiment(experiment_name, creation_mode=sql_client_utils.CreationMode(if_not_exists=True))

    def test_drop_experiment(self) -> None:
        # Test dropping an experiment
        experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")

        self.m_session.add_mock_sql("DROP EXPERIMENT TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT", self._create_mock_df())

        self.client.drop_experiment(experiment_name=experiment_name)

    def test_add_run(self) -> None:
        # Test adding a run to an experiment
        experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")
        run_name = sql_identifier.SqlIdentifier("TEST_RUN")

        # test live run (default)
        self.m_session.add_mock_sql(
            "ALTER EXPERIMENT TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT ADD LIVE RUN TEST_RUN", self._create_mock_df()
        )
        self.client.add_run(experiment_name=experiment_name, run_name=run_name)

        # test non-live run
        self.m_session.add_mock_sql(
            "ALTER EXPERIMENT TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT ADD  RUN TEST_RUN", self._create_mock_df()
        )
        self.client.add_run(experiment_name=experiment_name, run_name=run_name, live=False)

    def test_commit_run(self) -> None:
        # Test committing a run
        experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")
        run_name = sql_identifier.SqlIdentifier("TEST_RUN")

        self.m_session.add_mock_sql(
            "ALTER EXPERIMENT TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT COMMIT RUN TEST_RUN", self._create_mock_df()
        )

        self.client.commit_run(experiment_name=experiment_name, run_name=run_name)

    def test_drop_run(self) -> None:
        # Test dropping a run from an experiment
        experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")
        run_name = sql_identifier.SqlIdentifier("TEST_RUN")

        self.m_session.add_mock_sql(
            "ALTER EXPERIMENT TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT DROP RUN TEST_RUN", self._create_mock_df()
        )

        self.client.drop_run(experiment_name=experiment_name, run_name=run_name)

    def test_show_runs_in_experiment(self) -> None:
        # Test showing runs in an experiment without LIKE clause
        experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")

        self.m_session.add_mock_sql(
            "SHOW RUNS  IN EXPERIMENT TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT", self._create_mock_df()
        )

        self.client.show_runs_in_experiment(experiment_name=experiment_name)

    def test_show_runs_in_experiment_with_like(self) -> None:
        # Test showing runs in an experiment with LIKE clause
        experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")
        like_pattern = "TEST_%"

        # Mock result with filtered run data
        mock_runs = [
            row.Row(name="TEST_RUN_1", status="RUNNING"),
        ]

        self.m_session.add_mock_sql(
            "SHOW RUNS LIKE 'TEST_%' IN EXPERIMENT TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT",
            self._create_mock_df(result=mock_runs),
        )

        self.client.show_runs_in_experiment(experiment_name=experiment_name, like=like_pattern)


if __name__ == "__main__":
    absltest.main()
