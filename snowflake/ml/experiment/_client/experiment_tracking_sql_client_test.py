from typing import Optional, cast
from unittest import mock

from absl.testing import absltest

from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.experiment._client import (
    artifact,
    experiment_tracking_sql_client as sql_client,
)
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.ml.utils import sql_client as sql_client_utils
from snowflake.snowpark import file_operation, row, session


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

    def test_put_artifact(self) -> None:
        # Test uploading an artifact calls session.file.put with correct args and returns first result
        experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")
        run_name = sql_identifier.SqlIdentifier("TEST_RUN")
        artifact_path = "artifacts/models"
        file_path = "/local/path/to/model.bin"

        expected_stage_uri = self.client._build_snow_uri(experiment_name, run_name, artifact_path)

        # Monkey-patch file.put on the mock session
        called: dict[str, object] = {}

        expected_put_result = file_operation.PutResult(
            source=file_path,
            target=expected_stage_uri,
            source_size=10,
            target_size=10,
            source_compression="NONE",
            target_compression="NONE",
            status="UPLOADED",
            message="OK",
        )

        def _mock_put(
            local_file_name: str, stage_location: str, overwrite: bool, auto_compress: bool
        ) -> list[file_operation.PutResult]:
            called["local_file_name"] = local_file_name
            called["stage_location"] = stage_location
            called["overwrite"] = overwrite
            called["auto_compress"] = auto_compress
            return [expected_put_result]

        # Attach a simple mock object with a put method
        self.m_session.file = mock.Mock()
        self.m_session.file.put = _mock_put

        res = self.client.put_artifact(
            experiment_name=experiment_name,
            run_name=run_name,
            artifact_path=artifact_path,
            file_path=file_path,
        )

        # Validate call args
        self.assertEqual(called.get("local_file_name"), file_path)
        self.assertEqual(called.get("stage_location"), expected_stage_uri)
        self.assertTrue(called.get("overwrite"))
        self.assertFalse(called.get("auto_compress"))

        # Validate return value is the first element of put's result
        self.assertIs(res, expected_put_result)

    def test_get_artifact(self) -> None:
        # Test downloading an artifact calls session.file.get with correct args and returns list
        experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")
        run_name = sql_identifier.SqlIdentifier("TEST_RUN")
        artifact_path = "artifacts/models/model.bin"
        target_dir = "/tmp/downloads"

        expected_stage_uri = self.client._build_snow_uri(experiment_name, run_name, artifact_path)

        expected_get_result = file_operation.GetResult(
            file="/tmp/downloads/model.bin",
            size="10",
            status="DOWNLOADED",
            message="OK",
        )

        def _mock_get(stage_location: str, target_directory: str) -> list[file_operation.GetResult]:
            self.assertEqual(stage_location, expected_stage_uri)
            self.assertEqual(target_directory, target_dir)
            return [expected_get_result]

        # Attach a simple mock object with a get method
        self.m_session.file = mock.Mock()
        self.m_session.file.get = _mock_get

        res = self.client.get_artifact(
            experiment_name=experiment_name,
            run_name=run_name,
            artifact_path=artifact_path,
            target_path=target_dir,
        )

        self.assertEqual(res, expected_get_result)

    def test_build_snow_uri(self) -> None:
        # Test the _build_snow_uri helper method
        experiment_name = sql_identifier.SqlIdentifier("TEST_EXPERIMENT")
        run_name = sql_identifier.SqlIdentifier("TEST_RUN")

        # Test with artifact path
        artifact_path = "models/artifacts"
        expected_uri = "snow://experiment/TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT/versions/TEST_RUN/models/artifacts"
        actual_uri = self.client._build_snow_uri(experiment_name, run_name, artifact_path)
        self.assertEqual(actual_uri, expected_uri)

        # Test with empty artifact path
        artifact_path = ""
        expected_uri = "snow://experiment/TEST_DB.TEST_SCHEMA.TEST_EXPERIMENT/versions/TEST_RUN"
        actual_uri = self.client._build_snow_uri(experiment_name, run_name, artifact_path)
        self.assertEqual(actual_uri, expected_uri)

    def test_list_artifacts(self) -> None:
        experiment_name = sql_identifier.SqlIdentifier("EXP")
        run_name = sql_identifier.SqlIdentifier("RUN")
        artifact_path = "sub/dir"

        # Expect LIST query with built URI (prefix should be removed)
        expected_uri = self.client._build_snow_uri(experiment_name, run_name, artifact_path)
        self.m_session.add_mock_sql(
            f"LIST {expected_uri}",
            self._create_mock_df(
                result=[
                    row.Row(name=f"/versions/{run_name}/a.txt", size=10, md5="aaa", last_modified="2024-01-01"),
                    row.Row(name=f"/versions/{run_name}/sub/dir/b.txt", size=10, md5="bbb", last_modified="2024-01-01"),
                    row.Row(name=f"/versions/{run_name}/sub/dir/c.bin", size=20, md5="ccc", last_modified="2024-01-02"),
                ]
            ),
        )

        res = self.client.list_artifacts(
            experiment_name=experiment_name,
            run_name=run_name,
            artifact_path=artifact_path,
        )

        self.assertEqual(
            res,
            [
                artifact.ArtifactInfo(name="a.txt", size=10, md5="aaa", last_modified="2024-01-01"),
                artifact.ArtifactInfo(name="sub/dir/b.txt", size=10, md5="bbb", last_modified="2024-01-01"),
                artifact.ArtifactInfo(name="sub/dir/c.bin", size=20, md5="ccc", last_modified="2024-01-02"),
            ],
        )


if __name__ == "__main__":
    absltest.main()
