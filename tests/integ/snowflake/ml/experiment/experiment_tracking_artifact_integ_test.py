import os
import tempfile

from absl.testing import absltest

from tests.integ.snowflake.ml.experiment._integ_test_base import (
    ExperimentTrackingIntegTestBase,
)


class ExperimentArtifactIntegTest(ExperimentTrackingIntegTestBase):
    def test_log_artifact_file(self) -> None:
        experiment_name = "TEST_EXPERIMENT_LOG_ARTIFACT_FILE"
        run_name = "TEST_RUN_LOG_ARTIFACT_FILE"
        local_path = "tests/integ/snowflake/ml/experiment/test_artifact.json"

        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            self.assertEqual(0, len(self.exp.list_artifacts(run_name=run_name)))
            self.exp.log_artifact(local_path)

        # Test that the artifact is logged correctly
        artifacts = self.exp.list_artifacts(run_name=run_name)
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].name, "test_artifact.json")

        # Test that the artifact can be retrieved and that the content is the same
        with tempfile.TemporaryDirectory() as temp_dir:
            self.exp.download_artifacts(run_name=run_name, target_path=temp_dir)
            with (
                open(os.path.join(temp_dir, "test_artifact.json")) as uploaded_file,
                open(local_path) as original_file,
            ):
                self.assertEqual(uploaded_file.read(), original_file.read())

        # Test downloading a specific file path
        with tempfile.TemporaryDirectory() as temp_dir:
            self.exp.download_artifacts(run_name=run_name, target_path=temp_dir, artifact_path="test_artifact.json")
            self.assertEqual(os.listdir(temp_dir), ["test_artifact.json"])
            with (
                open(os.path.join(temp_dir, "test_artifact.json")) as uploaded_file,
                open(local_path) as original_file,
            ):
                self.assertEqual(uploaded_file.read(), original_file.read())

    def test_log_artifact_directory(self) -> None:
        experiment_name = "TEST_EXPERIMENT_LOG_ARTIFACT_DIR"
        run_name = "TEST_RUN_LOG_ARTIFACT_DIR"
        local_path = "tests/integ/snowflake/ml/experiment/test_artifact_dir"

        self.exp.set_experiment(experiment_name=experiment_name)
        with self.exp.start_run(run_name=run_name):
            self.exp.log_artifact(local_path)

        # Test that the artifacts are logged correctly
        expected_artifacts = [
            "artifact1.txt",
            "artifact2.py",
            "nested_dir/artifact3.md",
        ]
        artifacts = self.exp.list_artifacts(run_name=run_name)
        self.assertListEqual(expected_artifacts, [a.name for a in artifacts])

        # Test that artifacts can be retrieved and that the content is the same
        with tempfile.TemporaryDirectory() as temp_dir:
            self.exp.download_artifacts(run_name=run_name, target_path=temp_dir)
            for expected_artifact in expected_artifacts:
                with (
                    open(os.path.join(temp_dir, expected_artifact)) as uploaded_file,
                    open(os.path.join(local_path, expected_artifact)) as original_file,
                ):
                    self.assertEqual(uploaded_file.read(), original_file.read())

        # Test downloading a specific path, should only download the artifact3.md file
        with tempfile.TemporaryDirectory() as temp_dir:
            self.exp.download_artifacts(
                run_name=run_name, target_path=temp_dir, artifact_path="nested_dir/artifact3.md"
            )
            self.assertEqual(os.listdir(temp_dir), ["nested_dir"])
            with (
                open(os.path.join(temp_dir, "nested_dir/artifact3.md")) as uploaded_file,
                open(os.path.join(local_path, "nested_dir/artifact3.md")) as original_file,
            ):
                self.assertEqual(uploaded_file.read(), original_file.read())


if __name__ == "__main__":
    absltest.main()
