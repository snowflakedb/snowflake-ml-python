import os
import tempfile

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake.ml.model._deploy_client.image_builds import server_image_builder
from snowflake.ml.model._deploy_client.utils import constants


class ServerImageBuilderTestCase(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.unique_id = "mock_id"
        self.image_repo = "mock_image_repo"
        self.model_dir = "local/dir/model.zip"
        self.artifact_stage_location = "@stage/models/id"
        self.compute_pool = "test_pool"
        self.context_tarball_stage_location = f"{self.artifact_stage_location}/context.tar.gz"

    @mock.patch("snowflake.ml.model._deploy_client.image_builds.server_image_builder.snowpark.Session")  # type: ignore
    def test_construct_and_upload_docker_entrypoint_script(self, m_session_class: mock.MagicMock) -> None:
        m_session = m_session_class.return_value
        mock_file_put = mock.MagicMock()
        m_session.file.put = mock_file_put

        builder = server_image_builder.ServerImageBuilder(
            id=self.unique_id,
            image_repo=self.image_repo,
            model_dir=self.model_dir,
            session=m_session,
            artifact_stage_location=self.artifact_stage_location,
            compute_pool=self.compute_pool,
        )

        with tempfile.TemporaryDirectory() as context_dir:
            shell_file_path = os.path.join(context_dir, constants.KANIKO_SHELL_SCRIPT_NAME)
            fixture_path = os.path.join(os.path.dirname(__file__), "test_fixtures", "kaniko_shell_script_fixture.sh")
            builder._construct_and_upload_docker_entrypoint_script(
                context_dir=context_dir, context_tarball_stage_location=self.context_tarball_stage_location
            )
            m_session.file.put.assert_called_once_with(
                local_file_name=shell_file_path,
                stage_location=self.artifact_stage_location,
                auto_compress=False,
                overwrite=True,
            )

            with open(shell_file_path, encoding="utf-8") as shell_file, open(fixture_path, encoding="utf-8") as fixture:
                actual = shell_file.read()
                expected = fixture.read()
                self.assertEqual(actual, expected, "Generated image build shell script is not the same")


if __name__ == "__main__":
    absltest.main()
