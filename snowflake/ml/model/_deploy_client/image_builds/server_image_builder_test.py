import os
import tempfile

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake.ml.model._deploy_client.image_builds import server_image_builder
from snowflake.ml.model._deploy_client.utils import constants


class ServerImageBuilderTestCase(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.image_repo = "mock_image_repo"
        self.artifact_stage_location = "@stage/models/id"
        self.compute_pool = "test_pool"
        self.context_tarball_stage_location = f"{self.artifact_stage_location}/context.tar.gz"
        self.full_image_name = "org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest"
        self.eais = ["eai_1"]

    @mock.patch(  # type: ignore[misc]
        "snowflake.ml.model._deploy_client.image_builds.server_image_builder.snowpark.Session"
    )
    def test_construct_and_upload_docker_entrypoint_script(self, m_session_class: mock.MagicMock) -> None:
        m_session = m_session_class.return_value
        mock_file_put = mock.MagicMock()
        m_session.file.put = mock_file_put

        with tempfile.TemporaryDirectory() as context_dir:
            builder = server_image_builder.ServerImageBuilder(
                context_dir=context_dir,
                full_image_name=self.full_image_name,
                image_repo=self.image_repo,
                session=m_session,
                artifact_stage_location=self.artifact_stage_location,
                compute_pool=self.compute_pool,
                external_access_integrations=self.eais,
            )

            shell_file_path = os.path.join(context_dir, constants.KANIKO_SHELL_SCRIPT_NAME)
            fixture_path = os.path.join(os.path.dirname(__file__), "test_fixtures", "kaniko_shell_script_fixture.sh")
            builder._construct_and_upload_docker_entrypoint_script(
                context_tarball_stage_location=self.context_tarball_stage_location
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
