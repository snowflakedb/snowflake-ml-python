import subprocess
from typing import cast

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake import snowpark
from snowflake.ml.model._deploy_client.image_builds import client_image_builder
from snowflake.ml.test_utils import mock_session


class ClientImageBuilderTestCase(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = cast(snowpark.session.Session, mock_session.MockSession(conn=None, test_case=self))
        self.unique_id = "mock_id"
        self.image_repo = "mock_image_repo"
        self.model_dir = "local/dir/model.zip"
        self.use_gpu = True

        self.client_image_builder = client_image_builder.ClientImageBuilder(
            id=self.unique_id,
            image_repo=self.image_repo,
            model_dir=self.model_dir,
            session=self.m_session,
            use_gpu=self.use_gpu,
        )

    @mock.patch(
        "snowflake.ml.model._deploy_client.image_builds.client_image_builder.subprocess.check_call"
    )  # type: ignore
    def test_throw_exception_when_docker_is_not_running(self, m_check_call: mock.MagicMock) -> None:
        m_check_call.side_effect = subprocess.CalledProcessError(1, "docker info")
        with self.assertRaises(ConnectionError):
            self.client_image_builder.build_and_upload_image()

    @mock.patch(
        "snowflake.ml.model._deploy_client.image_builds.client_image_builder" ".docker_context.DockerContext"
    )  # type: ignore
    @mock.patch("tempfile.TemporaryDirectory")  # type: ignore
    def test_build(self, m_tempdir: mock.MagicMock, m_docker_context_class: mock.MagicMock) -> None:
        m_docker_context = m_docker_context_class.return_value
        m_context_dir = "mock_context_dir"
        # Modify the m_tempdir mock to return the desired TemporaryDirectory object
        m_tempdir.return_value.__enter__.return_value = m_context_dir
        m_docker_config_dir = "mock_docker_config_dir"

        with mock.patch.object(m_docker_context, "build") as m_build, mock.patch.object(
            self.client_image_builder, "_build_image_from_context"
        ) as m_build_image_from_context:
            self.client_image_builder._build(m_docker_config_dir)
            m_docker_context_class.assert_called_once_with(
                context_dir=m_context_dir, model_dir=self.model_dir, use_gpu=True
            )
            m_build.assert_called_once()
            m_build_image_from_context.assert_called_once_with(
                context_dir=m_context_dir, docker_config_dir=m_docker_config_dir
            )

    def test_build_image_from_context(self) -> None:
        with mock.patch.object(self.client_image_builder, "_run_docker_commands") as m_run_docker_commands:
            m_run_docker_commands.return_value = None
            m_context_dir = "fake_context_dir"
            m_docker_config_dir = "mock_docker_config_dir"
            self.client_image_builder._build_image_from_context(
                context_dir=m_context_dir, docker_config_dir=m_docker_config_dir
            )

            expected_commands = [
                "docker",
                "--config",
                m_docker_config_dir,
                "buildx",
                "build",
                "--platform",
                "linux/amd64",
                "--tag",
                f"{'/'.join([self.image_repo, self.unique_id])}:latest",
                m_context_dir,
            ]

            m_run_docker_commands.assert_called_once()
            actual_commands = m_run_docker_commands.call_args.args[0]
            self.assertListEqual(expected_commands, actual_commands)


if __name__ == "__main__":
    absltest.main()
