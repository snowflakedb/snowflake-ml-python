import docker
from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake.ml.model._deploy_client.image_builds import client_image_builder


class ClientImageBuilderTestCase(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.idempotent_key = "mock_idempotent_key"
        self.image_repo = "mock_image_repo"
        self.model_dir = "mock_model_dir"
        self.use_gpu = False

        self.client_image_builder = client_image_builder.ClientImageBuilder(
            id=self.idempotent_key,
            image_repo=self.image_repo,
            model_dir=self.model_dir,
            use_gpu=self.use_gpu,
        )

    @mock.patch("docker.from_env")  # type: ignore
    @mock.patch(
        "snowflake.ml.model._deploy_client.image_builds.client_image_builder" ".docker_context.DockerContext"
    )  # type: ignore
    @mock.patch("tempfile.TemporaryDirectory")  # type: ignore
    def test_build(self, m_tempdir: mock.MagicMock, m_docker_context_class: mock.MagicMock, _) -> None:
        m_docker_context = m_docker_context_class.return_value
        m_context_dir = "mock_context_dir"
        # Modify the m_tempdir mock to return the desired TemporaryDirectory object
        m_tempdir.return_value.__enter__.return_value = m_context_dir

        with mock.patch.object(m_docker_context, "build") as m_build, mock.patch.object(
            self.client_image_builder, "_build_image_from_context"
        ) as m_build_image_from_context:
            self.client_image_builder._build()

            m_docker_context_class.assert_called_once_with(
                context_dir=m_context_dir, model_dir=self.model_dir, use_gpu=self.use_gpu
            )
            m_build.assert_called_once()
            m_build_image_from_context.assert_called_once()

    @mock.patch("docker.from_env")  # type: ignore
    def test_build_image_from_context_with_docker_daemon_running(self, m_docker_from_env: mock.MagicMock) -> None:
        m_docker_client = m_docker_from_env.return_value
        m_context_dir = "mock_context_dir"
        m_docker_client.images.build.return_value = None
        self.client_image_builder._build_image_from_context(context_dir=m_context_dir)
        m_docker_client.images.build.assert_called_once_with(
            path=m_context_dir,
            tag=self.client_image_builder.image_tag,
            platform=client_image_builder.Platform.LINUX_AMD64,
        )

    def test_build_image_from_context_without_docker_daemon_running(self) -> None:
        with mock.patch("docker.from_env", side_effect=docker.errors.DockerException):
            with self.assertRaises(ConnectionError):
                self.client_image_builder._build_image_from_context(context_dir="dummy")


if __name__ == "__main__":
    absltest.main()
