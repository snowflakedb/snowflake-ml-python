from typing import cast

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake import snowpark
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.model._deploy_client.image_builds import client_image_builder
from snowflake.ml.test_utils import exception_utils, mock_session


class ClientImageBuilderTestCase(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = cast(snowpark.session.Session, mock_session.MockSession(conn=None, test_case=self))
        self.full_image_name = "mock_full_image_name"
        self.image_repo = "mock_image_repo"
        self.context_dir = "/tmp/context_dir"

        self.client_image_builder = client_image_builder.ClientImageBuilder(
            context_dir=self.context_dir,
            full_image_name=self.full_image_name,
            image_repo=self.image_repo,
            session=self.m_session,
        )

    def test_throw_exception_when_docker_is_not_running(self) -> None:
        with mock.patch.object(self.client_image_builder, "_run_docker_commands") as m_run_docker_commands:
            m_run_docker_commands.side_effect = snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_DOCKER_ERROR, original_exception=ConnectionError()
            )
            with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=ConnectionError):
                self.client_image_builder.build_and_upload_image()
            m_run_docker_commands.assert_called_once_with(["docker", "info"])

    def test_build(self) -> None:
        m_docker_config_dir = "mock_docker_config_dir"

        with mock.patch.object(self.client_image_builder, "_build_image_from_context") as m_build_image_from_context:
            self.client_image_builder._build_and_tag(m_docker_config_dir)
            m_build_image_from_context.assert_called_once_with(docker_config_dir=m_docker_config_dir)

    def test_build_image_from_context(self) -> None:
        with mock.patch.object(self.client_image_builder, "_run_docker_commands") as m_run_docker_commands:
            m_run_docker_commands.return_value = None
            m_docker_config_dir = "mock_docker_config_dir"
            self.client_image_builder._build_image_from_context(docker_config_dir=m_docker_config_dir)

            expected_commands = [
                "docker",
                "--config",
                m_docker_config_dir,
                "buildx",
                "build",
                "--platform",
                "linux/amd64",
                "--tag",
                self.full_image_name,
                self.context_dir,
            ]

            m_run_docker_commands.assert_called_once()
            actual_commands = m_run_docker_commands.call_args.args[0]
            self.assertListEqual(expected_commands, actual_commands)


if __name__ == "__main__":
    absltest.main()
