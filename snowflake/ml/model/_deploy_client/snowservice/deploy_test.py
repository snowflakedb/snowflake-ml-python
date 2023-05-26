from typing import Dict, cast

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake.ml.model._deploy_client.docker import client_image_builder
from snowflake.ml.model._deploy_client.snowservice.deploy import (
    SnowServiceDeployment,
    _deploy,
)
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import session


class DeployTestCase(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = cast(session.Session, mock_session.MockSession(conn=None, test_case=self))

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_model_id(self, m_deployment_class: mock.MagicMock) -> None:
        m_deployment = m_deployment_class.return_value

        _deploy(
            session=self.m_session,
            model_id="provided_model_id",
            service_func_name="mock_service_func",
            model_dir="mock_model_dir",
            options={},
        )

        m_deployment_class.assert_called_once_with(
            session=self.m_session,
            model_id="provided_model_id",
            service_func_name="mock_service_func",
            model_dir="mock_model_dir",
            image_builder=mock.ANY,
            options={},
        )
        m_deployment.deploy.assert_called_once()

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_empty_model_id(self, m_deployment_class: mock.MagicMock) -> None:
        with self.assertRaises(ValueError):
            _deploy(
                session=self.m_session,
                service_func_name="mock_service_func",
                model_id="",
                model_dir="mock_model_dir",
                options={},
            )

        m_deployment_class.assert_not_called()


class SnowServiceDeploymentTestCase(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = cast(session.Session, mock_session.MockSession(conn=None, test_case=self))
        self.m_image_builder = mock.create_autospec(client_image_builder.ClientImageBuilder)
        self.m_model_id = "provided_model_id"
        self.m_service_func_name = "provided_service_func_name"
        self.m_model_dir = "provided_model_dir"
        self.m_options: Dict[str, str] = {}

        self.deployment = SnowServiceDeployment(
            self.m_session,
            model_id=self.m_model_id,
            service_func_name=self.m_service_func_name,
            model_dir=self.m_model_dir,
            image_builder=self.m_image_builder,
            options=self.m_options,
        )

    def test_deploy(self) -> None:
        with mock.patch.object(
            self.deployment, "_build_and_upload_image"
        ) as m_build_and_upload_image, mock.patch.object(self.deployment, "_deploy_workflow") as m_deploy_workflow:
            self.deployment.deploy()
            m_build_and_upload_image.assert_called_once()
            m_deploy_workflow.assert_called_once()

    def test_build_and_upload_image(self) -> None:
        self.deployment._build_and_upload_image()
        self.m_image_builder.build_and_upload_image.assert_called_once()


if __name__ == "__main__":
    absltest.main()
