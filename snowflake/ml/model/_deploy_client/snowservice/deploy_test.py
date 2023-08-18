from typing import Any, Dict, cast

from absl.testing import absltest
from absl.testing.absltest import mock

from snowflake.ml.model._deploy_client.snowservice import deploy_options
from snowflake.ml.model._deploy_client.snowservice.deploy import (
    SnowServiceDeployment,
    _deploy,
    _get_or_create_image_repo,
)
from snowflake.ml.model._deploy_client.utils import constants
from snowflake.ml.test_utils import mock_session
from snowflake.snowpark import FileOperation, session


class Connection:
    def __init__(self, host: str, account: str, database: str, schema: str) -> None:
        self.host = host
        self.account = account
        self._database = database
        self._schema = schema


class DeployTestCase(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = cast(session.Session, mock_session.MockSession(conn=None, test_case=self))
        self.options: Dict[str, Any] = {
            "compute_pool": "mock_compute_pool",
            "image_repo": "mock_image_repo",
        }

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy._model")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.file_utils")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_model_id(
        self, m_deployment_class: mock.MagicMock, m_file_utils_class: mock.MagicMock, m_model_class: mock.MagicMock
    ) -> None:
        m_deployment = m_deployment_class.return_value
        m_file_utils = m_file_utils_class.return_value

        m_extracted_model_dir = "mock_extracted_model_dir"
        m_model_zip_stage_path = "@mock_model_zip_stage_path/model.zip"
        m_deployment_stage_path = "@mock_model_deployment_stage_path"

        with mock.patch.object(FileOperation, "get_stream", return_value=None):
            with mock.patch.object(m_file_utils, "unzip_stream_in_temp_dir", return_value=m_extracted_model_dir):
                _deploy(
                    session=self.m_session,
                    model_id="provided_model_id",
                    service_func_name="mock_service_func",
                    model_zip_stage_path=m_model_zip_stage_path,
                    deployment_stage_path=m_deployment_stage_path,
                    target_method=constants.PREDICT,
                    **self.options,
                )

                # TODO: for some reason mock is not wired up properly
                # m_model.load_model.assert_called_once_with(model_dir_path=m_extracted_model_dir, meta_only=True)

                m_deployment_class.assert_called_once_with(
                    session=self.m_session,
                    model_id="provided_model_id",
                    service_func_name="mock_service_func",
                    model_zip_stage_path=m_model_zip_stage_path,
                    deployment_stage_path=m_deployment_stage_path,
                    model_dir=mock.ANY,
                    target_method=constants.PREDICT,
                    options=mock.ANY,
                )
                m_deployment.deploy.assert_called_once()

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_empty_model_id(self, m_deployment_class: mock.MagicMock) -> None:
        with self.assertRaises(ValueError):
            _deploy(
                session=self.m_session,
                service_func_name="mock_service_func",
                model_id="",
                model_zip_stage_path="@mock_model_zip_stage_path/model.zip",
                deployment_stage_path="@mock_model_deployment_stage_path",
                target_method=constants.PREDICT,
                **self.options,
            )

        m_deployment_class.assert_not_called()

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_missing_required_options(self, m_deployment_class: mock.MagicMock) -> None:
        with self.assertRaisesRegex(ValueError, "compute_pool"):
            options: Dict[str, Any] = {}
            _deploy(
                session=self.m_session,
                service_func_name="mock_service_func",
                model_id="mock_model_id",
                model_zip_stage_path="@mock_model_zip_stage_path/model.zip",
                deployment_stage_path="@mock_model_deployment_stage_path",
                target_method=constants.PREDICT,
                **options,
            )
        m_deployment_class.assert_not_called()

    @mock.patch(
        "snowflake.ml.model._deploy_client.snowservice.deploy." "snowservice_client.SnowServiceClient"
    )  # type: ignore
    def test_get_or_create_image_repo(self, m_snowservice_client_class: mock.MagicMock) -> None:
        # Test when image repo url is provided.
        self.assertEqual(
            _get_or_create_image_repo(
                self.m_session, image_repo="org-account.registry-dev.snowflakecomputing.com/DB/SCHEMA/REPO"
            ),
            "org-account.registry-dev.snowflakecomputing.com/db/schema/repo",
        )

        # Test when session is missing component(db/schema etc) in order to construct image repo url
        with self.assertRaises(RuntimeError):
            _get_or_create_image_repo(self.m_session, image_repo=None)

        # Test constructing image repo from session object
        self.m_session._conn = mock.MagicMock()
        self.m_session._conn._conn = Connection(
            host="account.org.us-west-2.aws.snowflakecomputing.com", account="account", database="DB", schema="SCHEMA"
        )  # type: ignore

        m_snowservice_client = m_snowservice_client_class.return_value
        expected = f"org-account.registry.snowflakecomputing.com/db/schema/{constants.SNOWML_IMAGE_REPO}"
        self.assertEqual(_get_or_create_image_repo(self.m_session, image_repo=None), expected)
        m_snowservice_client.create_image_repo.assert_called_with(constants.SNOWML_IMAGE_REPO)


class SnowServiceDeploymentTestCase(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = cast(session.Session, mock_session.MockSession(conn=None, test_case=self))
        self.m_model_id = "provided_model_id"
        self.m_service_func_name = "provided_service_func_name"
        self.m_model_zip_stage_path = "@provided_model_zip_stage_path/model.zip"
        self.m_deployment_stage_path = "@mock_model_deployment_stage_path"
        self.m_model_dir = "tmp/local_model.zip"
        self.m_options = {
            "stage": "mock_stage",
            "compute_pool": "mock_compute_pool",
            "image_repo": "mock_image_repo",
        }

        self.deployment = SnowServiceDeployment(
            self.m_session,
            model_id=self.m_model_id,
            service_func_name=self.m_service_func_name,
            model_dir=self.m_model_dir,
            model_zip_stage_path=self.m_model_zip_stage_path,
            deployment_stage_path=self.m_deployment_stage_path,
            target_method=constants.PREDICT,
            options=deploy_options.SnowServiceDeployOptions.from_dict(self.m_options),
        )

    def test_deploy(self) -> None:
        with mock.patch.object(
            self.deployment, "_build_and_upload_image"
        ) as m_build_and_upload_image, mock.patch.object(self.deployment, "_deploy_workflow") as m_deploy_workflow:
            self.deployment.deploy()
            m_build_and_upload_image.assert_called_once()
            m_deploy_workflow.assert_called_once()

    @mock.patch(
        "snowflake.ml.model._deploy_client.snowservice.deploy.client_image_builder" ".ClientImageBuilder"
    )  # type: ignore
    def test_build_and_upload_image(self, client_image_builder_class: mock.MagicMock) -> None:
        m_image_builder = client_image_builder_class.return_value
        with mock.patch.object(
            m_image_builder, "build_and_upload_image", return_value="image_path"
        ) as mock_build_and_upload:
            res = self.deployment._build_and_upload_image()
            mock_build_and_upload.assert_called_once()
            self.assertEqual(res, "image_path")


if __name__ == "__main__":
    absltest.main()
