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
from snowflake.ml.test_utils import exception_utils, mock_data_frame, mock_session
from snowflake.snowpark import row, session


class Connection:
    def __init__(self, account: str, database: str, schema: str) -> None:
        self.account = account
        self._database = database
        self._schema = schema


class DeployTestCase(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.options: Dict[str, Any] = {
            "compute_pool": "mock_compute_pool",
            "image_repo": "mock_image_repo",
        }

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy._model_meta.ModelMetadata")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_model_id(self, m_deployment_class: mock.MagicMock, m_model_meta_class: mock.MagicMock) -> None:
        m_deployment = m_deployment_class.return_value
        m_model_meta = m_model_meta_class.return_value

        m_model_zip_stage_path = "@mock_model_zip_stage_path/model.zip"
        m_deployment_stage_path = "@mock_model_deployment_stage_path"

        _deploy(
            session=cast(session.Session, self.m_session),
            model_id="provided_model_id",
            model_meta=m_model_meta,
            service_func_name="mock_service_func",
            model_zip_stage_path=m_model_zip_stage_path,
            deployment_stage_path=m_deployment_stage_path,
            target_method=constants.PREDICT,
            **self.options,
        )

        m_deployment_class.assert_called_once_with(
            session=self.m_session,
            model_id="provided_model_id",
            service_func_name="mock_service_func",
            model_zip_stage_path=m_model_zip_stage_path,
            deployment_stage_path=m_deployment_stage_path,
            model_meta=m_model_meta,
            target_method=constants.PREDICT,
            options=mock.ANY,
        )
        m_deployment.deploy.assert_called_once()

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy._model_meta.ModelMetadata")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_empty_model_id(
        self, m_deployment_class: mock.MagicMock, m_model_meta_class: mock.MagicMock
    ) -> None:
        m_model_meta = m_model_meta_class.return_value
        with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=ValueError):
            _deploy(
                session=cast(session.Session, self.m_session),
                service_func_name="mock_service_func",
                model_id="",
                model_meta=m_model_meta,
                model_zip_stage_path="@mock_model_zip_stage_path/model.zip",
                deployment_stage_path="@mock_model_deployment_stage_path",
                target_method=constants.PREDICT,
                **self.options,
            )

        m_deployment_class.assert_not_called()

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy._model_meta.ModelMetadata")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_missing_required_options(
        self, m_deployment_class: mock.MagicMock, m_model_meta_class: mock.MagicMock
    ) -> None:
        m_model_meta = m_model_meta_class.return_value
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=ValueError, expected_regex="compute_pool"
        ):
            options: Dict[str, Any] = {}
            _deploy(
                session=cast(session.Session, self.m_session),
                service_func_name="mock_service_func",
                model_id="mock_model_id",
                model_meta=m_model_meta,
                model_zip_stage_path="@mock_model_zip_stage_path/model.zip",
                deployment_stage_path="@mock_model_deployment_stage_path",
                target_method=constants.PREDICT,
                **options,
            )
        m_deployment_class.assert_not_called()

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy._model_meta.ModelMetadata")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_over_requested_gpus(
        self, m_deployment_class: mock.MagicMock, m_model_meta_class: mock.MagicMock
    ) -> None:
        m_model_meta = m_model_meta_class.return_value
        with exception_utils.assert_snowml_exceptions(
            self, expected_original_error_type=RuntimeError, expected_regex="GPU request exceeds instance capability"
        ):
            self.m_session.add_mock_sql(
                query=f"DESC COMPUTE POOL {self.options['compute_pool']}",
                result=mock_data_frame.MockDataFrame(
                    [row.Row(name="MY_GPU_POOL", state="IDLE", min_nodes=1, max_nodes=1, instance_family="GPU_3")]
                ),
            )

            _deploy(
                session=cast(session.Session, self.m_session),
                service_func_name="mock_service_func",
                model_id="mock_model_id",
                model_meta=m_model_meta,
                model_zip_stage_path="@mock_model_zip_stage_path/model.zip",
                deployment_stage_path="@mock_model_deployment_stage_path",
                target_method=constants.PREDICT,
                num_gpus=2,
                **self.options,
            )
        m_deployment_class.assert_not_called()

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy._model_meta.ModelMetadata")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_over_requested_gpus_no_cuda(
        self, m_deployment_class: mock.MagicMock, m_model_meta_class: mock.MagicMock
    ) -> None:
        m_model_meta = m_model_meta_class.return_value
        m_model_meta.cuda_version = None
        with exception_utils.assert_snowml_exceptions(
            self,
            expected_original_error_type=ValueError,
            expected_regex="You are requesting GPUs for models that do not use a GPU or does not have CUDA version set",
        ):
            self.m_session.add_mock_sql(
                query=f"DESC COMPUTE POOL {self.options['compute_pool']}",
                result=mock_data_frame.MockDataFrame(
                    [row.Row(name="MY_GPU_POOL", state="IDLE", min_nodes=1, max_nodes=1, instance_family="GPU_7")]
                ),
            )
            _deploy(
                session=cast(session.Session, self.m_session),
                service_func_name="mock_service_func",
                model_id="mock_model_id",
                model_meta=m_model_meta,
                model_zip_stage_path="@mock_model_zip_stage_path/model.zip",
                deployment_stage_path="@mock_model_deployment_stage_path",
                target_method=constants.PREDICT,
                num_gpus=2,
                **self.options,
            )
        m_deployment_class.assert_not_called()

    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.copy.deepcopy")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy._model_meta.ModelMetadata")  # type: ignore
    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy.SnowServiceDeployment")  # type: ignore
    def test_deploy_with_gpu_validation_and_unknown_instance_type(
        self, m_deployment_class: mock.MagicMock, m_model_meta_class: mock.MagicMock, m_deepcopy_func: mock.MagicMock
    ) -> None:
        m_deployment = m_deployment_class.return_value
        m_model_meta = m_model_meta_class.return_value
        m_model_meta.cuda_version = "11.7"
        m_model_meta_deploy = m_deepcopy_func.return_value
        m_model_zip_stage_path = "@mock_model_zip_stage_path/model.zip"
        m_deployment_stage_path = "@mock_model_deployment_stage_path"

        unknown_instance_type = "GPU_UNKNOWN"
        self.m_session.add_mock_sql(
            query=f"DESC COMPUTE POOL {self.options['compute_pool']}",
            result=mock_data_frame.MockDataFrame(
                [row.Row(name="MY_GPU_POOL", state="IDLE", instance_family=unknown_instance_type)]
            ),
        )
        with self.assertLogs(level="INFO") as cm:
            _deploy(
                session=cast(session.Session, self.m_session),
                model_id="provided_model_id",
                model_meta=m_model_meta,
                service_func_name="mock_service_func",
                model_zip_stage_path=m_model_zip_stage_path,
                deployment_stage_path=m_deployment_stage_path,
                target_method=constants.PREDICT,
                num_gpus=2,
                **self.options,
            )

            self.assertListEqual(
                cm.output,
                [
                    "INFO:root:num_workers has been defaulted to 1 when using GPU.",
                    (
                        "WARNING:snowflake.ml.model._deploy_client.snowservice.deploy:Unknown "
                        "instance type: GPU_UNKNOWN, skipping GPU validation"
                    ),
                ],
            )

        m_deployment_class.assert_called_once_with(
            session=self.m_session,
            model_id="provided_model_id",
            model_meta=m_model_meta_deploy,
            service_func_name="mock_service_func",
            model_zip_stage_path=m_model_zip_stage_path,
            deployment_stage_path=m_deployment_stage_path,
            target_method=constants.PREDICT,
            options=mock.ANY,
        )
        m_deployment.deploy.assert_called_once()

    @mock.patch(
        "snowflake.ml.model._deploy_client.snowservice.deploy." "snowservice_client.SnowServiceClient"
    )  # type: ignore
    def test_get_or_create_image_repo(self, m_snowservice_client_class: mock.MagicMock) -> None:
        db = "mock_db"
        schema = "mock_schema"
        session_db = "session_db"
        session_schema = "session_schema"
        repo_url = f"org-account.registry.snowflakecomputing.com/{db}/{schema}/{constants.SNOWML_IMAGE_REPO}"
        session_repo_url = (
            f"org-account.registry-dev.snowflakecomputing.com"
            f"/{session_db}/{session_schema}/{constants.SNOWML_IMAGE_REPO}"
        )
        self.m_session.add_mock_sql(
            query=f"SHOW IMAGE REPOSITORIES LIKE '{constants.SNOWML_IMAGE_REPO}' IN SCHEMA {'.'.join([db, schema])}",
            result=mock_data_frame.MockDataFrame([row.Row(name=constants.SNOWML_IMAGE_REPO, repository_url=repo_url)]),
        )
        self.m_session.add_mock_sql(
            query=f"SHOW IMAGE REPOSITORIES LIKE '{constants.SNOWML_IMAGE_REPO}' "
            f"IN SCHEMA {'.'.join([session_db, session_schema])}",
            result=mock_data_frame.MockDataFrame(
                [row.Row(name=constants.SNOWML_IMAGE_REPO, repository_url=session_repo_url)]
            ),
        )
        self.m_session._conn = mock.MagicMock()
        self.m_session._conn._conn = Connection(account="account", database=session_db, schema=session_schema)

        # Test when image repo url is provided.
        self.assertEqual(
            _get_or_create_image_repo(
                session=cast(session.Session, self.m_session),
                service_func_name="func",
                image_repo="dummy.dummy.snowflakecomputing.com/DB/SCHEMA/REPO",
            ),
            "dummy.dummy.snowflakecomputing.com/db/schema/repo",
        )

        # Test when image repo is constructed from db/schema inferred from the service func.
        self.assertEqual(
            _get_or_create_image_repo(
                session=cast(session.Session, self.m_session), service_func_name=f"{db}.{schema}.func"
            ),
            repo_url,
        )
        m_snowservice_client = m_snowservice_client_class.return_value
        m_snowservice_client.create_image_repo.assert_called_with(f"{db}.{schema}.{constants.SNOWML_IMAGE_REPO}")

        # Test constructing image repo from session object, this happens when service func is missing db and/or schema.
        self.assertEqual(
            _get_or_create_image_repo(session=cast(session.Session, self.m_session), service_func_name="func"),
            session_repo_url,
        )
        m_snowservice_client.create_image_repo.assert_called_with(
            f"{session_db}.{session_schema}.{constants.SNOWML_IMAGE_REPO}"
        )

        with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
            # Cannot find image repo in the given db/schema.
            self.m_session.add_mock_sql(
                query=f"SHOW IMAGE REPOSITORIES LIKE "
                f"'{constants.SNOWML_IMAGE_REPO}' IN SCHEMA {'.'.join([db, schema])}",
                result=mock_data_frame.MockDataFrame([]),
            )
            _get_or_create_image_repo(
                session=cast(session.Session, self.m_session), service_func_name=f"{db}.{schema}.func"
            )


class SnowServiceDeploymentTestCase(absltest.TestCase):
    @mock.patch("snowflake.ml.model._deploy_client.snowservice.deploy._model_meta.ModelMetadata")  # type: ignore
    def setUp(self, m_model_meta_class: mock.MagicMock) -> None:
        super().setUp()
        self.m_session = cast(session.Session, mock_session.MockSession(conn=None, test_case=self))
        self.m_model_id = "provided_model_id"
        self.m_service_func_name = "mock_db.mock_schema.provided_service_func_name"
        self.m_model_zip_stage_path = "@provided_model_zip_stage_path/model.zip"
        self.m_deployment_stage_path = "@mock_model_deployment_stage_path"
        self.m_model_meta = m_model_meta_class.return_value
        self.m_options = {
            "stage": "mock_stage",
            "compute_pool": "mock_compute_pool",
            "image_repo": "mock_image_repo",
        }

        self.deployment = SnowServiceDeployment(
            self.m_session,
            model_id=self.m_model_id,
            service_func_name=self.m_service_func_name,
            model_meta=self.m_model_meta,
            model_zip_stage_path=self.m_model_zip_stage_path,
            deployment_stage_path=self.m_deployment_stage_path,
            target_method=constants.PREDICT,
            options=deploy_options.SnowServiceDeployOptions.from_dict(self.m_options),
        )

    def test_service_name(self) -> None:
        self.assertEqual(self.deployment._service_name, "mock_db.mock_schema.service_provided_model_id")

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
