import logging
import os
import posixpath
import string
import tempfile
from abc import ABC
from typing import Any, Dict, Optional, cast

from typing_extensions import Unpack

from snowflake.ml._internal import file_utils
from snowflake.ml.model import _model, _model_meta, type_hints
from snowflake.ml.model._deploy_client.image_builds import client_image_builder
from snowflake.ml.model._deploy_client.snowservice import deploy_options
from snowflake.ml.model._deploy_client.utils import constants, snowservice_client
from snowflake.snowpark import FileOperation, Session


def _deploy(
    session: Session,
    *,
    model_id: str,
    service_func_name: str,
    model_zip_stage_path: str,
    deployment_stage_path: str,
    **kwargs: Unpack[type_hints.SnowparkContainerServiceDeployOptions],
) -> _model_meta.ModelMetadata:
    """Entrypoint for model deployment to SnowService. This function will trigger a docker image build followed by
    workflow deployment to SnowService.

    Args:
        session: Snowpark session
        model_id: Unique hex string of length 32, provided by model registry.
        service_func_name: The service function name in SnowService associated with the created service.
        model_zip_stage_path: Path to model zip file in stage. Note that this path has a "@" prefix.
        deployment_stage_path: Path to stage containing deployment artifacts.
        **kwargs: various SnowService deployment options.

    Raises:
        ValueError: Raised when model_id is empty.
        ValueError: Raised when service_func_name is empty.
        ValueError: Raised when model_stage_file_path is empty.

    Returns:
        The metadata of the model that has been deployed.
    """
    snowpark_logger = logging.getLogger("snowflake.snowpark")
    snowflake_connector_logger = logging.getLogger("snowflake.connector")
    snowpark_log_level = snowpark_logger.level
    snowflake_connector_log_level = snowflake_connector_logger.level
    try:
        # Setting appropriate log level to prevent console from being polluted by vast amount of snowpark and snowflake
        # connector logging.
        snowpark_logger.setLevel(logging.WARNING)
        snowflake_connector_logger.setLevel(logging.WARNING)
        if not model_id:
            raise ValueError('Must provide a non-empty string for "model_id" when deploying to SnowService')
        if not service_func_name:
            raise ValueError('Must provide a non-empty string for "service_func_name" when deploying to SnowService')
        if not model_zip_stage_path:
            raise ValueError(
                'Must provide a non-empty string for "model_stage_file_path" when deploying to SnowService'
            )
        if not deployment_stage_path:
            raise ValueError(
                'Must provide a non-empty string for "deployment_stage_path" when deploying to SnowService'
            )

        # Remove full qualified name to avoid double quotes corrupting the service spec
        model_zip_stage_path = model_zip_stage_path.replace('"', "")
        deployment_stage_path = deployment_stage_path.replace('"', "")

        assert model_zip_stage_path.startswith("@"), f"stage path should start with @, actual: {model_zip_stage_path}"
        assert deployment_stage_path.startswith("@"), f"stage path should start with @, actual: {deployment_stage_path}"
        options = deploy_options.SnowServiceDeployOptions.from_dict(cast(Dict[str, Any], kwargs))

        # TODO[shchen]: SNOW-863701, Explore ways to prevent entire model zip being downloaded during deploy step
        #  (for both warehouse and snowservice deployment)
        # One alternative is for model registry to duplicate the model metadata and env dependency storage from model
        # zip so that we don't have to pull down the entire model zip.
        fo = FileOperation(session=session)
        zf = fo.get_stream(model_zip_stage_path)
        with file_utils.unzip_stream_in_temp_dir(stream=zf) as temp_local_model_dir_path:
            # Download the model zip file that is already uploaded to stage during model registry log_model step.
            # This is needed in order to obtain the conda and requirement file inside the model zip, as well as to
            # return the model object needed for deployment info tracking.
            ss_deployment = SnowServiceDeployment(
                session=session,
                model_id=model_id,
                service_func_name=service_func_name,
                model_zip_stage_path=model_zip_stage_path,  # Pass down model_zip_stage_path for service spec file
                deployment_stage_path=deployment_stage_path,
                model_dir=temp_local_model_dir_path,
                options=options,
            )
            ss_deployment.deploy()
            meta = _model.load_model(model_dir_path=temp_local_model_dir_path, meta_only=True)
            return meta
    finally:
        # Preserve the original logging level.
        snowpark_logger.setLevel(snowpark_log_level)
        snowflake_connector_logger.setLevel(snowflake_connector_log_level)


def _get_or_create_image_repo(session: Session, *, image_repo: Optional[str]) -> str:
    def _sanitize_dns_url(url: str) -> str:
        # Align with existing SnowService image registry url standard.
        return url.lower()

    if image_repo:
        return _sanitize_dns_url(image_repo)

    try:
        conn = session._conn._conn
        org = conn.host.split(".")[1]
        account = conn.account
        db = conn._database
        schema = conn._schema
        subdomain = constants.PROD_IMAGE_REGISTRY_SUBDOMAIN
        sanitized_url = _sanitize_dns_url(
            f"{org}-{account}.{subdomain}.{constants.PROD_IMAGE_REGISTRY_DOMAIN}/{db}/"
            f"{schema}/{constants.SNOWML_IMAGE_REPO}"
        )
        client = snowservice_client.SnowServiceClient(session)
        client.create_image_repo(constants.SNOWML_IMAGE_REPO)
        return sanitized_url
    except Exception:
        raise RuntimeError(
            "Failed to construct image repo URL, please ensure the following connections"
            "parameters are set in your session: ['host', 'account', 'database', 'schema']"
        )


class SnowServiceDeployment(ABC):
    """
    Class implementation that encapsulates image build and workflow deployment to SnowService
    """

    def __init__(
        self,
        session: Session,
        model_id: str,
        service_func_name: str,
        model_dir: str,
        model_zip_stage_path: str,
        deployment_stage_path: str,
        options: deploy_options.SnowServiceDeployOptions,
    ) -> None:
        """Initialization

        Args:
            session: Snowpark session
            model_id: Unique hex string of length 32, provided by model registry; if not provided, auto-generate one for
                        resource naming.The model_id serves as an idempotent key throughout the deployment workflow.
            service_func_name: The service function name in SnowService associated with the created service.
            model_dir: Local model directory, downloaded form stage and extracted.
            model_zip_stage_path: Path to model zip file in stage.
            deployment_stage_path: Path to stage containing deployment artifacts.
            options: A SnowServiceDeployOptions object containing deployment options.
        """

        self.session = session
        self.id = model_id
        self.service_func_name = service_func_name
        self.model_zip_stage_path = model_zip_stage_path
        self.model_dir = model_dir
        self.options = options
        self._service_name = f"service_{model_id}"
        # Spec file and future deployment related artifacts will be stored under {stage}/models/{model_id}
        self._model_artifact_stage_location = posixpath.join(deployment_stage_path, "models", self.id)

    def deploy(self) -> None:
        """
        This function triggers image build followed by workflow deployment to SnowService.
        """
        if self.options.prebuilt_snowflake_image:
            image = self.options.prebuilt_snowflake_image
            logging.warning(f"Skipped image build. Use prebuilt image: {self.options.prebuilt_snowflake_image}")
        else:
            logging.warning(
                "Building the Docker image and deploying to Snowpark Container Service. "
                "This process may take a few minutes."
            )
            image = self._build_and_upload_image()

            logging.warning(
                f"Image successfully built! To prevent the need for rebuilding the Docker image in future deployments, "
                f"simply specify 'prebuilt_snowflake_image': '{image}' in the options field of the deploy() function"
            )
        self._deploy_workflow(image)

    def _build_and_upload_image(self) -> str:
        """This function handles image build and upload to image registry.

        Returns:
            Path to the image in the remote image repository.
        """
        image_repo = _get_or_create_image_repo(self.session, image_repo=self.options.image_repo)
        image_builder = client_image_builder.ClientImageBuilder(
            id=self.id,
            image_repo=image_repo,
            model_dir=self.model_dir,
            session=self.session,
            use_gpu=True if self.options.use_gpu else False,
        )
        return image_builder.build_and_upload_image()

    def _prepare_and_upload_artifacts_to_stage(self, image: str) -> None:
        """Constructs and upload service spec to stage.

        Args:
            image: Name of the image to create SnowService container from.
        """

        with tempfile.TemporaryDirectory() as tempdir:
            spec_template_path = os.path.join(os.path.dirname(__file__), "templates/service_spec_template")
            spec_file_path = os.path.join(tempdir, f"{constants.SERVICE_SPEC}.yaml")

            with open(spec_template_path, encoding="utf-8") as template, open(
                spec_file_path, "w", encoding="utf-8"
            ) as spec_file:
                content = string.Template(template.read()).substitute(
                    {
                        "image": image,
                        "predict_endpoint_name": constants.PREDICT,
                        "model_stage": self.model_zip_stage_path[1:].split("/")[0],  # Reserve only the stage name
                        "model_zip_stage_path": self.model_zip_stage_path[1:],  # Remove the @ prefix
                        "inference_server_container_name": constants.INFERENCE_SERVER_CONTAINER,
                    }
                )
                spec_file.write(content)
                logging.info(f"Create service spec: \n {content}")

            self.session.file.put(
                local_file_name=spec_file_path,
                stage_location=self._model_artifact_stage_location,
                auto_compress=False,
                overwrite=True,
            )
            logging.info(
                f"Uploaded spec file {os.path.basename(spec_file_path)} " f"to {self._model_artifact_stage_location}"
            )

    def _deploy_workflow(self, image: str) -> None:
        """This function handles workflow deployment to SnowService with the given image.

        Args:
            image: Name of the image to create SnowService container from.
        """

        self._prepare_and_upload_artifacts_to_stage(image)
        client = snowservice_client.SnowServiceClient(self.session)
        spec_stage_location = posixpath.join(
            self._model_artifact_stage_location.rstrip("/"), f"{constants.SERVICE_SPEC}.yaml"
        )
        client.create_or_replace_service(
            service_name=self._service_name,
            compute_pool=self.options.compute_pool,
            spec_stage_location=spec_stage_location,
            min_instances=self.options.min_instances,
            max_instances=self.options.max_instances,
        )
        client.block_until_resource_is_ready(
            resource_name=self._service_name, resource_type=constants.ResourceType.SERVICE
        )
        client.create_or_replace_service_function(
            service_func_name=self.service_func_name,
            service_name=self._service_name,
            endpoint_name=constants.PREDICT,
        )
