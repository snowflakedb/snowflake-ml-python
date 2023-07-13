import logging
import os
import posixpath
import string
import tempfile
from abc import ABC
from typing import Any, Dict, cast

from typing_extensions import Unpack

from snowflake.ml.model._deploy_client.image_builds import (
    base_image_builder,
    client_image_builder,
)
from snowflake.ml.model._deploy_client.snowservice import deploy_options
from snowflake.ml.model._deploy_client.utils import constants, snowservice_client
from snowflake.snowpark import Session


def _deploy(
    session: Session,
    *,
    model_id: str,
    service_func_name: str,
    model_zip_stage_path: str,
    **kwargs: Unpack[deploy_options.SnowServiceDeployOptionsTypedHint],
) -> None:
    """Entrypoint for model deployment to SnowService. This function will trigger a docker image build followed by
    workflow deployment to SnowService.

    Args:
        session: Snowpark session
        model_id: Unique hex string of length 32, provided by model registry.
        service_func_name: The service function name in SnowService associated with the created service.
        model_zip_stage_path: Path to model zip file in stage. Note that this path has a "@" prefix.
        **kwargs: various SnowService deployment options.

    Raises:
        ValueError: Raised when model_id is empty.
        ValueError: Raised when service_func_name is empty.
        ValueError: Raised when model_stage_file_path is empty.
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
        assert model_zip_stage_path.startswith("@"), f"stage path should start with @, actual: {model_zip_stage_path}"
        options = deploy_options.SnowServiceDeployOptions.from_dict(cast(Dict[str, Any], kwargs))
        image_builder = client_image_builder.ClientImageBuilder(
            id=model_id, image_repo=options.image_repo, model_zip_stage_path=model_zip_stage_path, session=session
        )
        ss_deployment = SnowServiceDeployment(
            session=session,
            model_id=model_id,
            service_func_name=service_func_name,
            model_zip_stage_path=model_zip_stage_path,
            image_builder=image_builder,
            options=options,
        )
        ss_deployment.deploy()
    finally:
        # Preserve the original logging level.
        snowpark_logger.setLevel(snowpark_log_level)
        snowflake_connector_logger.setLevel(snowflake_connector_log_level)


class SnowServiceDeployment(ABC):
    """
    Class implementation that encapsulates image build and workflow deployment to SnowService

    #TODO[shchen], SNOW-830093 GPU support on model deployment to SnowService
    """

    def __init__(
        self,
        session: Session,
        model_id: str,
        service_func_name: str,
        model_zip_stage_path: str,
        image_builder: base_image_builder.ImageBuilder,
        options: deploy_options.SnowServiceDeployOptions,
    ) -> None:
        """Initialization

        Args:
            session: Snowpark session
            model_id: Unique hex string of length 32, provided by model registry; if not provided, auto-generate one for
                        resource naming.The model_id serves as an idempotent key throughout the deployment workflow.
            service_func_name: The service function name in SnowService associated with the created service.
            model_zip_stage_path: Path to model zip file in stage.
            image_builder: InferenceImageBuilder instance that handles image build and upload to image registry.
            options: A SnowServiceDeployOptions object containing deployment options.
        """

        self.session = session
        self.id = model_id
        self.service_func_name = service_func_name
        self.model_zip_stage_path = model_zip_stage_path
        self.image_builder = image_builder
        self.options = options
        self._service_name = f"service_{model_id}"
        # Spec file and future deployment related artifacts will be stored under {stage}/models/{model_id}
        self._model_artifact_stage_location = posixpath.join(options.stage, "models", self.id)

    def deploy(self) -> None:
        """
        This function triggers image build followed by workflow deployment to SnowService.
        """
        if self.options.prebuilt_snowflake_image:
            image = self.options.prebuilt_snowflake_image
            logging.info(f"Skipped image build. Use Snowflake prebuilt image: {self.options.prebuilt_snowflake_image}")
        else:
            image = self._build_and_upload_image()
        self._deploy_workflow(image)

    def _build_and_upload_image(self) -> str:
        """This function handles image build and upload to image registry.

        Returns:
            Path to the image in the remote image repository.
        """
        return self.image_builder.build_and_upload_image()

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
                        "stage": self.options.stage,
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
