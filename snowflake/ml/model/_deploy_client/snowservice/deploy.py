from abc import ABC
from typing import Any, Dict, cast

from typing_extensions import Unpack

from snowflake.ml.model._deploy_client.image_builds import (
    base_image_builder,
    client_image_builder,
)
from snowflake.ml.model._deploy_client.snowservice import deploy_options
from snowflake.snowpark import Session


def _deploy(
    session: Session,
    *,
    model_id: str,
    service_func_name: str,
    model_dir: str,
    **kwargs: Unpack[deploy_options.SnowServiceDeployOptionsTypedHint],
) -> None:
    """Entrypoint for model deployment to SnowService. This function will trigger a docker image build followed by
    workflow deployment to SnowService.

    Args:
        session: Snowpark session
        model_id: Unique hex string of length 32, provided by model registry.
        service_func_name: The service function name in SnowService associated with the created service.
        model_dir: Path to model directory.
        **kwargs: various SnowService deployment options.

    Raises:
        ValueError: Raised when model_id is empty.
        ValueError: Raised when service_func_name is empty.
        ValueError: Raised when model_dir is empty.
    """
    if not model_id:
        raise ValueError('Must provide a non-empty string for "model_id" when deploying to SnowService')
    if not service_func_name:
        raise ValueError('Must provide a non-empty string for "service_func_name" when deploying to SnowService')
    if not model_dir:
        raise ValueError('Must provide a non-empty string for "model_dir" when deploying to SnowService')
    options = deploy_options.SnowServiceDeployOptions.from_dict(cast(Dict[str, Any], kwargs))
    image_builder = client_image_builder.ClientImageBuilder(
        id=model_id, image_repo=options.image_repo, model_dir=model_dir
    )
    ss_deployment = SnowServiceDeployment(
        session=session,
        model_id=model_id,
        service_func_name=service_func_name,
        model_dir=model_dir,
        image_builder=image_builder,
        options=options,
    )
    ss_deployment.deploy()


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
        model_dir: str,
        image_builder: base_image_builder.ImageBuilder,
        options: deploy_options.SnowServiceDeployOptions,
    ) -> None:
        """Initialization

        Args:
            session: Snowpark session
            model_id: Unique hex string of length 32, provided by model registry; if not provided, auto-generate one for
                        resource naming.The model_id serves as an idempotent key throughout the deployment workflow.
            service_func_name: The service function name in SnowService associated with the created service.
            model_dir: Path to model directory.
            image_builder: InferenceImageBuilder instance that handles image build and upload to image registry.
            options: A SnowServiceDeployOptions object containing deployment options.
        """

        self.session = session
        self.id = model_id
        self.service_func_name = service_func_name
        self.model_dir = model_dir
        self.image_builder = image_builder
        self.options = options

        self._stage_location = "/".join([options.stage.rstrip("/"), "models", self.id])
        self._service_name = f"service_{model_id}"
        self._spec_file_location = "/".join([self._stage_location.rstrip("/"), f"{self.id}.yaml"])

    def deploy(self) -> None:
        """
        This function triggers image build followed by workflow deployment to SnowService.
        """
        self._build_and_upload_image()
        self._deploy_workflow()

    def _build_and_upload_image(self) -> None:
        """
        This function handles image build and upload to image registry.
        """
        self.image_builder.build_and_upload_image()

    def _deploy_workflow(self) -> None:
        """
        This function handles workflow deployment to SnowService.
        """
        pass
