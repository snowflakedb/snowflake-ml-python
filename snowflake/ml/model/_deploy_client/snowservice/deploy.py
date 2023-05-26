from abc import ABC
from typing import Dict

from snowflake.ml.model._deploy_client.docker import (
    base_image_builder,
    client_image_builder,
)
from snowflake.snowpark import Session


def _deploy(
    session: Session, *, model_id: str, service_func_name: str, model_dir: str, options: Dict[str, str]
) -> None:
    """Entrypoint for model deployment to SnowService. This function will trigger a docker image build followed by
    workflow deployment to SnowService.

    Args:
        session: Snowpark session
        model_id: Unique hex string of length 32, provided by model registry.
        service_func_name: The service function name in SnowService associated with the created service.
        model_dir: Path to model directory.
        options: various SnowService deployment options.

    Raises:
        ValueError: Raised when target method does not exist in model.
    """
    if not model_id:
        raise ValueError('Must provide a non-empty string for the "model_id" when deploying to SnowService')

    image_builder = client_image_builder.ClientImageBuilder()
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
    """

    def __init__(
        self,
        session: Session,
        *,
        model_id: str,
        service_func_name: str,
        model_dir: str,
        image_builder: base_image_builder.ImageBuilder,
        options: Dict[str, str]
    ) -> None:
        """Initialization.

        Args:
            session: Snowpark session
            model_id: Unique hex string of length 32, provided by model registry; if not provided, auto-generate one for
                    resource naming.The model_id serves as an idempotent key throughout the deployment workflow.
            service_func_name: The service function name in SnowService associated with the created service.
            model_dir: Path to model directory.
            image_builder: InferenceImageBuilder instance that handles image build and upload to image registry.
            options: various SnowService deployment options.
        """
        self.session = session
        self.idempotent_key = model_id
        self.service_func_name = service_func_name
        self.model_dir = model_dir
        self.image_builder = image_builder
        self.options = options

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
