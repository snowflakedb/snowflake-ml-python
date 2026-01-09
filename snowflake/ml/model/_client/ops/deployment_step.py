import enum
import hashlib
from typing import Optional


class DeploymentStep(enum.Enum):
    MODEL_BUILD = ("model-build", "model_build_")
    MODEL_INFERENCE = ("model-inference", None)
    MODEL_LOGGING = ("model-logging", "model_logging_")

    def __init__(self, container_name: str, service_name_prefix: Optional[str]) -> None:
        self._container_name = container_name
        self._service_name_prefix = service_name_prefix

    @property
    def container_name(self) -> str:
        """Get the container name for the deployment step."""
        return self._container_name

    @property
    def service_name_prefix(self) -> Optional[str]:
        """Get the service name prefix for the deployment step."""
        return self._service_name_prefix


def get_service_id_from_deployment_step(query_id: str, deployment_step: DeploymentStep) -> str:
    """Get the service ID through the server-side logic."""
    uuid = query_id.replace("-", "")
    big_int = int(uuid, 16)
    md5_hash = hashlib.md5(str(big_int).encode(), usedforsecurity=False).hexdigest()
    identifier = md5_hash[:8]
    service_name_prefix = deployment_step.service_name_prefix
    if service_name_prefix is None:
        # raise an exception if the service name prefix is None
        raise ValueError(f"Service name prefix is {service_name_prefix} for deployment step {deployment_step}.")
    return (service_name_prefix + identifier).upper()
