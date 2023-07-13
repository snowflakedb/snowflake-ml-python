from typing import Any, Dict, Optional, TypedDict

from typing_extensions import NotRequired

from snowflake.ml.model._deploy_client.utils import constants


class SnowServiceDeployOptionsTypedHint(TypedDict):
    """Deployment options for deploying to SnowService.

    stage: the name of the stage for uploading artifacts.
    compute_pool: SnowService compute pool name.
    image_repo: SnowService image repo path. e.g. "<image_registry>/<db>/<schema>/<repo>"
    min_instances: Minimum number of service replicas.
    max_instances: Maximum number of service replicas.
    endpoint: The specific name of the endpoint that the service function will communicate with. Default to
                "predict". This option is useful when service has multiple endpoints.
    overridden_base_image: When provided, it will override the base image.
    """

    stage: str
    compute_pool: str
    image_repo: str
    min_instances: NotRequired[int]
    max_instances: NotRequired[int]
    endpoint: NotRequired[str]
    overridden_base_image: NotRequired[str]


class SnowServiceDeployOptions:
    def __init__(
        self,
        stage: str,
        compute_pool: str,
        image_repo: str,
        *,
        min_instances: int = 1,
        max_instances: int = 1,
        endpoint: str = constants.PREDICT,
        overridden_base_image: Optional[str] = None,
        prebuilt_snowflake_image: Optional[str] = None,
    ) -> None:
        """Initialization

        Args:
            stage: the name of the stage for uploading artifacts.
            compute_pool: SnowService compute pool name.
            image_repo: SnowService image repo path. e.g. "<image_registry>/<db>/<schema>/<repo>"
            min_instances: Minimum number of service replicas.
            max_instances: Maximum number of service replicas.
            endpoint: The specific name of the endpoint that the service function will communicate with. Default to
                        "predict". This option is useful when service has multiple endpoints.
            overridden_base_image: When provided, it will override the base image.
            prebuilt_snowflake_image: When provided, the image building step is skipped, and the pre-built image from
                Snowflake is used as is. This option is for users who consistently use the same image for multiple use
                cases, allowing faster deployment. The snowflake image used for deployment is logged to the console for
                future use.
        """

        self.stage = stage
        self.compute_pool = compute_pool
        self.image_repo = image_repo
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.endpoint = endpoint
        self.overridden_base_image = overridden_base_image
        self.prebuilt_snowflake_image = prebuilt_snowflake_image

    @classmethod
    def from_dict(cls, options_dict: Dict[str, Any]) -> "SnowServiceDeployOptions":
        """Construct SnowServiceDeployOptions instance based from an option dictionary.

        Args:
            options_dict: The dict containing various deployment options.

        Raises:
            ValueError: When required option is missing.

        Returns:
            A SnowServiceDeployOptions object
        """
        required_options = [constants.STAGE, constants.COMPUTE_POOL, constants.IMAGE_REPO]
        missing_keys = [key for key in required_options if options_dict.get(key) is None]
        if missing_keys:
            raise ValueError(f"Must provide options when deploying to SnowService: {', '.join(missing_keys)}")
        # SnowService image repo cannot handle upper case repo name.
        options_dict[constants.IMAGE_REPO] = options_dict[constants.IMAGE_REPO].lower()
        return cls(**options_dict)
