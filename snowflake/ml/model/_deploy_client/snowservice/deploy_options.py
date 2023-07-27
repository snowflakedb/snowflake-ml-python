import inspect
from typing import Any, Dict, Optional

from snowflake.ml.model._deploy_client.utils import constants


class SnowServiceDeployOptions:
    def __init__(
        self,
        compute_pool: str,
        *,
        image_repo: Optional[str] = None,
        min_instances: Optional[int] = 1,
        max_instances: Optional[int] = 1,
        endpoint: Optional[str] = constants.PREDICT,
        prebuilt_snowflake_image: Optional[str] = None,
        use_gpu: Optional[bool] = False,
    ) -> None:
        """Initialization

        When updated, please ensure the type hint is updated accordingly at: //snowflake/ml/model/type_hints

        Args:
            compute_pool: SnowService compute pool name. Please refer to official doc for how to create a
                compute pool: https://docs.snowflake.com/LIMITEDACCESS/snowpark-containers/reference/compute-pool
            image_repo: SnowService image repo path. e.g. "<image_registry>/<db>/<schema>/<repo>". Default to auto
                inferred based on session information.
            min_instances: Minimum number of service replicas. Default to 1.
            max_instances: Maximum number of service replicas. Default to 1.
            endpoint: The specific name of the endpoint that the service function will communicate with. This option is
                useful when the service has multiple endpoints. Default to “predict”.
            prebuilt_snowflake_image: When provided, the image-building step is skipped, and the pre-built image from
                Snowflake is used as is. This option is for users who consistently use the same image for multiple use
                cases, allowing faster deployment. The snowflake image used for deployment is logged to the console for
                future use. Default to None.
            use_gpu: When set to True, a CUDA-enabled Docker image will be used to provide a runtime CUDA environment.
                Default to False.
        """

        self.compute_pool = compute_pool
        self.image_repo = image_repo
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.endpoint = endpoint
        self.prebuilt_snowflake_image = prebuilt_snowflake_image
        self.use_gpu = use_gpu

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
        required_options = [constants.COMPUTE_POOL]
        missing_keys = [key for key in required_options if options_dict.get(key) is None]
        if missing_keys:
            raise ValueError(f"Must provide options when deploying to SnowService: {', '.join(missing_keys)}")
        supported_options_keys = inspect.signature(cls.__init__).parameters.keys()
        filtered_options = {k: v for k, v in options_dict.items() if k in supported_options_keys}
        return cls(**filtered_options)
