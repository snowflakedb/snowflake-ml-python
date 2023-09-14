import copy
import logging
import os
import posixpath
import string
import tempfile
import time
from abc import ABC
from typing import Any, Dict, Optional, cast

import yaml
from typing_extensions import Unpack

from snowflake.ml._internal import env_utils
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import identifier, query_result_checker
from snowflake.ml.model import _model_meta, type_hints
from snowflake.ml.model._deploy_client.image_builds import (
    base_image_builder,
    client_image_builder,
    server_image_builder,
)
from snowflake.ml.model._deploy_client.snowservice import deploy_options, instance_types
from snowflake.ml.model._deploy_client.utils import constants, snowservice_client
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def _deploy(
    session: Session,
    *,
    model_id: str,
    model_meta: _model_meta.ModelMetadata,
    service_func_name: str,
    model_zip_stage_path: str,
    deployment_stage_path: str,
    target_method: str,
    **kwargs: Unpack[type_hints.SnowparkContainerServiceDeployOptions],
) -> None:
    """Entrypoint for model deployment to SnowService. This function will trigger a docker image build followed by
    workflow deployment to SnowService.

    Args:
        session: Snowpark session
        model_id: Unique hex string of length 32, provided by model registry.
        model_meta: Model Metadata.
        service_func_name: The service function name in SnowService associated with the created service.
        model_zip_stage_path: Path to model zip file in stage. Note that this path has a "@" prefix.
        deployment_stage_path: Path to stage containing deployment artifacts.
        target_method: The name of the target method to be deployed.
        **kwargs: various SnowService deployment options.

    Raises:
        SnowflakeMLException: Raised when model_id is empty.
        SnowflakeMLException: Raised when service_func_name is empty.
        SnowflakeMLException: Raised when model_stage_file_path is empty.
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
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    'Must provide a non-empty string for "model_id" when deploying to Snowpark Container Services'
                ),
            )
        if not service_func_name:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    'Must provide a non-empty string for "service_func_name"'
                    " when deploying to Snowpark Container Services"
                ),
            )
        if not model_zip_stage_path:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    'Must provide a non-empty string for "model_stage_file_path"'
                    " when deploying to Snowpark Container Services"
                ),
            )
        if not deployment_stage_path:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    'Must provide a non-empty string for "deployment_stage_path"'
                    " when deploying to Snowpark Container Services"
                ),
            )

        # Remove full qualified name to avoid double quotes corrupting the service spec
        model_zip_stage_path = model_zip_stage_path.replace('"', "")
        deployment_stage_path = deployment_stage_path.replace('"', "")

        assert model_zip_stage_path.startswith("@"), f"stage path should start with @, actual: {model_zip_stage_path}"
        assert deployment_stage_path.startswith("@"), f"stage path should start with @, actual: {deployment_stage_path}"
        options = deploy_options.SnowServiceDeployOptions.from_dict(cast(Dict[str, Any], kwargs))

        model_meta_deploy = copy.deepcopy(model_meta)
        if options.use_gpu:
            # Make mypy happy
            assert options.num_gpus is not None
            if model_meta.cuda_version is None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        "You are requesting GPUs for models that do not use a GPU or does not have CUDA version set."
                    ),
                )
            _validate_requested_gpus(session, request_gpus=options.num_gpus, compute_pool=options.compute_pool)
            if model_meta.cuda_version:
                (
                    model_meta_deploy._conda_dependencies,
                    model_meta_deploy._pip_requirements,
                ) = env_utils.generate_env_for_cuda(
                    model_meta._conda_dependencies, model_meta._pip_requirements, model_meta.cuda_version
                )
        else:
            # If user does not need GPU, we set this copies cuda_version to None, thus when Image builder gets a
            # not-None cuda_version, it gets to know that GPU is required.
            model_meta_deploy._cuda_version = None

        # Set conda-forge as backup channel for SPCS deployment
        if "conda-forge" not in model_meta_deploy._conda_dependencies:
            model_meta_deploy._conda_dependencies["conda-forge"] = []

        # TODO[shchen]: SNOW-863701, Explore ways to prevent entire model zip being downloaded during deploy step
        #  (for both warehouse and snowservice deployment)
        # One alternative is for model registry to duplicate the model metadata and env dependency storage from model
        # zip so that we don't have to pull down the entire model zip.
        ss_deployment = SnowServiceDeployment(
            session=session,
            model_id=model_id,
            model_meta=model_meta_deploy,
            service_func_name=service_func_name,
            model_zip_stage_path=model_zip_stage_path,  # Pass down model_zip_stage_path for service spec file
            deployment_stage_path=deployment_stage_path,
            target_method=target_method,
            options=options,
        )
        ss_deployment.deploy()
    finally:
        # Preserve the original logging level.
        snowpark_logger.setLevel(snowpark_log_level)
        snowflake_connector_logger.setLevel(snowflake_connector_log_level)


def _validate_requested_gpus(session: Session, *, request_gpus: int, compute_pool: str) -> None:
    # Remove full qualified name to avoid double quotes, which does not work well in desc compute pool syntax.
    compute_pool = compute_pool.replace('"', "")
    sql = f"DESC COMPUTE POOL {compute_pool}"
    result = (
        query_result_checker.SqlResultValidator(
            session=session,
            query=sql,
        )
        .has_column("instance_family")
        .has_dimensions(expected_rows=1)
        .validate()
    )
    instance_family = result[0]["instance_family"]
    if instance_family in instance_types.INSTANCE_TYPE_TO_GPU_COUNT:
        gpu_capacity = instance_types.INSTANCE_TYPE_TO_GPU_COUNT[instance_family]
        if request_gpus > gpu_capacity:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=RuntimeError(
                    f"GPU request exceeds instance capability; {instance_family} instance type has total "
                    f"capacity of {gpu_capacity} GPU, yet a request was made for {request_gpus} GPUs."
                ),
            )
    else:
        logger.warning(f"Unknown instance type: {instance_family}, skipping GPU validation")


def _get_or_create_image_repo(session: Session, *, service_func_name: str, image_repo: Optional[str] = None) -> str:
    def _sanitize_dns_url(url: str) -> str:
        # Align with existing SnowService image registry url standard.
        return url.lower()

    if image_repo:
        return _sanitize_dns_url(image_repo)

    try:
        conn = session._conn._conn
        # We try to use the same db and schema as the service function locates, as we could retrieve those information
        # if that is a fully qualified one. If not we use the current session one.
        (_db, _schema, _, _) = identifier.parse_schema_level_object_identifier(service_func_name)
        db = _db if _db is not None else conn._database
        schema = _schema if _schema is not None else conn._schema
        assert isinstance(db, str) and isinstance(schema, str)

        client = snowservice_client.SnowServiceClient(session)
        client.create_image_repo(identifier.get_schema_level_object_identifier(db, schema, constants.SNOWML_IMAGE_REPO))
        sql = f"SHOW IMAGE REPOSITORIES LIKE '{constants.SNOWML_IMAGE_REPO}' IN SCHEMA {'.'.join([db, schema])}"
        result = (
            query_result_checker.SqlResultValidator(
                session=session,
                query=sql,
            )
            .has_column("repository_url")
            .has_dimensions(expected_rows=1)
            .validate()
        )
        repository_url = result[0]["repository_url"]
        return str(repository_url)
    except Exception as e:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWPARK_CONTAINER_SERVICE_ERROR,
            original_exception=RuntimeError("Failed to retrieve image repo URL"),
        ) from e


class SnowServiceDeployment(ABC):
    """
    Class implementation that encapsulates image build and workflow deployment to SnowService
    """

    def __init__(
        self,
        session: Session,
        model_id: str,
        model_meta: _model_meta.ModelMetadata,
        service_func_name: str,
        model_zip_stage_path: str,
        deployment_stage_path: str,
        target_method: str,
        options: deploy_options.SnowServiceDeployOptions,
    ) -> None:
        """Initialization

        Args:
            session: Snowpark session
            model_id: Unique hex string of length 32, provided by model registry; if not provided, auto-generate one for
                        resource naming.The model_id serves as an idempotent key throughout the deployment workflow.
            model_meta: Model Metadata.
            service_func_name: The service function name in SnowService associated with the created service.
            model_zip_stage_path: Path to model zip file in stage.
            deployment_stage_path: Path to stage containing deployment artifacts.
            target_method: The name of the target method to be deployed.
            options: A SnowServiceDeployOptions object containing deployment options.
        """

        self.session = session
        self.id = model_id
        self.model_meta = model_meta
        self.service_func_name = service_func_name
        self.model_zip_stage_path = model_zip_stage_path
        self.options = options
        self.target_method = target_method
        (db, schema, _, _) = identifier.parse_schema_level_object_identifier(service_func_name)

        self._service_name = identifier.get_schema_level_object_identifier(db, schema, f"service_{model_id}")
        # Spec file and future deployment related artifacts will be stored under {stage}/models/{model_id}
        self._model_artifact_stage_location = posixpath.join(deployment_stage_path, "models", self.id)

    def deploy(self) -> None:
        """
        This function triggers image build followed by workflow deployment to SnowService.
        """
        if self.options.prebuilt_snowflake_image:
            image = self.options.prebuilt_snowflake_image
            logger.warning(f"Skipped image build. Use prebuilt image: {self.options.prebuilt_snowflake_image}")
        else:
            logger.warning(
                "Building the Docker image and deploying to Snowpark Container Service. "
                "This process may take a few minutes."
            )
            start = time.time()
            image = self._build_and_upload_image()
            end = time.time()
            logger.info(f"Time taken to build and upload image to registry: {end-start:.2f} seconds")

            logger.warning(
                f"Image successfully built! To prevent the need for rebuilding the Docker image in future deployments, "
                f"simply specify 'prebuilt_snowflake_image': '{image}' in the options field of the deploy() function"
            )
        self._deploy_workflow(image)

    def _build_and_upload_image(self) -> str:
        """This function handles image build and upload to image registry.

        Returns:
            Path to the image in the remote image repository.
        """
        image_repo = _get_or_create_image_repo(
            self.session, service_func_name=self.service_func_name, image_repo=self.options.image_repo
        )
        image_builder: base_image_builder.ImageBuilder
        if self.options.enable_remote_image_build:
            image_builder = server_image_builder.ServerImageBuilder(
                id=self.id,
                image_repo=image_repo,
                model_meta=self.model_meta,
                session=self.session,
                artifact_stage_location=self._model_artifact_stage_location,
                compute_pool=self.options.compute_pool,
            )
        else:
            image_builder = client_image_builder.ClientImageBuilder(
                id=self.id, image_repo=image_repo, model_meta=self.model_meta, session=self.session
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
                spec_file_path, "w+", encoding="utf-8"
            ) as spec_file:
                assert self.model_zip_stage_path.startswith("@")
                norm_stage_path = posixpath.normpath(identifier.remove_prefix(self.model_zip_stage_path, "@"))
                (db, schema, stage, path) = identifier.parse_schema_level_object_identifier(norm_stage_path)
                content = string.Template(template.read()).substitute(
                    {
                        "image": image,
                        "predict_endpoint_name": constants.PREDICT,
                        "model_stage": identifier.get_schema_level_object_identifier(db, schema, stage),
                        "model_zip_stage_path": norm_stage_path,
                        "inference_server_container_name": constants.INFERENCE_SERVER_CONTAINER,
                        "target_method": self.target_method,
                        "num_workers": self.options.num_workers,
                        "use_gpu": self.options.use_gpu,
                    }
                )
                content_dict = yaml.safe_load(content)
                if self.options.use_gpu:
                    container = content_dict["spec"]["container"][0]
                    # TODO[shchen]: SNOW-871538, external dependency that only single GPU is supported on SnowService.
                    # GPU limit has to be specified in order to trigger the workload to be run on GPU in SnowService.
                    container["resources"] = {
                        "limits": {"nvidia.com/gpu": self.options.num_gpus},
                        "requests": {"nvidia.com/gpu": self.options.num_gpus},
                    }

                    # Make LLM use case sequential
                    if any(
                        model_blob_meta.model_type == "huggingface_pipeline"
                        for model_blob_meta in self.model_meta.models.values()
                    ):
                        container["env"]["_CONCURRENT_REQUESTS_MAX"] = 1

                yaml.dump(content_dict, spec_file)
                spec_file.seek(0)
                logger.debug(f"Create service spec: \n {spec_file.read()}")

            self.session.file.put(
                local_file_name=spec_file_path,
                stage_location=self._model_artifact_stage_location,
                auto_compress=False,
                overwrite=True,
            )
            logger.debug(
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

        # To avoid too large batch in HF LLM case
        max_batch_rows = None
        if self.options.use_gpu:
            for model_blob_meta in self.model_meta.models.values():
                if model_blob_meta.model_type == "huggingface_pipeline":
                    batch_size = int(model_blob_meta.options.get("batch_size", 1))
                if max_batch_rows is None:
                    max_batch_rows = batch_size
                else:
                    max_batch_rows = min(batch_size, max_batch_rows)

        client.create_or_replace_service_function(
            service_func_name=self.service_func_name,
            service_name=self._service_name,
            endpoint_name=constants.PREDICT,
            max_batch_rows=max_batch_rows,
        )
