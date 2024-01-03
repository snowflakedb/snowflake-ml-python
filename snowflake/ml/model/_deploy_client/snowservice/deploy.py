import copy
import logging
import os
import posixpath
import string
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, cast

import importlib_resources
import yaml
from packaging import requirements
from typing_extensions import Unpack

from snowflake.ml._internal import env_utils, file_utils
from snowflake.ml._internal.container_services.image_registry import (
    registry_client as image_registry_client,
)
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import (
    identifier,
    query_result_checker,
    spcs_attribution_utils,
)
from snowflake.ml.model import type_hints
from snowflake.ml.model._deploy_client import snowservice
from snowflake.ml.model._deploy_client.image_builds import (
    base_image_builder,
    client_image_builder,
    docker_context,
    server_image_builder,
)
from snowflake.ml.model._deploy_client.snowservice import deploy_options, instance_types
from snowflake.ml.model._deploy_client.utils import constants, snowservice_client
from snowflake.ml.model._packager.model_meta import model_meta, model_meta_schema
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


@contextmanager
def _debug_aware_tmp_directory(debug_dir: Optional[str] = None) -> Generator[str, None, None]:
    """Debug-aware directory provider.

    Args:
        debug_dir: A folder for deploymement context.

    Yields:
        A directory path to write deployment artifacts
    """
    create_temp = False
    if debug_dir:
        directory_path = debug_dir
    else:
        temp_dir_context = tempfile.TemporaryDirectory()
        directory_path = temp_dir_context.name
        create_temp = True
    try:
        yield directory_path
    finally:
        if create_temp:
            temp_dir_context.cleanup()


def _deploy(
    session: Session,
    *,
    model_id: str,
    model_meta: model_meta.ModelMetadata,
    service_func_name: str,
    model_zip_stage_path: str,
    deployment_stage_path: str,
    target_method: str,
    **kwargs: Unpack[type_hints.SnowparkContainerServiceDeployOptions],
) -> type_hints.SnowparkContainerServiceDeployDetails:
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

    Returns:
        Deployment details for SPCS.

    Raises:
        SnowflakeMLException: Raised when model_id is empty.
        SnowflakeMLException: Raised when service_func_name is empty.
        SnowflakeMLException: Raised when model_stage_file_path is empty.
    """
    snowpark_logger = logging.getLogger("snowflake.snowpark")
    snowflake_connector_logger = logging.getLogger("snowflake.connector")
    snowpark_log_level = snowpark_logger.level
    snowflake_connector_log_level = snowflake_connector_logger.level

    query_result = (
        query_result_checker.SqlResultValidator(
            session,
            query="SHOW PARAMETERS LIKE 'PYTHON_CONNECTOR_QUERY_RESULT_FORMAT' IN SESSION",
        )
        .has_dimensions(expected_rows=1)
        .validate()
    )
    prev_format = query_result[0].value

    try:
        # Setting appropriate log level to prevent console from being polluted by vast amount of snowpark and snowflake
        # connector logging.
        snowpark_logger.setLevel(logging.WARNING)
        snowflake_connector_logger.setLevel(logging.WARNING)

        # Query format change is needed to ensure session token obtained from the session object is valid.
        session.sql("ALTER SESSION SET PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = 'json'").collect()
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
        # Set conda-forge as backup channel for SPCS deployment
        if "conda-forge" not in model_meta_deploy.env._conda_dependencies:
            model_meta_deploy.env._conda_dependencies["conda-forge"] = []
        # Snowflake connector needs pyarrow to work correctly.
        env_utils.append_conda_dependency(
            model_meta_deploy.env._conda_dependencies,
            (env_utils.DEFAULT_CHANNEL_NAME, requirements.Requirement("pyarrow")),
        )
        if options.use_gpu:
            # Make mypy happy
            assert options.num_gpus is not None
            if model_meta_deploy.env.cuda_version is None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        "You are requesting GPUs for models that do not use a GPU or does not have CUDA version set."
                    ),
                )
            if model_meta.env.cuda_version:
                model_meta_deploy.env.generate_env_for_cuda()
        else:
            # If user does not need GPU, we set this copies cuda_version to None, thus when Image builder gets a
            # not-None cuda_version, it gets to know that GPU is required.
            model_meta_deploy.env._cuda_version = None

        _validate_compute_pool(session, options=options)

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
        return ss_deployment.deploy()
    finally:
        session.sql(f"ALTER SESSION SET PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = '{prev_format}'").collect()
        # Preserve the original logging level.
        snowpark_logger.setLevel(snowpark_log_level)
        snowflake_connector_logger.setLevel(snowflake_connector_log_level)


def _validate_compute_pool(session: Session, *, options: deploy_options.SnowServiceDeployOptions) -> None:
    # Remove full qualified name to avoid double quotes, which does not work well in desc compute pool syntax.
    compute_pool = options.compute_pool.replace('"', "")
    sql = f"DESC COMPUTE POOL {compute_pool}"
    result = (
        query_result_checker.SqlResultValidator(
            session=session,
            query=sql,
        )
        .has_column("instance_family")
        .has_column("state")
        .has_column("auto_resume")
        .has_dimensions(expected_rows=1)
        .validate()
    )

    state = result[0]["state"]
    auto_resume = bool(result[0]["auto_resume"])

    if state == "SUSPENDED":
        if not auto_resume:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_SNOWPARK_COMPUTE_POOL,
                original_exception=RuntimeError(
                    "The compute pool you are requesting to use is suspended without auto-resume enabled"
                ),
            )

    elif state not in ["ACTIVE", "IDLE"]:
        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_SNOWPARK_COMPUTE_POOL,
            original_exception=RuntimeError(
                "The compute pool you are requesting to use is not in the ACTIVE/IDLE status."
            ),
        )

    if options.use_gpu:
        assert options.num_gpus is not None
        request_gpus = options.num_gpus
        instance_family = result[0]["instance_family"]
        if instance_family in instance_types.INSTANCE_TYPE_TO_GPU_COUNT:
            gpu_capacity = instance_types.INSTANCE_TYPE_TO_GPU_COUNT[instance_family]
            if request_gpus > gpu_capacity:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_SNOWPARK_COMPUTE_POOL,
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


class SnowServiceDeployment:
    """
    Class implementation that encapsulates image build and workflow deployment to SnowService
    """

    def __init__(
        self,
        session: Session,
        model_id: str,
        model_meta: model_meta.ModelMetadata,
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
        self.debug_dir: Optional[str] = None
        if self.options.debug_mode:
            self.debug_dir = tempfile.mkdtemp()
            logger.warning(f"Debug model is enabled, deployment artifacts will be available in {self.debug_dir}")

    def deploy(self) -> type_hints.SnowparkContainerServiceDeployDetails:
        """
        This function triggers image build followed by workflow deployment to SnowService.

        Returns:
            Deployment details.
        """
        if self.options.prebuilt_snowflake_image:
            logger.warning(f"Skipped image build. Use prebuilt image: {self.options.prebuilt_snowflake_image}")
            service_function_sql = self._deploy_workflow(self.options.prebuilt_snowflake_image)
        else:
            with _debug_aware_tmp_directory(debug_dir=self.debug_dir) as context_dir:
                extra_kwargs = {}
                if self.options.model_in_image:
                    extra_kwargs = {
                        "session": self.session,
                        "model_zip_stage_path": self.model_zip_stage_path,
                    }
                dc = docker_context.DockerContext(
                    context_dir=context_dir,
                    model_meta=self.model_meta,
                    **extra_kwargs,  # type: ignore[arg-type]
                )
                dc.build()
                image_repo = _get_or_create_image_repo(
                    self.session, service_func_name=self.service_func_name, image_repo=self.options.image_repo
                )
                full_image_name = self._get_full_image_name(image_repo=image_repo, context_dir=context_dir)
                registry_client = image_registry_client.ImageRegistryClient(self.session, full_image_name)

                if not self.options.force_image_build and registry_client.image_exists(full_image_name=full_image_name):
                    logger.warning(
                        f"Similar environment detected. Using existing image {full_image_name} to skip image "
                        f"build. To disable this feature, set 'force_image_build=True' in deployment options"
                    )
                else:
                    logger.warning(
                        "Building the Docker image and deploying to Snowpark Container Service. "
                        "This process may take anywhere from a few minutes to a longer period for GPU-based models."
                    )
                    start = time.time()
                    self._build_and_upload_image(
                        context_dir=context_dir, image_repo=image_repo, full_image_name=full_image_name
                    )
                    end = time.time()
                    logger.info(f"Time taken to build and upload image to registry: {end - start:.2f} seconds")
                    logger.warning(
                        f"Image successfully built! For future model deployments, the image will be reused if "
                        f"possible, saving model deployment time. To enforce using the same image, include "
                        f"'prebuilt_snowflake_image': '{full_image_name}' in the deploy() function's options."
                    )

                # Adding the model name as an additional tag to the existing image, excluding the version to prevent
                # excessive tags and also due to version not available in current model metadata. This will allow
                # users to associate images with specific models and perform relevant image registry actions. In the
                # event that model dependencies change across versions, a new image hash will be computed, resulting in
                # a new image.
                try:
                    registry_client.add_tag_to_remote_image(
                        original_full_image_name=full_image_name, new_tag=self.model_meta.name
                    )
                except Exception as e:
                    # Proceed to the deployment with a warning message.
                    logger.warning(f"Failed to add tag {self.model_meta.name} to image {full_image_name}: {str(e)}")
                service_function_sql = self._deploy_workflow(full_image_name)

        rows = self.session.sql(f"DESCRIBE SERVICE {self._service_name}").collect()
        service_info = rows[0].as_dict() if rows and rows[0] else None
        return type_hints.SnowparkContainerServiceDeployDetails(
            service_info=service_info,
            service_function_sql=service_function_sql,
        )

    def _get_full_image_name(self, image_repo: str, context_dir: str) -> str:
        """Return a valid full image name that consists of image name and tag. e.g
        org-account.registry.snowflakecomputing.com/db/schema/repo/image:latest

        Args:
            image_repo: image repo path, e.g. org-account.registry.snowflakecomputing.com/db/schema/repo
            context_dir: the local docker context directory, which consists everything needed to build the docker image.

        Returns:
            Full image name.
        """
        image_repo = _get_or_create_image_repo(
            self.session, service_func_name=self.service_func_name, image_repo=self.options.image_repo
        )

        # We skip "MODEL_METADATA_FILE" as it contains information that will always lead to cache misses.  This isn't an
        # issue because model dependency is also captured in the model env/ folder, which will be hashed. The aim is to
        # reuse the same Docker image even if the user logs a similar model without new dependencies.
        docker_context_dir_hash = file_utils.hash_directory(
            context_dir, ignore_hidden=True, excluded_files=[model_meta.MODEL_METADATA_FILE]
        )
        # By default, we associate a 'latest' tag with each of our created images for easy existence checking.
        # Additional tags are added for readability.
        return f"{image_repo}/{docker_context_dir_hash}:{constants.LATEST_IMAGE_TAG}"

    def _build_and_upload_image(self, context_dir: str, image_repo: str, full_image_name: str) -> None:
        """Handles image build and upload to image registry.

        Args:
            context_dir: the local docker context directory, which consists everything needed to build the docker image.
            image_repo: image repo path, e.g. org-account.registry.snowflakecomputing.com/db/schema/repo
            full_image_name: Full image name consists of image name and image tag.
        """
        image_builder: base_image_builder.ImageBuilder
        if self.options.enable_remote_image_build:
            image_builder = server_image_builder.ServerImageBuilder(
                context_dir=context_dir,
                full_image_name=full_image_name,
                image_repo=image_repo,
                session=self.session,
                artifact_stage_location=self._model_artifact_stage_location,
                compute_pool=self.options.compute_pool,
            )
        else:
            image_builder = client_image_builder.ClientImageBuilder(
                context_dir=context_dir, full_image_name=full_image_name, image_repo=image_repo, session=self.session
            )
        image_builder.build_and_upload_image()

    def _prepare_and_upload_artifacts_to_stage(self, image: str) -> None:
        """Constructs and upload service spec to stage.

        Args:
            image: Name of the image to create SnowService container from.
        """
        if self.options.model_in_image:
            spec_template = (
                importlib_resources.files(snowservice)
                .joinpath("templates/service_spec_template_with_model")  # type: ignore[no-untyped-call]
                .read_text("utf-8")
            )
        else:
            spec_template = (
                importlib_resources.files(snowservice)
                .joinpath("templates/service_spec_template")  # type: ignore[no-untyped-call]
                .read_text("utf-8")
            )

        with _debug_aware_tmp_directory(self.debug_dir) as dir_path:
            spec_file_path = os.path.join(dir_path, f"{constants.SERVICE_SPEC}.yaml")

            with open(spec_file_path, "w+", encoding="utf-8") as spec_file:
                assert self.model_zip_stage_path.startswith("@")
                norm_stage_path = posixpath.normpath(identifier.remove_prefix(self.model_zip_stage_path, "@"))
                # Ensure model stage path has root prefix as stage mount will it mount it to root.
                absolute_model_stage_path = os.path.join("/", norm_stage_path)
                (db, schema, stage, path) = identifier.parse_schema_level_object_identifier(norm_stage_path)
                substitutes = {
                    "image": image,
                    "predict_endpoint_name": constants.PREDICT,
                    "model_stage": identifier.get_schema_level_object_identifier(db, schema, stage),
                    "model_zip_stage_path": absolute_model_stage_path,
                    "inference_server_container_name": constants.INFERENCE_SERVER_CONTAINER,
                    "target_method": self.target_method,
                    "num_workers": self.options.num_workers,
                    "use_gpu": self.options.use_gpu,
                    "enable_ingress": self.options.enable_ingress,
                }
                if self.options.model_in_image:
                    del substitutes["model_stage"]
                    del substitutes["model_zip_stage_path"]
                content = string.Template(spec_template).substitute(substitutes)
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
                        model_blob_meta.model_type == "huggingface_pipeline" or model_blob_meta.model_type == "llm"
                        for model_blob_meta in self.model_meta.models.values()
                    ):
                        container["env"]["_CONCURRENT_REQUESTS_MAX"] = 1

                yaml.dump(content_dict, spec_file)
                logger.debug("Create service spec: \n, %s", content_dict)

            self.session.file.put(
                local_file_name=spec_file_path,
                stage_location=self._model_artifact_stage_location,
                auto_compress=False,
                overwrite=True,
            )
            logger.debug(
                f"Uploaded spec file {os.path.basename(spec_file_path)} " f"to {self._model_artifact_stage_location}"
            )

    def _get_max_batch_rows(self) -> Optional[int]:
        # To avoid too large batch in HF LLM case
        max_batch_rows = None
        if self.options.use_gpu:
            for model_blob_meta in self.model_meta.models.values():
                batch_size = None
                if model_blob_meta.model_type == "huggingface_pipeline":
                    model_blob_options_hf = cast(
                        model_meta_schema.HuggingFacePipelineModelBlobOptions, model_blob_meta.options
                    )
                    batch_size = model_blob_options_hf["batch_size"]
                if model_blob_meta.model_type == "llm":
                    model_blob_options_llm = cast(model_meta_schema.LLMModelBlobOptions, model_blob_meta.options)
                    batch_size = model_blob_options_llm["batch_size"]
                if batch_size:
                    if max_batch_rows is None:
                        max_batch_rows = batch_size
                    else:
                        max_batch_rows = min(batch_size, max_batch_rows)
        return max_batch_rows

    def _deploy_workflow(self, image: str) -> str:
        """This function handles workflow deployment to SnowService with the given image.

        Args:
            image: Name of the image to create SnowService container from.

        Returns:
            service function sql
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
        logger.info(f"Wait for service {self._service_name} to become ready...")
        client.block_until_resource_is_ready(
            resource_name=self._service_name, resource_type=constants.ResourceType.SERVICE
        )
        logger.info(f"Service {self._service_name} is ready. Creating service function...")

        spcs_attribution_utils.record_service_start(self.session, self._service_name)

        service_function_sql = client.create_or_replace_service_function(
            service_func_name=self.service_func_name,
            service_name=self._service_name,
            endpoint_name=constants.PREDICT,
            max_batch_rows=self._get_max_batch_rows(),
        )
        logger.info(f"Service function {self.service_func_name} is created. Deployment completed successfully!")
        return service_function_sql
