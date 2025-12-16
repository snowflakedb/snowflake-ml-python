import logging
from typing import Any, Optional, Union

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.human_readable_id import hrid_generator
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.model import inference_engine_utils
from snowflake.ml.model._client.ops import service_ops
from snowflake.ml.model.models import huggingface
from snowflake.snowpark import async_job, session

logger = logging.getLogger(__name__)


class HuggingFacePipelineModel(huggingface.TransformersPipeline):
    def __init__(
        self,
        task: Optional[str] = None,
        model: Optional[str] = None,
        *,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        trust_remote_code: Optional[bool] = None,
        model_kwargs: Optional[dict[str, Any]] = None,
        download_snapshot: bool = True,
        # repo snapshot download args
        allow_patterns: Optional[Union[list[str], str]] = None,
        ignore_patterns: Optional[Union[list[str], str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Utility factory method to build a wrapper over transformers [`Pipeline`].
        When deploying, this wrapper will create a real pipeline object and loading tokenizers and models.

        For pipelines docs, please refer:
        https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline

        Args:
            task: The task that pipeline will be used. If None it would be inferred from model.
                For available tasks, please refer Transformers's documentation. Defaults to None.
            model: The model that will be used by the pipeline to make predictions. This can only be a model identifier
                currently. If not provided, the default for the `task` will be loaded. Defaults to None.
            revision: When passing a task name or a string model identifier: The specific model version to use. It can
                be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and
                other artifacts on huggingface.co, so `revision` can be any identifier allowed by git. Defaults to None.
            token: The token to use as HTTP bearer authorization for remote files. Defaults to None.
            trust_remote_code: Whether or not to allow for custom code defined on the Hub in their own modeling,
                configuration, tokenization or even pipeline files. This option should only be set to `True` for
                repositories you trust and in which you have read the code, as it will execute code present on the Hub.
                Defaults to None.
            model_kwargs: Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,`.
                Defaults to None.
            download_snapshot: Whether to download the HuggingFace repository. Defaults to True.
            allow_patterns: If provided, only files matching at least one pattern are downloaded.
            ignore_patterns: If provided, files matching any of the patterns are not downloaded.
            kwargs: Additional keyword arguments passed along to the specific pipeline init (see the documentation for
                the corresponding pipeline class for possible values).

        Return:
            A wrapper over transformers [`Pipeline`].
        """
        logger.warning("HuggingFacePipelineModel is deprecated. Please use TransformersPipeline instead.")
        super().__init__(
            task=task,
            model=model,
            revision=revision,
            token_or_secret=token,
            trust_remote_code=trust_remote_code,
            model_kwargs=model_kwargs,
            compute_pool_for_log=None,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            **kwargs,
        )
        self.token = token

    @telemetry.send_api_usage_telemetry(
        project=huggingface._TELEMETRY_PROJECT,
        subproject=huggingface._TELEMETRY_SUBPROJECT,
        func_params_to_log=[
            "service_name",
            "image_build_compute_pool",
            "service_compute_pool",
            "image_repo",
            "gpu_requests",
            "num_workers",
            "max_batch_rows",
        ],
    )
    @snowpark._internal.utils.private_preview(version="1.9.1")
    def log_model_and_create_service(
        self,
        *,
        session: session.Session,
        # registry.log_model parameters
        model_name: str,
        version_name: Optional[str] = None,
        pip_requirements: Optional[list[str]] = None,
        conda_dependencies: Optional[list[str]] = None,
        comment: Optional[str] = None,
        # model_version_impl.create_service parameters
        service_name: str,
        service_compute_pool: str,
        image_repo: Optional[str] = None,
        image_build_compute_pool: Optional[str] = None,
        ingress_enabled: bool = False,
        max_instances: int = 1,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        gpu_requests: Optional[Union[str, int]] = None,
        num_workers: Optional[int] = None,
        max_batch_rows: Optional[int] = None,
        force_rebuild: bool = False,
        build_external_access_integrations: Optional[list[str]] = None,
        block: bool = True,
        inference_engine_options: Optional[dict[str, Any]] = None,
        experimental_options: Optional[dict[str, Any]] = None,
    ) -> Union[str, async_job.AsyncJob]:
        """Logs a Hugging Face model and creates a service in Snowflake.

        Args:
            session: The Snowflake session object.
            model_name: The name of the model in Snowflake.
            version_name: The version name of the model. Defaults to None.
            pip_requirements: Pip requirements for the model. Defaults to None.
            conda_dependencies: Conda dependencies for the model. Defaults to None.
            comment: Comment for the model. Defaults to None.
            service_name: The name of the service to create.
            service_compute_pool: The compute pool for the service.
            image_repo: The name of the image repository. This can be None, in that case a default hidden image
                repository will be used.
            image_build_compute_pool: The name of the compute pool used to build the model inference image. It uses
            the service compute pool if None.
            ingress_enabled: Whether ingress is enabled. Defaults to False.
            max_instances: Maximum number of instances. Defaults to 1.
            cpu_requests: CPU requests configuration. Defaults to None.
            memory_requests: Memory requests configuration. Defaults to None.
            gpu_requests: GPU requests configuration. Defaults to None.
            num_workers: Number of workers. Defaults to None.
            max_batch_rows: Maximum batch rows. Defaults to None.
            force_rebuild: Whether to force rebuild the image. Defaults to False.
            build_external_access_integrations: External access integrations for building the image. Defaults to None.
            block: Whether to block the operation. Defaults to True.
            inference_engine_options: Options for the service creation with custom inference engine. Defaults to None.
            experimental_options: Experimental options for the service creation. Defaults to None.

        Raises:
            ValueError: if database and schema name is not provided and session doesn't have a
            database and schema name.
            exceptions.SnowparkSQLException: if service already exists.

        Returns:
            The service ID or an async job object.

        .. # noqa: DAR003
        """
        statement_params = telemetry.get_statement_params(
            project=huggingface._TELEMETRY_PROJECT,
            subproject=huggingface._TELEMETRY_SUBPROJECT,
        )

        database_name_id, schema_name_id, model_name_id = sql_identifier.parse_fully_qualified_name(model_name)
        session_database_name = session.get_current_database()
        session_schema_name = session.get_current_schema()
        if database_name_id is None:
            if session_database_name is None:
                raise ValueError("Either database needs to be provided or needs to be available in session.")
            database_name_id = sql_identifier.SqlIdentifier(session_database_name)
        if schema_name_id is None:
            if session_schema_name is None:
                raise ValueError("Either schema needs to be provided or needs to be available in session.")
            schema_name_id = sql_identifier.SqlIdentifier(session_schema_name)

        if version_name is None:
            name_generator = hrid_generator.HRID16()
            version_name = name_generator.generate()[1]

        service_db_id, service_schema_id, service_id = sql_identifier.parse_fully_qualified_name(service_name)

        service_operator = service_ops.ServiceOperator(
            session=session,
            database_name=database_name_id,
            schema_name=schema_name_id,
        )
        logger.info(f"A service job is going to register the hf model as: {model_name}.{version_name}")

        # Check if model is HuggingFace text-generation before doing inference engine checks
        inference_engine_args = None
        if inference_engine_options:
            if self.task != "text-generation":
                raise ValueError(
                    "Currently, InferenceEngine using inference_engine_options is only supported for "
                    "HuggingFace text-generation models."
                )

            inference_engine_args = inference_engine_utils._get_inference_engine_args(inference_engine_options)

            # Enrich inference engine args if inference engine is specified
            if inference_engine_args is not None:
                inference_engine_args = inference_engine_utils._enrich_inference_engine_args(
                    inference_engine_args,
                    gpu_requests,
                )

        from snowflake.ml.model import event_handler
        from snowflake.snowpark import exceptions

        hf_event_handler = event_handler.ModelEventHandler()
        with hf_event_handler.status("Creating HuggingFace model service", total=6, block=block) as status:
            try:
                result = service_operator.create_service(
                    database_name=database_name_id,
                    schema_name=schema_name_id,
                    model_name=model_name_id,
                    version_name=sql_identifier.SqlIdentifier(version_name),
                    service_database_name=service_db_id,
                    service_schema_name=service_schema_id,
                    service_name=service_id,
                    image_build_compute_pool_name=(
                        sql_identifier.SqlIdentifier(image_build_compute_pool)
                        if image_build_compute_pool
                        else sql_identifier.SqlIdentifier(service_compute_pool)
                    ),
                    service_compute_pool_name=sql_identifier.SqlIdentifier(service_compute_pool),
                    image_repo_name=image_repo,
                    ingress_enabled=ingress_enabled,
                    max_instances=max_instances,
                    cpu_requests=cpu_requests,
                    memory_requests=memory_requests,
                    gpu_requests=gpu_requests,
                    num_workers=num_workers,
                    max_batch_rows=max_batch_rows,
                    force_rebuild=force_rebuild,
                    build_external_access_integrations=(
                        None
                        if build_external_access_integrations is None
                        else [sql_identifier.SqlIdentifier(eai) for eai in build_external_access_integrations]
                    ),
                    block=block,
                    progress_status=status,
                    statement_params=statement_params,
                    # hf model
                    hf_model_args=service_ops.HFModelArgs(
                        hf_model_name=self.model,
                        hf_task=self.task,
                        hf_tokenizer=self.tokenizer,
                        hf_revision=self.revision,
                        hf_token=self.token,
                        hf_trust_remote_code=bool(self.trust_remote_code),
                        hf_model_kwargs=self.model_kwargs,
                        pip_requirements=pip_requirements,
                        conda_dependencies=conda_dependencies,
                        comment=comment,
                        # TODO: remove warehouse in the next release
                        warehouse=session.get_current_warehouse(),
                    ),
                    # inference engine
                    inference_engine_args=inference_engine_args,
                )
                status.update(label="HuggingFace model service created successfully", state="complete", expanded=False)
                return result
            except exceptions.SnowparkSQLException as e:
                # Check if the error is because the service already exists
                if "already exists" in str(e).lower() or "100132" in str(
                    e
                ):  # 100132 is Snowflake error code for object already exists
                    # Update progress to show service already exists (preserve exception behavior)
                    status.update("service already exists")
                    status.complete()  # Complete progress to full state
                    status.update(label="Service already exists", state="error", expanded=False)
                    # Re-raise the exception to preserve existing API behavior
                    raise
                else:
                    # Re-raise other SQL exceptions
                    status.update(label="Service creation failed", state="error", expanded=False)
                    raise
