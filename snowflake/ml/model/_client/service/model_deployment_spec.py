import json
import pathlib
import warnings
from typing import Any, Optional, Union

import yaml

from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.ml.model import inference_engine as inference_engine_module
from snowflake.ml.model._client.service import model_deployment_spec_schema


class ModelDeploymentSpec:
    """Class to construct deploy.yml file for Model container services deployment.

    Attributes:
        workspace_path: A local path where model related files should be dumped to.
    """

    DEPLOY_SPEC_FILE_REL_PATH = "deploy.yml"

    def __init__(self, workspace_path: Optional[pathlib.Path] = None) -> None:
        self.workspace_path = workspace_path
        self._models: list[model_deployment_spec_schema.Model] = []
        self._image_build: Optional[model_deployment_spec_schema.ImageBuild] = None
        self._service: Optional[model_deployment_spec_schema.Service] = None
        self._job: Optional[model_deployment_spec_schema.Job] = None
        self._model_loggings: Optional[list[model_deployment_spec_schema.ModelLogging]] = None
        # this is referring to custom inference engine spec (vllm, sglang, etc)
        self._inference_engine_spec: Optional[model_deployment_spec_schema.InferenceEngineSpec] = None
        self._inference_spec: dict[str, Any] = {}  # Common inference spec for service/job

        self.database: Optional[sql_identifier.SqlIdentifier] = None
        self.schema: Optional[sql_identifier.SqlIdentifier] = None

    def clear(self) -> None:
        """Reset the deployment spec to its initial state."""
        self._models = []
        self._image_build = None
        self._service = None
        self._job = None
        self._model_loggings = None
        self._inference_spec = {}
        self.database = None
        self.schema = None

    def add_model_spec(
        self,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
    ) -> "ModelDeploymentSpec":
        """Add model specification to the deployment spec.

        Args:
            database_name: Database name containing the model.
            schema_name: Schema name containing the model.
            model_name: Name of the model.
            version_name: Version of the model.

        Returns:
            Self for chaining.
        """
        fq_model_name = identifier.get_schema_level_object_identifier(
            database_name.identifier(), schema_name.identifier(), model_name.identifier()
        )
        if not self.database:
            self.database = database_name
        if not self.schema:
            self.schema = schema_name
        model = model_deployment_spec_schema.Model(name=fq_model_name, version=version_name.identifier())
        self._models.append(model)
        return self

    def add_image_build_spec(
        self,
        image_build_compute_pool_name: Optional[sql_identifier.SqlIdentifier] = None,
        fully_qualified_image_repo_name: Optional[str] = None,
        force_rebuild: bool = False,
        external_access_integrations: Optional[list[sql_identifier.SqlIdentifier]] = None,
    ) -> "ModelDeploymentSpec":
        """Add image build specification to the deployment spec.

        Args:
            image_build_compute_pool_name: Compute pool for image building.
            fully_qualified_image_repo_name: Fully qualified name of the image repository.
            force_rebuild: Whether to force rebuilding the image.
            external_access_integrations: List of external access integrations.

        Returns:
            Self for chaining.
        """
        if (
            image_build_compute_pool_name is not None
            or fully_qualified_image_repo_name is not None
            or force_rebuild is True
            or external_access_integrations is not None
        ):
            self._image_build = model_deployment_spec_schema.ImageBuild(
                compute_pool=(
                    None if image_build_compute_pool_name is None else image_build_compute_pool_name.identifier()
                ),
                image_repo=fully_qualified_image_repo_name,
                force_rebuild=force_rebuild,
                external_access_integrations=(
                    [eai.identifier() for eai in external_access_integrations] if external_access_integrations else None
                ),
            )
        return self

    def _add_inference_spec(
        self,
        cpu: Optional[str],
        memory: Optional[str],
        gpu: Optional[Union[str, int]],
        num_workers: Optional[int],
        max_batch_rows: Optional[int],
    ) -> None:
        """Internal helper to store common inference specs."""
        if cpu:
            self._inference_spec["cpu"] = cpu
        if memory:
            self._inference_spec["memory"] = memory
        if gpu:
            if isinstance(gpu, int):
                gpu_str = str(gpu)
            else:
                gpu_str = gpu
            self._inference_spec["gpu"] = gpu_str
        if num_workers:
            self._inference_spec["num_workers"] = num_workers
        if max_batch_rows:
            self._inference_spec["max_batch_rows"] = max_batch_rows

    def add_service_spec(
        self,
        service_name: sql_identifier.SqlIdentifier,
        inference_compute_pool_name: sql_identifier.SqlIdentifier,
        service_database_name: Optional[sql_identifier.SqlIdentifier] = None,
        service_schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        ingress_enabled: bool = True,
        max_instances: int = 1,
        cpu: Optional[str] = None,
        memory: Optional[str] = None,
        gpu: Optional[Union[str, int]] = None,
        num_workers: Optional[int] = None,
        max_batch_rows: Optional[int] = None,
    ) -> "ModelDeploymentSpec":
        """Add service specification to the deployment spec.

        Args:
            service_name: Name of the service.
            inference_compute_pool_name: Compute pool for inference.
            service_database_name: Database name for the service.
            service_schema_name: Schema name for the service.
            ingress_enabled: Whether ingress is enabled.
            max_instances: Maximum number of service instances.
            cpu: CPU requirement.
            memory: Memory requirement.
            gpu: GPU requirement.
            num_workers: Number of workers.
            max_batch_rows: Maximum batch rows for inference.

        Raises:
            ValueError: If a job spec already exists.

        Returns:
            Self for chaining.
        """
        if self._job:
            raise ValueError("Cannot add a service spec when a job spec already exists.")

        saved_service_database = service_database_name or self.database
        saved_service_schema = service_schema_name or self.schema
        assert saved_service_database is not None
        assert saved_service_schema is not None
        fq_service_name = identifier.get_schema_level_object_identifier(
            saved_service_database.identifier(), saved_service_schema.identifier(), service_name.identifier()
        )

        self._add_inference_spec(cpu, memory, gpu, num_workers, max_batch_rows)

        self._service = model_deployment_spec_schema.Service(
            name=fq_service_name,
            compute_pool=inference_compute_pool_name.identifier(),
            ingress_enabled=ingress_enabled,
            max_instances=max_instances,
            **self._inference_spec,
        )
        return self

    def add_job_spec(
        self,
        job_name: sql_identifier.SqlIdentifier,
        inference_compute_pool_name: sql_identifier.SqlIdentifier,
        function_name: str,
        input_stage_location: str,
        output_stage_location: str,
        completion_filename: str,
        input_file_pattern: str,
        warehouse: sql_identifier.SqlIdentifier,
        job_database_name: Optional[sql_identifier.SqlIdentifier] = None,
        job_schema_name: Optional[sql_identifier.SqlIdentifier] = None,
        cpu: Optional[str] = None,
        memory: Optional[str] = None,
        gpu: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_batch_rows: Optional[int] = None,
        replicas: Optional[int] = None,
    ) -> "ModelDeploymentSpec":
        """Add job specification to the deployment spec.

        Args:
            job_name: Name of the job.
            inference_compute_pool_name: Compute pool for inference.
            warehouse: Warehouse for the job.
            function_name: Function name.
            input_stage_location: Stage location for input data.
            output_stage_location: Stage location for output data.
            job_database_name: Database name for the job.
            job_schema_name: Schema name for the job.
            input_file_pattern: Pattern for input files (optional).
            completion_filename: Name of completion file (default: "completion.txt").
            cpu: CPU requirement.
            memory: Memory requirement.
            gpu: GPU requirement.
            num_workers: Number of workers.
            max_batch_rows: Maximum batch rows for inference.
            replicas: Number of replicas.

        Raises:
            ValueError: If a service spec already exists.

        Returns:
            Self for chaining.
        """
        if self._service:
            raise ValueError("Cannot add a job spec when a service spec already exists.")

        saved_job_database = job_database_name or self.database
        saved_job_schema = job_schema_name or self.schema

        assert saved_job_database is not None
        assert saved_job_schema is not None

        fq_job_name = identifier.get_schema_level_object_identifier(
            saved_job_database.identifier(), saved_job_schema.identifier(), job_name.identifier()
        )

        self._add_inference_spec(cpu, memory, gpu, num_workers, max_batch_rows)

        self._job = model_deployment_spec_schema.Job(
            name=fq_job_name,
            compute_pool=inference_compute_pool_name.identifier(),
            warehouse=warehouse.identifier() if warehouse else None,
            function_name=function_name,
            input=model_deployment_spec_schema.Input(
                input_stage_location=input_stage_location, input_file_pattern=input_file_pattern
            ),
            output=model_deployment_spec_schema.Output(
                output_stage_location=output_stage_location,
                completion_filename=completion_filename,
            ),
            replicas=replicas,
            **self._inference_spec,
        )
        return self

    def add_hf_logger_spec(
        self,
        hf_model_name: str,
        hf_task: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_tokenizer: Optional[str] = None,
        hf_revision: Optional[str] = None,
        hf_trust_remote_code: Optional[bool] = False,
        pip_requirements: Optional[list[str]] = None,
        conda_dependencies: Optional[list[str]] = None,
        target_platforms: Optional[list[str]] = None,
        comment: Optional[str] = None,
        warehouse: Optional[str] = None,
        **kwargs: Any,
    ) -> "ModelDeploymentSpec":
        """Add Hugging Face logger specification.

        Args:
            hf_model_name: Hugging Face model name.
            hf_task: Hugging Face task.
            hf_token: Hugging Face token.
            hf_tokenizer: Hugging Face tokenizer.
            hf_revision: Hugging Face model revision.
            hf_trust_remote_code: Whether to trust remote code.
            pip_requirements: List of pip requirements.
            conda_dependencies: List of conda dependencies.
            target_platforms: List of target platforms.
            comment: Comment for the model.
            warehouse: Warehouse used to log the model.
            **kwargs: Additional Hugging Face model arguments.

        Raises:
            ValueError: If Hugging Face model name is missing when other HF parameters are provided.

        Returns:
            Self for chaining.
        """
        # Validation moved here from save
        if (
            any(
                [
                    hf_task,
                    hf_token,
                    hf_tokenizer,
                    hf_revision,
                    hf_trust_remote_code,
                    pip_requirements,
                ]
            )
            and not hf_model_name
        ):
            # This condition might be redundant now as hf_model_name is mandatory
            raise ValueError("Hugging Face model name is required when using Hugging Face model deployment.")

        log_model_args = model_deployment_spec_schema.LogModelArgs(
            pip_requirements=pip_requirements,
            conda_dependencies=conda_dependencies,
            target_platforms=target_platforms,
            comment=comment,
            warehouse=warehouse,
        )
        hf_model = model_deployment_spec_schema.HuggingFaceModel(
            hf_model_name=hf_model_name,
            task=hf_task,
            token=hf_token,
            tokenizer=hf_tokenizer,
            trust_remote_code=hf_trust_remote_code,
            revision=hf_revision,
            hf_model_kwargs=json.dumps(kwargs),
        )
        model_logging = model_deployment_spec_schema.ModelLogging(
            log_model_args=log_model_args,
            hf_model=hf_model,
        )
        if self._model_loggings is None:
            self._model_loggings = [model_logging]
        else:
            self._model_loggings.append(model_logging)
        return self

    def add_inference_engine_spec(
        self,
        inference_engine: inference_engine_module.InferenceEngine,
        inference_engine_args: Optional[list[str]] = None,
    ) -> "ModelDeploymentSpec":
        """Add inference engine specification. This must be called after self.add_service_spec().

        Args:
            inference_engine: Inference engine.
            inference_engine_args: Inference engine arguments.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If inference engine specification is called before add_service_spec().
            ValueError: If the argument does not have a '--' prefix.
        """
        # TODO: needs to eventually support job deployment spec
        if self._service is None:
            raise ValueError("Inference engine specification must be called after add_service_spec().")

        if inference_engine_args is None:
            inference_engine_args = []

        # Validate inference engine
        if inference_engine == inference_engine_module.InferenceEngine.VLLM:
            # Block list for VLLM args that should not be user-configurable
            # make this a set for faster lookup
            block_list = {
                "--host",
                "--port",
                "--allowed-headers",
                "--api-key",
                "--lora-modules",
                "--prompt-adapter",
                "--ssl-keyfile",
                "--ssl-certfile",
                "--ssl-ca-certs",
                "--enable-ssl-refresh",
                "--ssl-cert-reqs",
                "--root-path",
                "--middleware",
                "--disable-frontend-multiprocessing",
                "--enable-request-id-headers",
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "--tool-parser-plugin",
                "--log-config-file",
            }

            filtered_args = []
            for arg in inference_engine_args:
                # Check if the argument has a '--' prefix
                if not arg.startswith("--"):
                    raise ValueError(
                        f"""The argument {arg} is not allowed for configuration in Snowflake ML's
                        {inference_engine.value} inference engine. Maybe you forgot to add '--' prefix?""",
                    )

                # Filter out blocked args and warn user
                if arg.split("=")[0] in block_list:
                    warnings.warn(
                        f"""The argument {arg} is not allowed for configuration in Snowflake ML's
                        {inference_engine.value} inference engine. It will be ignored.""",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    filtered_args.append(arg)

            inference_engine_args = filtered_args

        self._service.inference_engine_spec = model_deployment_spec_schema.InferenceEngineSpec(
            # convert to string to be saved in the deployment spec
            inference_engine_name=inference_engine.value,
            inference_engine_args=inference_engine_args,
        )
        return self

    def save(self) -> str:
        """Constructs the final deployment spec from added components and saves it.

        Raises:
            ValueError: If required components are missing or conflicting specs are added.
            RuntimeError: If no service or job spec is found despite validation.

        Returns:
            The path to the saved YAML file as a string, or the YAML content as a string
            if workspace_path was not provided.
        """
        # Validations
        if not self._models:
            raise ValueError("Model specification is required. Call add_model_spec().")
        if not self._service and not self._job:
            raise ValueError(
                "Either service or job specification is required. Call add_service_spec() or add_job_spec()."
            )
        if self._service and self._job:
            # This case should be prevented by checks in add_service_spec/add_job_spec, but double-check
            raise ValueError("Cannot have both service and job specifications.")

        # Construct the final spec object
        if self._service:
            model_deployment_spec: Union[
                model_deployment_spec_schema.ModelServiceDeploymentSpec,
                model_deployment_spec_schema.ModelJobDeploymentSpec,
            ] = model_deployment_spec_schema.ModelServiceDeploymentSpec(
                models=self._models,
                image_build=self._image_build,
                service=self._service,
                model_loggings=self._model_loggings,
            )
        elif self._job:
            model_deployment_spec = model_deployment_spec_schema.ModelJobDeploymentSpec(
                models=self._models,
                image_build=self._image_build,
                job=self._job,
                model_loggings=self._model_loggings,
            )
        else:
            # Should not happen due to earlier validation
            raise RuntimeError("Internal error: No service or job spec found despite validation.")

        # Serialize and save/return
        yaml_content = model_deployment_spec.model_dump(exclude_none=True)

        if self.workspace_path is None:
            return yaml.safe_dump(yaml_content)

        file_path = self.workspace_path / self.DEPLOY_SPEC_FILE_REL_PATH
        with file_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)
        return str(file_path.resolve())
