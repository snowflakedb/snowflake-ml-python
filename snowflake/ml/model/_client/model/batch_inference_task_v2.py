from __future__ import annotations

from typing import Any, Optional

from snowflake.ml._internal import telemetry
from snowflake.ml.model._client.model import (
    batch_inference_job_specs,
    model_version_impl,
)
from snowflake.ml.model._client.ops import service_ops

try:
    from snowflake.core.task.dagv1 import DAGTask

    _HAS_SNOWFLAKE_CORE = True
except ModuleNotFoundError:
    DAGTask = object  # type: ignore[assignment, misc]
    _HAS_SNOWFLAKE_CORE = False

_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"


class BatchInferenceTask(DAGTask):
    """A ``DAGTask`` that runs a batch inference job for a registered model version.

    Construct it inside a ``with DAG(...)`` block (or pass ``dag=`` explicitly) and
    chain it with other tasks using ``>>``. Requires the ``snowflake.core`` package
    to be installed.

    The job runs synchronously, so the DAG step completes when the job does. Each run
    gets a server-generated job name, and results are written under
    ``<output_spec.stage_location>/<job_name>/``. The job publishes that location as
    the task return value, so a successor can read it with
    ``SYSTEM$GET_PREDECESSOR_RETURN_VALUE()`` rather than hardcoding the path.

    The DAG must supply a warehouse, via ``DAG(warehouse=...)`` or ``warehouse=`` on the
    task; serverless DAGs are unsupported.

    Prefer fully qualified stage paths: an unqualified path is resolved against the
    session namespace when the task is built, but against the task owner's namespace
    when it runs.
    """

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
        func_params_to_log=[
            "name",
            "compute_pool",
            "input_spec",
            "output_spec",
            "resources_spec",
            "inference_spec",
            "image_build_spec",
            "replicas",
        ],
    )
    def __init__(
        self,
        name: str,
        *,
        model_version: model_version_impl.ModelVersion,
        compute_pool: str,
        output_spec: batch_inference_job_specs.OutputSpec,
        query: Optional[str] = None,
        input_stage_location: Optional[str] = None,
        input_spec: Optional[batch_inference_job_specs.InputSpec] = None,
        resources_spec: Optional[batch_inference_job_specs.ResourcesSpec] = None,
        inference_spec: Optional[batch_inference_job_specs.InferenceSpec] = None,
        image_build_spec: Optional[batch_inference_job_specs.ImageBuildSpec] = None,
        function_name: Optional[str] = None,
        replicas: Optional[int] = None,
        **dagtask_kwargs: Any,
    ) -> None:
        """Build a batch inference task.

        Args:
            name: Task name within the DAG.
            model_version: The model version to run.
            compute_pool: Compute pool used by the job and the image build.
            output_spec: Output block. ``stage_location`` is treated as a base; results are
                written under ``<stage_location>/<job_name>/``.
            query: SQL query producing the input rows. Provide exactly one of ``query`` or
                ``input_stage_location``.
            input_stage_location: Existing stage path holding the input data. Provide exactly
                one of ``query`` or ``input_stage_location``.
            input_spec: Optional input block.
            resources_spec: Optional resources block.
            inference_spec: Optional inference block.
            image_build_spec: Optional image build block.
            function_name: Model function name. Resolved against the model's function list
                when omitted.
            replicas: Optional ``REPLICAS`` value.
            dagtask_kwargs: Passed through to ``DAGTask`` (``dag``, ``condition``,
                ``warehouse``, ``session_parameters``, and so on).

        Raises:
            ImportError: If the ``snowflake.core`` package is not installed.
            TypeError: If ``definition`` is passed; the task builds its own.
        """
        if not _HAS_SNOWFLAKE_CORE:
            raise ImportError(
                "BatchInferenceTask requires the `snowflake.core` package. "
                "Install it with `pip install snowflake.core`."
            )
        if "definition" in dagtask_kwargs:
            raise TypeError("BatchInferenceTask builds its own task definition; do not pass `definition=`.")

        self._model_version = model_version
        self._compute_pool = compute_pool
        self._output_spec = output_spec
        self._query = query
        self._input_stage_location = input_stage_location
        self._input_spec = input_spec
        self._resources_spec = resources_spec
        self._inference_spec = inference_spec
        self._image_build_spec = image_build_spec
        self._function_name = function_name
        self._replicas = replicas

        super().__init__(name, definition=self._to_sql(), **dagtask_kwargs)

    def _to_sql(self) -> str:
        """Build the ``EXECUTE INFERENCE JOB SERVICE`` command that becomes the task definition.

        Returns:
            The command text.

        Raises:
            ValueError: If not exactly one of ``query`` / ``input_stage_location`` is provided.
        """
        if (self._query is None) == (self._input_stage_location is None):
            raise ValueError("Exactly one of query or input_stage_location must be provided.")

        target_function_info = self._model_version._validate_batch_inference_request(
            input_spec=self._input_spec,
            resources_spec=self._resources_spec,
            function_name=self._function_name,
        )

        return service_ops.build_batch_inference_task_definition(
            session=self._model_version._service_ops._session,
            model_fqn=self._model_version.fully_qualified_model_name,
            version_name=self._model_version.version_name,
            compute_pool=self._compute_pool,
            function_name=target_function_info["target_method"],
            query=self._query,
            input_stage_location=self._input_stage_location,
            input_spec=self._input_spec,
            output_spec=self._output_spec,
            resources_spec=self._resources_spec,
            inference_spec=self._inference_spec,
            image_build_spec=self._image_build_spec,
            replicas=self._replicas,
        )
