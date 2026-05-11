from __future__ import annotations

from typing import Any, Optional

import yaml

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import identifier, sql_identifier
from snowflake.ml.model._client.model import (
    batch_inference_serialization,
    batch_inference_specs,
    model_version_impl,
)
from snowflake.snowpark import dataframe

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
    """

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
        func_params_to_log=[
            "name",
            "compute_pool",
            "input_spec",
            "output_spec",
            "job_spec",
            "inference_engine_options",
        ],
    )
    def __init__(
        self,
        name: str,
        *,
        model_version: model_version_impl.ModelVersion,
        X: dataframe.DataFrame,
        compute_pool: str,
        output_spec: batch_inference_specs.OutputSpec,
        input_spec: Optional[batch_inference_specs.InputSpec] = None,
        job_spec: Optional[batch_inference_specs.JobSpec] = None,
        inference_engine_options: Optional[dict[str, Any]] = None,
        **dagtask_kwargs: Any,
    ) -> None:
        if not _HAS_SNOWFLAKE_CORE:
            raise ImportError(
                "BatchInferenceTask requires the `snowflake.core` package. "
                "Install it with `pip install snowflake.core`."
            )
        if "definition" in dagtask_kwargs:
            raise TypeError("BatchInferenceTask builds its own task definition; do not pass `definition=`.")

        self._fully_qualified_model_name: str = model_version.fully_qualified_model_name
        self._version_name: str = model_version.version_name
        self._compute_pool = compute_pool
        self._output_spec = output_spec
        self._input_spec = input_spec if input_spec is not None else batch_inference_specs.InputSpec()
        self._job_spec = job_spec if job_spec is not None else batch_inference_specs.JobSpec()
        self._inference_engine_options = inference_engine_options

        target_function_name = self._job_spec.function_name if self._job_spec.function_name else None
        target_function_info = model_version._get_function_info(function_name=target_function_name)
        self._function_name: str = target_function_info["target_method"]

        if self._job_spec.warehouse is not None:
            self._warehouse: str = self._job_spec.warehouse
        else:
            session_warehouse = model_version._service_ops._session.get_current_warehouse()
            if session_warehouse is None:
                raise ValueError("Warehouse is not set. Please set the warehouse field in the JobSpec.")
            self._warehouse = session_warehouse

        self._queries: list[str] = list(X.queries["queries"])
        self._post_actions: list[str] = list(X.queries["post_actions"])

        sql = self._to_sql()
        super().__init__(name, definition=sql, **dagtask_kwargs)

    def _to_sql(self) -> str:
        self._validate()
        spec_dict = self._build_spec_dict()
        yaml_str = yaml.safe_dump(spec_dict, default_flow_style=False, sort_keys=False)
        return f"CALL SYSTEM$DEPLOY_MODEL($${yaml_str}$$)"

    def _build_spec_dict(self) -> dict[str, Any]:
        models = [{"name": self._fully_qualified_model_name, "version": self._version_name}]

        db_id, schema_id, _ = sql_identifier.parse_fully_qualified_name(self._fully_qualified_model_name)

        job_name: Optional[str] = None
        name_prefix: Optional[str] = None
        if self._job_spec.job_name is not None:
            parsed_job_db, parsed_job_schema, parsed_job_name = sql_identifier.parse_fully_qualified_name(
                self._job_spec.job_name
            )
            job_db = parsed_job_db or db_id
            job_schema = parsed_job_schema or schema_id
            assert job_db is not None and job_schema is not None
            job_name = identifier.get_schema_level_object_identifier(
                job_db.identifier(), job_schema.identifier(), parsed_job_name.identifier()
            )
        elif self._job_spec.job_name_prefix is not None:
            assert db_id is not None and schema_id is not None
            job_name_prefix_qualified = identifier.get_schema_level_object_identifier(
                db_id.identifier(), schema_id.identifier(), self._job_spec.job_name_prefix + "_"
            )
            name_prefix = job_name_prefix_qualified

        function_name = self._function_name

        if self._output_spec.base_stage_location is not None:
            base_stage_location = self._output_spec.base_stage_location
            if not base_stage_location.endswith("/"):
                base_stage_location += "/"

            input_dict: dict[str, Any] = {
                "queries": self._queries,
                "post_actions": self._post_actions,
                "input_file_pattern": "*",
            }
            output_dict: dict[str, Any] = {
                "base_stage_location": base_stage_location,
                "completion_filename": "_SUCCESS",
            }
        else:
            assert self._output_spec.stage_location is not None
            output_stage = self._output_spec.stage_location
            if not output_stage.endswith("/"):
                output_stage += "/"

            input_stage_location = f"{output_stage}_temporary/"
            input_dict = {
                "input_stage_location": input_stage_location,
                "queries": self._queries,
                "post_actions": self._post_actions,
                "input_file_pattern": "*",
            }
            output_dict = {
                "output_stage_location": output_stage,
                "completion_filename": "_SUCCESS",
            }

        params_encoded = batch_inference_serialization.encode_params(self._input_spec.params)
        if params_encoded is not None:
            input_dict["params"] = params_encoded

        column_handling_encoded = batch_inference_serialization.encode_column_handling(self._input_spec.column_handling)
        if column_handling_encoded is not None:
            input_dict["column_handling"] = column_handling_encoded

        if self._input_spec.partition_column is not None:
            input_dict["partition_columns"] = [self._input_spec.partition_column]

        job_dict: dict[str, Any] = {
            "compute_pool": self._compute_pool,
            "warehouse": self._warehouse,
            "function_name": function_name,
            "input": input_dict,
            "output": output_dict,
            "sync": True,
        }

        if job_name is not None:
            job_dict["name"] = job_name
        if name_prefix is not None:
            job_dict["name_prefix"] = name_prefix

        if self._job_spec.cpu_requests is not None:
            job_dict["cpu"] = self._job_spec.cpu_requests
        if self._job_spec.memory_requests is not None:
            job_dict["memory"] = self._job_spec.memory_requests
        if self._job_spec.gpu_requests is not None:
            job_dict["gpu"] = self._job_spec.gpu_requests
        if self._job_spec.num_workers is not None:
            job_dict["num_workers"] = self._job_spec.num_workers
        if self._job_spec.max_batch_rows is not None:
            job_dict["max_batch_rows"] = self._job_spec.max_batch_rows
        if self._job_spec.replicas is not None:
            job_dict["replicas"] = self._job_spec.replicas

        spec: dict[str, Any] = {
            "models": models,
            "job": job_dict,
        }

        if self._inference_engine_options is not None:
            engine = self._inference_engine_options["engine"]
            engine_spec: dict[str, Any] = {"inference_engine_name": engine.value}
            engine_args = self._inference_engine_options.get("engine_args_override")
            if engine_args:
                engine_spec["inference_engine_args"] = engine_args
            job_dict["inference_engine_spec"] = engine_spec
        else:
            image_build: dict[str, Any] = {"compute_pool": self._compute_pool}
            if self._job_spec.image_repo is not None:
                image_build["image_repo"] = self._job_spec.image_repo
            if self._job_spec.force_rebuild:
                image_build["force_rebuild"] = True
            spec["image_build"] = image_build

        return spec

    def _validate(self) -> None:
        pass
