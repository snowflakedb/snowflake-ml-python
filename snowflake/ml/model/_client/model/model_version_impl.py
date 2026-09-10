import enum
import pathlib
import tempfile
import warnings
from typing import Any, Callable, Union, overload

import pandas as pd

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.feature_store import feature_view
from snowflake.ml.jobs import job
from snowflake.ml.lineage import lineage_node
from snowflake.ml.model import inference_engine, openai_signatures, task, type_hints
from snowflake.ml.model._client.model import (
    _loaded_model_telemetry,
    batch_inference_job_specs,
    inference_engine_utils,
)
from snowflake.ml.model._client.model_spec import model_spec
from snowflake.ml.model._client.ops import metadata_ops, model_ops, service_ops
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._model_composer.model_method import utils as model_method_utils
from snowflake.ml.model._packager.model_handlers import huggingface, snowmlmodel
from snowflake.ml.model._signatures import core
from snowflake.snowpark import Session, async_job, dataframe

_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"
VLLM_SUPPORTED_TASKS = [
    "text-generation",
    "image-text-to-text",
    "video-text-to-text",
    "audio-text-to-text",
]
VALID_OPENAI_SIGNATURES = [
    openai_signatures.OPENAI_CHAT_SIGNATURE,
    openai_signatures.OPENAI_CHAT_SIGNATURE_WITH_CONTENT_FORMAT_STRING,
    openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE,
    openai_signatures.OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING,
]


class ExportMode(enum.Enum):
    MODEL = "model"
    FULL = "full"


class ModelVersion(lineage_node.LineageNode):
    """Model Version Object representing a specific version of the model that could be run."""

    _model_ops: model_ops.ModelOperator
    _service_ops: service_ops.ServiceOperator
    _model_name: sql_identifier.SqlIdentifier
    _version_name: sql_identifier.SqlIdentifier
    _functions: list[model_manifest_schema.ModelFunctionInfo]
    _model_spec: model_spec.ModelSpec | None
    _target_platforms: list[str] | None

    def __init__(self) -> None:
        raise RuntimeError("ModelVersion's initializer is not meant to be used. Use `version` from model instead.")

    def _repr_html_(self) -> str:
        """Generate an HTML representation of the model version.

        Returns:
            str: HTML string containing formatted model version details.
        """
        from snowflake.ml.utils import html_utils

        # Get task
        try:
            task = self.get_model_task().value
        except Exception:
            task = (
                html_utils.create_error_message("Not available")
                .replace('<em style="color: #888; font-style: italic;">', "")
                .replace("</em>", "")
            )

        # Get functions info for display
        try:
            functions = self.show_functions()
            if not functions:
                functions_html = html_utils.create_error_message("No functions available")
            else:
                functions_list = []
                for func in functions:
                    try:
                        sig_html = func["signature"]._repr_html_()
                    except Exception:
                        # Fallback to simple display if can't display signature
                        sig_html = f"<pre style='margin: 5px 0;'>{func['signature']}</pre>"

                    function_content = f"""
                        <div style="margin: 5px 0;">
                            <strong>Target Method:</strong> {func['target_method']}
                        </div>
                        <div style="margin: 5px 0;">
                            <strong>Function Type:</strong> {func.get('target_method_function_type', 'N/A')}
                        </div>
                        <div style="margin: 5px 0;">
                            <strong>Partitioned:</strong> {func.get('is_partitioned', False)}
                        </div>
                        <div style="margin: 10px 0;">
                            <strong>Signature:</strong>
                            {sig_html}
                        </div>
                    """

                    functions_list.append(
                        html_utils.create_collapsible_section(
                            title=func["name"], content=function_content, open_by_default=False
                        )
                    )
                functions_html = "".join(functions_list)
        except Exception:
            functions_html = html_utils.create_error_message("Error retrieving functions")

        # Get metrics for display
        try:
            metrics = self.show_metrics()
            if not metrics:
                metrics_html = html_utils.create_error_message("No metrics available")
            else:
                metrics_html = ""
                for metric_name, value in metrics.items():
                    metrics_html += html_utils.create_metric_item(metric_name, value)
        except Exception:
            metrics_html = html_utils.create_error_message("Error retrieving metrics")

        # Create main content sections
        main_info = html_utils.create_grid_section(
            [
                ("Model Name", self.model_name),
                ("Version", f'<strong style="color: #28a745;">{self.version_name}</strong>'),
                ("Full Name", self.fully_qualified_model_name),
                ("Description", self.description),
                ("Task", task),
            ]
        )

        functions_section = html_utils.create_section_header("Functions") + html_utils.create_content_section(
            functions_html
        )

        metrics_section = html_utils.create_section_header("Metrics") + html_utils.create_content_section(metrics_html)

        content = main_info + functions_section + metrics_section

        return html_utils.create_base_container("Model Version Details", content)

    @classmethod
    def _ref(
        cls,
        model_ops: model_ops.ModelOperator,
        *,
        service_ops: service_ops.ServiceOperator,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
        retry: bool | None = False,
    ) -> "ModelVersion":
        self: "ModelVersion" = object.__new__(cls)
        self._model_ops = model_ops
        self._service_ops = service_ops
        self._model_name = model_name
        self._version_name = version_name
        self._functions = self._get_functions(**({"retry": True} if retry else {}))
        self._model_spec = None
        self._target_platforms = None
        super(cls, cls).__init__(
            self,
            session=model_ops._session,
            name=model_ops._model_client.fully_qualified_object_name(
                database_name=None, schema_name=None, object_name=model_name
            ),
            domain="model",
            version=version_name,
        )
        return self

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ModelVersion):
            return False
        return (
            self._model_ops == __value._model_ops
            and self._service_ops == __value._service_ops
            and self._model_name == __value._model_name
            and self._version_name == __value._version_name
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n" f"  name='{self.model_name}',\n" f"  version='{self._version_name}',\n" f")"
        )

    @property
    def model_name(self) -> str:
        """Return the name of the model to which the model version belongs, usable as a reference in SQL."""
        return self._model_name.identifier()

    @property
    def version_name(self) -> str:
        """Return the name of the version to which the model version belongs, usable as a reference in SQL."""
        return self._version_name.identifier()

    @property
    def fully_qualified_model_name(self) -> str:
        """Return the fully qualified name of the model to which the model version belongs."""
        return self._model_ops._model_version_client.fully_qualified_object_name(None, None, self._model_name)

    @property
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def description(self) -> str:
        """The description for the model version. This is an alias of `comment`."""
        return self.comment

    @description.setter
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def description(self, description: str) -> None:
        self.comment = description

    @property
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def comment(self) -> str:
        """The comment to the model version."""
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops.get_comment(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )

    @comment.setter
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def comment(self, comment: str) -> None:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops.set_comment(
            comment=comment,
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def show_metrics(self) -> dict[str, Any]:
        """Show all metrics logged with the model version.

        Returns:
            A dictionary showing the metrics.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops._metadata_ops.load(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )["metrics"]

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def get_metric(self, metric_name: str) -> Any:
        """Get the value of a specific metric.

        Args:
            metric_name: The name of the metric.

        Raises:
            KeyError: When the requested metric name does not exist.

        Returns:
            The value of the metric.
        """
        metrics = self.show_metrics()
        if metric_name not in metrics:
            raise KeyError(f"Cannot find metric with name {metric_name}.")
        return metrics[metric_name]

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def set_metric(self, metric_name: str, value: Any) -> None:
        """Set the value of a specific metric.

        Args:
            metric_name: The name of the metric.
            value: The value of the metric.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        metrics = self.show_metrics()
        metrics[metric_name] = value
        self._model_ops._metadata_ops.save(
            metadata_ops.ModelVersionMetadataSchema(metrics=metrics),
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def set_alias(self, alias_name: str) -> None:
        """Set alias to a model version.

        Args:
            alias_name: Alias to the model version.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        alias_name = sql_identifier.SqlIdentifier(alias_name)
        self._model_ops.set_alias(
            alias_name=alias_name,
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def unset_alias(self, version_or_alias: str) -> None:
        """unset alias to a model version.

        Args:
            version_or_alias: The name of the version or alias to a version.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        self._model_ops.unset_alias(
            version_or_alias_name=sql_identifier.SqlIdentifier(version_or_alias),
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def delete_metric(self, metric_name: str) -> None:
        """Delete a metric from metric storage.

        Args:
            metric_name: The name of the metric to be deleted.

        Raises:
            KeyError: When the requested metric name does not exist.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        metrics = self.show_metrics()
        if metric_name not in metrics:
            raise KeyError(f"Cannot find metric with name {metric_name}.")
        del metrics[metric_name]
        self._model_ops._metadata_ops.save(
            metadata_ops.ModelVersionMetadataSchema(metrics=metrics),
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )

    def _get_functions(self, *, retry: bool | None = False) -> list[model_manifest_schema.ModelFunctionInfo]:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops.get_functions(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
            **({"retry": True} if retry else {}),
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def get_model_task(self) -> task.Task:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops.get_model_task(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def show_functions(self) -> list[model_manifest_schema.ModelFunctionInfo]:
        """Show all functions information in a model version that is callable.

        Returns:
            A list of ModelFunctionInfo objects containing the following information:

            - name: The name of the function to be called (both in SQL and in Python SDK).
            - target_method: The original method name in the logged Python object.
            - signature: Python signature of the original method.
        """
        return self._functions

    def _get_model_spec(self, statement_params: dict[str, Any] | None = None) -> model_spec.ModelSpec:
        """Fetch and cache the model spec for this model version.

        Args:
            statement_params: Optional dictionary of statement parameters to include
                in the SQL command to fetch the model spec.

        Returns:
            The model spec for this model version.
        """
        if self._model_spec is None:
            self._model_spec = self._model_ops._fetch_model_spec(
                database_name=None,
                schema_name=None,
                model_name=self._model_name,
                version_name=self._version_name,
                statement_params=statement_params,
            )
        return self._model_spec

    @overload
    def run(
        self,
        X: pd.DataFrame | dataframe.DataFrame,
        *,
        function_name: str | None = None,
        partition_column: str | None = None,
        strict_input_validation: bool = False,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame | dataframe.DataFrame:
        """Invoke a method in a model version object.

        Args:
            X: The input data, which could be a pandas DataFrame or Snowpark DataFrame.
            function_name: The function name to run. It is the name used to call a function in SQL.
                Defaults to None. It can only be None if there is only 1 method.
            partition_column: The partition column name to partition by.
            strict_input_validation: Enable stricter validation for the input data. This will result value range based
                type validation to make sure your input data won't overflow when providing to the model.
            params: Optional dictionary of model inference parameters (e.g., temperature, top_k for LLMs).
                These are passed as keyword arguments to the model's inference method. Defaults to None.
        """
        ...

    @overload
    def run(
        self,
        X: pd.DataFrame | dataframe.DataFrame,
        *,
        service_name: str,
        function_name: str | None = None,
        strict_input_validation: bool = False,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame | dataframe.DataFrame:
        """Invoke a method in a model version object via a service.

        Args:
            X: The input data, which could be a pandas DataFrame or Snowpark DataFrame.
            service_name: The service name.
            function_name: The function name to run. It is the name used to call a function in SQL.
            strict_input_validation: Enable stricter validation for the input data. This will result value range based
                type validation to make sure your input data won't overflow when providing to the model.
            params: Optional dictionary of model inference parameters (e.g., temperature, top_k for LLMs).
                These are passed as keyword arguments to the model's inference method. Defaults to None.
        """
        ...

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
        func_params_to_log=["function_name", "service_name", "params"],
    )
    def run(
        self,
        X: Union[pd.DataFrame, "dataframe.DataFrame"],
        *,
        service_name: str | None = None,
        function_name: str | None = None,
        partition_column: str | None = None,
        strict_input_validation: bool = False,
        params: dict[str, Any] | None = None,
    ) -> Union[pd.DataFrame, "dataframe.DataFrame"]:
        """Invoke a method in a model version object via the warehouse or a service.

        Args:
            X: The input data, which could be a pandas DataFrame or Snowpark DataFrame.
            service_name: The service name. If None, the function is invoked via the warehouse. Otherwise, the function
                is invoked via the given service.
            function_name: The function name to run. It is the name used to call a function in SQL.
            partition_column: The partition column name to partition by.
            strict_input_validation: Enable stricter validation for the input data. This will result value range based
                type validation to make sure your input data won't overflow when providing to the model.
            params: Optional dictionary of model inference parameters (e.g., temperature, top_k for LLMs).
                These are passed as keyword arguments to the model's inference method. Defaults to None.

        Returns:
            The prediction data. It would be the same type dataframe as your input.

        Raises:
            ValueError: When the model does not support running on warehouse and no service name is provided.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )

        if partition_column is not None:
            # Partition column must be a valid identifier
            partition_column = sql_identifier.SqlIdentifier(partition_column)

        target_function_info = self._get_function_info(function_name=function_name)

        if service_name:
            database_name_id, schema_name_id, service_name_id = sql_identifier.parse_fully_qualified_name(service_name)

            return self._model_ops.invoke_method(
                method_name=sql_identifier.SqlIdentifier(target_function_info["name"]),
                signature=target_function_info["signature"],
                X=X,
                database_name=database_name_id,
                schema_name=schema_name_id,
                service_name=service_name_id,
                strict_input_validation=strict_input_validation,
                statement_params=statement_params,
                params=params,
                is_object_output=target_function_info["is_object_output"],
            )

        if self._model_spec is None:
            self._model_spec, self._target_platforms = self._model_ops._fetch_model_spec_and_target_platforms(
                database_name=None,
                schema_name=None,
                model_name=self._model_name,
                version_name=self._version_name,
                statement_params=statement_params,
            )

        if (
            self._target_platforms is not None
            and len(self._target_platforms) > 0
            and type_hints.TargetPlatform.WAREHOUSE.value not in self._target_platforms
        ):
            raise ValueError(
                f"The model {self.fully_qualified_model_name} version {self.version_name} "
                "is not logged for inference in Warehouse. "
                "To run the model in Warehouse, please log the model again using `log_model` API with "
                '`target_platforms=["WAREHOUSE"]` or '
                '`target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"]` and rerun the command. '
                "To run the model in Snowpark Container Services, the `service_name` argument must be provided. "
                "You can create a service using the `create_service` API. "
                "For inference in Warehouse, see https://docs.snowflake.com/en/developer-guide/"
                "snowflake-ml/model-registry/warehouse#inference-from-python. "
                "For inference in Snowpark Container Services, see https://docs.snowflake.com/en/developer-guide/"
                "snowflake-ml/model-registry/container#python."
            )

        explain_case_sensitive = self._determine_explain_case_sensitivity(target_function_info, statement_params)

        return self._model_ops.invoke_method(
            method_name=sql_identifier.SqlIdentifier(target_function_info["name"]),
            method_function_type=target_function_info["target_method_function_type"],
            signature=target_function_info["signature"],
            X=X,
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            strict_input_validation=strict_input_validation,
            partition_column=partition_column,
            statement_params=statement_params,
            is_partitioned=target_function_info["is_partitioned"],
            explain_case_sensitive=explain_case_sensitive,
            params=params,
            is_object_output=target_function_info["is_object_output"],
        )

    def _determine_explain_case_sensitivity(
        self,
        target_function_info: model_manifest_schema.ModelFunctionInfo,
        statement_params: dict[str, Any] | None = None,
    ) -> bool:
        parsed_model_spec = self._get_model_spec(statement_params)
        return model_method_utils.determine_explain_case_sensitive_from_method_options(
            parsed_model_spec.method_options,
            target_function_info["name"],
            default=parsed_model_spec.case_sensitive,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
        func_params_to_log=[
            "compute_pool",
            "input_spec",
            "output_spec",
            "resources_spec",
            "inference_spec",
            "image_build_spec",
            "replicas",
        ],
    )
    def run_batch(
        self,
        X: dataframe.DataFrame | None = None,
        *,
        input_stage_location: str | None = None,
        compute_pool: str,
        output_spec: batch_inference_job_specs.OutputSpec,
        input_spec: batch_inference_job_specs.InputSpec | None = None,
        resources_spec: batch_inference_job_specs.ResourcesSpec | None = None,
        inference_spec: batch_inference_job_specs.InferenceSpec | None = None,
        image_build_spec: batch_inference_job_specs.ImageBuildSpec | None = None,
        function_name: str | None = None,
        job_name: str | None = None,
        replicas: int | None = None,
        async_: bool = True,
    ) -> job.MLJob[Any]:
        """Run batch inference on a model as a Snowflake job.

        Args:
            X: Optional Snowpark DataFrame with the input rows. Provide exactly one of ``X`` or
                ``input_stage_location``.
            input_stage_location: Optional existing stage path holding the input data. Provide exactly
                one of ``X`` or ``input_stage_location``.
            compute_pool: Compute pool used by the SPCS job and image build.
            output_spec: Output block. ``stage_location`` is treated as a base;
                results are written under a per-job subdirectory,
                ``<stage_location>/<job_name>/``.
            input_spec: Input block.
            resources_spec: Resources block.
            inference_spec: Inference block.
            image_build_spec: Image build block.
            function_name: Model function name. Resolved against the model's
                function list when omitted.
            job_name: Optional fully qualified job name. When omitted the
                server generates a name.
            replicas: Optional ``REPLICAS`` value.
            async_: ``ASYNC`` clause value. Defaults to ``True``.

        Returns:
            MLJob handle for the launched batch inference job.

        Raises:
            ValueError: If not exactly one of ``X`` / ``input_stage_location`` is provided, or if
                ``input_spec.partition_column`` is supplied for a HuggingFace pipeline model or a
                FUNCTION-type method, or if the partition column collides with a partitioned model output.

        Example:
            >>> from snowflake.ml.model.batch_inference import (
            ...     InferenceSpec,
            ...     InputSpec,
            ...     OutputSpec,
            ...     ResourcesSpec,
            ...     SaveMode,
            ... )
            >>>
            >>> # Input rows can come from a table, a query, or Parquet files in a stage.
            >>> input_df = session.table("my_input_table")
            >>>
            >>> job = model_version.run_batch(
            ...     input_df,
            ...     compute_pool="my_compute_pool",
            ...     output_spec=OutputSpec(
            ...         stage_location="@my_db.public.my_stage/predictions/",
            ...         mode=SaveMode.OVERWRITE,
            ...     ),
            ... )
            >>>
            >>> # Results land under a per-job subdirectory named for the unqualified job name,
            >>> # which is the trailing identifier of the fully qualified job.id.
            >>> job_name = job.id.split(".")[-1].strip('"')
            >>> output_location = f"@my_db.public.my_stage/predictions/{job_name}/"
            >>>
            >>> # Size the job, pass model parameters, and pin the image repo.
            >>> job = model_version.run_batch(
            ...     input_df,
            ...     compute_pool="my_gpu_pool",
            ...     output_spec=OutputSpec(stage_location="@my_db.public.my_stage/predictions/"),
            ...     input_spec=InputSpec(params={"temperature": 0.7, "top_k": 50}),
            ...     resources_spec=ResourcesSpec(cpu_requests="2", memory_requests="8Gi", gpu_requests="1"),
            ...     inference_spec=InferenceSpec(num_workers=4),
            ...     function_name="predict",
            ...     replicas=2,
            ... )
            >>>
            >>> # Read input that is already staged, instead of materializing a DataFrame.
            >>> job = model_version.run_batch(
            ...     input_stage_location="@my_db.public.input_stage/batch_01/",
            ...     compute_pool="my_compute_pool",
            ...     output_spec=OutputSpec(stage_location="@my_db.public.my_stage/predictions/"),
            ... )

        Note:
            When ``X`` is provided, the rows are written as Parquet to a reserved subdirectory of
            ``output_spec.stage_location`` before the job starts. An ``input_stage_location`` is read
            in place and must not sit inside the output location.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )

        if (X is None) == (input_stage_location is None):
            raise ValueError("Exactly one of X or input_stage_location must be provided.")

        target_function_info = self._validate_batch_inference_request(
            input_spec=input_spec,
            resources_spec=resources_spec,
            function_name=function_name,
            statement_params=statement_params,
        )

        return self._service_ops.execute_inference_job_service(
            X=X,
            input_stage_location=input_stage_location,
            model_name=self._model_name,
            version_name=self._version_name,
            compute_pool_name=sql_identifier.SqlIdentifier(compute_pool),
            input_spec=input_spec,
            output_spec=output_spec,
            resources_spec=resources_spec,
            inference_spec=inference_spec,
            image_build_spec=image_build_spec,
            function_name=target_function_info["target_method"],
            job_name=job_name,
            replicas=replicas,
            async_=async_,
            statement_params=statement_params,
        )

    def _validate_batch_inference_request(
        self,
        *,
        input_spec: batch_inference_job_specs.InputSpec | None,
        resources_spec: batch_inference_job_specs.ResourcesSpec | None,
        function_name: str | None,
        statement_params: dict[str, Any] | None = None,
    ) -> model_manifest_schema.ModelFunctionInfo:
        """Resolve the target function and reject unsupported batch inference requests.

        Args:
            input_spec: Optional input block; its ``partition_column`` drives the partitioning rules.
            resources_spec: Optional resources block; its ``gpu_requests`` drives the GPU check.
            function_name: Optional model function name. Resolved against the model's function list
                when omitted.
            statement_params: Optional statement params for telemetry.

        Returns:
            The resolved function info.

        Raises:
            ValueError: If ``partition_column`` is supplied for a HuggingFace pipeline model or a
                FUNCTION-type method, or if the partition column collides with a partitioned model
                output.
        """
        effective_input_spec = input_spec if input_spec is not None else batch_inference_job_specs.InputSpec()
        partition_columns = (
            [effective_input_spec.partition_column] if effective_input_spec.partition_column is not None else None
        )

        if partition_columns is not None:
            parsed_model_spec = self._get_model_spec(statement_params)
            if parsed_model_spec.model_type == huggingface.TransformersPipelineHandler.HANDLER_TYPE:
                raise ValueError(
                    "partition_column is not supported for HuggingFace pipeline models in batch inference jobs. "
                    "Please remove the partition_column from input_spec."
                )

        gpu_requests = resources_spec.gpu_requests if resources_spec is not None else None
        self._throw_error_if_gpu_is_not_supported(gpu_requests, statement_params)

        target_function_info = self._get_function_info(function_name=function_name)

        if (
            partition_columns is not None
            and target_function_info["target_method_function_type"]
            == model_manifest_schema.ModelMethodFunctionTypes.FUNCTION.value
        ):
            raise ValueError(
                "partition_column is not supported for FUNCTION type methods in batch inference jobs. "
                "Only TABLE_FUNCTION type methods support partitioning."
            )

        if partition_columns is not None and target_function_info["is_partitioned"]:
            output_cols_upper = {spec.name.upper() for spec in target_function_info["signature"].outputs}
            partition_cols_upper = {p.upper() for p in partition_columns}
            collisions = sorted(partition_cols_upper & output_cols_upper)
            if collisions:
                raise ValueError(
                    f"Partitioned model output includes the partition column(s) {collisions}. "
                    f"Batch inference automatically appends the partition column to the output of "
                    f"partitioned models; please remove {collisions} from the model output "
                    f"and re-register the model."
                )

        return target_function_info

    def _get_function_info(self, function_name: str | None) -> model_manifest_schema.ModelFunctionInfo:
        functions: list[model_manifest_schema.ModelFunctionInfo] = self._functions

        if function_name:
            req_method_name = sql_identifier.SqlIdentifier(function_name).identifier()
            find_method: Callable[[model_manifest_schema.ModelFunctionInfo], bool] = (
                lambda method: method["name"] == req_method_name
            )
            target_function_info = next(
                filter(find_method, functions),
                None,
            )
            if target_function_info is None:
                raise ValueError(
                    f"There is no method with name {function_name} available in the model"
                    f" {self.fully_qualified_model_name} version {self.version_name}"
                )
        elif len(functions) != 1:
            raise ValueError(
                f"There are more than 1 target methods available in the model {self.fully_qualified_model_name}"
                f" version {self.version_name}. Please specify a `function_name` when calling the `run` method."
            )
        else:
            target_function_info = functions[0]

        return target_function_info

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT, subproject=_TELEMETRY_SUBPROJECT, func_params_to_log=["export_mode"]
    )
    def export(self, target_path: str, *, export_mode: ExportMode = ExportMode.MODEL) -> None:
        """Export model files to a local directory.

        Args:
            target_path: Path to a local directory to export files to. A directory will be created if does not exist.
            export_mode: The mode to export the model. Defaults to ExportMode.MODEL.
                ExportMode.MODEL: All model files including environment to load the model and model weights.
                ExportMode.FULL: Additional files to run the model in Warehouse, besides all files in MODEL mode,

        Raises:
            ValueError: Raised when the target path is a file or an non-empty folder.
        """
        target_local_path = pathlib.Path(target_path)
        if target_local_path.is_file() or any(target_local_path.iterdir()):
            raise ValueError(f"Target path {target_local_path} is a file or an non-empty folder.")

        target_local_path.mkdir(parents=False, exist_ok=True)
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        self._model_ops.download_files(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            target_path=target_local_path,
            mode=export_mode.value,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT, subproject=_TELEMETRY_SUBPROJECT, func_params_to_log=["force", "options"]
    )
    def load(
        self,
        *,
        force: bool = False,
        options: type_hints.ModelLoadOption | None = None,
    ) -> type_hints.SupportedModelType:
        """Load the underlying original Python object back from a model.
            This operation requires to have the exact the same environment as the one when logging the model, otherwise,
            the model might be not functional or some other problems might occur.

        Args:
            force: Bypass the best-effort environment validation. Defaults to False.
            options: Options to specify when loading the model, check `snowflake.ml.model.type_hints` for available
                options. Defaults to None.

        Raises:
            ValueError: Raised when the best-effort environment validation fails.

        Returns:
            The original Python object loaded from the model object.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        self._enforce_owner_only("load", statement_params=statement_params)
        if not force:
            with tempfile.TemporaryDirectory() as tmp_workspace_for_validation:
                ws_path_for_validation = pathlib.Path(tmp_workspace_for_validation)
                self._model_ops.download_files(
                    database_name=None,
                    schema_name=None,
                    model_name=self._model_name,
                    version_name=self._version_name,
                    target_path=ws_path_for_validation,
                    mode="minimal",
                    statement_params=statement_params,
                )
                pk_for_validation = model_composer.ModelComposer.load(
                    ws_path_for_validation, meta_only=True, options=options
                )
                assert pk_for_validation.meta, (
                    "Unable to load model metadata for validation. "
                    f"model_name={self._model_name}, version_name={self._version_name}"
                )

                validation_errors = pk_for_validation.meta.env.validate_with_local_env(
                    check_snowpark_ml_version=(
                        pk_for_validation.meta.model_type == snowmlmodel.SnowMLModelHandler.HANDLER_TYPE
                    )
                )
                if validation_errors:
                    raise ValueError(
                        f"Unable to load this model due to following validation errors: {validation_errors}. "
                        "Make sure your local environment is the same as that when you logged the model, "
                        "or if you believe it should work, specify `force=True` to bypass this check."
                    )

        warnings.warn(
            "Loading model requires to have the exact the same environment as the one when "
            "logging the model, otherwise, the model might be not functional or "
            "some other problems might occur.",
            category=RuntimeWarning,
            stacklevel=2,
        )

        # We need the folder to be existed.
        workspace = pathlib.Path(tempfile.mkdtemp())
        self._model_ops.download_files(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            target_path=workspace,
            mode="model",
            statement_params=statement_params,
        )
        warnings.warn(
            "Loading the model into memory. It will execute node in the model files.",
            category=RuntimeWarning,
            stacklevel=2,
        )
        pk = model_composer.ModelComposer.load(workspace, meta_only=False, options=options)
        assert pk.model, (
            "Unable to load model. "
            f"model_name={self._model_name}, version_name={self._version_name}, metadata={pk.meta}"
        )
        return _loaded_model_telemetry.instrument_for_telemetry(
            pk.model,
            model_name=self._model_name.identifier(),
            version_name=self._version_name.identifier(),
        )

    def _enforce_owner_only(self, operation: str, *, statement_params: dict[str, Any] | None = None) -> None:
        owner = self._model_ops.get_model_owner(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            statement_params=statement_params,
        )
        current_role_raw = self._model_ops._session.get_current_role()
        if not current_role_raw:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INSUFFICIENT_PRIVILEGES,
                original_exception=RuntimeError(
                    f"model registry: cannot {operation} this model — no active role on the session."
                ),
            )
        current_role = sql_identifier.SqlIdentifier(current_role_raw)
        if current_role != owner:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INSUFFICIENT_PRIVILEGES,
                original_exception=RuntimeError(
                    f"model registry: cannot {operation} this model — only the model's owner role is permitted. "
                    "You may use export() instead."
                ),
            )

    @staticmethod
    def _load_from_lineage_node(session: Session, name: str, version: str) -> "ModelVersion":
        database_name_id, schema_name_id, model_name_id = sql_identifier.parse_fully_qualified_name(name)
        if not database_name_id or not schema_name_id:
            raise ValueError("name should be fully qualifed.")

        return ModelVersion._ref(
            model_ops.ModelOperator(
                session,
                database_name=database_name_id,
                schema_name=schema_name_id,
            ),
            service_ops=service_ops.ServiceOperator(
                session,
                database_name=database_name_id,
                schema_name=schema_name_id,
            ),
            model_name=model_name_id,
            version_name=sql_identifier.SqlIdentifier(version),
        )

    def _can_run_on_gpu(
        self,
        statement_params: dict[str, Any] | None = None,
    ) -> bool:
        """Check if the model has GPU runtime support.

        Args:
            statement_params: Optional dictionary of statement parameters to include
                in the SQL command to fetch model spec.

        Returns:
            True if the model has GPU runtime configured, False otherwise.
        """
        return self._get_model_spec(statement_params).supports_gpu

    def _throw_error_if_gpu_is_not_supported(
        self,
        gpu_requests: str | int | None = None,
        statement_params: dict[str, Any] | None = None,
    ) -> None:
        """Check if the model has GPU runtime support.

        Args:
            gpu_requests: The gpu limit for GPU based inference. Can be integer, fractional or string values. Use CPU
                if None.
            statement_params: Optional dictionary of statement parameters to include
                in the SQL command to fetch model spec.

        Raises:
            ValueError: If the model does not have GPU runtime support.
        """
        if gpu_requests is not None and not self._can_run_on_gpu(statement_params):
            raise ValueError(
                f"GPU resources requested (gpu_requests={gpu_requests}), but the model "
                f"{self.fully_qualified_model_name} version {self.version_name} does not have GPU runtime support. "
                "Please ensure the model was logged with GPU runtime configuration or do not provide gpu_requests. "
                "To log the model with GPU runtime configuration, provide `cuda_version` in the `options` while calling"
                " the `log_model` function."
            )

    def _prepare_inference_engine_args(
        self,
        inference_engine_options: dict[str, Any] | None,
        statement_params: dict[str, Any] | None = None,
    ) -> service_ops.InferenceEngineArgs | None:
        """Prepare and validate inference engine arguments.

        This method handles the common logic for processing inference engine options:
        1. Parse inference engine options into InferenceEngineArgs
        2. Validate that the model is a HuggingFace text-generation model (if inference engine is specified)

        Args:
            inference_engine_options: Optional dictionary containing inference engine configuration.
            statement_params: Optional dictionary of statement parameters for SQL commands.

        Returns:
            Prepared InferenceEngineArgs or None if no inference engine is specified.
        """
        inference_engine_args = inference_engine_utils._get_inference_engine_args(inference_engine_options)

        if (
            inference_engine_args is not None
            and inference_engine_args.inference_engine == inference_engine.InferenceEngine.VLLM
        ):
            # Validate that model is HuggingFace vLLM supported model and is logged with
            # OpenAI compatible signature.
            self._check_huggingface_vllm_supported_model(statement_params)

        return inference_engine_args

    def _check_huggingface_vllm_supported_model(
        self,
        statement_params: dict[str, Any] | None = None,
    ) -> None:
        """Check if the model is a HuggingFace pipeline with vLLM supported task
        and is logged with OpenAI compatible signature.

        Args:
            statement_params: Optional dictionary of statement parameters to include
                in the SQL command to fetch model spec.

        Raises:
            ValueError: If the model is not a HuggingFace vLLM supported model or
                if the model is not logged with OpenAI compatible signature.
        """
        # Fetch model spec
        parsed_model_spec = self._get_model_spec(statement_params)

        # Check if model_type is huggingface_pipeline
        model_type = parsed_model_spec.model_type
        if model_type != "huggingface_pipeline":
            raise ValueError(
                f"Inference engine is only supported for HuggingFace vLLM supported models. "
                f"Found model_type: {model_type}"
            )

        is_vllm_supported_task = False
        found_tasks: list[str] = []

        # As long as the model supports vLLM supported task, we can use it
        for model_task in parsed_model_spec.model_tasks.values():
            if model_task:
                found_tasks.append(model_task)
                if model_task in VLLM_SUPPORTED_TASKS:
                    is_vllm_supported_task = True
                    break

        if not is_vllm_supported_task:
            tasks_str = ", ".join(found_tasks)
            found_tasks_str = (
                f"Found task(s): {tasks_str} in model spec." if found_tasks else "No task found in model spec."
            )
            supported_tasks_str = ", ".join(VLLM_SUPPORTED_TASKS)
            raise ValueError(
                f"Inference engine is only supported for vLLM supported tasks. {supported_tasks_str}. {found_tasks_str}"
            )

        # Check if the model is logged with OpenAI compatible signature.
        signatures_dict = parsed_model_spec.signatures

        # Deserialize signatures from model spec to ModelSignature objects for proper semantic comparison.
        deserialized_signatures = {
            func_name: core.ModelSignature.from_dict(sig_dict) for func_name, sig_dict in signatures_dict.items()
        }

        # Check if the model is logged with OpenAI compatible signature that contains messages feature group.
        if not any(
            isinstance(spec, core.FeatureGroupSpec) and spec.name == "messages"
            for sig in deserialized_signatures.values()
            for spec in sig.inputs
        ):
            raise ValueError(
                "Inference engine requires the model to be logged with one of the following signatures: "
                f"{VALID_OPENAI_SIGNATURES}. Please log the model again with one of these supported signatures."
                f"Found signatures: {signatures_dict}. "
            )

    @overload
    def create_service(
        self,
        *,
        service_name: str,
        image_build_compute_pool: str | None = None,
        service_compute_pool: str,
        image_repo: str | None = None,
        ingress_enabled: bool = False,
        min_instances: int = 0,
        max_instances: int = 1,
        cpu_requests: str | None = None,
        memory_requests: str | None = None,
        gpu_requests: str | None = None,
        num_workers: int | None = None,
        max_batch_rows: int | None = None,
        force_rebuild: bool = False,
        build_external_access_integration: str | None = None,
        block: bool = True,
        autocapture: bool | None = None,
        inference_engine_options: dict[str, Any] | None = None,
        experimental_options: dict[str, Any] | None = None,
        feature_sources_per_function: dict[str, list[feature_view.FeatureView]] | None = None,
    ) -> str | async_job.AsyncJob:
        """Create an inference service with the given spec.

        Args:
            service_name: The name of the service, can be fully qualified. If not fully qualified, the database or
                schema of the model will be used.
            image_build_compute_pool: The name of the compute pool used to build the model inference image. It uses
                the service compute pool if None.
            service_compute_pool: The name of the compute pool used to run the inference service.
            image_repo: The name of the image repository, can be fully qualified. If not fully qualified, the database
                or schema of the model will be used. This can be None, in that case a default hidden image repository
                will be used.
            ingress_enabled: If true, creates an service endpoint associated with the service. User must have
                BIND SERVICE ENDPOINT privilege on the account.
            min_instances: The minimum number of instances for the inference service. The service will automatically
                scale between min_instances and max_instances based on traffic and hardware utilization. If set to
                0 (default), the service will automatically suspend after a period of inactivity.
            max_instances: The maximum number of instances for the inference service.
            cpu_requests: The cpu limit for CPU based inference. Can be an integer, fractional or string values. If
                None, we attempt to utilize all the vCPU of the node.
            memory_requests: The memory limit with for CPU based inference. Can be an integer or a fractional value, but
                requires a unit (GiB, MiB). If None, we attempt to utilize all the memory of the node.
            gpu_requests: The gpu limit for GPU based inference. Can be integer, fractional or string values. Use CPU
                if None.
            num_workers: The number of workers to run the inference service for handling requests in parallel within an
                instance of the service. By default, it is set to 2*vCPU+1 of the node for CPU based inference and 1 for
                GPU based inference. For GPU based inference, please see best practices before playing with this value.
            max_batch_rows: The maximum number of rows to batch for inference. Auto determined if None. Minimum 32.
            force_rebuild: Whether to force a model inference image rebuild.
            build_external_access_integration: (Deprecated) The external access integration for image build. This is
                usually permitting access to conda & PyPI repositories.
            block: A bool value indicating whether this function will wait until the service is available.
                When it is ``False``, this function executes the underlying service creation asynchronously
                and returns an :class:`AsyncJob`.
            autocapture: Whether inference autocapture is enabled on the service. If true, inference data will be
                captured in the model inference table.
            inference_engine_options: Options for the service creation with custom inference engine.
                Supports `engine` and `engine_args_override`.
                `engine` is the type of the inference engine to use. Accepts an
                :class:`~snowflake.ml.model.inference_engine.InferenceEngine` enum member or a
                case-insensitive string such as ``"vllm"`` or ``"python_generic"``.
                `engine_args_override` is a list of string arguments to pass to the inference engine.
            experimental_options: Experimental options for the service creation.
            feature_sources_per_function: Optional mapping from model function name (e.g. ``"predict"``) to the list of
                registered :class:`FeatureView` objects whose columns should be looked up at inference time. The model
                service will fetch any missing feature columns from these sources before invoking the model. Currently
                only one FeatureView per function is supported.
        """
        ...

    @overload
    def create_service(
        self,
        *,
        service_name: str,
        image_build_compute_pool: str | None = None,
        service_compute_pool: str,
        image_repo: str | None = None,
        ingress_enabled: bool = False,
        min_instances: int = 0,
        max_instances: int = 1,
        cpu_requests: str | None = None,
        memory_requests: str | None = None,
        gpu_requests: str | None = None,
        num_workers: int | None = None,
        max_batch_rows: int | None = None,
        force_rebuild: bool = False,
        build_external_access_integrations: list[str] | None = None,
        block: bool = True,
        autocapture: bool | None = None,
        inference_engine_options: dict[str, Any] | None = None,
        experimental_options: dict[str, Any] | None = None,
        feature_sources_per_function: dict[str, list[feature_view.FeatureView]] | None = None,
    ) -> str | async_job.AsyncJob:
        """Create an inference service with the given spec.

        Args:
            service_name: The name of the service, can be fully qualified. If not fully qualified, the database or
                schema of the model will be used.
            image_build_compute_pool: The name of the compute pool used to build the model inference image. It uses
                the service compute pool if None.
            service_compute_pool: The name of the compute pool used to run the inference service.
            image_repo: The name of the image repository, can be fully qualified. If not fully qualified, the database
                or schema of the model will be used. This can be None, in that case a default hidden image repository
                will be used.
            ingress_enabled: If true, creates an service endpoint associated with the service. User must have
                BIND SERVICE ENDPOINT privilege on the account.
            min_instances: The minimum number of instances for the inference service. The service will automatically
                scale between min_instances and max_instances based on traffic and hardware utilization. If set to
                0 (default), the service will automatically suspend after a period of inactivity.
            max_instances: The maximum number of instances for the inference service.
            cpu_requests: The cpu limit for CPU based inference. Can be an integer, fractional or string values. If
                None, we attempt to utilize all the vCPU of the node.
            memory_requests: The memory limit with for CPU based inference. Can be an integer or a fractional value, but
                requires a unit (GiB, MiB). If None, we attempt to utilize all the memory of the node.
            gpu_requests: The gpu limit for GPU based inference. Can be integer, fractional or string values. Use CPU
                if None.
            num_workers: The number of workers to run the inference service for handling requests in parallel within an
                instance of the service. By default, it is set to 2*vCPU+1 of the node for CPU based inference and 1 for
                GPU based inference. For GPU based inference, please see best practices before playing with this value.
            max_batch_rows: The maximum number of rows to batch for inference. Auto determined if None. Minimum 32.
            force_rebuild: Whether to force a model inference image rebuild.
            build_external_access_integrations: The external access integrations for image build. This is usually
                permitting access to conda & PyPI repositories.
            block: A bool value indicating whether this function will wait until the service is available.
                When it is ``False``, this function executes the underlying service creation asynchronously
                and returns an :class:`AsyncJob`.
            autocapture: Whether inference autocapture is enabled on the service. If true, inference data will be
                captured in the model inference table.
            inference_engine_options: Options for the service creation with custom inference engine.
                Supports `engine` and `engine_args_override`.
                `engine` is the type of the inference engine to use. Accepts an
                :class:`~snowflake.ml.model.inference_engine.InferenceEngine` enum member or a
                case-insensitive string such as ``"vllm"`` or ``"python_generic"``.
                `engine_args_override` is a list of string arguments to pass to the inference engine.
            experimental_options: Experimental options for the service creation.
            feature_sources_per_function: Optional mapping from model function name (e.g. ``"predict"``) to the list of
                registered :class:`FeatureView` objects whose columns should be looked up at inference time. The model
                service will fetch any missing feature columns from these sources before invoking the model. Currently
                only one FeatureView per function is supported.
        """
        ...

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
        func_params_to_log=[
            "service_name",
            "image_build_compute_pool",
            "service_compute_pool",
            "image_repo_database",
            "image_repo_schema",
            "image_repo",
            "gpu_requests",
            "num_workers",
            "max_batch_rows",
        ],
    )
    def create_service(
        self,
        *,
        service_name: str,
        image_build_compute_pool: str | None = None,
        service_compute_pool: str,
        image_repo: str | None = None,
        ingress_enabled: bool = False,
        min_instances: int = 0,
        max_instances: int = 1,
        cpu_requests: str | None = None,
        memory_requests: str | None = None,
        gpu_requests: str | int | None = None,
        num_workers: int | None = None,
        max_batch_rows: int | None = None,
        force_rebuild: bool = False,
        build_external_access_integration: str | None = None,
        build_external_access_integrations: list[str] | None = None,
        block: bool = True,
        autocapture: bool | None = None,
        inference_engine_options: dict[str, Any] | None = None,
        experimental_options: dict[str, Any] | None = None,
        feature_sources_per_function: dict[str, list[feature_view.FeatureView]] | None = None,
    ) -> str | async_job.AsyncJob:
        """Create an inference service with the given spec.

        Args:
            service_name: The name of the service, can be fully qualified. If not fully qualified, the database or
                schema of the model will be used.
            image_build_compute_pool: The name of the compute pool used to build the model inference image. It uses
                the service compute pool if None.
            service_compute_pool: The name of the compute pool used to run the inference service.
            image_repo: The name of the image repository, can be fully qualified. If not fully qualified, the database
                or schema of the model will be used. This can be None, in that case a default hidden image repository
                will be used.
            ingress_enabled: If true, creates an service endpoint associated with the service. User must have
                BIND SERVICE ENDPOINT privilege on the account.
            min_instances: The minimum number of instances for the inference service. The service will automatically
                scale between min_instances and max_instances based on traffic and hardware utilization. If set to
                0 (default), the service will automatically suspend after a period of inactivity.
            max_instances: The maximum number of instances for the inference service.
            cpu_requests: The cpu limit for CPU based inference. Can be an integer, fractional or string values. If
                None, we attempt to utilize all the vCPU of the node.
            memory_requests: The memory limit with for CPU based inference. Can be an integer or a fractional value, but
                requires a unit (GiB, MiB). If None, we attempt to utilize all the memory of the node.
            gpu_requests: The gpu limit for GPU based inference. Can be integer, fractional or string values. Use CPU
                if None.
            num_workers: The number of workers to run the inference service for handling requests in parallel within an
                instance of the service. By default, it is set to 2*vCPU+1 of the node for CPU based inference and 1 for
                GPU based inference. For GPU based inference, please see best practices before playing with this value.
            max_batch_rows: The maximum number of rows to batch for inference. Auto determined if None. Minimum 32.
            force_rebuild: Whether to force a model inference image rebuild.
            build_external_access_integration: (Deprecated) The external access integration for image build. This is
                usually permitting access to conda & PyPI repositories.
            build_external_access_integrations: The external access integrations for image build. This is usually
                permitting access to conda & PyPI repositories.
            block: A bool value indicating whether this function will wait until the service is available.
                When it is False, this function executes the underlying service creation asynchronously
                and returns an AsyncJob.
            autocapture: Whether inference autocapture is enabled on the service. If true, inference data will be
                captured in the model inference table.
            inference_engine_options: Options for the service creation with custom inference engine.
                Supports `engine` and `engine_args_override`.
                `engine` is the type of the inference engine to use. Accepts an
                :class:`~snowflake.ml.model.inference_engine.InferenceEngine` enum member or a
                case-insensitive string such as ``"vllm"`` or ``"python_generic"``.
                `engine_args_override` is a list of string arguments to pass to the inference engine.
            experimental_options: Experimental options for the service creation.
            feature_sources_per_function: Optional mapping from model function name (e.g. ``"predict"``) to the list of
                registered :class:`FeatureView` objects whose columns should be looked up at inference time. The model
                service will fetch any missing feature columns from these sources before invoking the model. Currently
                only one FeatureView per function is supported.


        Raises:
            ValueError: Illegal external access integration arguments.
            ValueError: If GPU resources are requested but the model does not have GPU runtime support.
            ValueError: If all model methods are TABLE_FUNCTION type, which is not supported for
                online inference.
            exceptions.SnowparkSQLException: if service already exists.

        Returns:
            If `block=True`, return result information about service creation from server.
            Otherwise, return the service creation AsyncJob.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )

        if build_external_access_integration is not None:
            msg = (
                "`build_external_access_integration` is deprecated. "
                "Please use `build_external_access_integrations` instead."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            if build_external_access_integrations is not None:
                msg = (
                    "`build_external_access_integration` and `build_external_access_integrations` cannot be set at the"
                    "same time. Please use `build_external_access_integrations` only."
                )
                raise ValueError(msg)
            build_external_access_integrations = [build_external_access_integration]

        service_db_id, service_schema_id, service_id = sql_identifier.parse_fully_qualified_name(service_name)

        # Check for TABLE_FUNCTION methods — online inference doesn't support table functions
        table_function_methods = [
            f["name"]
            for f in self._functions
            if f["target_method_function_type"] == model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value
        ]
        if table_function_methods:
            non_table_function_methods = [
                f["name"]
                for f in self._functions
                if f["target_method_function_type"]
                != model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION.value
            ]
            msg = (
                "Online inference services do not support TABLE_FUNCTION methods. "
                f"The following methods have TABLE_FUNCTION type: {', '.join(table_function_methods)}. "
            )
            if non_table_function_methods:
                warnings.warn(
                    msg + "These methods will not be available in the service.",
                    stacklevel=2,
                )
            else:
                raise ValueError(msg + "Consider using batch inference jobs instead.")

        # Validate GPU support if GPU resources are requested
        self._throw_error_if_gpu_is_not_supported(gpu_requests, statement_params)

        inference_engine_args = self._prepare_inference_engine_args(
            inference_engine_options,
            statement_params,
        )

        from snowflake.ml.model import event_handler
        from snowflake.snowpark import exceptions

        model_event_handler = event_handler.ModelEventHandler()

        with model_event_handler.status("Creating model inference service", total=6, block=block) as status:
            try:
                result = self._service_ops.create_service(
                    database_name=None,
                    schema_name=None,
                    model_name=self._model_name,
                    version_name=self._version_name,
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
                    min_instances=min_instances,
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
                    statement_params=statement_params,
                    progress_status=status,
                    inference_engine_args=inference_engine_args,
                    autocapture=autocapture,
                    feature_sources_per_function=feature_sources_per_function,
                )
                status.update(label="Model service created successfully", state="complete", expanded=False)
                return result
            except exceptions.SnowparkSQLException as e:
                # Check if the error is because the service already exists
                if "already exists" in str(e).lower() or "100132" in str(
                    e
                ):  # 100132 is Snowflake error code for object already exists
                    status.update("service already exists")
                    status.complete()
                    status.update(label="Service already exists", state="error", expanded=False)
                    raise
                else:
                    status.update(label="Service creation failed", state="error", expanded=False)
                    raise

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def list_services(
        self,
    ) -> pd.DataFrame:
        """List all the service names using this model version.

        Returns:
            List of details about all the services associated with this model version. The details include:
              name: The name of the service.
              status: The status of the service.
              inference_endpoint: The public endpoint of the service, if enabled and services is not in PENDING state.
                This will give privatelink endpoint if the session is created with privatelink connection
              internal_endpoint: The internal endpoint of the service, if services is not in PENDING state.
              autocapture_enabled: Whether service has autocapture enabled, if it is set in service proxy spec.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )

        return pd.DataFrame(
            self._model_ops.show_services(
                database_name=None,
                schema_name=None,
                model_name=self._model_name,
                version_name=self._version_name,
                statement_params=statement_params,
            )
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def delete_service(
        self,
        service_name: str,
    ) -> None:
        """Drops the given service.

        Args:
            service_name: The name of the service, can be fully qualified. If not fully qualified, the database or
                schema of the model will be used.

        Raises:
            ValueError: If the service does not exist or operation is not permitted by user or service does not belong
                to this model.
        """
        if not service_name:
            raise ValueError("service_name cannot be empty.")

        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )

        database_name_id, schema_name_id, service_name_id = sql_identifier.parse_fully_qualified_name(service_name)
        self._model_ops.delete_service(
            database_name=None,
            schema_name=None,
            model_name=self._model_name,
            version_name=self._version_name,
            service_database_name=database_name_id,
            service_schema_name=schema_name_id,
            service_name=service_name_id,
            statement_params=statement_params,
        )


lineage_node.DOMAIN_LINEAGE_REGISTRY["model"] = ModelVersion
