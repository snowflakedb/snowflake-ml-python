import enum
import pathlib
import tempfile
import uuid
import warnings
from typing import Any, Callable, Optional, Union, overload

import pandas as pd

from snowflake.ml import jobs
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.lineage import lineage_node
from snowflake.ml.model import task, type_hints
from snowflake.ml.model._client.model import (
    batch_inference_specs,
    inference_engine_utils,
)
from snowflake.ml.model._client.ops import metadata_ops, model_ops, service_ops
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.model._model_composer.model_method import utils as model_method_utils
from snowflake.ml.model._packager.model_handlers import snowmlmodel
from snowflake.ml.model._packager.model_meta import model_meta_schema
from snowflake.snowpark import Session, async_job, dataframe

_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"
_BATCH_INFERENCE_JOB_ID_PREFIX = "BATCH_INFERENCE_"
_BATCH_INFERENCE_TEMPORARY_FOLDER = "_temporary"


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
    _model_spec: Optional[model_meta_schema.ModelMetadataDict]

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
    ) -> "ModelVersion":
        self: "ModelVersion" = object.__new__(cls)
        self._model_ops = model_ops
        self._service_ops = service_ops
        self._model_name = model_name
        self._version_name = version_name
        self._functions = self._get_functions()
        self._model_spec = None
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

    def _get_functions(self) -> list[model_manifest_schema.ModelFunctionInfo]:
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

    def _get_model_spec(self, statement_params: Optional[dict[str, Any]] = None) -> model_meta_schema.ModelMetadataDict:
        """Fetch and cache the model spec for this model version.

        Args:
            statement_params: Optional dictionary of statement parameters to include
                in the SQL command to fetch the model spec.

        Returns:
            The model spec as a dictionary for this model version.
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
        X: Union[pd.DataFrame, dataframe.DataFrame],
        *,
        function_name: Optional[str] = None,
        partition_column: Optional[str] = None,
        strict_input_validation: bool = False,
    ) -> Union[pd.DataFrame, dataframe.DataFrame]:
        """Invoke a method in a model version object.

        Args:
            X: The input data, which could be a pandas DataFrame or Snowpark DataFrame.
            function_name: The function name to run. It is the name used to call a function in SQL.
                Defaults to None. It can only be None if there is only 1 method.
            partition_column: The partition column name to partition by.
            strict_input_validation: Enable stricter validation for the input data. This will result value range based
                type validation to make sure your input data won't overflow when providing to the model.
        """
        ...

    @overload
    def run(
        self,
        X: Union[pd.DataFrame, dataframe.DataFrame],
        *,
        service_name: str,
        function_name: Optional[str] = None,
        strict_input_validation: bool = False,
    ) -> Union[pd.DataFrame, dataframe.DataFrame]:
        """Invoke a method in a model version object via a service.

        Args:
            X: The input data, which could be a pandas DataFrame or Snowpark DataFrame.
            service_name: The service name.
            function_name: The function name to run. It is the name used to call a function in SQL.
            strict_input_validation: Enable stricter validation for the input data. This will result value range based
                type validation to make sure your input data won't overflow when providing to the model.
        """
        ...

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
        func_params_to_log=["function_name", "service_name"],
    )
    def run(
        self,
        X: Union[pd.DataFrame, "dataframe.DataFrame"],
        *,
        service_name: Optional[str] = None,
        function_name: Optional[str] = None,
        partition_column: Optional[str] = None,
        strict_input_validation: bool = False,
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

        Returns:
            The prediction data. It would be the same type dataframe as your input.
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
            )
        else:
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
            )

    def _determine_explain_case_sensitivity(
        self,
        target_function_info: model_manifest_schema.ModelFunctionInfo,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> bool:
        model_spec = self._get_model_spec(statement_params)
        method_options = model_spec.get("method_options", {})
        return model_method_utils.determine_explain_case_sensitive_from_method_options(
            method_options, target_function_info["name"]
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
        func_params_to_log=[
            "compute_pool",
            "output_spec",
            "job_spec",
        ],
    )
    def _run_batch(
        self,
        *,
        compute_pool: str,
        input_spec: dataframe.DataFrame,
        output_spec: batch_inference_specs.OutputSpec,
        job_spec: Optional[batch_inference_specs.JobSpec] = None,
    ) -> jobs.MLJob[Any]:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )

        if job_spec is None:
            job_spec = batch_inference_specs.JobSpec()

        warehouse = job_spec.warehouse or self._service_ops._session.get_current_warehouse()
        if warehouse is None:
            raise ValueError("Warehouse is not set. Please set the warehouse field in the JobSpec.")

        # use a temporary folder in the output stage to store the intermediate output from the dataframe
        output_stage_location = output_spec.stage_location
        if not output_stage_location.endswith("/"):
            output_stage_location += "/"
        input_stage_location = f"{output_stage_location}{_BATCH_INFERENCE_TEMPORARY_FOLDER}/"

        self._service_ops._enforce_save_mode(output_spec.mode, output_stage_location)

        try:
            input_spec.write.copy_into_location(location=input_stage_location, file_format_type="parquet", header=True)
        # todo: be specific about the type of errors to provide better error messages.
        except Exception as e:
            raise RuntimeError(f"Failed to process input_spec: {e}")

        if job_spec.job_name is None:
            # Same as the MLJob ID generation logic with a different prefix
            job_name = f"{_BATCH_INFERENCE_JOB_ID_PREFIX}{str(uuid.uuid4()).replace('-', '_').upper()}"
        else:
            job_name = job_spec.job_name

        return self._service_ops.invoke_batch_job_method(
            # model version info
            model_name=self._model_name,
            version_name=self._version_name,
            # job spec
            function_name=self._get_function_info(function_name=job_spec.function_name)["target_method"],
            compute_pool_name=sql_identifier.SqlIdentifier(compute_pool),
            force_rebuild=job_spec.force_rebuild,
            image_repo_name=job_spec.image_repo,
            num_workers=job_spec.num_workers,
            max_batch_rows=job_spec.max_batch_rows,
            warehouse=sql_identifier.SqlIdentifier(warehouse),
            cpu_requests=job_spec.cpu_requests,
            memory_requests=job_spec.memory_requests,
            gpu_requests=job_spec.gpu_requests,
            job_name=job_name,
            replicas=job_spec.replicas,
            # input and output
            input_stage_location=input_stage_location,
            input_file_pattern="*",
            output_stage_location=output_stage_location,
            completion_filename="_SUCCESS",
            # misc
            statement_params=statement_params,
        )

    def _get_function_info(self, function_name: Optional[str]) -> model_manifest_schema.ModelFunctionInfo:
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
        options: Optional[type_hints.ModelLoadOption] = None,
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
        pk = model_composer.ModelComposer.load(workspace, meta_only=False, options=options)
        assert pk.model, (
            "Unable to load model. "
            f"model_name={self._model_name}, version_name={self._version_name}, metadata={pk.meta}"
        )
        return pk.model

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

    def _check_huggingface_text_generation_model(
        self,
        statement_params: Optional[dict[str, Any]] = None,
    ) -> None:
        """Check if the model is a HuggingFace pipeline with text-generation task.

        Args:
            statement_params: Optional dictionary of statement parameters to include
                in the SQL command to fetch model spec.

        Raises:
            ValueError: If the model is not a HuggingFace text-generation model.
        """
        # Fetch model spec
        model_spec = self._get_model_spec(statement_params)

        # Check if model_type is huggingface_pipeline
        model_type = model_spec.get("model_type")
        if model_type != "huggingface_pipeline":
            raise ValueError(
                f"Inference engine is only supported for HuggingFace text-generation models. "
                f"Found model_type: {model_type}"
            )

        # Check if model supports text-generation task
        # There should only be one model in the list because we don't support multiple models in a single model spec
        models = model_spec.get("models", {})
        is_text_generation = False
        found_tasks: list[str] = []

        # As long as the model supports text-generation task, we can use it
        for _, model_info in models.items():
            options = model_info.get("options", {})
            task = options.get("task")
            if task:
                found_tasks.append(str(task))
                if task == "text-generation":
                    is_text_generation = True
                    break

        if not is_text_generation:
            tasks_str = ", ".join(found_tasks)
            found_tasks_str = (
                f"Found task(s): {tasks_str} in model spec." if found_tasks else "No task found in model spec."
            )
            raise ValueError(f"Inference engine is only supported for task 'text-generation'. {found_tasks_str}")

    @overload
    def create_service(
        self,
        *,
        service_name: str,
        image_build_compute_pool: Optional[str] = None,
        service_compute_pool: str,
        image_repo: Optional[str] = None,
        ingress_enabled: bool = False,
        max_instances: int = 1,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        gpu_requests: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_batch_rows: Optional[int] = None,
        force_rebuild: bool = False,
        build_external_access_integration: Optional[str] = None,
        block: bool = True,
        experimental_options: Optional[dict[str, Any]] = None,
    ) -> Union[str, async_job.AsyncJob]:
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
            max_instances: The maximum number of inference service instances to run. The same value it set to
                MIN_INSTANCES property of the service.
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
            experimental_options: Experimental options for the service creation with custom inference engine.
                Currently, only `inference_engine` and `inference_engine_args_override` are supported.
                `inference_engine` is the name of the inference engine to use.
                `inference_engine_args_override` is a list of string arguments to pass to the inference engine.
        """
        ...

    @overload
    def create_service(
        self,
        *,
        service_name: str,
        image_build_compute_pool: Optional[str] = None,
        service_compute_pool: str,
        image_repo: Optional[str] = None,
        ingress_enabled: bool = False,
        max_instances: int = 1,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        gpu_requests: Optional[str] = None,
        num_workers: Optional[int] = None,
        max_batch_rows: Optional[int] = None,
        force_rebuild: bool = False,
        build_external_access_integrations: Optional[list[str]] = None,
        block: bool = True,
        experimental_options: Optional[dict[str, Any]] = None,
    ) -> Union[str, async_job.AsyncJob]:
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
            max_instances: The maximum number of inference service instances to run. The same value it set to
                MIN_INSTANCES property of the service.
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
            experimental_options: Experimental options for the service creation with custom inference engine.
                Currently, only `inference_engine` and `inference_engine_args_override` are supported.
                `inference_engine` is the name of the inference engine to use.
                `inference_engine_args_override` is a list of string arguments to pass to the inference engine.
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
        image_build_compute_pool: Optional[str] = None,
        service_compute_pool: str,
        image_repo: Optional[str] = None,
        ingress_enabled: bool = False,
        max_instances: int = 1,
        cpu_requests: Optional[str] = None,
        memory_requests: Optional[str] = None,
        gpu_requests: Optional[Union[str, int]] = None,
        num_workers: Optional[int] = None,
        max_batch_rows: Optional[int] = None,
        force_rebuild: bool = False,
        build_external_access_integration: Optional[str] = None,
        build_external_access_integrations: Optional[list[str]] = None,
        block: bool = True,
        experimental_options: Optional[dict[str, Any]] = None,
    ) -> Union[str, async_job.AsyncJob]:
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
            max_instances: The maximum number of inference service instances to run. The same value it set to
                MIN_INSTANCES property of the service.
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
            experimental_options: Experimental options for the service creation with custom inference engine.
                Currently, only `inference_engine` and `inference_engine_args_override` are supported.
                `inference_engine` is the name of the inference engine to use.
                `inference_engine_args_override` is a list of string arguments to pass to the inference engine.


        Raises:
            ValueError: Illegal external access integration arguments.
            exceptions.SnowparkSQLException: if service already exists.

        Returns:
            If `block=True`, return result information about service creation from server.
            Otherwise, return the service creation AsyncJob.

        Raises:
            ValueError: Illegal external access integration arguments.
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

        # Check if model is HuggingFace text-generation before doing inference engine checks
        if experimental_options:
            self._check_huggingface_text_generation_model(statement_params)

        inference_engine_args = inference_engine_utils._get_inference_engine_args(experimental_options)

        # Enrich inference engine args if inference engine is specified
        if inference_engine_args is not None:
            inference_engine_args = inference_engine_utils._enrich_inference_engine_args(
                inference_engine_args,
                gpu_requests,
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
            List of service_names: The name of the service, can be fully qualified. If not fully qualified, the database
                or schema of the model will be used.
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
