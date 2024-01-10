import re
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from snowflake import connector
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature
from snowflake.ml.model._client.ops import metadata_ops, model_ops
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.snowpark import dataframe

_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"


class ModelVersion:
    """Model Version Object representing a specific version of the model that could be run."""

    _model_ops: model_ops.ModelOperator
    _model_name: sql_identifier.SqlIdentifier
    _version_name: sql_identifier.SqlIdentifier

    def __init__(self) -> None:
        raise RuntimeError("ModelVersion's initializer is not meant to be used. Use `version` from model instead.")

    @classmethod
    def _ref(
        cls,
        model_ops: model_ops.ModelOperator,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
    ) -> "ModelVersion":
        self: "ModelVersion" = object.__new__(cls)
        self._model_ops = model_ops
        self._model_name = model_name
        self._version_name = version_name
        return self

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ModelVersion):
            return False
        return (
            self._model_ops == __value._model_ops
            and self._model_name == __value._model_name
            and self._version_name == __value._version_name
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
        return self._model_ops._model_version_client.fully_qualified_model_name(self._model_name)

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
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def show_metrics(self) -> Dict[str, Any]:
        """Show all metrics logged with the model version.

        Returns:
            A dictionary showing the metrics.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops._metadata_ops.load(
            model_name=self._model_name, version_name=self._version_name, statement_params=statement_params
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
            model_name=self._model_name,
            version_name=self._version_name,
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
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )

    # Only used when the model does not contains user_data with client SDK information.
    def _legacy_show_functions(self) -> List[model_manifest_schema.ModelFunctionInfo]:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        manifest = self._model_ops.get_model_version_manifest(
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )
        model_meta = self._model_ops.get_model_version_native_packing_meta(
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )
        return_functions_info: List[model_manifest_schema.ModelFunctionInfo] = []
        for method in manifest["methods"]:
            # Method's name is resolved so we need to use case_sensitive as True to get the user-facing identifier.
            method_name = sql_identifier.SqlIdentifier(method["name"], case_sensitive=True).identifier()
            # Method's handler is `functions.<target_method>.infer`
            assert re.match(
                r"^functions\.([^\d\W]\w*)\.infer$", method["handler"]
            ), f"Get unexpected handler name {method['handler']}"
            target_method = method["handler"].split(".")[1]
            signature_dict = model_meta["signatures"][target_method]
            fi = model_manifest_schema.ModelFunctionInfo(
                name=method_name,
                target_method=target_method,
                signature=model_signature.ModelSignature.from_dict(signature_dict),
            )
            return_functions_info.append(fi)
        return return_functions_info

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def show_functions(self) -> List[model_manifest_schema.ModelFunctionInfo]:
        """Show all functions information in a model version that is callable.

        Returns:
            A list of ModelFunctionInfo objects containing the following information:

            - name: The name of the function to be called (both in SQL and in Python SDK).
            - target_method: The original method name in the logged Python object.
            - signature: Python signature of the original method.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        try:
            client_data = self._model_ops.get_client_data_in_user_data(
                model_name=self._model_name,
                version_name=self._version_name,
                statement_params=statement_params,
            )
            return [
                model_manifest_schema.ModelFunctionInfo(
                    name=fi["name"],
                    target_method=fi["target_method"],
                    signature=model_signature.ModelSignature.from_dict(fi["signature"]),
                )
                for fi in client_data["functions"]
            ]
        except (NotImplementedError, ValueError, connector.DataError):
            return self._legacy_show_functions()

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def run(
        self,
        X: Union[pd.DataFrame, dataframe.DataFrame],
        *,
        function_name: Optional[str] = None,
    ) -> Union[pd.DataFrame, dataframe.DataFrame]:
        """Invoke a method in a model version object.

        Args:
            X: The input data, which could be a pandas DataFrame or Snowpark DataFrame.
            function_name: The function name to run. It is the name used to call a function in SQL.
                Defaults to None. It can only be None if there is only 1 method.

        Raises:
            ValueError: When no method with the corresponding name is available.
            ValueError: When there are more than 1 target methods available in the model but no function name specified.

        Returns:
            The prediction data. It would be the same type dataframe as your input.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )

        functions: List[model_manifest_schema.ModelFunctionInfo] = self.show_functions()
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
                f" version {self.version_name}. Please specify a `method_name` when calling the `run` method."
            )
        else:
            target_function_info = functions[0]
        return self._model_ops.invoke_method(
            method_name=sql_identifier.SqlIdentifier(target_function_info["name"]),
            signature=target_function_info["signature"],
            X=X,
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )
