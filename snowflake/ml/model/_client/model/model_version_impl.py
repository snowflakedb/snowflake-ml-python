import re
from typing import Callable, List, Optional, Union

import pandas as pd

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model import model_signature
from snowflake.ml.model._client.model import model_method_info
from snowflake.ml.model._client.ops import model_ops
from snowflake.snowpark import dataframe, session

_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"


class ModelVersion:
    """Model Version Object representing a specific version of the model that could be run."""

    def __init__(
        self,
        model_ops: model_ops.ModelOperator,
        *,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
    ) -> None:
        self._model_ops = model_ops
        self._model_name = model_name
        self._version_name = version_name

    @classmethod
    def create(
        cls,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        model_name: sql_identifier.SqlIdentifier,
        version_name: sql_identifier.SqlIdentifier,
    ) -> "ModelVersion":
        return cls(
            model_ops.ModelOperator(session, database_name=database_name, schema_name=schema_name),
            model_name=model_name,
            version_name=version_name,
        )

    @property
    def model_name(self) -> str:
        return self._model_name.identifier()

    @property
    def version_name(self) -> str:
        return self._version_name.identifier()

    @property
    def fully_qualified_model_name(self) -> str:
        return self._model_ops._model_version_client.fully_qualified_model_name(self._model_name)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def list_methods(self) -> List[model_method_info.ModelMethodInfo]:
        """List all method information in a model version that is callable.

        Returns:
            A list of ModelMethodInfo object containing the following information:
            - name: The name of the method to be called (both in SQL and in Python SDK).
            - target_method: The original method name in the logged Python object.
            - Signature: Python signature of the original method.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        # TODO(SNOW-986673, SNOW-986675): Avoid parsing manifest and meta file and put Python signature into user_data.
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
        return_methods_info: List[model_method_info.ModelMethodInfo] = []
        for method in manifest["methods"]:
            # Method's name is resolved so we need to use case_sensitive as True to get the user-facing identifier.
            method_name = sql_identifier.SqlIdentifier(method["name"], case_sensitive=True).identifier()
            # Method's handler is `functions.<target_method>.infer`
            assert re.match(
                r"^functions\.([^\d\W]\w*)\.infer$", method["handler"]
            ), f"Get unexpected handler name {method['handler']}"
            target_method = method["handler"].split(".")[1]
            signature_dict = model_meta["signatures"][target_method]
            method_info = model_method_info.ModelMethodInfo(
                name=method_name,
                target_method=target_method,
                signature=model_signature.ModelSignature.from_dict(signature_dict),
            )
            return_methods_info.append(method_info)

        return return_methods_info

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def run(
        self,
        X: Union[pd.DataFrame, dataframe.DataFrame],
        *,
        method_name: Optional[str] = None,
    ) -> Union[pd.DataFrame, dataframe.DataFrame]:
        """Invoke a method in a model version object

        Args:
            X: The input data. Could be pandas DataFrame or Snowpark DataFrame
            method_name: The method name to run. It is the name you will use to call a method in SQL. Defaults to None.
                It can only be None if there is only 1 method.

        Raises:
            ValueError: No method with the corresponding name is available.
            ValueError: There are more than 1 target methods available in the model but no method name specified.

        Returns:
            The prediction data.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )

        methods: List[model_method_info.ModelMethodInfo] = self.list_methods()
        if method_name:
            req_method_name = sql_identifier.SqlIdentifier(method_name).identifier()
            find_method: Callable[[model_method_info.ModelMethodInfo], bool] = (
                lambda method: method["name"] == req_method_name
            )
            target_method_info = next(
                filter(find_method, methods),
                None,
            )
            if target_method_info is None:
                raise ValueError(
                    f"There is no method with name {method_name} available in the model"
                    f" {self.fully_qualified_model_name} version {self.version_name}"
                )
        elif len(methods) != 1:
            raise ValueError(
                f"There are more than 1 target methods available in the model {self.fully_qualified_model_name}"
                f" version {self.version_name}. Please specify a `method_name` when calling the `run` method."
            )
        else:
            target_method_info = methods[0]
        return self._model_ops.invoke_method(
            method_name=sql_identifier.SqlIdentifier(target_method_info["name"]),
            signature=target_method_info["signature"],
            X=X,
            model_name=self._model_name,
            version_name=self._version_name,
            statement_params=statement_params,
        )
