from typing import List, Union

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._client.ops import model_ops

_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"


class Model:
    """Model Object containing multiple versions. Mapping to SQL's MODEL object."""

    _model_ops: model_ops.ModelOperator
    _model_name: sql_identifier.SqlIdentifier

    def __init__(self) -> None:
        raise RuntimeError("Model's initializer is not meant to be used. Use `get_model` from registry instead.")

    @classmethod
    def _ref(
        cls,
        model_ops: model_ops.ModelOperator,
        *,
        model_name: sql_identifier.SqlIdentifier,
    ) -> "Model":
        self: "Model" = object.__new__(cls)
        self._model_ops = model_ops
        self._model_name = model_name
        return self

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Model):
            return False
        return self._model_ops == __value._model_ops and self._model_name == __value._model_name

    @property
    def name(self) -> str:
        return self._model_name.identifier()

    @property
    def fully_qualified_name(self) -> str:
        return self._model_ops._model_version_client.fully_qualified_model_name(self._model_name)

    @property
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def description(self) -> str:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops.get_comment(
            model_name=self._model_name,
            statement_params=statement_params,
        )

    @description.setter
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def description(self, description: str) -> None:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        return self._model_ops.set_comment(
            comment=description,
            model_name=self._model_name,
            statement_params=statement_params,
        )

    @property
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def default(self) -> model_version_impl.ModelVersion:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
            class_name=self.__class__.__name__,
        )
        default_version_name = self._model_ops._model_version_client.get_default_version(
            model_name=self._model_name, statement_params=statement_params
        )
        return self.version(default_version_name)

    @default.setter
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def default(self, version: Union[str, model_version_impl.ModelVersion]) -> None:
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
            class_name=self.__class__.__name__,
        )
        if isinstance(version, str):
            version_name = sql_identifier.SqlIdentifier(version)
        else:
            version_name = version._version_name
        self._model_ops._model_version_client.set_default_version(
            model_name=self._model_name, version_name=version_name, statement_params=statement_params
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def version(self, version_name: str) -> model_version_impl.ModelVersion:
        """Get a model version object given a version name in the model.

        Args:
            version_name: The name of version

        Raises:
            ValueError: Raised when the version requested does not exist.

        Returns:
            The model version object.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        version_id = sql_identifier.SqlIdentifier(version_name)
        if self._model_ops.validate_existence(
            model_name=self._model_name,
            version_name=version_id,
            statement_params=statement_params,
        ):
            return model_version_impl.ModelVersion._ref(
                self._model_ops,
                model_name=self._model_name,
                version_name=version_id,
            )
        else:
            raise ValueError(
                f"Unable to find version with name {version_id.identifier()} in model {self.fully_qualified_name}"
            )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def list_versions(self) -> List[model_version_impl.ModelVersion]:
        """List all versions in the model.

        Returns:
            A List of ModelVersion object representing all versions in the model.
        """
        statement_params = telemetry.get_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
        )
        version_names = self._model_ops.list_models_or_versions(
            model_name=self._model_name,
            statement_params=statement_params,
        )
        return [
            model_version_impl.ModelVersion._ref(
                self._model_ops,
                model_name=self._model_name,
                version_name=version_name,
            )
            for version_name in version_names
        ]

    def delete_version(self, version_name: str) -> None:
        raise NotImplementedError("Deleting version has not been supported yet.")
