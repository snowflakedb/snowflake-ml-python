from typing import List

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._client.ops import model_ops
from snowflake.snowpark import session

_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelManagement"


class Model:
    """Model Object containing multiple versions. Mapping to SQL's MODEL object."""

    def __init__(
        self,
        model_ops: model_ops.ModelOperator,
        *,
        model_name: sql_identifier.SqlIdentifier,
    ) -> None:
        self._model_ops = model_ops
        self._model_name = model_name

    @classmethod
    def create(
        cls,
        session: session.Session,
        *,
        database_name: sql_identifier.SqlIdentifier,
        schema_name: sql_identifier.SqlIdentifier,
        model_name: sql_identifier.SqlIdentifier,
    ) -> "Model":
        return cls(
            model_ops.ModelOperator(session, database_name=database_name, schema_name=schema_name),
            model_name=model_name,
        )

    @property
    def name(self) -> str:
        return self._model_name.identifier()

    @property
    def fully_qualified_name(self) -> str:
        return self._model_ops._model_version_client.fully_qualified_model_name(self._model_name)

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
            return model_version_impl.ModelVersion(
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
            model_version_impl.ModelVersion(
                self._model_ops,
                model_name=self._model_name,
                version_name=version_name,
            )
            for version_name in version_names
        ]
