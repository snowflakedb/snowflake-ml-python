from snowflake.ml._internal.utils import sql_identifier
from snowflake.ml.model._client.model import model_version_impl
from snowflake.ml.model._client.ops import model_ops
from snowflake.snowpark import session


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

    def version(self, version_name: str) -> model_version_impl.ModelVersion:
        """Get a model version object given a version name in the model.

        Args:
            version_name: The name of version

        Returns:
            The model version object.
        """
        return model_version_impl.ModelVersion(
            self._model_ops,
            model_name=self._model_name,
            version_name=sql_identifier.SqlIdentifier(version_name),
        )
