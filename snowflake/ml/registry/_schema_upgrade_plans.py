from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from snowflake import snowpark
from snowflake.ml._internal.utils import table_manager
from snowflake.ml.registry import _initial_schema


class BaseSchemaUpgradePlans(ABC):
    """Abstract Class for specifying schema upgrades for registry."""

    def __init__(
        self,
        session: snowpark.Session,
        database_name: str,
        schema_name: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._session = session
        self._database = database_name
        self._schema = schema_name
        self._statement_params = statement_params

    @abstractmethod
    def upgrade(self) -> None:
        """Convert schema from previous version to `_current_version`."""
        pass


class AddTrainingDatasetIdIfNotExists(BaseSchemaUpgradePlans):
    """Add Column TRAINING_DATASET_ID in registry schema table."""

    def __init__(
        self,
        session: snowpark.Session,
        database_name: str,
        schema_name: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(session, database_name, schema_name, statement_params)

    def upgrade(self) -> None:
        full_schema_path = f"{self._database}.{self._schema}"
        table_schema_dict = table_manager.get_table_schema(
            self._session, _initial_schema._MODELS_TABLE_NAME, full_schema_path
        )
        new_column = "TRAINING_DATASET_ID"
        if new_column not in table_schema_dict:
            self._session.sql(
                f"""ALTER TABLE {self._database}.{self._schema}.{_initial_schema._MODELS_TABLE_NAME}
                    ADD COLUMN {new_column} VARCHAR
                """
            ).collect(statement_params=self._statement_params)


class ReplaceTrainingDatasetIdWithArtifactIds(BaseSchemaUpgradePlans):
    """Drop column `TRAINING_DATASET_ID`, add `ARTIFACT_IDS`."""

    def __init__(
        self,
        session: snowpark.Session,
        database_name: str,
        schema_name: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(session, database_name, schema_name, statement_params)

    def upgrade(self) -> None:
        full_schema_path = f"{self._database}.{self._schema}"
        old_column = "TRAINING_DATASET_ID"
        self._session.sql(
            f"""ALTER TABLE {full_schema_path}.{_initial_schema._MODELS_TABLE_NAME}
                DROP COLUMN {old_column}
            """
        ).collect(statement_params=self._statement_params)

        new_column = "ARTIFACT_IDS"
        self._session.sql(
            f"""ALTER TABLE {full_schema_path}.{_initial_schema._MODELS_TABLE_NAME}
                ADD COLUMN {new_column} ARRAY
            """
        ).collect(statement_params=self._statement_params)


class ChangeArtifactSpecFromObjectToVarchar(BaseSchemaUpgradePlans):
    """Change artifact spec type from object to varchar. It's fine to drop the column as it's empty."""

    def __init__(
        self,
        session: snowpark.Session,
        database_name: str,
        schema_name: str,
        statement_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(session, database_name, schema_name, statement_params)

    def upgrade(self) -> None:
        full_schema_path = f"{self._database}.{self._schema}"
        update_col = "ARTIFACT_SPEC"
        self._session.sql(
            f"""ALTER TABLE {full_schema_path}.{_initial_schema._ARTIFACT_TABLE_NAME}
                DROP COLUMN {update_col}
            """
        ).collect(statement_params=self._statement_params)

        self._session.sql(
            f"""ALTER TABLE {full_schema_path}.{_initial_schema._ARTIFACT_TABLE_NAME}
                ADD COLUMN {update_col} VARCHAR
            """
        ).collect(statement_params=self._statement_params)

        self._session.sql(
            f"""COMMENT ON COLUMN {full_schema_path}.{_initial_schema._ARTIFACT_TABLE_NAME}.{update_col} IS
                'This column is VARCHAR but supposed to store a valid JSON object'
            """
        ).collect(statement_params=self._statement_params)
