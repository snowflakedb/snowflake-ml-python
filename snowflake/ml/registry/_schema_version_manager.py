from typing import Any, Dict, List, Optional, Tuple

from snowflake import snowpark
from snowflake.ml._internal.utils import identifier, query_result_checker, table_manager
from snowflake.ml.registry import _initial_schema, _schema

_SCHEMA_VERSION_TABLE_NAME: str = "_SYSTEM_REGISTRY_SCHEMA_VERSION"

_SCHEMA_VERSION_TABLE_SCHEMA: List[Tuple[str, str]] = [
    ("VERSION", "NUMBER"),
    ("CREATION_TIME", "TIMESTAMP_TZ"),
    ("INFO", "OBJECT"),
]


class SchemaVersionManager:
    """Registry Schema Version Manager to deal with schema evolution."""

    def __init__(self, session: snowpark.Session, database: str, schema: str) -> None:
        """SchemaVersionManager constructor.

        Args:
            session: Snowpark session
            database: Database in which registry is being managed.
            schema: Schema in which registry is being managed.
        """
        self._session = session
        self._database = database
        self._schema = schema

    def get_deployed_version(self, statement_params: Optional[Dict[str, Any]] = None) -> int:
        """Get current version of deployed schema.

        Args:
            statement_params: Statement parameters for telemetry tracking.

        Returns:
            Version of deployed schema.
        """
        if not table_manager.validate_table_exist(
            self._session, _SCHEMA_VERSION_TABLE_NAME, self._get_qualified_schema()
        ):
            return _initial_schema._INITIAL_VERSION

        result = (
            query_result_checker.SqlResultValidator(
                session=self._session,
                query=f"""SELECT MAX(VERSION) AS MAX_VERSION
                        FROM {self._get_qualified_schema()}.{_SCHEMA_VERSION_TABLE_NAME}
                        """,
                statement_params=statement_params,
            )
            .has_dimensions(expected_rows=1, expected_cols=1)
            .has_column("MAX_VERSION")
            .validate()
        )
        cur_version = result[0]["MAX_VERSION"]
        return int(cur_version)

    def validate_schema_version(self, statement_params: Optional[Dict[str, Any]] = None) -> None:
        """Checks if currently deployed schema is up to date.

        Args:
            statement_params: Statement parameters for telemetry tracking.

        Raises:
            RuntimeError: if deployed schema different from registry schema.
        """
        deployed_version = self.get_deployed_version(statement_params)
        if deployed_version > _schema._CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Deployed registry schema version ({deployed_version}) is ahead of current "
                f"package ({_schema._CURRENT_SCHEMA_VERSION}). Please update the package."
            )
        elif deployed_version < _schema._CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Registry schema version ({_schema._CURRENT_SCHEMA_VERSION}) is ahead of deployed "
                f"schema ({deployed_version}). Please call create_model_registry() to upgrade."
            )

    def try_upgrade(self, statement_params: Optional[Dict[str, Any]] = None) -> None:
        """Upgrade deployed schema to current.

        Args:
            statement_params: Statement parameters for telemetry tracking.

        Raises:
            RuntimeError: Deployed schema is newer than package.
        """
        deployed_version = self.get_deployed_version(statement_params)
        if deployed_version > _schema._CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Deployed registry schema version ({deployed_version}) is ahead of current "
                f"package ({_schema._CURRENT_SCHEMA_VERSION}). Please update the package."
            )

        any_upgraded = False
        for cur_version in range(deployed_version, _schema._CURRENT_SCHEMA_VERSION):
            assert cur_version + 1 in _schema._SCHEMA_UPGRADE_PLANS, "version number not exist."
            plan = _schema._SCHEMA_UPGRADE_PLANS[cur_version + 1](
                self._session, self._database, self._schema, statement_params
            )
            plan.upgrade()
            any_upgraded = True

        self._validate_schema()

        if any_upgraded:
            self._create_or_update_version_table(statement_params)

    def _create_or_update_version_table(self, statement_params: Optional[Dict[str, Any]] = None) -> None:
        if not table_manager.validate_table_exist(
            self._session, _SCHEMA_VERSION_TABLE_NAME, self._get_qualified_schema()
        ):
            table_manager.create_single_table(
                session=self._session,
                database_name=self._database,
                schema_name=self._schema,
                table_name=_SCHEMA_VERSION_TABLE_NAME,
                table_schema=_SCHEMA_VERSION_TABLE_SCHEMA,
                statement_params=statement_params,
            )
        query_result_checker.SqlResultValidator(
            session=self._session,
            query=f"""INSERT INTO {self._get_qualified_schema_version_table()} (VERSION, CREATION_TIME)
                VALUES ({_schema._CURRENT_SCHEMA_VERSION}, CURRENT_TIMESTAMP())
            """,
            statement_params=statement_params,
        ).insertion_success(expected_num_rows=1).validate()

    def _validate_schema(self) -> None:
        for table_name in _initial_schema._INITIAL_TABLE_SCHEMAS:
            if table_name not in _schema._CURRENT_TABLE_SCHEMAS:
                # This table must be deleted by transformations.
                if table_manager.validate_table_exist(
                    self._session,
                    table_name,
                    table_manager.get_fully_qualified_schema_name(self._database, self._schema),
                ):
                    raise RuntimeError(
                        f"Schema transformation error. A table '{table_name}' found, which should not exist."
                    )

        exclude_cols = ["PRIMARY KEY"]
        for table_name, expected_schema in _schema._CURRENT_TABLE_SCHEMAS.items():
            deployed_schema_dict = table_manager.get_table_schema(
                self._session, table_name, self._get_qualified_schema()
            )

            # TODO check type as well.
            for col_name, _ in expected_schema:
                if col_name not in deployed_schema_dict and col_name not in exclude_cols:
                    raise RuntimeError(f"Schema table: {table_name} doesn't have required column:'{col_name}'.")

    def _get_qualified_schema(self) -> str:
        return table_manager.get_fully_qualified_schema_name(self._database, self._schema)

    def _get_qualified_schema_version_table(self) -> str:
        return table_manager.get_fully_qualified_table_name(
            self._database,
            self._schema,
            identifier.get_inferred_name(_SCHEMA_VERSION_TABLE_NAME),
        )
