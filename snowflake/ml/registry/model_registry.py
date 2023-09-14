import inspect
import json
import os
import posixpath
import sys
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from uuid import uuid1

from absl import logging

from snowflake import connector, snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.utils import (
    formatting,
    identifier,
    query_result_checker,
    table_manager,
    uri,
)
from snowflake.ml.model import (
    _deployer,
    _model as model_api,
    deploy_platforms,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.registry import _ml_artifact, _schema
from snowflake.ml.training_dataset import training_dataset
from snowflake.snowpark._internal import utils as snowpark_utils

if TYPE_CHECKING:
    import pandas as pd

_DEFAULT_REGISTRY_NAME: str = "_SYSTEM_MODEL_REGISTRY"
_DEFAULT_SCHEMA_NAME: str = "_SYSTEM_MODEL_REGISTRY_SCHEMA"
_MODELS_TABLE_NAME: str = "_SYSTEM_REGISTRY_MODELS"
_METADATA_TABLE_NAME: str = "_SYSTEM_REGISTRY_METADATA"
_DEPLOYMENT_TABLE_NAME: str = "_SYSTEM_REGISTRY_DEPLOYMENTS"

# Metadata operation types.
_SET_METADATA_OPERATION: str = "SET"
_ADD_METADATA_OPERATION: str = "ADD"
_DROP_METADATA_OPERATION: str = "DROP"

# Metadata types.
_METADATA_ATTRIBUTE_DESCRIPTION: str = "DESCRIPTION"
_METADATA_ATTRIBUTE_METRICS: str = "METRICS"
_METADATA_ATTRIBUTE_REGISTRATION: str = "REGISTRATION"
_METADATA_ATTRIBUTE_TAGS: str = "TAGS"
_METADATA_ATTRIBUTE_DEPLOYMENT: str = "DEPLOYMENTS"
_METADATA_ATTRIBUTE_DELETION: str = "DELETION"

# Leaving out REGISTRATION/DEPLOYMENT events as they will be handled differently from all mutable attributes.
_LIST_METADATA_ATTRIBUTE: List[str] = [
    _METADATA_ATTRIBUTE_DESCRIPTION,
    _METADATA_ATTRIBUTE_METRICS,
    _METADATA_ATTRIBUTE_TAGS,
]
_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelRegistry"

_STAGE_PREFIX = "@"


def _create_registry_database(
    session: snowpark.Session,
    database_name: str,
    statement_params: Dict[str, Any],
) -> None:
    """Private helper to create the model registry database.

    The creation will be skipped if the target database already exists.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        statement_params: Function usage statement parameters used in sql query executions.
    """
    registry_databases = session.sql(f"SHOW DATABASES LIKE '{identifier.get_unescaped_names(database_name)}'").collect(
        statement_params=statement_params
    )
    if len(registry_databases) > 0:
        logging.warning(f"The database {database_name} already exists. Skipping creation.")
        return

    session.sql(f"CREATE DATABASE {database_name}").collect(statement_params=statement_params)


def _create_registry_schema(
    session: snowpark.Session, database_name: str, schema_name: str, statement_params: Dict[str, Any]
) -> None:
    """Private helper to create the model registry schema.

    The creation will be skipped if the target schema already exists.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.
        statement_params: Function usage statement parameters used in sql query executions.
    """
    # The default PUBLIC schema is created by default so it might already exist even in a new database.
    registry_schemas = session.sql(
        f"SHOW SCHEMAS LIKE '{identifier.get_unescaped_names(schema_name)}' IN DATABASE {database_name}"
    ).collect(statement_params=statement_params)

    if len(registry_schemas) > 0:
        logging.warning(
            f"The schema {table_manager.get_fully_qualified_schema_name(database_name, schema_name)} already exists. "
            + "Skipping creation."
        )
        return

    session.sql(f"CREATE SCHEMA {table_manager.get_fully_qualified_schema_name(database_name, schema_name)}").collect(
        statement_params=statement_params
    )


def _create_registry_tables(
    session: snowpark.Session,
    database_name: str,
    schema_name: str,
    registry_table_name: str,
    metadata_table_name: str,
    deployment_table_name: str,
    artifact_table_name: str,
    statement_params: Dict[str, Any],
) -> None:
    """Private helper to create the model registry required tables.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.
        registry_table_name: Name for the main model registry table.
        metadata_table_name: Name for the metadata table used by the model registry.
        deployment_table_name: Name for the deployment event table.
        artifact_table_name: Name for the artifact table.
        statement_params: Function usage statement parameters used in sql query executions.
    """

    # Create model registry table to store immutable properties of models
    fully_qualified_registry_table_name = table_manager.create_single_registry_table(
        session=session,
        database_name=database_name,
        schema_name=schema_name,
        table_name=registry_table_name,
        table_schema=_schema._REGISTRY_TABLE_SCHEMA,
        statement_params=statement_params,
    )

    # Create model metadata table to store mutable properties of models
    metadata_table_schema = [
        (k, v.format(registry_table_name=fully_qualified_registry_table_name))
        for k, v in _schema._METADATA_TABLE_SCHEMA
    ]
    table_manager.create_single_registry_table(
        session=session,
        database_name=database_name,
        schema_name=schema_name,
        table_name=metadata_table_name,
        table_schema=metadata_table_schema,
        statement_params=statement_params,
    )

    # Create model deployment table to store deployment events of models
    deployment_table_schema = [
        (k, v.format(registry_table_name=fully_qualified_registry_table_name))
        for k, v in _schema._DEPLOYMENTS_TABLE_SCHEMA
    ]
    table_manager.create_single_registry_table(
        session=session,
        database_name=database_name,
        schema_name=schema_name,
        table_name=deployment_table_name,
        table_schema=deployment_table_schema,
        statement_params=statement_params,
    )

    _ml_artifact.create_ml_artifact_table(
        session=session,
        database_name=database_name,
        schema_name=schema_name,
        statement_params=statement_params,
    )


def _create_registry_views(
    session: snowpark.Session,
    database_name: str,
    schema_name: str,
    registry_table_name: str,
    metadata_table_name: str,
    deployment_table_name: str,
    artifact_table_name: str,
    statement_params: Dict[str, Any],
) -> None:
    """Create views on underlying ModelRegistry tables.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.
        registry_table_name: Name for the main model registry table.
        metadata_table_name: Name for the metadata table used by the model registry.
        deployment_table_name: Name for the deployment event table.
        artifact_table_name: Name for the artifact table.
        statement_params: Function usage statement parameters used in sql query executions.
    """
    fully_qualified_schema_name = table_manager.get_fully_qualified_schema_name(database_name, schema_name)

    # From the documentation: Each DDL statement executes as a separate transaction. Races should not be an issue.
    # https://docs.snowflake.com/en/sql-reference/transactions.html#ddl

    # Create a view on active permanent deployments.
    _create_active_permanent_deployment_view(
        session, fully_qualified_schema_name, registry_table_name, deployment_table_name, statement_params
    )

    # Create views on most recent metadata items.
    metadata_view_name_prefix = identifier.concat_names([metadata_table_name, "_LAST_"])
    metadata_view_template = formatting.unwrap(
        """CREATE OR REPLACE VIEW {database}.{schema}.{attribute_view} COPY GRANTS AS
            SELECT DISTINCT MODEL_ID, {select_expression} AS {final_attribute_name} FROM {metadata_table}
            WHERE ATTRIBUTE_NAME = '{attribute_name}'"""
    )

    # Create a separate view for the most recent item in each metadata column.
    metadata_view_names = []
    metadata_select_fields = []
    for attribute_name in _LIST_METADATA_ATTRIBUTE:
        view_name = identifier.concat_names([metadata_view_name_prefix, attribute_name])
        select_expression = f"(LAST_VALUE(VALUE) OVER (PARTITION BY MODEL_ID ORDER BY SEQUENCE_ID))['{attribute_name}']"
        sql = metadata_view_template.format(
            database=database_name,
            schema=schema_name,
            select_expression=select_expression,
            attribute_view=view_name,
            attribute_name=attribute_name,
            final_attribute_name=attribute_name,
            metadata_table=metadata_table_name,
        )
        session.sql(sql).collect(statement_params=statement_params)
        metadata_view_names.append(view_name)
        metadata_select_fields.append(f"{view_name}.{attribute_name} AS {attribute_name}")

    # Create a special view for the registration timestamp.
    attribute_name = _METADATA_ATTRIBUTE_REGISTRATION
    final_attribute_name = identifier.concat_names([attribute_name, "_TIMESTAMP"])
    view_name = identifier.concat_names([metadata_view_name_prefix, attribute_name])
    create_registration_view_sql = metadata_view_template.format(
        database=database_name,
        schema=schema_name,
        select_expression="EVENT_TIMESTAMP",
        attribute_view=view_name,
        attribute_name=attribute_name,
        final_attribute_name=final_attribute_name,
        metadata_table=metadata_table_name,
    )
    session.sql(create_registration_view_sql).collect(statement_params=statement_params)
    metadata_view_names.append(view_name)
    metadata_select_fields.append(f"{view_name}.{final_attribute_name} AS {final_attribute_name}")

    metadata_views_join = " ".join(
        [
            "LEFT JOIN {view} ON ({view}.MODEL_ID = {registry_table}.ID)".format(
                view=view, registry_table=registry_table_name
            )
            for view in metadata_view_names
        ]
    )

    # Create view to combine all attributes.
    registry_view_name = identifier.concat_names([registry_table_name, "_VIEW"])
    metadata_select_fields_formatted = ",".join(metadata_select_fields)
    session.sql(
        f"""CREATE OR REPLACE VIEW {fully_qualified_schema_name}.{registry_view_name} COPY GRANTS AS
                SELECT {registry_table_name}.*, {metadata_select_fields_formatted}
                FROM {registry_table_name} {metadata_views_join}"""
    ).collect(statement_params=statement_params)

    # Create artifact view. it joins artifact tables with registry table on model id.
    artifact_view_name = identifier.concat_names([artifact_table_name, "_VIEW"])
    session.sql(
        f"""CREATE OR REPLACE VIEW {fully_qualified_schema_name}.{artifact_view_name} COPY GRANTS AS
                SELECT
                    {registry_table_name}.NAME AS MODEL_NAME,
                    {registry_table_name}.VERSION AS MODEL_VERSION,
                    {artifact_table_name}.*
                FROM {registry_table_name}
                LEFT JOIN {artifact_table_name}
                ON ({registry_table_name}.TRAINING_DATASET_ID = {artifact_table_name}.ID)
                WHERE {artifact_table_name}.TYPE = 'TRAINING_DATASET'
        """
    ).collect(statement_params=statement_params)


def _create_active_permanent_deployment_view(
    session: snowpark.Session,
    fully_qualified_schema_name: str,
    registry_table_name: str,
    deployment_table_name: str,
    statement_params: Dict[str, Any],
) -> None:
    """Create a view which lists all available permanent deployments.

    Args:
        session: Session object to communicate with Snowflake.
        fully_qualified_schema_name: Schema name to the target table.
        registry_table_name: Name for the main model registry table.
        deployment_table_name: Name of the deployment table.
        statement_params: Function usage statement parameters used in sql query executions.
    """

    # Create a view on active permanent deployments
    # Active deployments are those whose last operation is not DROP.
    active_deployments_view_name = identifier.concat_names([deployment_table_name, "_VIEW"])
    active_deployments_view_expr = f"""
        CREATE OR REPLACE VIEW {fully_qualified_schema_name}.{active_deployments_view_name}
        COPY GRANTS AS
        SELECT
            DEPLOYMENT_NAME,
            MODEL_ID,
            {registry_table_name}.NAME as MODEL_NAME,
            {registry_table_name}.VERSION as MODEL_VERSION,
            {deployment_table_name}.CREATION_TIME as CREATION_TIME,
            TARGET_METHOD,
            TARGET_PLATFORM,
            SIGNATURE,
            OPTIONS,
            STAGE_PATH,
            ROLE
        FROM {deployment_table_name}
        LEFT JOIN {registry_table_name}
            ON {deployment_table_name}.MODEL_ID = {registry_table_name}.ID
    """
    session.sql(active_deployments_view_expr).collect(statement_params=statement_params)


class ModelRegistry:
    """Model Management API."""

    def __init__(
        self,
        *,
        session: snowpark.Session,
        database_name: str = _DEFAULT_REGISTRY_NAME,
        schema_name: str = _DEFAULT_SCHEMA_NAME,
        create_if_not_exists: bool = False,
    ) -> None:
        """
        Opens an already-created registry.

        Args:
            session: Session object to communicate with Snowflake.
            database_name: Desired name of the model registry database.
            schema_name: Desired name of the schema used by this model registry inside the database.
            create_if_not_exists: create model registry if it's not exists already.
        """
        statement_params = self._get_statement_params(inspect.currentframe())

        if create_if_not_exists:
            create_model_registry(session=session, database_name=database_name, schema_name=schema_name)

        self._name = identifier.get_inferred_name(database_name)
        self._schema = identifier.get_inferred_name(schema_name)
        self._registry_table = identifier.get_inferred_name(_MODELS_TABLE_NAME)
        self._registry_table_view = identifier.concat_names([self._registry_table, "_VIEW"])
        self._metadata_table = identifier.get_inferred_name(_METADATA_TABLE_NAME)
        self._deployment_table = identifier.get_inferred_name(_DEPLOYMENT_TABLE_NAME)
        self._permanent_deployment_view = identifier.concat_names([self._deployment_table, "_VIEW"])
        self._permanent_deployment_stage = identifier.concat_names([self._deployment_table, "_STAGE"])
        self._artifact_table = identifier.get_inferred_name(_ml_artifact._ARTIFACT_TABLE_NAME)
        self._artifact_view = identifier.concat_names([self._artifact_table, "_VIEW"])
        self._session = session

        # A in-memory deployment info cache to store information of temporary deployments
        # TODO(zhe): Use a temporary table to replace the in-memory cache.
        self._temporary_deployments: Dict[str, _deployer.Deployment] = {}

        self._check_access(statement_params)

    # Private methods

    def _check_access(self, statement_params: Dict[str, Any]) -> None:
        """Check access db/schema/tables."""
        # Check that the required tables exist and are accessible by the current role.

        query_result_checker.SqlResultValidator(
            self._session, query=f"SHOW DATABASES LIKE '{identifier.get_unescaped_names(self._name)}'"
        ).has_dimensions(expected_rows=1).validate()

        query_result_checker.SqlResultValidator(
            self._session,
            query=f"SHOW SCHEMAS LIKE '{identifier.get_unescaped_names(self._schema)}' IN DATABASE {self._name}",
        ).has_dimensions(expected_rows=1).validate()

        table_manager.validate_table_exist(
            self._session, identifier.get_unescaped_names(self._registry_table), self._fully_qualified_schema_name()
        )

        schema_remains_same = self._validate_registry_table_schema(add_if_not_exists=["TRAINING_DATASET_ID"])
        if not schema_remains_same:
            _create_registry_views(
                self._session,
                self._name,
                self._schema,
                self._registry_table,
                self._metadata_table,
                self._deployment_table,
                self._artifact_table,
                statement_params,
            )

        table_manager.validate_table_exist(
            self._session, identifier.get_unescaped_names(self._metadata_table), self._fully_qualified_schema_name()
        )

        table_manager.validate_table_exist(
            self._session, identifier.get_unescaped_names(self._deployment_table), self._fully_qualified_schema_name()
        )

        # TODO(zzhu): Also check validity of views.

    # TODO checks type as well. note type in _schema doesn't match with it appears in 'DESC TABLE'.
    # We need another layer of mapping. This function can also be extended to other tables as well.
    def _validate_registry_table_schema(self, add_if_not_exists: List[str]) -> bool:
        """Validate the table schema and check for any missing columns.

        Args:
            add_if_not_exists: column names that will be created if not found in existing tables.

        Returns:
            True if table schema remains, False otherwise.

        Raises:
            TypeError: required column not exists in schema table and not defined in add_if_not_exists.
        """

        valid_cols = [t[0] for t in _schema._REGISTRY_TABLE_SCHEMA]
        for k in add_if_not_exists:
            assert k in valid_cols

        actual_table_rows = self._session.sql(f"DESC TABLE {self._fully_qualified_registry_table_name()}").collect()
        actual_schema_dict = {}
        for row in actual_table_rows:
            actual_schema_dict[row["name"]] = row["type"]

        schema_remains_same = True
        for col_name, col_type in _schema._REGISTRY_TABLE_SCHEMA:
            if col_name not in actual_schema_dict:
                if col_name not in add_if_not_exists:
                    raise TypeError(
                        f"Registry table:{self._fully_qualified_registry_table_name()}"
                        f" doesn't have required column:'{col_name}'."
                    )
                else:
                    self._session.sql(
                        f"""ALTER TABLE {self._fully_qualified_registry_table_name()}
                                ADD COLUMN {col_name} {col_type}"""
                    ).collect()
                    schema_remains_same = False

        return schema_remains_same

    def _get_statement_params(self, frame: Optional[types.FrameType]) -> Dict[str, Any]:
        return telemetry.get_function_usage_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
            function_name=telemetry.get_statement_params_full_func_name(frame, "ModelRegistry"),
        )

    def _get_new_unique_identifier(self) -> str:
        """Create new unique identifier.

        Returns:
            String identifier."""
        return uuid1().hex

    def _fully_qualified_registry_table_name(self) -> str:
        """Get the fully qualified name to the current registry table."""
        return table_manager.get_fully_qualified_table_name(self._name, self._schema, self._registry_table)

    def _fully_qualified_registry_view_name(self) -> str:
        """Get the fully qualified name to the current registry view."""
        return table_manager.get_fully_qualified_table_name(self._name, self._schema, self._registry_table_view)

    def _fully_qualified_metadata_table_name(self) -> str:
        """Get the fully qualified name to the current metadata table."""
        return table_manager.get_fully_qualified_table_name(self._name, self._schema, self._metadata_table)

    def _fully_qualified_deployment_table_name(self) -> str:
        """Get the fully qualified name to the current deployment table."""
        return table_manager.get_fully_qualified_table_name(self._name, self._schema, self._deployment_table)

    def _fully_qualified_permanent_deployment_view_name(self) -> str:
        """Get the fully qualified name to the permanent deployment view."""
        return table_manager.get_fully_qualified_table_name(self._name, self._schema, self._permanent_deployment_view)

    def _fully_qualified_artifact_view_name(self) -> str:
        return table_manager.get_fully_qualified_table_name(self._name, self._schema, self._artifact_view)

    def _fully_qualified_schema_name(self) -> str:
        """Get the fully qualified name to the current registry schema."""
        return table_manager.get_fully_qualified_schema_name(self._name, self._schema)

    def _fully_qualified_deployment_name(self, deployment_name: str) -> str:
        """Get the fully qualified name to the given deployment."""
        return table_manager.get_fully_qualified_table_name(self._name, self._schema, deployment_name)

    def _insert_registry_entry(
        self, *, id: str, name: str, version: str, properties: Dict[str, Any]
    ) -> List[snowpark.Row]:
        """Insert a new row into the model registry table.

        Args:
            id: Model id to register.
            name: Model Name string.
            version: Model Version string.
            properties: Dictionary of properties corresponding to table columns.

        Returns:
            snowpark.Dataframe with the result of the operation.

        Raises:
            DataError: Mismatch between different id fields.
        """
        if not id:
            raise connector.DataError("Model ID is required but none given.")
        mandatory_args = {"ID": id, "NAME": name, "VERSION": version}
        for k, v in mandatory_args.items():
            if k not in properties:
                properties[k] = v
            else:
                if v and v != properties[k]:
                    raise connector.DataError(
                        formatting.unwrap(
                            f"""Parameter '{k.lower()}' is given and parameter 'properties' has the field '{k}' set but
                            the values do not match: {k.lower()}=="{v}" properties['{k}']=="{properties[k]}"."""
                        )
                    )
        # Could do a multi-table insert here with some pros and cons:
        # [PRO] Atomic insert across multiple tables.
        # [CON] Code logic becomes messy depending on which fields are set.
        # [CON] Harder to re-use existing methods like set_model_name.
        # Context: https://docs.snowflake.com/en/sql-reference/sql/insert-multi-table.html
        return table_manager.insert_table_entry(
            self._session, table=self._fully_qualified_registry_table_name(), columns=properties
        )

    def _insert_metadata_entry(self, *, id: str, attribute: str, value: Any, operation: str) -> List[snowpark.Row]:
        """Insert a new row into the model metadata table.

        Args:
            id: Model id to register.
            attribute: name of the metadata attribute
            value: new value of the metadata attribute
            operation: the operation type of the metadata entry.

        Returns:
            snowpark.DataFrame with the result of the operation.

        Raises:
            DataError: Missing ID field.
        """
        if not id:
            raise connector.DataError("Model ID is required but none given.")

        columns: Dict[str, Any] = {}
        columns["EVENT_TIMESTAMP"] = formatting.SqlStr("CURRENT_TIMESTAMP()")
        columns["EVENT_ID"] = self._get_new_unique_identifier()
        columns["MODEL_ID"] = id
        columns["ROLE"] = self._session.get_current_role()
        columns["OPERATION"] = operation
        columns["ATTRIBUTE_NAME"] = attribute
        columns["VALUE"] = value

        return table_manager.insert_table_entry(
            self._session, table=self._fully_qualified_metadata_table_name(), columns=columns
        )

    def _insert_deployment_entry(
        self,
        *,
        id: str,
        name: str,
        platform: str,
        stage_path: str,
        signature: Dict[str, Any],
        target_method: str,
        options: Optional[
            Union[model_types.WarehouseDeployOptions, model_types.SnowparkContainerServiceDeployOptions]
        ] = None,
    ) -> List[snowpark.Row]:
        """Insert a new row into the model deployment table.

        Each row in the deployment table is a deployment event.

        Args:
            id: Model id of the deployed model.
            name: Name of the deployment.
            platform: The deployment target destination.
            stage_path: The stage location where the deployment UDF is stored.
            signature: The model signature.
            target_method: The method name which is used for the deployment.
            options: The deployment options.

        Returns:
            A list of snowpark rows which is the insertion result.

        Raises:
            DataError: Missing ID field.
        """
        if not id:
            raise connector.DataError("Model ID is required but none given.")

        columns: Dict[str, Any] = {}
        columns["CREATION_TIME"] = formatting.SqlStr("CURRENT_TIMESTAMP()")
        columns["MODEL_ID"] = id
        columns["DEPLOYMENT_NAME"] = name
        columns["TARGET_PLATFORM"] = platform
        columns["STAGE_PATH"] = stage_path
        columns["ROLE"] = self._session.get_current_role()
        columns["SIGNATURE"] = signature
        columns["TARGET_METHOD"] = target_method
        columns["OPTIONS"] = options

        return table_manager.insert_table_entry(
            self._session, table=self._fully_qualified_deployment_table_name(), columns=columns
        )

    def _prepare_deployment_stage(self) -> str:
        """Create a stage in the model registry for storing all permanent deployments.

        Returns:
            Path to the stage that was created.
        """
        schema = self._fully_qualified_schema_name()
        fully_qualified_deployment_stage_name = f"{schema}.{self._permanent_deployment_stage}"
        statement_params = self._get_statement_params(inspect.currentframe())
        self._session.sql(
            f"CREATE STAGE IF NOT EXISTS {fully_qualified_deployment_stage_name} "
            f"ENCRYPTION = (TYPE= 'SNOWFLAKE_SSE')"
        ).collect(statement_params=statement_params)
        return f"@{fully_qualified_deployment_stage_name}"

    def _prepare_model_stage(self, model_id: str) -> str:
        """Create a stage in the model registry for storing the model with the given id.

        Creating a permanent stage here since we do not have a way to switch a stage from temporary to permanent.
        This can result in orphaned stages in case the process fails. It might be better to try to create a
        temporary stage, attempt to perform all operations and convert the temp stage into permanent once the
        operation is complete.

        Args:
            model_id: Internal model ID string.

        Returns:
            Name of the stage that was created.

        Raises:
            DatabaseError: Indicates that something went wrong when creating the stage.
        """
        schema = self._fully_qualified_schema_name()

        # Uppercase the model_stage_name to avoid having to quote the the stage name.
        stage_name = model_id.upper()

        model_stage_name = f"SNOWML_MODEL_{stage_name}"
        fully_qualified_model_stage_name = f"{schema}.{model_stage_name}"
        statement_params = self._get_statement_params(inspect.currentframe())

        create_stage_result = self._session.sql(
            f"CREATE OR REPLACE STAGE {fully_qualified_model_stage_name} ENCRYPTION = (TYPE= 'SNOWFLAKE_SSE')"
        ).collect(statement_params=statement_params)
        if not create_stage_result:
            raise connector.DatabaseError("Unable to create stage for model. Operation returned not result.")
        if len(create_stage_result) != 1:
            raise connector.DatabaseError(
                "Unable to create stage for model. Creating the model stage returned unexpected result: {}.".format(
                    str(create_stage_result)
                )
            )

        return fully_qualified_model_stage_name

    def _get_fully_qualified_stage_name_from_uri(self, model_uri: str) -> Optional[str]:
        """Get fully qualified stage path pointed by the URI.

        Args:
            model_uri: URI for which stage file is needed.

        Returns:
            The fully qualified Snowflake stage location encoded by the given URI. Returns None if the URI is not
                pointing to a Snowflake stage.
        """
        raw_stage_path = uri.get_snowflake_stage_path_from_uri(model_uri)
        if not raw_stage_path:
            return None
        (db, schema, stage, _) = identifier.parse_schema_level_object_identifier(raw_stage_path)
        return identifier.get_schema_level_object_identifier(db, schema, stage)

    def _list_selected_models(
        self, *, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> snowpark.DataFrame:
        """Retrieve the Snowpark dataframe of models matching the specified ID or (name and version).

        Args:
            id: Model ID string. Required if either name or version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if id is None.

        Returns:
            A Snowpark dataframe representing the models that match the given constraints.
        """
        models = self.list_models()

        if id:
            filtered_models = models.filter(snowpark.Column("ID") == id)
        else:
            self._model_identifier_is_nonempty_or_raise(model_name, model_version)

            # The following two asserts is to satisfy mypy.
            assert model_name
            assert model_version

            filtered_models = models.filter(snowpark.Column("NAME") == model_name).filter(
                snowpark.Column("VERSION") == model_version
            )

        return cast(snowpark.DataFrame, filtered_models)

    def _validate_exact_one_result(
        self, selected_model: snowpark.DataFrame, model_identifier: str
    ) -> List[snowpark.Row]:
        """Validate the filtered model has exactly one result.

        Args:
            selected_model: A snowpark dataframe representing the models that are filtered out.
            model_identifier: A string which is used to filter the model.

        Returns:
            A snowpark row which contains the metadata of the filtered model

        Raises:
            KeyError: The target model doesn't exist.
            DataError: The target model is not unique.
        """
        statement_params = self._get_statement_params(inspect.currentframe())
        model_info = None
        try:
            model_info = (
                query_result_checker.ResultValidator(result=selected_model.collect(statement_params=statement_params))
                .has_dimensions(expected_rows=1)
                .validate()
            )
        except connector.DataError:
            if model_info is None or len(model_info) == 0:
                raise KeyError(f"The model {model_identifier} does not exist in the current registry.")
            else:
                raise connector.DataError(
                    formatting.unwrap(
                        f"""There are {len(model_info)} models {model_identifier}. This might indicate a problem with
                            the integrity of the model registry data."""
                    )
                )
        return model_info

    def _get_metadata_attribute(
        self,
        attribute: str,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Any:
        """Get the value of the given metadata attribute for target model with given (model name + model version) or id.

        Args:
            attribute: Name of the attribute to get.
            id: Model ID string. Required if either name or version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            The value of the attribute that was requested. Can be None if the attribute is not set.
        """
        selected_models = self._list_selected_models(id=id, model_name=model_name, model_version=model_version)
        identifier = f"id {id}" if id else f"{model_name}/{model_version}"
        model_info = self._validate_exact_one_result(selected_models, identifier)
        return model_info[0][attribute]

    def _set_metadata_attribute(
        self,
        attribute: str,
        value: Any,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        operation: str = _SET_METADATA_OPERATION,
        enable_model_presence_check: bool = True,
    ) -> None:
        """Set the value of the given metadata attribute for target model with given (model name + model version) or id.

        Args:
            attribute: Name of the attribute to set.
            value: Value to set.
            id: Model ID string. Required if either name or version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.
            operation: the operation type of the metadata entry.
            enable_model_presence_check: If True, we will check if the model with the given ID is currently registered
                before setting the metadata attribute. False by default meaning that by default we will check.

        Raises:
            DataError: Failed to set the metadata attribute.
            KeyError: The target model doesn't exist
        """
        selected_models = self._list_selected_models(id=id, model_name=model_name, model_version=model_version)
        identifier = f"id {id}" if id else f"{model_name}/{model_version}"
        try:
            model_info = self._validate_exact_one_result(selected_models, identifier)
        except KeyError as e:
            # If the target model doesn't exist, raise the error only if enable_model_presence_check is True.
            if enable_model_presence_check:
                raise e

        if not id:
            id = model_info[0]["ID"]
        assert id is not None

        try:
            self._insert_metadata_entry(id=id, attribute=attribute, value={attribute: value}, operation=operation)
        except connector.DataError:
            raise connector.DataError(f"Setting {attribute} for mode id {id} failed.")

    def _model_identifier_is_nonempty_or_raise(self, model_name: Optional[str], model_version: Optional[str]) -> None:
        """Validate model_name and model_version are non-empty strings.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Raises:
            ValueError: Raised when either model_name and model_version is empty.
        """
        if not model_name or not model_version:
            raise ValueError("model_name and model_version have to be non-empty strings.")

    def _get_model_id(self, model_name: str, model_version: str) -> str:
        """Get ID of the model with the given (model name + model version).

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            Id of the model.

        Raises:
            DataError: The requested model could not be found.
        """
        result = self._get_metadata_attribute("ID", model_name=model_name, model_version=model_version)
        if not result:
            raise connector.DataError(f"Model {model_name}/{model_version} doesn't exist.")
        return str(result)

    def _get_model_path(
        self, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> str:
        """Get the stage path for the model with the given (model name + model version) or `id` from the registry.

        Args:
            id: Id of the model to deploy. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if id is None.

        Returns:
            str: Stage path for the model.

        Raises:
            DataError: When the model cannot be found or not be restored.
            NotImplementedError: For models that span multiple files.
        """
        statement_params = self._get_statement_params(inspect.currentframe())
        selected_models = self._list_selected_models(id=id, model_name=model_name, model_version=model_version)
        identifier = f"id {id}" if id else f"{model_name}/{model_version}"
        model_info = self._validate_exact_one_result(selected_models, identifier)
        if not id:
            id = model_info[0]["ID"]
        model_uri = model_info[0]["URI"]

        if not uri.is_snowflake_stage_uri(model_uri):
            raise connector.DataError(
                f"Artifacts with URI scheme {uri.get_uri_scheme(model_uri)} are currently not supported."
            )

        model_stage_path = self._get_fully_qualified_stage_name_from_uri(model_uri=model_uri)

        # Currently we assume only the model is on the stage.
        model_file_list = self._session.sql(f"LIST @{model_stage_path}").collect(statement_params=statement_params)
        if len(model_file_list) == 0:
            raise connector.DataError(f"No files in model artifact for id {id} located at {model_uri}.")
        if len(model_file_list) > 1:
            raise NotImplementedError("Restoring models consisting of multiple files is currently not supported.")
        return f"{self._fully_qualified_schema_name()}.{model_file_list[0].name}"

    def _log_model_path(
        self,
        model_name: str,
        model_version: str,
    ) -> Tuple[str, str]:
        """Generate a path in the Model Registry to store a model.

        Args:
            model_name: The given name for the model.
            model_version: Version string to be set for the model.

        Returns:
            String of the auto-generate unique model identifier and path to store it.
        """
        model_id = self._get_new_unique_identifier()

        # Copy model from local disk to remote stage.
        # TODO(zhe): Check if we could use the same stage for multiple models.
        fully_qualified_model_stage_name = self._prepare_model_stage(model_id=model_id)

        return model_id, fully_qualified_model_stage_name

    def _register_model_with_id(
        self,
        model_name: str,
        model_version: str,
        model_id: str,
        *,
        type: str,
        uri: str,
        input_spec: Optional[Dict[str, str]] = None,
        output_spec: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        training_dataset: Optional[training_dataset.TrainingDataset] = None,
    ) -> None:
        """Helper function to register model metadata.

        Args:
            model_name: Name to be set for the model. The model name can NOT be changed after registration. The
                combination of name and version is expected to be unique inside the registry.
            model_version: Version string to be set for the model. The model version string can NOT be changed after
                model registration. The combination of name and version is expected to be unique inside the registry.
            model_id: The internal id for the model.
            type: Type of the model. Only a subset of types are supported natively.
            uri: Resource identifier pointing to the model artifact. There are no restrictions on the URI format,
                however only a limited set of URI schemes is supported natively.
            input_spec: The expected input schema of the model. Dictionary where the keys are
                expected column names and the values are the value types.
            output_spec: The expected output schema of the model. Dictionary where the keys
                are expected column names and the values are the value types.
            description: A description for the model. The description can be changed later.
            tags: Key-value pairs of tags to be set for this model. Tags can be modified
                after model registration.
            training_dataset: An object contains training dataset metadata.

        Raises:
            DataError: The given model already exists.
            DatabaseError: Unable to register the model properties into table.
        """
        new_model: Dict[Any, Any] = {}
        new_model["ID"] = model_id
        new_model["NAME"] = model_name
        new_model["VERSION"] = model_version
        new_model["TYPE"] = type
        new_model["URI"] = uri
        new_model["INPUT_SPEC"] = input_spec
        new_model["OUTPUT_SPEC"] = output_spec
        new_model["CREATION_TIME"] = formatting.SqlStr("CURRENT_TIMESTAMP()")
        new_model["CREATION_ROLE"] = self._session.get_current_role()
        new_model["CREATION_ENVIRONMENT_SPEC"] = {"python": ".".join(map(str, sys.version_info[:3]))}
        if training_dataset is not None:
            _ml_artifact.add_artifact(
                session=self._session,
                database_name=self._name,
                schema_name=self._schema,
                artifact_id=training_dataset.id(),
                artifact_type=_ml_artifact.ArtifactType.TRAINING_DATASET,
                artifact_name=training_dataset.id(),
                artifact_version="",
                artifact_spec=json.loads(training_dataset.to_json()),
            )
            new_model["TRAINING_DATASET_ID"] = training_dataset.id()
        else:
            new_model["TRAINING_DATASET_ID"] = None

        existing_model_nums = self._list_selected_models(model_name=model_name, model_version=model_version).count()
        if existing_model_nums:
            raise connector.DataError(
                f"Model {model_name}/{model_version} already exists. Unable to register the model."
            )

        if self._insert_registry_entry(id=model_id, name=model_name, version=model_version, properties=new_model):
            self._set_metadata_attribute(
                model_name=model_name,
                model_version=model_version,
                attribute=_METADATA_ATTRIBUTE_REGISTRATION,
                value=new_model,
            )
            if description:
                self.set_model_description(model_name=model_name, model_version=model_version, description=description)
            if tags:
                self._set_metadata_attribute(
                    _METADATA_ATTRIBUTE_TAGS, value=tags, model_name=model_name, model_version=model_version
                )
        else:
            raise connector.DatabaseError("Failed to insert the model properties to the registry table.")

    # Registry operations

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def list_models(self) -> snowpark.DataFrame:
        """Lists models contained in the registry.

        Returns:
            snowpark.DataFrame with the list of models. Access is read-only through the snowpark.DataFrame.
            The resulting snowpark.dataframe will have an "id" column that uniquely identifies each model and can be
            used to reference the model when performing operations.
        """
        # Explicitly not calling collect.
        return self._session.sql(
            "SELECT * FROM {database}.{schema}.{view}".format(
                database=self._name, schema=self._schema, view=self._registry_table_view
            )
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def set_tag(
        self,
        model_name: str,
        model_version: str,
        tag_name: str,
        tag_value: Optional[str] = None,
    ) -> None:
        """Set model tag to the model with value.

        If the model tag already exists, the tag value will be overwritten.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            tag_name: Desired tag name string.
            tag_value: (optional) New tag value string. If no value is given the value of the tag will be set to None.
        """
        # This method uses a read-modify-write pattern for setting tags.
        # TODO(amauser): Investigate the use of transactions to avoid race conditions.
        model_tags = self.get_tags(model_name=model_name, model_version=model_version)
        model_tags[tag_name] = tag_value
        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_TAGS, model_tags, model_name=model_name, model_version=model_version
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def remove_tag(self, model_name: str, model_version: str, tag_name: str) -> None:
        """Remove target model tag.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            tag_name: Desired tag name string.

        Raises:
            DataError: If the model does not have the requested tag.
        """
        # This method uses a read-modify-write pattern for updating tags.

        model_tags = self.get_tags(model_name=model_name, model_version=model_version)
        try:
            del model_tags[tag_name]
        except KeyError:
            raise connector.DataError(
                f"Model {model_name}/{model_version} has no tag named {tag_name}. Full list of tags: {model_tags}"
            )

        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_TAGS, model_tags, model_name=model_name, model_version=model_version
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def has_tag(
        self,
        model_name: str,
        model_version: str,
        tag_name: str,
        tag_value: Optional[str] = None,
    ) -> bool:
        """Check if a model has a tag with the given name and value.

        If no value is given, any value for the tag will return true.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            tag_name: Desired tag name string.
            tag_value: (optional) Tag value to check. If not value is given, only the presence of the tag will be
                checked.

        Returns:
            True if the tag or tag and value combination is present for the model with the given id, False otherwise.
        """
        tags = self.get_tags(model_name=model_name, model_version=model_version)
        has_tag = tag_name in tags
        if tag_value is None:
            return has_tag
        return has_tag and tags[tag_name] == str(tag_value)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def get_tag_value(self, model_name: str, model_version: str, tag_name: str) -> Any:
        """Return the value of the tag for the model.

        The returned value can be None. If the tag does not exist, KeyError will be raised.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            tag_name: Desired tag name string.

        Returns:
            Value string of the tag or None, if no value is set for the tag.
        """
        return self.get_tags(model_name=model_name, model_version=model_version)[tag_name]

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def get_tags(self, model_name: Optional[str] = None, model_version: Optional[str] = None) -> Dict[str, Any]:
        """Get all tags and values stored for the target model.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            String-to-string dictionary containing all tags and values. The resulting dictionary can be empty.
        """
        # Snowpark snowpark.dataframe returns dictionary objects as strings. We need to convert it back to a dictionary
        # here.
        result = self._get_metadata_attribute(
            _METADATA_ATTRIBUTE_TAGS, model_name=model_name, model_version=model_version
        )

        if result:
            ret: Dict[str, Optional[str]] = json.loads(result)
            return ret
        else:
            return dict()

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def get_model_description(self, model_name: str, model_version: str) -> Optional[str]:
        """Get the description of the model.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            Description of the model or None.
        """
        result = self._get_metadata_attribute(
            _METADATA_ATTRIBUTE_DESCRIPTION, model_name=model_name, model_version=model_version
        )
        return None if result is None else json.loads(result)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def set_model_description(
        self,
        model_name: str,
        model_version: str,
        description: str,
    ) -> None:
        """Set the description of the model.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            description: Desired new model description.
        """
        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_DESCRIPTION, description, model_name=model_name, model_version=model_version
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def get_history(self) -> snowpark.DataFrame:
        """Return a dataframe with the history of operations performed on the model registry.

        The returned dataframe is order by time and can be filtered further.

        Returns:
            snowpark.DataFrame with the history of the model.
        """
        res = (
            self._session.table(self._fully_qualified_metadata_table_name())
            .order_by("EVENT_TIMESTAMP")
            .select_expr(
                "EVENT_TIMESTAMP",
                "EVENT_ID",
                "MODEL_ID",
                "ROLE",
                "OPERATION",
                "ATTRIBUTE_NAME",
                "VALUE[ATTRIBUTE_NAME]",
            )
        )
        return cast(snowpark.DataFrame, res)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def get_model_history(
        self,
        model_name: str,
        model_version: str,
    ) -> snowpark.DataFrame:
        """Return a dataframe with the history of operations performed on the desired model.

        The returned dataframe is order by time and can be filtered further.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            snowpark.DataFrame with the history of the model.
        """
        id = self._get_model_id(model_name=model_name, model_version=model_version)
        return cast(snowpark.DataFrame, self.get_history().filter(snowpark.Column("MODEL_ID") == id))

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def set_metric(
        self,
        model_name: str,
        model_version: str,
        metric_name: str,
        metric_value: object,
    ) -> None:
        """Set scalar model metric to value.

        If a metric with that name already exists for the model, the metric value will be overwritten.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            metric_name: Desired metric name.
            metric_value: New metric value.
        """
        # This method uses a read-modify-write pattern for setting tags.
        # TODO(amauser): Investigate the use of transactions to avoid race conditions.
        model_metrics = self.get_metrics(model_name=model_name, model_version=model_version)
        model_metrics[metric_name] = metric_value
        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_METRICS, model_metrics, model_name=model_name, model_version=model_version
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def remove_metric(
        self,
        model_name: str,
        model_version: str,
        metric_name: str,
    ) -> None:
        """Remove a specific metric entry from the model.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            metric_name: Desired metric name.

        Raises:
            DataError: If the model does not have the requested metric.
        """
        # This method uses a read-modify-write pattern for updating tags.

        model_metrics = self.get_metrics(model_name=model_name, model_version=model_version)
        try:
            del model_metrics[metric_name]
        except KeyError:
            raise connector.DataError(
                f"Model {model_name}/{model_version} has no metric named {metric_name}. "
                f"Full list of metrics: {model_metrics}"
            )

        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_METRICS, model_metrics, model_name=model_name, model_version=model_version
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def has_metric(self, model_name: str, model_version: str, metric_name: str) -> bool:
        """Check if a model has a metric with the given name.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            metric_name: Desired metric name.

        Returns:
            True if the metric is present for the model with the given id, False otherwise.
        """
        metrics = self.get_metrics(model_name=model_name, model_version=model_version)
        return metric_name in metrics

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def get_metric_value(self, model_name: str, model_version: str, metric_name: str) -> object:
        """Return the value of the given metric for the model.

        The returned value can be None. If the metric does not exist, KeyError will be raised.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            metric_name: Desired metric name.

        Returns:
            Value of the metric. Can be None if the metric was set to None.
        """
        return self.get_metrics(model_name=model_name, model_version=model_version)[metric_name]

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def get_metrics(self, model_name: str, model_version: str) -> Dict[str, object]:
        """Get all metrics and values stored for the given model.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            String-to-float dictionary containing all metrics and values. The resulting dictionary can be empty.
        """
        # Snowpark snowpark.dataframe returns dictionary objects as strings. We need to convert it back to a dictionary
        # here.
        result = self._get_metadata_attribute(
            _METADATA_ATTRIBUTE_METRICS, model_name=model_name, model_version=model_version
        )

        if result:
            ret: Dict[str, object] = json.loads(result)
            return ret
        else:
            return dict()

    # Combined Registry and Repository operations.
    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def log_model(
        self,
        model_name: str,
        model_version: str,
        *,
        model: Any,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[Any] = None,
        training_dataset: Optional[training_dataset.TrainingDataset] = None,
        code_paths: Optional[List[str]] = None,
        options: Optional[model_types.BaseModelSaveOption] = None,
    ) -> Optional["ModelReference"]:
        """Uploads and register a model to the Model Registry.

        Args:
            model_name: The given name for the model. The combination (name + version) must be unique for each model.
            model_version: Version string to be set for the model. The combination (name + version) must be unique for
                each model.
            model: Local model object in a supported format.
            description: A description for the model. The description can be changed later.
            tags: string-to-string dictionary of tag names and values to be set for the model.
            conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to
                specify a dependency. It is a recommended way to specify your dependencies using conda. When channel is
                not specified, defaults channel will be used. When deploying to Snowflake Warehouse, defaults channel
                would be replaced with the Snowflake Anaconda channel.
            pip_requirements: List of PIP package specs. Model will not be able to deploy to the warehouse if there is
                pip requirements.
            signatures: Signatures of the model, which is a mapping from target method name to signatures of input and
                output, which could be inferred by calling `infer_signature` method with sample input data or training
                dataset.
            sample_input_data: Sample of the input data for the model.
            training_dataset: A training dataset metadata object.
            code_paths: Directory of code to import when loading and deploying the model.
            options: Additional options when saving the model.

        Raises:
            DataError: Raised when the given model exists.
            ValueError: Raised in following cases:
                1) both sample_input_data and training_dataset are provided;
                2) signatures and sample_input_data/training_dataset are both not provided and
                    model is not a snowflake estimator.
            Exception: Raised when there is any error raised when saving the model.

        Returns:
            Model Reference . None if failed.
        """
        # Ideally, the whole operation should be a single transaction. Currently, transactions do not support stage
        # operations.

        self._model_identifier_is_nonempty_or_raise(model_name, model_version)

        if sample_input_data is not None and training_dataset is not None:
            raise ValueError("Only one of sample_input_data and training_dataset should be provided.")

        if training_dataset is not None:
            sample_input_data = training_dataset.df
            if training_dataset.timestamp_col is not None:
                sample_input_data = sample_input_data.drop(training_dataset.timestamp_col)
            if training_dataset.label_cols is not None:
                sample_input_data = sample_input_data.drop(training_dataset.label_cols)

        existing_model_nums = self._list_selected_models(model_name=model_name, model_version=model_version).count()
        if existing_model_nums:
            raise connector.DataError(f"Model {model_name}/{model_version} already exists. Unable to log the model.")
        model_id, fully_qualified_model_stage_name = self._log_model_path(
            model_name=model_name,
            model_version=model_version,
        )
        model_stage_file_path = posixpath.join(f"{_STAGE_PREFIX}{fully_qualified_model_stage_name}", f"{model_id}.zip")
        model = cast(model_types.SupportedModelType, model)
        try:
            model_metadata = model_api.save_model(  # type: ignore[call-overload, misc]
                name=model_name,
                session=self._session,
                model_stage_file_path=model_stage_file_path,
                model=model,
                signatures=signatures,
                metadata=tags,
                conda_dependencies=conda_dependencies,
                pip_requirements=pip_requirements,
                sample_input=sample_input_data,
                code_paths=code_paths,
                options=options,
            )
        except Exception:
            # When model saving fails, clean up the model stage.
            query_result_checker.SqlResultValidator(
                self._session, f"DROP STAGE {fully_qualified_model_stage_name}"
            ).has_dimensions(expected_rows=1, expected_cols=1).validate()
            raise

        self._register_model_with_id(
            model_name=model_name,
            model_version=model_version,
            model_id=model_id,
            type=model_metadata.model_type,
            uri=uri.get_uri_from_snowflake_stage_path(model_stage_file_path),
            description=description,
            tags=tags,
            training_dataset=training_dataset,
        )

        return ModelReference(registry=self, model_name=model_name, model_version=model_version)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def load_model(self, model_name: str, model_version: str) -> Any:
        """Loads the model with the given (model_name + model_version) from the registry into memory.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            Restored model object.
        """
        remote_model_path = self._get_model_path(model_name=model_name, model_version=model_version)
        restored_model = None

        restored_model, _ = model_api.load_model(session=self._session, model_stage_file_path=remote_model_path)

        return restored_model

    # Repository Operations

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def deploy(
        self,
        model_name: str,
        model_version: str,
        *,
        deployment_name: str,
        target_method: Optional[str] = None,
        permanent: bool = False,
        platform: deploy_platforms.TargetPlatform = deploy_platforms.TargetPlatform.WAREHOUSE,
        options: Optional[
            Union[model_types.WarehouseDeployOptions, model_types.SnowparkContainerServiceDeployOptions]
        ] = None,
    ) -> None:
        """Deploy the model with the given deployment name.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            deployment_name: name of the generated UDF.
            target_method: The method name to use in deployment. Can be omitted if only 1 method in the model.
            permanent: Whether the deployment is permanent or not. Permanent deployment will generate a permanent UDF.
                (Only applicable for Warehouse deployment)
            platform: Target platform to deploy the model to. Currently supported platforms are
                ['warehouse', 'SNOWPARK_CONTAINER_SERVICES']
            options: Optional options for model deployment. Defaults to None.

        Raises:
            RuntimeError: Raised when parameters are not properly enabled when deploying to Warehouse with temporary UDF
        """
        if options is None:
            options = {}

        deployment_stage_path = ""

        if platform == deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES:
            permanent = True
            options = cast(model_types.SnowparkContainerServiceDeployOptions, options)
            deployment_stage_path = f"{self._prepare_deployment_stage()}/{deployment_name}/"
        elif platform == deploy_platforms.TargetPlatform.WAREHOUSE:
            options = cast(model_types.WarehouseDeployOptions, options)
            if permanent:
                # Every deployment-generated UDF should reside in its own unique directory. As long as each deployment
                # is allocated a distinct directory, multiple deployments can coexist within the same stage.
                # Given that each permanent deployment possesses a unique deployment_name, sharing the same stage does
                # not present any issues
                deployment_stage_path = (
                    options.get("permanent_udf_stage_location")
                    or f"{self._prepare_deployment_stage()}/{deployment_name}/"
                )
                options["permanent_udf_stage_location"] = deployment_stage_path

        remote_model_path = "@" + self._get_model_path(model_name=model_name, model_version=model_version)
        model_id = self._get_model_id(model_name, model_version)

        # https://snowflakecomputing.atlassian.net/browse/SNOW-858376
        # During temporary deployment on the Warehouse, Snowpark creates an unencrypted temporary stage for UDF-related
        # artifacts. However, UDF generation fails when importing from a mix of encrypted and unencrypted stages.
        # The following workaround copies model between stages (PrPr as of July 7th, 2023) to transfer the SSE
        # encrypted model zip from model stage to the temporary unencrypted stage.
        if not permanent and platform == deploy_platforms.TargetPlatform.WAREHOUSE:
            schema = self._fully_qualified_schema_name()
            unencrypted_stage = (
                f"@{schema}.{snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)}"
            )
            self._session.sql(f"CREATE TEMPORARY STAGE {unencrypted_stage[1:]}").collect()
            try:
                self._session.sql(f"COPY FILES INTO {unencrypted_stage} from {remote_model_path}").collect()
            except Exception:
                raise RuntimeError(
                    "Temporary deployment to the warehouse is currently not supported. Please use "
                    "permanent deployment by setting the 'permanent' parameter to True"
                )
            remote_model_path = f"{unencrypted_stage}/{os.path.basename(remote_model_path)}"

        # Step 1: Deploy to get the UDF
        deployment_info = _deployer.deploy(
            session=self._session,
            name=self._fully_qualified_deployment_name(deployment_name),
            platform=platform,
            target_method=target_method,
            model_stage_file_path=remote_model_path,
            deployment_stage_path=deployment_stage_path,
            model_id=model_id,
            options=options,
        )

        # Step 2: Record the deployment

        # Assert to convince mypy.
        assert deployment_info
        if permanent:
            self._insert_deployment_entry(
                id=model_id,
                name=deployment_name,
                platform=deployment_info["platform"].value,
                stage_path=deployment_stage_path,
                signature=deployment_info["signature"].to_dict(),
                target_method=deployment_info["target_method"],
                options=options,
            )

        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_DEPLOYMENT,
            {"name": deployment_name, "permanent": permanent},
            id=model_id,
            operation=_ADD_METADATA_OPERATION,
        )

        # Store temporary deployment information in the in-memory cache. This allows for future referencing and
        # tracking of its availability status.
        if not permanent:
            self._temporary_deployments[deployment_name] = deployment_info

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="1.0.1")
    def list_deployments(self, model_name: str, model_version: str) -> snowpark.DataFrame:
        """List all permanent deployments that originated from the given model.

        Temporary deployment info are currently not supported for listing.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            A snowpark dataframe that contains all deployments that associated with the given model.
        """
        deployments_df = (
            self._session.sql(f"SELECT * FROM {self._fully_qualified_permanent_deployment_view_name()}")
            .filter(snowpark.Column("MODEL_NAME") == model_name)
            .filter(snowpark.Column("MODEL_VERSION") == model_version)
        )
        res = deployments_df.select(
            deployments_df["MODEL_NAME"],
            deployments_df["MODEL_VERSION"],
            deployments_df["DEPLOYMENT_NAME"],
            deployments_df["CREATION_TIME"],
            deployments_df["TARGET_METHOD"],
            deployments_df["TARGET_PLATFORM"],
            deployments_df["SIGNATURE"],
            deployments_df["OPTIONS"],
            deployments_df["STAGE_PATH"],
            deployments_df["ROLE"],
        )
        return cast(snowpark.DataFrame, res)

    @snowpark._internal.utils.private_preview(version="1.0.1")
    def list_artifacts(self, model_name: str, model_version: str) -> snowpark.DataFrame:
        """List all artifacts that associated with given model.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            A snowpark dataframe that contains all artifacts that associated with the given model.
        """
        artifacts = (
            self._session.sql(f"SELECT * FROM {self._fully_qualified_artifact_view_name()}")
            .filter(snowpark.Column("MODEL_NAME") == model_name)
            .filter(snowpark.Column("MODEL_VERSION") == model_version)
        )
        return cast(snowpark.DataFrame, artifacts)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="1.0.1")
    def get_deployment(self, model_name: str, model_version: str, *, deployment_name: str) -> snowpark.DataFrame:
        """Get the permanent deployment with target name of the given model.

        Temporary deployment info are currently not supported.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            deployment_name: Deployment name string.

        Returns:
            A snowpark dataframe that contains the information of the target deployment.

        Raises:
            KeyError: Raised if the target deployment is not found.
        """
        deployment = self.list_deployments(model_name, model_version).filter(
            snowpark.Column("DEPLOYMENT_NAME") == deployment_name
        )
        if deployment.count() == 0:
            raise KeyError(
                f"Unable to find deployment named {deployment_name} in the model {model_name}/{model_version}."
            )
        return cast(snowpark.DataFrame, deployment)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="1.0.1")
    def get_training_dataset(self, model_name: str, model_version: str) -> Optional[training_dataset.TrainingDataset]:
        """Get training dataset of the model with the given (model name + model version).

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            Training dataset of the model or none if not found.
        """
        artifacts = (
            self.list_artifacts(model_name, model_version)
            .filter(snowpark.Column("TYPE") == _ml_artifact.ArtifactType.TRAINING_DATASET.value)
            .collect()
        )

        return (
            training_dataset.TrainingDataset.from_json(artifacts[0]["ARTIFACT_SPEC"], self._session)
            if len(artifacts) != 0
            else None
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="1.0.1")
    def delete_deployment(self, model_name: str, model_version: str, *, deployment_name: str) -> None:
        """Delete the target permanent deployment of the given model.

        Deleting temporary deployment are currently not supported.
        Temporary deployment will get cleaned automatically when the current session closed.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            deployment_name: Name of the deployment that is getting deleted.

        Raises:
            KeyError: Raised if the target deployment is not found.
        """
        deployment = (
            self._session.sql(f"SELECT * FROM {self._fully_qualified_permanent_deployment_view_name()}")
            .filter(snowpark.Column("DEPLOYMENT_NAME") == deployment_name)
            .filter(snowpark.Column("MODEL_NAME") == model_name)
            .filter(snowpark.Column("MODEL_VERSION") == model_version)
        ).collect()
        if len(deployment) == 0:
            raise KeyError(
                f"Unable to find deployment named {deployment_name} in the model {model_name}/{model_version}."
            )
        deployment = deployment[0]

        # TODO(SNOW-759526): The following sequence should be a transaction.
        # Step 1: Drop the UDF
        self._session.sql(
            f"DROP FUNCTION IF EXISTS {self._fully_qualified_deployment_name(deployment_name)}(OBJECT)"
        ).collect()

        # Step 2: Remove the staged artifact
        self._session.sql(f"REMOVE {deployment['STAGE_PATH']}").collect()

        # Step 3: Delete the deployment from the deployment table
        query_result_checker.SqlResultValidator(
            self._session,
            f"""DELETE FROM {self._fully_qualified_deployment_table_name()}
                WHERE MODEL_ID='{deployment['MODEL_ID']}' AND DEPLOYMENT_NAME='{deployment_name}'
            """,
        ).deletion_success(expected_num_rows=1).validate()

        # Step 4: Record the delete event
        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_DEPLOYMENT,
            {"name": deployment_name},
            id=deployment["MODEL_ID"],
            operation=_DROP_METADATA_OPERATION,
        )

        # Optional Step 5: Delete Snowpark container service.
        if deployment["TARGET_PLATFORM"] == deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES.value:
            service_name = f"service_{deployment['MODEL_ID']}"
            query_result_checker.SqlResultValidator(
                self._session,
                f"DROP SERVICE IF EXISTS {service_name}",
            ).validate()

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def delete_model(
        self,
        model_name: str,
        model_version: str,
        delete_artifact: bool = True,
    ) -> None:
        """Delete model with the given ID from the registry.

        The history of the model will still be preserved.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.
            delete_artifact: If True, the underlying model artifact will also be deleted, not just the entry in
                the registry table.
        """

        # Check that a model with the given ID exists and there is only one of them.
        # TODO(amauser): The following sequence should be a transaction. Transactions currently cannot contain DDL
        # statements.
        model_info = None
        selected_models = self._list_selected_models(model_name=model_name, model_version=model_version)
        identifier = f"{model_name}/{model_version}"
        model_info = self._validate_exact_one_result(selected_models, identifier)
        id = model_info[0]["ID"]
        model_uri = model_info[0]["URI"]

        # Step 1/3: Delete the registry entry.
        query_result_checker.SqlResultValidator(
            self._session, f"DELETE FROM {self._fully_qualified_registry_table_name()} WHERE ID='{id}'"
        ).deletion_success(expected_num_rows=1).validate()

        # Step 2/3: Delete the artifact (if desired).
        if delete_artifact:
            if uri.is_snowflake_stage_uri(model_uri):
                stage_path = self._get_fully_qualified_stage_name_from_uri(model_uri)
                query_result_checker.SqlResultValidator(self._session, f"DROP STAGE {stage_path}").has_dimensions(
                    expected_rows=1, expected_cols=1
                ).validate()

        # Step 3/3: Record the deletion event.
        self._set_metadata_attribute(
            id=id,
            attribute=_METADATA_ATTRIBUTE_DELETION,
            value={"delete_artifact": True, "URI": model_uri},
            enable_model_presence_check=False,
        )


class ModelReference:
    """Wrapper class for ModelReference objects that proxy model metadata operations."""

    def _remove_arg_from_docstring(self, arg: str, docstring: Optional[str]) -> Optional[str]:
        """Remove the given parameter from a function docstring (Google convention)."""
        if docstring is None:
            return None
        docstring_lines = docstring.split("\n")

        args_section_start = None
        args_section_end = None
        args_section_indent = None
        arg_start = None
        arg_end = None
        arg_indent = None
        for i in range(len(docstring_lines)):
            line = docstring_lines[i]
            lstrip_line = line.lstrip()
            indent = len(line) - len(lstrip_line)

            if line.strip() == "Args:":
                # Starting the Args section of the docstring (assuming Google-style).
                args_section_start = i
                # logging.info("TEST: args_section_start=" + str(args_section_start))
                args_section_indent = indent
                continue

            # logging.info("TEST: " + lstrip_line)
            if args_section_start and lstrip_line.startswith(f"{arg}:"):
                # This is the arg we are looking for.
                arg_start = i
                # logging.info("TEST: arg_start=" + str(arg_start))
                arg_indent = indent
                continue

            if arg_start and not arg_end and indent == arg_indent:
                # We got the next arg, previous line was the last of the cut out arg docstring
                # and we do have other args. Saving arg_end for python slice/range notation.
                arg_end = i
                continue

            if arg_start and (len(lstrip_line) == 0 or indent == args_section_indent):
                # Arg section ends.
                args_section_end = i
                arg_end = arg_end if arg_end else i
                # We have learned everything we need to know, no need to continue.
                break

        if arg_start and not arg_end:
            arg_end = len(docstring_lines)

        if args_section_start and not args_section_end:
            args_section_end = len(docstring_lines)

        # Determine which lines from the "Args:" section of the docstring to skip or if we
        # should skip the entire section.
        keep_lines = set(range(len(docstring_lines)))
        if args_section_start:
            if arg_start == args_section_start + 1 and arg_end == args_section_end:
                # Removed arg was the only arg, remove the entire section.
                assert args_section_end
                keep_lines.difference_update(range(args_section_start, args_section_end))
            else:
                # Just remove the arg.
                assert arg_start
                assert arg_end
                keep_lines.difference_update(range(arg_start, arg_end))

        return "\n".join([docstring_lines[i] for i in sorted(keep_lines)])

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        model_name: str,
        model_version: str,
    ) -> None:
        self._registry = registry
        self._id = registry._get_model_id(model_name=model_name, model_version=model_version)
        self._model_name = model_name
        self._model_version = model_version

        # Wrap all functions of the ModelRegistry that have an "id" parameter and bind that parameter
        # the the "_id" member of this class.
        if hasattr(self.__class__, "init_complete"):
            # Already did the generation of wrapped method.
            return

        for name, obj in self._registry.__class__.__dict__.items():
            if (
                not inspect.isfunction(obj)
                or "model_name" not in inspect.signature(obj).parameters
                or "model_version" not in inspect.signature(obj).parameters
            ):
                continue

            # Ensure that we are not silently overwriting existing functions.
            assert not hasattr(self.__class__, name)

            def build_method(m: Callable[..., Any]) -> Callable[..., Any]:
                return lambda self, *args, **kwargs: m(
                    self._registry, self._model_name, self._model_version, *args, **kwargs
                )

            method = build_method(m=obj)
            setattr(self.__class__, name, method)

            docstring = self._remove_arg_from_docstring("model_name", obj.__doc__)
            if docstring and "model_version" in docstring:
                docstring = self._remove_arg_from_docstring("model_version", docstring)
            setattr(self.__class__.__dict__[name], "__doc__", docstring)  # NoQA

        setattr(self.__class__, "init_complete", True)  # NoQA

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def get_name(self) -> str:
        return self._model_name

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def get_version(self) -> str:
        return self._model_version

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    @snowpark._internal.utils.private_preview(version="0.2.0")
    def predict(self, deployment_name: str, data: Any) -> "pd.DataFrame":
        """Predict using the deployed model in Snowflake.

        Args:
            deployment_name: name of the generated UDF.
            data: Data to run predict.

        Raises:
            ValueError: The deployment with given name haven't been deployed.

        Returns:
            A dataframe containing the result of prediction.
        """

        # We will search temporary deployments from the local in-memory cache.
        # If there is no hit, we try to search the remote deployment table.
        di = self._registry._temporary_deployments.get(deployment_name)

        statement_params = telemetry.get_function_usage_statement_params(
            project=_TELEMETRY_PROJECT,
            subproject=_TELEMETRY_SUBPROJECT,
            function_name=telemetry.get_statement_params_full_func_name(
                inspect.currentframe(), self.__class__.__name__
            ),
        )

        if di:
            return _deployer.predict(
                session=self._registry._session, deployment=di, X=data, statement_params=statement_params
            )

        try:
            # Mypy enforce to refer to the registry for calling the function
            deployment = self._registry.get_deployment(
                self._model_name, self._model_version, deployment_name=deployment_name
            ).collect(statement_params=statement_params)[0]
            platform = deploy_platforms.TargetPlatform(deployment["TARGET_PLATFORM"])
            target_method = deployment["TARGET_METHOD"]
            signature = model_signature.ModelSignature.from_dict(json.loads(deployment["SIGNATURE"]))
            options_dict = cast(Dict[str, Any], json.loads(deployment["OPTIONS"]))
            platform_options = {
                deploy_platforms.TargetPlatform.WAREHOUSE: model_types.WarehouseDeployOptions,
                deploy_platforms.TargetPlatform.SNOWPARK_CONTAINER_SERVICES: (
                    model_types.SnowparkContainerServiceDeployOptions
                ),
            }

            if platform not in platform_options:
                raise ValueError(f"Unsupported target Platform: {platform}")
            options = platform_options[platform](options_dict)
            di = _deployer.Deployment(
                name=self._registry._fully_qualified_deployment_name(deployment_name),
                platform=platform,
                target_method=target_method,
                signature=signature,
                options=options,
            )
            return _deployer.predict(
                session=self._registry._session, deployment=di, X=data, statement_params=statement_params
            )
        except KeyError:
            raise ValueError(f"The deployment with name {deployment_name} haven't been deployed")


@telemetry.send_api_usage_telemetry(
    project=_TELEMETRY_PROJECT,
    subproject=_TELEMETRY_SUBPROJECT,
)
@snowpark._internal.utils.private_preview(version="0.2.0")
def create_model_registry(
    *,
    session: snowpark.Session,
    database_name: str = _DEFAULT_REGISTRY_NAME,
    schema_name: str = _DEFAULT_SCHEMA_NAME,
) -> bool:
    """Setup a new model registry. This should be run once per model registry by an administrator role.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.

    Returns:
        True if the creation of the model registry internal data structures was successful,
        False otherwise.
    """
    # Get the db & schema of the current session
    old_db = session.get_current_database()
    old_schema = session.get_current_schema()

    # These might be exposed as parameters in the future.
    database_name = identifier.get_inferred_name(database_name)
    schema_name = identifier.get_inferred_name(schema_name)
    registry_table_name = identifier.get_inferred_name(_MODELS_TABLE_NAME)
    metadata_table_name = identifier.get_inferred_name(_METADATA_TABLE_NAME)
    deployment_table_name = identifier.get_inferred_name(_DEPLOYMENT_TABLE_NAME)
    artifact_table_name = identifier.get_inferred_name(_ml_artifact._ARTIFACT_TABLE_NAME)

    statement_params = telemetry.get_function_usage_statement_params(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
        function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), ""),
    )
    try:
        _create_registry_database(session, database_name, statement_params)
        _create_registry_schema(session, database_name, schema_name, statement_params)
        _create_registry_tables(
            session,
            database_name,
            schema_name,
            registry_table_name,
            metadata_table_name,
            deployment_table_name,
            artifact_table_name,
            statement_params,
        )
        _create_registry_views(
            session,
            database_name,
            schema_name,
            registry_table_name,
            metadata_table_name,
            deployment_table_name,
            artifact_table_name,
            statement_params,
        )
    finally:
        # Restore the db & schema to the original ones
        if old_db is not None:
            session.use_database(old_db)
        if old_schema is not None:
            session.use_schema(old_schema)
    return True
