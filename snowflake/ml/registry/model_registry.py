import inspect
import json
import os
import sys
import tempfile
import types
import zipfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from uuid import uuid1

import joblib
from absl import logging

from snowflake import connector, snowpark
from snowflake.ml._internal import file_utils, telemetry
from snowflake.ml._internal.utils import formatting, query_result_checker, uri
from snowflake.ml.model import (
    _deployer,
    _model as model_api,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.registry import _schema

if TYPE_CHECKING:
    import pandas as pd

_DEFAULT_REGISTRY_NAME: str = "MODEL_REGISTRY"
_DEFAULT_PROJECT_NAME: str = "PUBLIC"
_DEFAULT_TASK_NAME: str = "MODELS"
_DEFAULT_METADATA_NAME: str = "METADATA"

# Metadata operation types.
_SET_METADATA_OPERATION: str = "SET"

# Metadata types.
_METADATA_ATTRIBUTE_DESCRIPTION: str = "DESCRIPTION"
_METADATA_ATTRIBUTE_METRICS: str = "METRICS"
_METADATA_ATTRIBUTE_REGISTRATION: str = "REGISTRATION"
_METADATA_ATTRIBUTE_TAGS: str = "TAGS"
_METADATA_ATTRIBUTE_DELETION: str = "DELETION"

# Leaving out REGISTRATION evnts as they will be handled differently from all mutable attributes.
_LIST_METADATA_ATTRIBUTE: List[str] = [
    _METADATA_ATTRIBUTE_DESCRIPTION,
    _METADATA_ATTRIBUTE_METRICS,
    _METADATA_ATTRIBUTE_TAGS,
]
_TELEMETRY_PROJECT = "MLOps"
_TELEMETRY_SUBPROJECT = "ModelRegistry"


def create_model_registry(
    session: snowpark.Session,
    database_name: str = _DEFAULT_REGISTRY_NAME,
) -> bool:
    """Setup a new model registry. This should be run once per model registry by an administrator role.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.

    Returns:
        True if the creation of the model registry internal data structures was successful,
        False otherwise.
    """

    # These might be exposed as parameters in the future.
    schema_name = _DEFAULT_PROJECT_NAME
    registry_table_name = _DEFAULT_TASK_NAME
    metadata_table_name = _DEFAULT_METADATA_NAME

    create_ok = _create_registry_database(session, database_name, schema_name, registry_table_name, metadata_table_name)
    if create_ok:
        _create_registry_views(session, database_name, schema_name, registry_table_name, metadata_table_name)
    return create_ok


def _create_registry_database(
    session: snowpark.Session,
    database_name: str,
    schema_name: str,
    registry_table_name: str,
    metadata_table_name: str,
) -> bool:
    """Private helper to create the model registry internal data structures.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the database.
        registry_table_name: Name for the main model registry table.
        metadata_table_name: Name for the metadata table used by the model registry.

    Returns:
        True if the creation of the model registry internal data structures was successful,
        False otherwise.
    """
    fully_qualified_schema_name = f'"{database_name}"."{schema_name}"'
    fully_qualified_registry_table_name = f'{fully_qualified_schema_name}."{registry_table_name}"'
    fully_qualified_metadata_table_name = f'{fully_qualified_schema_name}."{metadata_table_name}"'
    statement_params = telemetry.get_function_usage_statement_params(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
        function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), ""),
    )

    registry_databases = session.sql(f"SHOW DATABASES LIKE '{database_name}'").collect(
        statement_params=statement_params
    )
    if len(registry_databases) > 0:
        logging.warning(f"The database {database_name} already exists. Skipping creation.")
        return False

    session.sql(f'CREATE DATABASE "{database_name}"').collect(statement_params=statement_params)

    # The PUBLIC schema is created by default so it might already exist even in a new database.
    registry_schemas = session.sql(f"SHOW SCHEMAS LIKE '{schema_name}' IN DATABASE \"{database_name}\"").collect(
        statement_params=statement_params
    )
    if len(registry_schemas) == 0:
        session.sql(f'CREATE SCHEMA "{database_name}"."{schema_name}"').collect(statement_params=statement_params)

    registry_schema_string = ", ".join([f"{k} {v}" for k, v in _schema._REGISTRY_TABLE_SCHEMA.items()])
    session.sql(f"CREATE TABLE {fully_qualified_registry_table_name} ({registry_schema_string})").collect(
        statement_params=statement_params
    )
    metadata_schema_string = ", ".join(
        [
            f"{k} {v.format(registry_table_name=fully_qualified_registry_table_name)}"
            for k, v in _schema._METADATA_TABLE_SCHEMA.items()
        ]
    )
    session.sql(f"CREATE TABLE {fully_qualified_metadata_table_name} ({metadata_schema_string})").collect(
        statement_params=statement_params
    )
    return True


def _create_registry_views(
    session: snowpark.Session,
    database_name: str,
    schema_name: str,
    registry_table_name: str,
    metadata_table_name: str,
) -> None:
    """Create views on underlying ModelRegistry tables.

    Args:
        session: Session object to communicate with Snowflake.
        database_name: Desired name of the model registry database.
        schema_name: Desired name of the schema used by this model registry inside the databse.
        registry_table_name: Name for the main model registry table.
        metadata_table_name: Name for the metadata table used by the model registry.
    """
    fully_qualified_schema_name = f'"{database_name}"."{schema_name}"'

    statement_params = telemetry.get_function_usage_statement_params(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
        function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), ""),
    )
    # From the documentation: Each DDL statement executes as a separate transaction. Races should not be an issue.
    # https://docs.snowflake.com/en/sql-reference/transactions.html#ddl

    # Create views on most recent metadata items.
    metadata_view_name_prefix = metadata_table_name + "_LAST_"
    metadata_view_template = formatting.unwrap(
        """CREATE OR REPLACE VIEW "{database}"."{schema}"."{attribute_view}" COPY GRANTS AS
            SELECT DISTINCT MODEL_ID, {select_expression} AS {final_attribute_name} FROM "{metadata_table}"
            WHERE ATTRIBUTE_NAME = '{attribute_name}'"""
    )

    # Create a separate view for the most recent item in each metadata column.
    metadata_view_names = []
    metadata_select_fields = []
    for attribute_name in _LIST_METADATA_ATTRIBUTE:
        view_name = f"{metadata_view_name_prefix}{attribute_name}"
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
        metadata_select_fields.append(f'"{view_name}".{attribute_name} AS {attribute_name}')

    # Create a special view for the registration timestamp.
    attribute_name = _METADATA_ATTRIBUTE_REGISTRATION
    final_attribute_name = attribute_name + "_TIMESTAMP"
    view_name = f"{metadata_view_name_prefix}{attribute_name}"
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
    metadata_select_fields.append(f'"{view_name}".{final_attribute_name} AS {final_attribute_name}')

    metadata_views_join = " ".join(
        [
            'LEFT JOIN "{view}" ON ("{view}".MODEL_ID = "{registry_table}".ID)'.format(
                view=view, registry_table=registry_table_name
            )
            for view in metadata_view_names
        ]
    )

    # Create view to combine all attributes.
    registry_view_name = registry_table_name + "_VIEW"
    metadata_select_fields_formatted = ",".join(metadata_select_fields)
    session.sql(
        f"""CREATE OR REPLACE VIEW {fully_qualified_schema_name}."{registry_view_name}" COPY GRANTS AS
                SELECT "{registry_table_name}".*, {metadata_select_fields_formatted}
                FROM "{registry_table_name}" {metadata_views_join}"""
    ).collect(statement_params=statement_params)


class ModelRegistry:
    """Model Management API."""

    def __init__(self, *, session: snowpark.Session, name: str = _DEFAULT_REGISTRY_NAME) -> None:
        self._name = name
        self._schema = _DEFAULT_PROJECT_NAME
        self._registry_table = _DEFAULT_TASK_NAME
        self._registry_table_view = self._registry_table + "_VIEW"
        self._metadata_table = _DEFAULT_METADATA_NAME

        self._session = session
        self._deploy_api = _deployer.Deployer(session=self._session, manager=_deployer.LocalDeploymentManager())

        self._open(name=name)

    # Private methods

    def _open(self, *, name: str = _DEFAULT_REGISTRY_NAME) -> None:
        """Open a model registry.

        If no name is give, the default registry will be used.

        Args:
            name: (optional) Name of the Model Registry to open.
        """
        self._name = name
        # Check that the required tables exist and are accessible by the current role.

        query_result_checker.SqlResultValidator(
            self._session, query=f"SHOW DATABASES LIKE '{self._name}'"
        ).has_dimensions(expected_rows=1).validate()

        query_result_checker.SqlResultValidator(
            self._session, query=f"SHOW SCHEMAS LIKE '{self._schema}' IN DATABASE \"{self._name}\""
        ).has_dimensions(expected_rows=1).validate()

        query_result_checker.SqlResultValidator(
            self._session, query=f"SHOW TABLES LIKE '{self._registry_table}' IN {self._fully_qualified_schema_name()}"
        ).has_dimensions(expected_rows=1).validate()

        query_result_checker.SqlResultValidator(
            self._session, query=f"SHOW TABLES LIKE '{self._metadata_table}' IN {self._fully_qualified_schema_name()}"
        ).has_dimensions(expected_rows=1).validate()

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
        return f'"{self._name}"."{self._schema}"."{self._registry_table}"'

    def _fully_qualified_metadata_table_name(self) -> str:
        """Get the fully qualified name to the current metadata table."""
        return f'"{self._name}"."{self._schema}"."{self._metadata_table}"'

    def _fully_qualified_schema_name(self) -> str:
        """Get the fully qualified name to the current registry schema."""
        return f'"{self._name}"."{self._schema}"'

    def _insert_table_entry(self, *, table: str, columns: Dict[str, Any]) -> List[snowpark.Row]:
        """Insert an entry into an internal Model Registry table.

        Args:
            table: Name of the table to insert into.
            columns: Key-value pairs of columns and values to be inserted into the table.

        Returns:
            Result of the operation as returned by the Snowpark session (snowpark.DataFrame).
        """
        sorted_columns = sorted(columns.items())

        sql = "INSERT INTO {table} ( {columns} ) SELECT {values}".format(
            table=table,
            columns=",".join([x[0] for x in sorted_columns]),
            values=",".join([formatting.format_value_for_select(x[1]) for x in sorted_columns]),
        )
        return (
            query_result_checker.SqlResultValidator(self._session, sql)
            .insertion_success(expected_num_rows=1)
            .validate()
        )

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
                            f"""Parameter '{k.lower()}' is given and parameter 'properties' has the field '{k}' set but the values
                            do not match: {k.lower()}=="{v}" properties['{k}']=="{properties[k]}"."""
                        )
                    )
        # Could do a multi-table insert here with some pros and cons:
        # [PRO] Atomic insert across multiple tables.
        # [CON] Code logic becomes messy depending on which fields are set.
        # [CON] Harder to re-use existing methods like set_model_name.
        # Context: https://docs.snowflake.com/en/sql-reference/sql/insert-multi-table.html
        return self._insert_table_entry(table=self._fully_qualified_registry_table_name(), columns=properties)

    def _insert_metadata_entry(self, *, id: str, attribute: str, value: Any) -> List[snowpark.Row]:
        """Insert a new row into the model metadata table.

        Args:
            id: Model id to register.
            attribute: name of the metadata attribute
            value: new value of the metadata attribute

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
        columns["OPERATION"] = _SET_METADATA_OPERATION
        columns["ATTRIBUTE_NAME"] = attribute
        columns["VALUE"] = value

        return self._insert_table_entry(table=self._fully_qualified_metadata_table_name(), columns=columns)

    def _prepare_model_stage(self, *, model_name: str, model_version: str) -> str:
        """Create a stage in the model registry for storing the model with the given id.

        Creating a permanent stage here since we do not have a way to swtich a stage from temporary to permanent.
        This can result in orphaned stages in case the process fails. It might be better to try to create a
        temporary stage, attempt to perform all operations and convert the temp stage into permanent once the
        operation is complete.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            Name of the stage that was created.

        Raises:
            DatabaseError: Indicates that something went wrong when creating the stage.
        """
        schema = self._fully_qualified_schema_name()

        stage_name = f"{model_name}_{model_version}".replace("-", "_").upper()

        # Replacing dashes and uppercasing the model_stage_name to avoid having to quote the the stage name.
        model_stage_name = f"SNOWML_MODEL_{stage_name}"
        fully_qualified_model_stage_name = f"{schema}.{model_stage_name}"
        statement_params = self._get_statement_params(inspect.currentframe())

        create_stage_result = self._session.sql(f"CREATE OR REPLACE STAGE {fully_qualified_model_stage_name}").collect(
            statement_params=statement_params
        )
        if not create_stage_result:
            raise connector.DatabaseError("Unable to create stage for model. Operation returned not result.")
        if len(create_stage_result) != 1:
            raise connector.DatabaseError(
                "Unable to create stage for model. Creating the model stage returned unexpected result: {}.".format(
                    str(create_stage_result)
                )
            )
        if create_stage_result[0]["status"] != f"Stage area {model_stage_name} successfully created.":
            raise connector.DatabaseError(
                "Unable to create stage for model. Return status of operation was: {}".format(
                    create_stage_result[0]["status"]
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
        raw_stage_name = uri.get_snowflake_stage_path_from_uri(model_uri)
        if not raw_stage_name:
            return None
        model_stage_name = raw_stage_name.split(".")[-1]
        qualified_stage_path = f"{self._fully_qualified_schema_name()}.{model_stage_name}"
        return qualified_stage_path

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

        Raises:
            DataError: Model ID or (Model Name + Model Version) is not given.
        """
        if not (id or (model_name and model_version)):
            raise connector.DataError("Either (Model Name + Model Version) or Model ID is required, but none is given.")

        models = self.list_models()

        if id:
            filtered_models = models.filter(snowpark.Column("ID") == id)
        else:
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
        enable_model_presence_check: bool = True,
    ) -> None:
        """Set the value of the given metadata attribute for targat model with given (model name + model version) or id.

        Args:
            attribute: Name of the attribute to set.
            value: Value to set.
            id: Model ID string. Required if either name or version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.
            enable_model_presence_check: If True, we will check if the model with the given ID is currently registered
                before setting the metadata attribute. False by default meaning that by default we will check.

        Raises:
            DataError: Failed to set the meatdata attribute.
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
            self._insert_metadata_entry(id=id, attribute=attribute, value={attribute: value})
        except connector.DataError:
            raise connector.DataError(f"Setting model name for mode id {id} failed.")

    def _model_identifier_is_nonempty_or_raise(self, model_name: str, model_version: str) -> None:
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

    # Registry operations

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def list_models(self) -> snowpark.DataFrame:
        """Lists models contained in the registry.

        Returns:
            snowpark.DataFrame with the list of models. Access is read-only through the snowpark.DataFrame.
            The resulting snowpark.dataframe will have an "id" column that uniquely identifies each model and can be
            used to reference the model when performing operations.
        """
        # Explicitly not calling collect.
        return self._session.sql(
            'SELECT * FROM "{database}"."{schema}"."{view}"'.format(
                database=self._name, schema=self._schema, view=self._registry_table_view
            )
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
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
            raise connector.DataError(f"Model id {id} has not tag named {tag_name}. Full list of tags: {model_tags}")

        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_TAGS, model_tags, model_name=model_name, model_version=model_version
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
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
    def get_tags(self, model_name: str = None, model_version: str = None) -> Dict[str, Any]:
        """Get all tags and values stored for the target model.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            String-to-string dictionary containing all tags and values. The resulting dictionary can be empty.
        """
        # Snowpark snowpark.dataframes returns dictionary objects as strings. We need to convert it back to a dictionary
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
    def get_model_description(self, model_name: str, model_version: str) -> Optional[str]:
        """Get the description of the model.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            Descrption of the model or None.
        """
        result = self._get_metadata_attribute(
            _METADATA_ATTRIBUTE_DESCRIPTION, model_name=model_name, model_version=model_version
        )
        return None if result is None else str(result)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
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
    def get_metric_value(self, model_name: str, model_version: str, metric_name: str) -> Optional[object]:
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
    def get_metrics(self, model_name: str, model_version: str) -> Dict[str, object]:
        """Get all metrics and values stored for the given model.

        Args:
            model_name: Model Name string.
            model_version: Model Version string.

        Returns:
            String-to-float dictionary containing all metrics and values. The resulting dictionary can be empty.
        """
        # Snowpark snowpark.dataframes returns dictionary objects as strings. We need to convert it back to a dictionary
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
    ) -> Optional[str]:
        """Uploads and register a model to the Model Registry.

        Args:
            model_name: The given name for the model. The combination (name + version) must be unique for each model.
            model_version: Version string to be set for the model. The combination (name + version) must be unique for
                each model.
            model: Local model object in a supported format.
            description: A desription for the model. The description can be changed later.
            tags: string-to-string dictonary of tag names and values to be set for the model.
            conda_dependencies: List of Conda package specs. Use "[channel::]package [operator version]" syntax to
                specify a dependency. It is a recommended way to specify your dependencies using conda. When channel is
                not specified, defaults channel will be used. When deploying to Snowflake Warehouse, defaults channel
                would be replaced with the Snowflake Anaconda channel.
            pip_requirements: List of PIP package specs. Model will not be able to deploy to the warehouse if there is
                pip requirements.
            signatures: Signatures of the model, which is a mapping from target method name to signatures of input and
                output, which could be inferred by calling `infer_signature` method with sample input data.
            sample_input_data: Sample of the input data for the model.

        Raises:
            TypeError: Raised when both signatures and sample_input_data is not presented. Will be captured locally.

        Returns:
            String of the auto-generate unique model identifier. None if failed.
        """
        # TODO(amauser): We should never return None, investigate and update the return type accordingly.
        id = None
        # Ideally, the whole operation should be a single transaction. Currently, transactions do not support stage
        # operations.
        # Save model to local disk.
        is_native_model_format = False

        self._model_identifier_is_nonempty_or_raise(model_name, model_version)

        with tempfile.TemporaryDirectory() as tmpdir:
            model = cast(model_types.SupportedModelType, model)
            try:
                if signatures:
                    model_api.save_model(
                        name=model_name,
                        model_dir_path=tmpdir,
                        model=model,
                        signatures=signatures,
                        metadata=tags,
                        conda_dependencies=conda_dependencies,
                        pip_requirements=pip_requirements,
                    )
                elif sample_input_data is not None:
                    model_api.save_model(
                        name=model_name,
                        model_dir_path=tmpdir,
                        model=model,
                        metadata=tags,
                        conda_dependencies=conda_dependencies,
                        pip_requirements=pip_requirements,
                        sample_input=sample_input_data,
                    )
                else:
                    raise TypeError("Either signature or sample input data should exist for native model packaging.")
                id = self.log_model_path(
                    model_name=model_name,
                    model_version=model_version,
                    path=tmpdir,
                    type="snowflake_native",
                    description=description,
                    tags=tags,  # TODO: Inherent model type enum.
                )
                is_native_model_format = True
            except (AttributeError, TypeError):
                pass

        if not is_native_model_format:
            with tempfile.NamedTemporaryFile(delete=True) as local_model_file:
                joblib.dump(model, local_model_file)
                local_model_file.flush()

                id = self.log_model_path(
                    model_name=model_name,
                    model_version=model_version,
                    path=local_model_file.name,
                    type=model.__class__.__name__,
                    description=description,
                    tags=tags,
                )

        return id

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def register_model(
        self,
        model_name: str,
        model_version: str,
        *,
        type: str,
        uri: str,
        input_spec: Optional[Dict[str, str]] = None,
        output_spec: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a new model in the ModelRegistry.

        This operation will only create the metadata and not handle any model artifacts. A URI is expected to be given
        that points the the actual model artifact.

        Args:
            model_name: Name to be set for the model. The model name can NOT be changed after registration. The
                combination of name and version is expected to be unique inside the registry.
            model_version: Version string to be set for the model. The model version string can NOT be changed after
                model registration. The combination of name and version is expected to be unique inside the registry.
            type: Type of the model. Only a subset of types are supported natively.
            uri: Resource identifier pointing to the model artifact. There are no restrictions on the URI format,
                however only a limited set of URI schemes is supported natively.
            input_spec: The expected input schema of the model. Dictionary where the keys are
                expected column names and the values are the value types.
            output_spec: The expected output schema of the model. Dictionary where the keys
                are expected column names and the values are the value types.
            description: A desription for the model. The description can be changed later.
            tags: Key-value pairs of tags to be set for this model. Tags can be modified
                after model registration.

        Returns:
            The model id string, which is unique identifier to be used for the model. None will be returned if the
            operation failed.

        Raises:
            DataError: The given model already exists.
            DatabaseError: Unable to register the model properties into table.
        """
        # TODO(Zhe SNOW-813224): Remove input_spec and output_spec. Use signature instead.

        self._model_identifier_is_nonempty_or_raise(model_name, model_version)

        # Create registry entry.

        id = self._get_new_unique_identifier()

        new_model: Dict[Any, Any] = {}
        new_model["ID"] = id
        new_model["NAME"] = model_name
        new_model["VERSION"] = model_version
        new_model["INPUT_SPEC"] = input_spec
        new_model["OUTPUT_SPEC"] = output_spec
        new_model["TYPE"] = type
        new_model["CREATION_TIME"] = formatting.SqlStr("CURRENT_TIMESTAMP()")
        new_model["CREATION_ROLE"] = self._session.get_current_role()
        new_model["CREATION_ENVIRONMENT_SPEC"] = {"python": ".".join(map(str, sys.version_info[:3]))}
        new_model["URI"] = uri

        existing_model_nums = self._list_selected_models(model_name=model_name, model_version=model_version).count()
        if existing_model_nums:
            raise connector.DataError(
                f"Model {model_name}/{model_version} already exists. Unable to register the model."
            )

        if self._insert_registry_entry(id=id, name=model_name, version=model_version, properties=new_model):
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
            return id
        else:
            raise connector.DatabaseError("Failed to insert the model properties to the registry table.")

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def log_model_path(
        self,
        model_name: str,
        model_version: str,
        *,
        path: str,
        type: str,
        description: Optional[str] = None,
        tags: Optional[Dict[Any, Any]] = None,
    ) -> str:
        """Uploads and register a model to the Model Registry from a local file path.

        If `path` is a directory all files will be uploaded recursively, preserving the relative directory structure.
        Symbolic links will be followed.

        NOTE: If any symlinks under `path` point to a parent directory, this can lead to infinite recursion.

        Args:
            model_name: The given name for the model.
            model_version: Version string to be set for the model.
            path: Local file path to be uploaded.
            type: Type of the model to be added.
            description: A desription for the model. The description can be changed later.
            tags: string-to-string dictonary of tag names and values to be set for the model.

        Returns:
            String of the auto-generate unique model identifier.
        """
        self._model_identifier_is_nonempty_or_raise(model_name, model_version)

        # Copy model from local disk to remote stage.
        fully_qualified_model_stage_name = self._prepare_model_stage(model_name=model_name, model_version=model_version)

        # Check if directory or file and adapt accordingly.
        # TODO: Unify and explicit about compression for both file and directory.
        if os.path.isfile(path):
            self._session.file.put(path, f"{fully_qualified_model_stage_name}/data")
        elif os.path.isdir(path):
            with file_utils.zip_file_or_directory_to_stream(path, path) as input_stream:
                self._session._conn.upload_stream(
                    input_stream=input_stream,
                    stage_location=fully_qualified_model_stage_name,
                    dest_filename=f"{os.path.basename(path)}.zip",
                    dest_prefix="",
                    source_compression="DEFLATE",
                    compress_data=False,
                    overwrite=True,
                    is_in_udf=True,
                )
        id = self.register_model(
            model_name=model_name,
            model_version=model_version,
            type=type,
            uri=uri.get_uri_from_snowflake_stage_path(fully_qualified_model_stage_name),
            description=description,
            tags=tags,
        )

        return id

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
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
        with tempfile.TemporaryDirectory() as local_model_directory:
            self._session.file.get(remote_model_path, local_model_directory)
            is_native_model_format = False
            local_path = os.path.join(local_model_directory, os.path.basename(remote_model_path))
            try:
                if zipfile.is_zipfile(local_path):
                    extracted_dir = os.path.join(local_model_directory, "extracted")
                    with zipfile.ZipFile(local_path, "r") as myzip:
                        if len(myzip.namelist()) > 1:
                            myzip.extractall(extracted_dir)
                            restored_model, _meta = model_api.load_model(extracted_dir)
                            is_native_model_format = True
            except TypeError:
                pass
            if not is_native_model_format:
                restored_model = joblib.load(
                    os.path.join(local_model_directory, os.path.basename(os.path.basename(remote_model_path)))
                )

        return restored_model

    # Repository Operations

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def deploy(
        self,
        model_name: str,
        model_version: str,
        *,
        deployment_name: str,
        target_method: str,
        options: Optional[model_types.DeployOptions] = None,
    ) -> None:
        """Deploy the model with the the given deployment name.

        Args:
            deployment_name: name of the generated UDF.
            target_method: The method name to use in deployment.
            model_name: Model Name string.
            model_version: Model Version string.
            options: Optional options for model deployment. Defaults to None.

        Raises:
            TypeError: The model with given id does not conform to native model format.
        """
        if options is None:
            options = {}

        remote_model_path = self._get_model_path(model_name=model_name, model_version=model_version)
        with tempfile.TemporaryDirectory() as local_model_directory:
            self._session.file.get(remote_model_path, local_model_directory)
            is_native_model_format = False
            local_path = os.path.join(local_model_directory, os.path.basename(remote_model_path))
            try:
                if zipfile.is_zipfile(local_path):
                    extracted_dir = os.path.join(local_model_directory, "extracted")
                    with zipfile.ZipFile(local_path, "r") as myzip:
                        if len(myzip.namelist()) > 1:
                            myzip.extractall(extracted_dir)
                            self._deploy_api.create_deployment(
                                name=deployment_name,
                                model_dir_path=extracted_dir,
                                platform=_deployer.TargetPlatform.WAREHOUSE,
                                target_method=target_method,
                                options=options,
                            )
                            is_native_model_format = True
            except TypeError:
                pass
            if not is_native_model_format:
                raise TypeError("Deployment is only supported for native model format.")

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
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

        di = self._deploy_api.get_deployment(name=deployment_name)
        if di is None:
            raise ValueError(f"The deployment with name {deployment_name} haven't been deployed")

        return self._deploy_api.predict(di["name"], data)

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
                query_result_checker.SqlResultValidator(self._session, f"DROP STAGE {stage_path}").has_value_match(
                    row_idx=0, col_idx=0, expected_value="successfully dropped."
                ).validate()

        # Step 3/3: Record the deletion event.
        self._set_metadata_attribute(
            id=id,
            attribute=_METADATA_ATTRIBUTE_DELETION,
            value={"delete_artifact": True, "URI": model_uri},
            enable_model_presence_check=False,
        )


_TEMPLATE_MODEL_REF_METHOD_DEFN = """
@telemetry.send_api_usage_telemetry(project='{project}', subproject='{subproject}')
def {name}{signature}:
    return self._registry.{name}({arguments})
"""


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
        id: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._id = id if id else registry._get_model_id(model_name=model_name, model_version=model_version)
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

            # logging.info("TEST: Adding function: " + name)
            old_sig = inspect.signature(obj)
            removed_none_type = map(
                lambda x: x.replace(annotation=str(x.annotation)),
                filter(lambda p: p.name not in ["model_name", "model_version"], old_sig.parameters.values()),
            )
            new_sig = old_sig.replace(
                parameters=list(removed_none_type), return_annotation=str(old_sig.return_annotation)
            )
            arguments = ", ".join(
                ["model_name=self._model_name"]
                + ["model_version=self._model_version"]
                + [
                    "{p.name}={p.name}".format(p=p)
                    for p in filter(
                        lambda p: p.name not in ["id", "model_name", "model_version", "self"],
                        old_sig.parameters.values(),
                    )
                ]
            )
            docstring = self._remove_arg_from_docstring("model_name", obj.__doc__)
            if docstring and "model_version" in docstring:
                docstring = self._remove_arg_from_docstring("model_version", docstring)
            exec(
                _TEMPLATE_MODEL_REF_METHOD_DEFN.format(
                    name=name,
                    signature=new_sig,
                    arguments=arguments,
                    project=_TELEMETRY_PROJECT,
                    subproject=_TELEMETRY_SUBPROJECT,
                )
            )
            setattr(self.__class__, name, locals()[name])
            setattr(self.__class__.__dict__[name], "__doc__", docstring)  # NoQA

        setattr(self.__class__, "init_complete", True)  # NoQA
