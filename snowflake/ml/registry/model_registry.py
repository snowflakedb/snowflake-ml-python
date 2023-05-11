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
from snowflake.ml._internal.utils import formatting, query_result_checker, uri
from snowflake.ml.model import (
    _deployer,
    _model as model_api,
    model_signature,
    type_hints as model_types,
)
from snowflake.ml.registry import _schema
from snowflake.ml.utils import telemetry
from snowflake.snowpark._internal import utils

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

        self.open(name=name)

    # Private methods

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

    def _prepare_model_stage(self, *, id: str) -> str:
        """Create a stage in the model registry for storing the model with the given id.

        Creating a permanent stage here since we do not have a way to swtich a stage from temporary to permanent.
        This can result in orphaned stages in case the process fails. It might be better to try to create a
        temporary stage, attempt to perform all operations and convert the temp stage into permanent once the
        operation is complete.

        Args:
            id: Identifier string of the model intended to be stored in the stage.

        Returns:
            Name of the stage that was created.

        Raises:
            DatabaseError: Indicates that something went wrong when creating the stage.
        """
        schema = self._fully_qualified_schema_name()

        # Replacing dashes and uppercasing the model_stage_name to avoid having to quote the the stage name.
        model_stage_name = "SNOWML_MODEL_{safe_id}".format(safe_id=id.replace("-", "_").upper())
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

    def _list_selected_models(
        self, *, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> Any:
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

        return filtered_models

    def _get_metadata_attribute(
        self,
        attribute: str,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> Any:
        """Get the value of the given metadata attribute for target model with (model name + model version) or id.

        Args:
            attribute: Name of the attribute to get.
            id: Model ID string. Required if either name or version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            The value of the attribute that was requested. Can be None if the attribute is not set.

        Raises:
            DataError: The given model identifier points to more than one models.
        """
        statement_params = self._get_statement_params(inspect.currentframe())
        selected_models = self._list_selected_models(id=id, model_name=model_name, model_version=model_version)
        result = selected_models.select(attribute).collect(statement_params=statement_params)

        if len(result) > 1:
            identifier = f"id {id}" if id else f"{model_name}/{model_version}"
            raise connector.DataError(f"Model {identifier} existed {len(result)} times. It should only exist once.")
        elif len(result) == 1 and attribute in result[0]:
            return result[0][attribute]
        else:
            return None

    def _set_metadata_attribute(
        self,
        attribute: str,
        value: Any,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        enable_model_presence_check: bool = True,
    ) -> None:
        """Set the value of the given metadata attribute for model id.

        Args:
            attribute: Name of the attribute to set.
            value: Value to set.
            id: Model ID string. Required if either name or version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.
            enable_model_presence_check: If True, we will check if the model with the given ID is currently registered
                before setting the metadata attribute. False by default meaning that by default we will check.

        Raises:
            DataError: The requested model id could not be found or is ambiguous.
        """
        statement_params = self._get_statement_params(inspect.currentframe())
        selected_models = self._list_selected_models(id=id, model_name=model_name, model_version=model_version)
        number_of_entries_filtered = selected_models.count(statement_params=statement_params)

        identifier = f"id {id}" if id else f"{model_name}/{model_version}"
        if enable_model_presence_check and number_of_entries_filtered == 0:
            raise connector.DataError(f"Model {identifier} was not found in the registry.")
        elif number_of_entries_filtered > 1:
            raise connector.DataError(
                f"Model {identifier} existed {number_of_entries_filtered} times. It should only exist once."
            )

        if not id:
            res = selected_models.select("ID").collect(statement_params=statement_params)
            id = res[0]["ID"]
        assert id is not None

        try:
            self._insert_metadata_entry(id=id, attribute=attribute, value={attribute: value})
        except connector.DataError:
            raise connector.DataError(f"Setting model name for mode id {id} failed.")

    # Registry operations

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def open(self, *, name: str = _DEFAULT_REGISTRY_NAME) -> None:
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
    def _get_model_id(self, *, model_name: Optional[str], model_version: Optional[str]) -> str:
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

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def set_tag(
        self,
        name: str,
        value: str,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """Set model tag to with value.

        If the model tag already exists, the tag value will be overwritten.

        Args:
            name: Desired tag name.
            value: (optional) New tag value. If no value is given the value of the tag will be set to None.
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.
        """
        # This method uses a read-modify-write pattern for setting tags.
        # TODO(amauser): Investigate the use of transactions to avoid race conditions.
        model_tags = self.get_tags(id=id, model_name=model_name, model_version=model_version)
        model_tags[name] = value
        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_TAGS, model_tags, id=id, model_name=model_name, model_version=model_version
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def remove_tag(
        self, name: str, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> None:
        """Remove target model tag.

        Args:
            name: Desired tag name.
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Raises:
            DataError: If the model does not have the requested tag.
        """
        # This method uses a read-modify-write pattern for updating tags.

        model_tags = self.get_tags(id=id, model_name=model_name, model_version=model_version)
        try:
            del model_tags[name]
        except KeyError:
            raise connector.DataError(f"Model id {id} has not tag named {name}. Full list of tags: {model_tags}")

        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_TAGS, model_tags, id=id, model_name=model_name, model_version=model_version
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def has_tag(
        self,
        name: str,
        value: Optional[str] = None,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> bool:
        """Check if a model has a tag with the given name and value.

        If no value is given, any value for the tag will return true.

        Args:
            name: Desired tag name.
            value: (optional) Tag value to check. If not value is given, only the presence of the tag will be
                checked.
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            True if the tag or tag and value combination is present for the model with the given id, False otherwise.
        """
        tags = self.get_tags(id=id, model_name=model_name, model_version=model_version)
        return name in tags and tags[name] == str(value)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def get_tag_value(
        self, name: str, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> Optional[str]:
        """Return the value of the tag for the model.

        The returned value can be None. If the tag does not exist, KeyError will be raised.

        Args:
            name: Desired tag name.
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            Value string of the tag or None, if no value is set for the tag.
        """
        return self.get_tags(id=id, model_name=model_name, model_version=model_version)[name]

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def get_tags(
        self, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> Dict[str, str]:
        """Get all tags and values stored for the given (model name + model version) or model id.

        Args:
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            String-to-string dictionary containing all tags and values. The resulting dictionary can be empty.
        """
        # Snowpark snowpark.dataframes returns dictionary objects as strings. We need to convert it back to a dictionary
        # here.
        result = self._get_metadata_attribute(
            _METADATA_ATTRIBUTE_TAGS, id=id, model_name=model_name, model_version=model_version
        )

        if result:
            ret: Dict[str, str] = json.loads(result)
            return ret
        else:
            return dict()

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def get_model_description(
        self, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> Optional[str]:
        """Get the description of the model with the given (model name + model version) or id.

        Args:
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            Descrption of the model or None.
        """
        result = self._get_metadata_attribute(
            _METADATA_ATTRIBUTE_DESCRIPTION, id=id, model_name=model_name, model_version=model_version
        )
        return None if result is None else str(result)

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def set_model_description(
        self,
        *,
        description: str,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """Set the description of the model with the given id.

        Args:
            description: Desired new model description.
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.
        """
        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_DESCRIPTION, description, id=id, model_name=model_name, model_version=model_version
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
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> snowpark.DataFrame:
        """Return a dataframe with the history of operations performed on the desired model.

        The returned dataframe is order by time and can be filtered further.

        Args:
            id: Id of the model to retrieve the history for. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            snowpark.DataFrame with the history of the model.
        """
        if not id:
            id = self._get_model_id(model_name=model_name, model_version=model_version)
        return cast(snowpark.DataFrame, self.get_history().filter(snowpark.Column("MODEL_ID") == id))

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def set_metric(
        self,
        name: str,
        value: object,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """Set scalar model metric to value.

        If a metric with that name already exists for the model, the metric value will be overwritten.

        Args:
            name: Desired metric name.
            value: New metric value.
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.
        """
        # This method uses a read-modify-write pattern for setting tags.
        # TODO(amauser): Investigate the use of transactions to avoid race conditions.
        model_metrics = self.get_metrics(id=id, model_name=model_name, model_version=model_version)
        model_metrics[name] = value
        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_METRICS, model_metrics, id=id, model_name=model_name, model_version=model_version
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def remove_metric(
        self,
        name: str,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """Remove a specific metric entry from the model.

        Args:
            name: Desired tag name.
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Raises:
            DataError: If the model does not have the requested metric.
        """
        # This method uses a read-modify-write pattern for updating tags.

        model_metrics = self.get_metrics(id=id, model_name=model_name, model_version=model_version)
        try:
            del model_metrics[name]
        except KeyError:
            raise connector.DataError(
                f"Model id {id} has no metric named {name}. Full list of metrics: {model_metrics}"
            )

        self._set_metadata_attribute(
            _METADATA_ATTRIBUTE_METRICS, model_metrics, id=id, model_name=model_name, model_version=model_version
        )

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def has_metric(
        self, name: str, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> bool:
        """Check if a model has a metric with the given name.

        Args:
            name: Desired metric name.
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            True if the metric is present for the model with the given id, False otherwise.
        """
        metrics = self.get_metrics(id=id, model_name=model_name, model_version=model_version)
        return name in metrics

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def get_metric_value(
        self, name: str, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> Optional[object]:
        """Return the value of the given metric for the model.

        The returned value can be None. If the metric does not exist, KeyError will be raised.

        Args:
            name: Desired tag name.
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            Value of the metric. Can be None if the metric was set to None.
        """
        return self.get_metrics(id=id, model_name=model_name, model_version=model_version)[name]

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def get_metrics(
        self, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> Dict[str, object]:
        """Get all metrics and values stored for the given (model name + model version) or model id.

        Args:
            id: Model ID string. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            String-to-float dictionary containing all metrics and values. The resulting dictionary can be empty.
        """
        # Snowpark snowpark.dataframes returns dictionary objects as strings. We need to convert it back to a dictionary
        # here.
        result = self._get_metadata_attribute(
            _METADATA_ATTRIBUTE_METRICS, id=id, model_name=model_name, model_version=model_version
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
        *,
        model: Any,
        name: str,
        version: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        conda_dependencies: Optional[List[str]] = None,
        pip_requirements: Optional[List[str]] = None,
        signatures: Optional[Dict[str, model_signature.ModelSignature]] = None,
        sample_input_data: Optional[Any] = None,
    ) -> Optional[str]:
        """Uploads and register a model to the Model Registry.

        Args:
            model: Local model object in a supported format.
            name: The given name for the model. The combination (name + version) must be unique for each model.
            version: Version string to be set for the model. The combination (name + version) must be unique for each
                model.
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

        Returns:
            String of the auto-generate unique model identifier. None if failed.
        """
        # TODO(amauser): We should never return None, investigate and update the return type accordingly.
        id = None
        # Ideally, the whole operation should be a single transaction. Currently, transactions do not support stage
        # operations.
        # Save model to local disk.
        is_native_model_format = False

        with tempfile.TemporaryDirectory() as tmpdir:
            model = cast(model_types.SupportedModelType, model)
            try:
                if signatures:
                    model_api.save_model(
                        name=name if name else self._get_new_unique_identifier(),
                        model_dir_path=tmpdir,
                        model=model,
                        signatures=signatures,
                        metadata=tags,
                        conda_dependencies=conda_dependencies,
                        pip_requirements=pip_requirements,
                    )
                elif sample_input_data is not None:
                    model_api.save_model(
                        name=name if name else self._get_new_unique_identifier(),
                        model_dir_path=tmpdir,
                        model=model,
                        metadata=tags,
                        conda_dependencies=conda_dependencies,
                        pip_requirements=pip_requirements,
                        sample_input=sample_input_data,
                    )
                id = self.log_model_path(
                    path=tmpdir,
                    type="snowflake_native",
                    name=name,
                    version=version,
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
                    path=local_model_file.name,
                    type=model.__class__.__name__,
                    name=name,
                    version=version,
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
        *,
        id: str,
        type: str,
        uri: str,
        name: str,
        version: str,
        input_spec: Optional[Dict[str, str]] = None,
        output_spec: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Register a new model in the ModelRegistry.

        This operation will only create the metadata and not handle any model artifacts. A URI is expected to be given
        that points the the actual model artifact.

        Args:
            id: Unique identifier to be used for the model. This is required to be unique within the registry and
                uniqueness will be verified. The model id is immutable once set.
            type: Type of the model. Only a subset of types are supported natively.
            uri: Resource identifier pointing to the model artifact. There are no restrictions on the URI format,
                however only a limited set of URI schemes is supported natively.
            name: Name to be set for the model. The model name can be changed after registration and is not
                required to be unique inside the registry. The combination of name and version is expected to be unique
                inside the registry.
            version: Version string to be set for the model. The model version string can NOT be changed after model
                registration and is not required to be unique inside the registry. The combination of name and version
                is expected to be unique inside the registry.
            input_spec: The expected input schema of the model. Dictionary where the keys are
                expected column names and the values are the value types.
            output_spec: The expected output schema of the model. Dictionary where the keys
                are expected column names and the values are the value types.
            description: A desription for the model. The description can be changed later.
            tags: Key-value pairs of tags to be set for this model. Tags can be modified
                after model registration.

        Returns:
            True if the operation was successful.

        Raises:
            DataError: The given model already exists.
        """
        # TODO(Zhe SNOW-813224): Remove input_spec and output_spec. Use signature instead.

        # Create registry entry.

        new_model: Dict[Any, Any] = {}
        new_model["ID"] = id
        new_model["NAME"] = name
        new_model["VERSION"] = version
        new_model["INPUT_SPEC"] = input_spec
        new_model["OUTPUT_SPEC"] = output_spec
        new_model["TYPE"] = type
        new_model["CREATION_TIME"] = formatting.SqlStr("CURRENT_TIMESTAMP()")
        new_model["CREATION_ROLE"] = self._session.get_current_role()
        new_model["CREATION_ENVIRONMENT_SPEC"] = {"python": ".".join(map(str, sys.version_info[:3]))}
        new_model["URI"] = uri

        existing_model_nums = self._list_selected_models(id=id, model_name=name, model_version=version).count()
        if existing_model_nums:
            raise connector.DataError(f"Model {name}/{version} already exists. Unable to register the model.")

        if self._insert_registry_entry(id=id, name=name, version=version, properties=new_model):
            self._set_metadata_attribute(id=id, attribute=_METADATA_ATTRIBUTE_REGISTRATION, value=new_model)
            if description:
                self.set_model_description(id=id, description=description)
            if tags:
                self._set_metadata_attribute(id=id, attribute=_METADATA_ATTRIBUTE_TAGS, value=tags)
            return True
        else:
            return False

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def log_model_path(
        self,
        *,
        path: str,
        type: str,
        name: str,
        version: str,
        description: Optional[str] = None,
        tags: Optional[Dict[Any, Any]] = None,
    ) -> str:
        """Uploads and register a model to the Model Registry from a local file path.

        If `path` is a directory all files will be uploaded recursively, preserving the relative directory structure.
        Symbolic links will be followed.

        NOTE: If any symlinks under `path` point to a parent directory, this can lead to infinite recursion.

        Args:
            path: Local file path to be uploaded.
            type: Type of the model to be added.
            name: The given name for the model.
            version: Version string to be set for the model.
            description: A desription for the model. The description can be changed later.
            tags: string-to-string dictonary of tag names and values to be set for the model.

        Returns:
            String of the auto-generate unique model identifier.
        """
        id = self._get_new_unique_identifier()

        # Copy model from local disk to remote stage.
        fully_qualified_model_stage_name = self._prepare_model_stage(id=id)

        # Check if directory or file and adapt accordingly.
        # TODO: Unify and explicit about compression for both file and directory.
        if os.path.isfile(path):
            self._session.file.put(path, f"{fully_qualified_model_stage_name}/data")
        elif os.path.isdir(path):
            with utils.zip_file_or_directory_to_stream(path, path, add_init_py=True) as input_stream:
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
        self.register_model(
            id=id,
            type=type,
            uri=uri.get_uri_from_snowflake_stage_path(fully_qualified_model_stage_name),
            name=name if name else fully_qualified_model_stage_name,
            version=version,
            description=description,
            tags=tags,
        )

        return id

    def _get_model_path(
        self, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> str:
        """Get the stage path for the model with the given (model name + model version) or `id` from the registry.

        Args:
            id: Id of the model to deploy. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if id is None.

        Raises:
            DataError: When the model cannot be found or not be restored.
            NotImplementedError: For models that span multiple files.

        Returns:
            str: Stage path for the model.
        """
        statement_params = self._get_statement_params(inspect.currentframe())
        selected_models = self._list_selected_models(id=id, model_name=model_name, model_version=model_version)
        model_uri_result = selected_models.select("ID", "URI").collect(statement_params=statement_params)

        table_name = self._fully_qualified_registry_table_name()
        if len(model_uri_result) == 0:
            raise connector.DataError(f"Model with id {id} not found in ModelRegistry {table_name}.")

        if len(model_uri_result) > 1:
            raise connector.DataError(
                f"Model with id {id} exist multiple ({len(model_uri_result)}) times in ModelRegistry " "{table_name}."
            )

        model_uri = model_uri_result[0].URI

        if not uri.is_snowflake_stage_uri(model_uri):
            raise connector.DataError(
                f"Artifacts with URI scheme {uri.get_uri_scheme(model_uri)} are currently not supported."
            )

        model_stage_name = uri.get_snowflake_stage_path_from_uri(model_uri)

        # Currently we assume only the model is on the stage.
        model_file_list = self._session.sql(f"LIST @{model_stage_name}").collect(statement_params=statement_params)
        if len(model_file_list) == 0:
            raise connector.DataError(f"No files in model artifact for id {id} located at {model_uri}.")
        if len(model_file_list) > 1:
            raise NotImplementedError("Restoring models consisting of multiple files is currently not supported.")
        return f"{self._fully_qualified_schema_name()}.{model_file_list[0].name}"

    @telemetry.send_api_usage_telemetry(
        project=_TELEMETRY_PROJECT,
        subproject=_TELEMETRY_SUBPROJECT,
    )
    def load_model(
        self, *, id: Optional[str] = None, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> Any:
        """Loads the model with the given (model_name + model_version) or `id` from the registry into memory.

        Args:
            id: Model identifier. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.

        Returns:
            Restored model object.
        """
        remote_model_path = self._get_model_path(id=id, model_name=model_name, model_version=model_version)
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
        *,
        deployment_name: str,
        target_method: str,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        options: Optional[model_types.DeployOptions] = None,
    ) -> None:
        """Deploy the model with the the given deployment name.

        Args:
            deployment_name: name of the generated UDF.
            target_method: The method name to use in deployment.
            id: Id of the model to deploy. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.
            options: Optional options for model deployment. Defaults to None.

        Raises:
            TypeError: The model with given id does not conform to native model format.
        """
        if options is None:
            options = {}

        remote_model_path = self._get_model_path(id=id, model_name=model_name, model_version=model_version)
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
        delete_artifact: bool = True,
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """Delete model with the given ID from the registry.

        The history of the model will still be preserved.

        Args:
            delete_artifact: If True, the underlying model artifact will also be deleted, not just the entry in
                the registry table.
            id: Id of the model to delete. Required if either model name or model version is None.
            model_name: Model Name string. Required if id is None.
            model_version: Model Version string. Required if version is None.


        Raises:
            KeyError: Model with the given ID does not exist in the registry.
        """

        # Check that a model with the given ID exists and there is only one of them.
        # TODO(amauser): The following sequence should be a transaction. Transactions currently cannot contain DDL
        # statements.
        model_info = None
        try:
            selected_models = self._list_selected_models(id=id, model_name=model_name, model_version=model_version)
            model_info = (
                query_result_checker.ResultValidator(result=selected_models.collect())
                .has_dimensions(expected_rows=1)
                .validate()
            )
        except connector.DataError:
            identifier = f"with id {id}" if id else f"named {model_name}/{model_version}"
            if model_info is None or len(model_info) == 0:
                raise KeyError(f"The model {identifier} does not exist in the current registry.")
            else:
                raise KeyError(
                    formatting.unwrap(
                        f"""There are {len(model_info)} models {identifier}. This might indicate a problem with
                            the integrity of the model registry data."""
                    )
                )
        if not id:
            id = model_info[0]["ID"]
        model_uri = model_info[0]["URI"]

        # Step 1/3: Delete the registry entry.
        query_result_checker.SqlResultValidator(
            self._session, f"DELETE FROM {self._fully_qualified_registry_table_name()} WHERE ID='{id}'"
        ).deletion_success(expected_num_rows=1).validate()

        # Step 2/3: Delete the artifact (if desired).
        if delete_artifact:
            if uri.is_snowflake_stage_uri(model_uri):
                stage_name = uri.get_snowflake_stage_path_from_uri(model_uri)
                query_result_checker.SqlResultValidator(self._session, f"DROP STAGE {stage_name}").has_value_match(
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
        id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._id = id if id else registry._get_model_id(model_name=model_name, model_version=model_version)

        # Wrap all functions of the ModelRegistry that have an "id" parameter and bind that parameter
        # the the "_id" member of this class.
        if hasattr(self.__class__, "init_complete"):
            # Already did the generation of wrapped method.
            return

        for name, obj in self._registry.__class__.__dict__.items():
            if not inspect.isfunction(obj) or "id" not in inspect.signature(obj).parameters:
                continue

            # Ensure that we are not silently overwriting existing functions.
            assert not hasattr(self.__class__, name)

            # logging.info("TEST: Adding function: " + name)
            old_sig = inspect.signature(obj)
            removed_none_type = map(
                lambda x: x.replace(annotation=str(x.annotation)),
                filter(lambda p: p.name not in ["id"], old_sig.parameters.values()),
            )
            new_sig = old_sig.replace(
                parameters=list(removed_none_type), return_annotation=str(old_sig.return_annotation)
            )
            arguments = ", ".join(
                ["id=self._id"]
                + [
                    "{p.name}={p.name}".format(p=p)
                    for p in filter(
                        lambda p: p.name not in ["id", "model_name", "model_version", "self"],
                        old_sig.parameters.values(),
                    )
                ]
            )
            docstring = self._remove_arg_from_docstring("id", obj.__doc__)
            if docstring and "model_name" in docstring:
                docstring = self._remove_arg_from_docstring("model_name", docstring)
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
