from __future__ import annotations

import datetime
import functools
import json
import logging
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import packaging.version as pkg_version
import snowflake.ml.version as snowml_version
from pytimeparse.timeparse import timeparse
from typing_extensions import Concatenate, ParamSpec

from snowflake.ml import dataset
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import (
    dataset_errors,
    error_codes,
    exceptions as snowml_exceptions,
    sql_error_codes,
)
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.utils.sql_identifier import (
    SqlIdentifier,
    to_sql_identifiers,
)
from snowflake.ml.dataset.dataset_metadata import FeatureStoreMetadata
from snowflake.ml.feature_store.entity import _ENTITY_NAME_LENGTH_LIMIT, Entity
from snowflake.ml.feature_store.feature_view import (
    _FEATURE_OBJ_TYPE,
    _FEATURE_VIEW_NAME_DELIMITER,
    _LEGACY_TIMESTAMP_COL_PLACEHOLDER_VALS,
    FeatureView,
    FeatureViewSlice,
    FeatureViewStatus,
    FeatureViewVersion,
    _FeatureViewMetadata,
)
from snowflake.ml.utils import sql_client
from snowflake.snowpark import DataFrame, Row, Session, functions as F
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

_Args = ParamSpec("_Args")
_RT = TypeVar("_RT")

logger = logging.getLogger(__name__)

_ENTITY_TAG_PREFIX = "SNOWML_FEATURE_STORE_ENTITY_"
_FEATURE_STORE_OBJECT_TAG = "SNOWML_FEATURE_STORE_OBJECT"
_FEATURE_VIEW_METADATA_TAG = "SNOWML_FEATURE_VIEW_METADATA"


@dataclass(frozen=True)
class _FeatureStoreObjInfo:
    type: _FeatureStoreObjTypes
    pkg_version: str

    def to_json(self) -> str:
        state_dict = self.__dict__.copy()
        state_dict["type"] = state_dict["type"].value
        return json.dumps(state_dict)

    @classmethod
    def from_json(cls, json_str: str) -> _FeatureStoreObjInfo:
        json_dict = json.loads(json_str)
        # since we may introduce new fields in the json blob in the future,
        # in order to guarantee compatibility, we need to select ones that can be
        # decoded in the current version
        state_dict = {}
        state_dict["type"] = _FeatureStoreObjTypes.parse(json_dict["type"])
        state_dict["pkg_version"] = json_dict["pkg_version"]
        return cls(**state_dict)  # type: ignore[arg-type]


class _FeatureStoreObjTypes(Enum):
    UNKNOWN = "UNKNOWN"  # for forward compatibility
    MANAGED_FEATURE_VIEW = "MANAGED_FEATURE_VIEW"
    EXTERNAL_FEATURE_VIEW = "EXTERNAL_FEATURE_VIEW"
    FEATURE_VIEW_REFRESH_TASK = "FEATURE_VIEW_REFRESH_TASK"
    TRAINING_DATA = "TRAINING_DATA"

    @classmethod
    def parse(cls, val: str) -> _FeatureStoreObjTypes:
        try:
            return cls(val)
        except ValueError:
            return cls.UNKNOWN


_PROJECT = "FeatureStore"
_DT_OR_VIEW_QUERY_PATTERN = re.compile(
    r"""CREATE\ (OR\ REPLACE\ )?(?P<obj_type>(DYNAMIC\ TABLE|VIEW))\ .*
        COMMENT\ =\ '(?P<comment>.*)'\s*
        TAG.*?{fv_metadata_tag}\ =\ '(?P<fv_metadata>.*?)',?.*?
        AS\ (?P<query>.*)
    """.format(
        fv_metadata_tag=_FEATURE_VIEW_METADATA_TAG,
    ),
    flags=re.DOTALL | re.IGNORECASE | re.X,
)

_DT_INITIALIZE_PATTERN = re.compile(
    r"""CREATE\ DYNAMIC\ TABLE\ .*
        initialize\ =\ '(?P<initialize>.*)'\ .*?
        AS\ .*
    """,
    flags=re.DOTALL | re.IGNORECASE | re.X,
)

_LIST_FEATURE_VIEW_SCHEMA = StructType(
    [
        StructField("name", StringType()),
        StructField("version", StringType()),
        StructField("database_name", StringType()),
        StructField("schema_name", StringType()),
        StructField("created_on", TimestampType()),
        StructField("owner", StringType()),
        StructField("desc", StringType()),
        StructField("entities", ArrayType(StringType())),
        StructField("refresh_freq", StringType()),
        StructField("refresh_mode", StringType()),
        StructField("scheduling_state", StringType()),
        StructField("warehouse", StringType()),
        StructField("cluster_by", StringType()),
    ]
)


CreationMode = sql_client.CreationOption
CreationMode.__module__ = __name__


@dataclass(frozen=True)
class _FeatureStoreConfig:
    database: SqlIdentifier
    schema: SqlIdentifier

    @property
    def full_schema_path(self) -> str:
        return f"{self.database}.{self.schema}"


def switch_warehouse(
    f: Callable[Concatenate[FeatureStore, _Args], _RT]
) -> Callable[Concatenate[FeatureStore, _Args], _RT]:
    @functools.wraps(f)
    def wrapper(self: FeatureStore, /, *args: _Args.args, **kargs: _Args.kwargs) -> _RT:
        original_warehouse = self._session.get_current_warehouse()
        if original_warehouse is not None:
            original_warehouse = SqlIdentifier(original_warehouse)
        warehouse_updated = False
        try:
            if original_warehouse != self._default_warehouse:
                self._session.use_warehouse(self._default_warehouse)
                warehouse_updated = True
            return f(self, *args, **kargs)
        finally:
            if warehouse_updated and original_warehouse is not None:
                self._session.use_warehouse(original_warehouse)

    return wrapper


def dispatch_decorator() -> Callable[
    [Callable[Concatenate[FeatureStore, _Args], _RT]],
    Callable[Concatenate[FeatureStore, _Args], _RT],
]:
    def decorator(
        f: Callable[Concatenate[FeatureStore, _Args], _RT]
    ) -> Callable[Concatenate[FeatureStore, _Args], _RT]:
        @telemetry.send_api_usage_telemetry(project=_PROJECT)
        @switch_warehouse
        @functools.wraps(f)
        def wrap(self: FeatureStore, /, *args: _Args.args, **kargs: _Args.kwargs) -> _RT:
            return f(self, *args, **kargs)

        return wrap

    return decorator


class FeatureStore:
    """
    FeatureStore provides APIs to create, materialize, retrieve and manage feature pipelines.
    """

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def __init__(
        self,
        session: Session,
        database: str,
        name: str,
        default_warehouse: str,
        *,
        creation_mode: CreationMode = CreationMode.FAIL_IF_NOT_EXIST,
    ) -> None:
        """
        Creates a FeatureStore instance.

        Args:
            session: Snowpark Session to interact with Snowflake backend.
            database: Database to create the FeatureStore instance.
            name: Target FeatureStore name, maps to a schema in the database.
            default_warehouse: Default warehouse for feature store compute.
            creation_mode: If FAIL_IF_NOT_EXIST, feature store throws when required resources not already exist; If
                CREATE_IF_NOT_EXIST, feature store will create required resources if they not already exist. Required
                resources include schema and tags. Note database must already exist in either mode.

        Raises:
            SnowflakeMLException: [ValueError] default_warehouse does not exist.
            SnowflakeMLException: [ValueError] Required resources not exist when mode is FAIL_IF_NOT_EXIST.
            SnowflakeMLException: [RuntimeError] Failed to find resources.
            SnowflakeMLException: [RuntimeError] Failed to create feature store.

        Example::

            >>> from snowflake.ml.feature_store import (
            ...     FeatureStore,
            ...     CreationMode,
            ... )
            <BLANKLINE>
            >>> # Create a new Feature Store:
            >>> fs = FeatureStore(
            ...     session=session,
            ...     database="MYDB",
            ...     name="MYSCHEMA",
            ...     default_warehouse="MYWH",
            ...     creation_mode=CreationMode.CREATE_IF_NOT_EXIST
            ... )
            <BLANKLINE>
            >>> # Connect to an existing Feature Store:
            >>> fs = FeatureStore(
            ...     session=session,
            ...     database="MYDB",
            ...     name="MYSCHEMA",
            ...     default_warehouse="MYWH",
            ...     creation_mode=CreationMode.FAIL_IF_NOT_EXIST
            ... )

        """

        database = SqlIdentifier(database)
        name = SqlIdentifier(name)

        self._telemetry_stmp = telemetry.get_function_usage_statement_params(_PROJECT)
        self._session: Session = session
        self._config = _FeatureStoreConfig(
            database=database,
            schema=name,
        )
        self._asof_join_enabled = None

        # A dict from object name to tuple of search space and object domain.
        # search space used in query "SHOW <object_TYPE> LIKE <object_name> IN <search_space>"
        # object domain used in query "TAG_REFERENCE(<object_name>, <object_domain>)"
        self._obj_search_spaces = {
            "DATASETS": (self._config.full_schema_path, "DATASET"),
            "DYNAMIC TABLES": (self._config.full_schema_path, "TABLE"),
            "VIEWS": (self._config.full_schema_path, "TABLE"),
            "SCHEMAS": (f"DATABASE {self._config.database}", "SCHEMA"),
            "TAGS": (self._config.full_schema_path, None),
            "TASKS": (self._config.full_schema_path, "TASK"),
            "WAREHOUSES": (None, None),
        }

        self.update_default_warehouse(default_warehouse)

        self._check_database_exists_or_throw()
        if creation_mode == CreationMode.FAIL_IF_NOT_EXIST:
            self._check_internal_objects_exist_or_throw()

        else:
            try:
                # Explicitly check if schema exists first since we may not have CREATE SCHEMA privilege
                if len(self._find_object("SCHEMAS", self._config.schema)) == 0:
                    self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self._config.full_schema_path}").collect(
                        statement_params=self._telemetry_stmp
                    )
                for tag in to_sql_identifiers([_FEATURE_VIEW_METADATA_TAG, _FEATURE_STORE_OBJECT_TAG]):
                    self._session.sql(f"CREATE TAG IF NOT EXISTS {self._get_fully_qualified_name(tag)}").collect(
                        statement_params=self._telemetry_stmp
                    )
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(f"Failed to create feature store {name}: {e}."),
                ) from e
        self._check_feature_store_object_versions()
        logger.info(f"Successfully connected to feature store: {self._config.full_schema_path}.")

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def update_default_warehouse(self, warehouse_name: str) -> None:
        """Update default warehouse for feature store.

        Args:
            warehouse_name: Name of warehouse.

        Raises:
            SnowflakeMLException: If warehouse does not exists.

        Example::

            >>> fs = FeatureStore(...)
            >>> fs.update_default_warehouse("MYWH_2")
            >>> draft_fv = FeatureView("my_fv", ...)
            >>> registered_fv = fs.register_feature_view(draft_fv, '2.0')
            >>> print(registered_fv.warehouse)
            MYWH_2

        """
        warehouse = SqlIdentifier(warehouse_name)
        warehouse_result = self._find_object("WAREHOUSES", warehouse)
        if len(warehouse_result) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Cannot find warehouse {warehouse}"),
            )

        self._default_warehouse = warehouse

    @dispatch_decorator()
    def register_entity(self, entity: Entity) -> Entity:
        """
        Register Entity in the FeatureStore.

        Args:
            entity: Entity object to be registered.

        Returns:
            A registered entity object.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to find resources.

        Example::

            >>> fs = FeatureStore(...)
            >>> e = Entity('BAR', ['A'], desc='entity bar')
            >>> fs.register_entity(e)
            >>> fs.list_entities().show()
            --------------------------------------------------
            |"NAME"  |"JOIN_KEYS"  |"DESC"      |"OWNER"     |
            --------------------------------------------------
            |BAR     |["A"]        |entity bar  |REGTEST_RL  |
            --------------------------------------------------

        """
        tag_name = self._get_entity_name(entity.name)
        found_rows = self._find_object("TAGS", tag_name)
        if len(found_rows) > 0:
            warnings.warn(
                f"Entity {entity.name} already exists. Skip registration.",
                stacklevel=2,
                category=UserWarning,
            )
            return entity

        # allowed_values will add double-quotes around each value, thus use resolved str here.
        join_keys = [f"{key.resolved()}" for key in entity.join_keys]
        join_keys_str = ",".join(join_keys)
        full_tag_name = self._get_fully_qualified_name(tag_name)
        try:
            self._session.sql(
                f"""CREATE TAG IF NOT EXISTS {full_tag_name}
                    ALLOWED_VALUES '{join_keys_str}'
                    COMMENT = '{entity.desc}'
                """
            ).collect(statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to register entity `{entity.name}`: {e}."),
            ) from e

        logger.info(f"Registered Entity {entity}.")

        return self.get_entity(entity.name)

    def update_entity(self, name: str, *, desc: Optional[str] = None) -> Optional[Entity]:
        """Update a registered entity with provided information.

        Args:
            name: Name of entity to update.
            desc: Optional new description to apply. Default to None.

        Raises:
            SnowflakeMLException: Error happen when updating.

        Returns:
            A new entity with updated information or None if the entity doesn't exist.

        Example::

            >>> fs = FeatureStore(...)
            <BLANKLINE>
            >>> e = Entity(name='foo', join_keys=['COL_1'], desc='old desc')
            >>> fs.list_entities().show()
            ------------------------------------------------
            |"NAME"  |"JOIN_KEYS"  |"DESC"    |"OWNER"     |
            ------------------------------------------------
            |FOO     |["COL_1"]    |old desc  |REGTEST_RL  |
            ------------------------------------------------
            <BLANKLINE>
            >>> fs.update_entity('foo', desc='NEW DESC')
            >>> fs.list_entities().show()
            ------------------------------------------------
            |"NAME"  |"JOIN_KEYS"  |"DESC"    |"OWNER"     |
            ------------------------------------------------
            |FOO     |["COL_1"]    |NEW DESC  |REGTEST_RL  |
            ------------------------------------------------

        """
        name = SqlIdentifier(name)
        found_rows = (
            self.list_entities().filter(F.col("NAME") == name.resolved()).collect(statement_params=self._telemetry_stmp)
        )

        if len(found_rows) == 0:
            warnings.warn(
                f"Entity {name} does not exist.",
                stacklevel=2,
                category=UserWarning,
            )
            return None

        new_desc = desc if desc is not None else found_rows[0]["DESC"]

        try:
            full_name = f"{self._config.full_schema_path}.{self._get_entity_name(name)}"
            self._session.sql(f"ALTER TAG {full_name} SET COMMENT = '{new_desc}'").collect(
                statement_params=self._telemetry_stmp
            )
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to update entity `{name}`: {e}."),
            ) from e

        logger.info(f"Successfully updated Entity {name}.")
        return self.get_entity(name)

    # TODO: add support to update column desc once SNOW-894249 is fixed
    @dispatch_decorator()
    def register_feature_view(
        self,
        feature_view: FeatureView,
        version: str,
        *,
        block: bool = True,
        overwrite: bool = False,
    ) -> FeatureView:
        """
        Materialize a FeatureView to Snowflake backend.
        Incremental maintenance for updates on the source data will be automated if refresh_freq is set.
        NOTE: Each new materialization will trigger a full FeatureView history refresh for the data included in the
              FeatureView.

        Args:
            feature_view: FeatureView instance to materialize.
            version: version of the registered FeatureView.
                NOTE: Version only accepts letters, numbers and underscore. Also version will be capitalized.
            block: Specify whether the FeatureView backend materialization should be blocking or not. If blocking then
                the API will wait until the initial FeatureView data is generated. Default to true.
            overwrite: Overwrite the existing FeatureView with same version. This is the same as dropping the
                FeatureView first then recreate. NOTE: there will be backfill cost associated if the FeatureView is
                being continuously maintained.

        Returns:
            A materialized FeatureView object.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView entity has not been registered.
            SnowflakeMLException: [ValueError] Warehouse or default warehouse is not specified.
            SnowflakeMLException: [RuntimeError] Failed to create dynamic table, task, or view.
            SnowflakeMLException: [RuntimeError] Failed to find resources.

        Example::

            >>> fs = FeatureStore(...)
            >>> # draft_fv is a local object that hasn't materiaized to Snowflake backend yet.
            >>> feature_df = session.sql("select f_1, f_2 from source_table")
            >>> draft_fv = FeatureView("my_fv", [entities], feature_df)
            >>> print(draft_fv.status)
            FeatureViewStatus.DRAFT
            <BLANKLINE>
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |        |           |                    |
            -------------------------------------------
            <BLANKLINE>
            >>> # registered_fv is a local object that maps to a Snowflake backend object.
            >>> registered_fv = fs.register_feature_view(draft_fv, "v1")
            >>> print(registered_fv.status)
            FeatureViewStatus.ACTIVE
            <BLANKLINE>
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |MY_FV   |v1         |ACTIVE              |
            -------------------------------------------

        """
        version = FeatureViewVersion(version)

        if feature_view.status != FeatureViewStatus.DRAFT:
            try:
                return self._get_feature_view_if_exists(feature_view.name, str(version))
            except Exception:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(
                        f"FeatureView {feature_view.name}/{feature_view.version} status is {feature_view.status}, "
                        + "but it doesn't exist."
                    ),
                )

        for e in feature_view.entities:
            if not self._validate_entity_exists(e.name):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(f"Entity {e.name} has not been registered."),
                )

        feature_view_name = FeatureView._get_physical_name(feature_view.name, version)
        if not overwrite:
            try:
                return self._get_feature_view_if_exists(feature_view.name, str(version))
            except Exception:
                pass

        fully_qualified_name = self._get_fully_qualified_name(feature_view_name)
        refresh_freq = feature_view.refresh_freq

        if refresh_freq is not None:
            obj_info = _FeatureStoreObjInfo(_FeatureStoreObjTypes.MANAGED_FEATURE_VIEW, snowml_version.VERSION)
        else:
            obj_info = _FeatureStoreObjInfo(_FeatureStoreObjTypes.EXTERNAL_FEATURE_VIEW, snowml_version.VERSION)

        tagging_clause = [
            f"{self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)} = '{obj_info.to_json()}'",
            f"{self._get_fully_qualified_name(_FEATURE_VIEW_METADATA_TAG)} = '{feature_view._metadata().to_json()}'",
        ]
        for e in feature_view.entities:
            join_keys = [f"{key.resolved()}" for key in e.join_keys]
            tagging_clause.append(
                f"{self._get_fully_qualified_name(self._get_entity_name(e.name))} = '{','.join(join_keys)}'"
            )
        tagging_clause_str = ",\n".join(tagging_clause)

        def create_col_desc(col: StructField) -> str:
            desc = feature_view.feature_descs.get(SqlIdentifier(col.name), None)  # type: ignore[union-attr]
            desc = "" if desc is None else f"COMMENT '{desc}'"
            return f"{col.name} {desc}"

        column_descs = (
            ", ".join([f"{create_col_desc(col)}" for col in feature_view.output_schema.fields])
            if feature_view.feature_descs is not None
            else ""
        )

        if refresh_freq is not None:
            schedule_task = refresh_freq != "DOWNSTREAM" and timeparse(refresh_freq) is None
            self._create_dynamic_table(
                feature_view_name,
                feature_view,
                fully_qualified_name,
                column_descs,
                tagging_clause_str,
                schedule_task,
                feature_view.warehouse if feature_view.warehouse is not None else self._default_warehouse,
                block,
                overwrite,
            )
        else:
            try:
                overwrite_clause = " OR REPLACE" if overwrite else ""
                query = f"""CREATE{overwrite_clause} VIEW {fully_qualified_name} ({column_descs})
                    COMMENT = '{feature_view.desc}'
                    TAG (
                        {tagging_clause_str}
                    )
                    AS {feature_view.query}
                """
                self._session.sql(query).collect(statement_params=self._telemetry_stmp)
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(f"Create view {fully_qualified_name} [\n{query}\n] failed: {e}"),
                ) from e

        logger.info(f"Registered FeatureView {feature_view.name}/{version} successfully.")
        return self.get_feature_view(feature_view.name, str(version))

    @overload
    def update_feature_view(
        self,
        name: str,
        version: str,
        *,
        refresh_freq: Optional[str] = None,
        warehouse: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> FeatureView:
        ...

    @overload
    def update_feature_view(
        self,
        name: FeatureView,
        version: Optional[str] = None,
        *,
        refresh_freq: Optional[str] = None,
        warehouse: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> FeatureView:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def update_feature_view(
        self,
        name: Union[FeatureView, str],
        version: Optional[str] = None,
        *,
        refresh_freq: Optional[str] = None,
        warehouse: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> FeatureView:
        """Update a registered feature view.
            Check feature_view.py for which fields are allowed to be updated after registration.

        Args:
            name: FeatureView object or name to suspend.
            version: Optional version of feature view. Must set when argument feature_view is a str.
            refresh_freq: updated refresh frequency.
            warehouse: updated warehouse.
            desc: description of feature view.

        Returns:
            Updated FeatureView.

        Example::

            >>> fs = FeatureStore(...)
            >>> fv = FeatureView(
            ...     name='foo',
            ...     entities=[e1, e2],
            ...     feature_df=session.sql('...'),
            ...     desc='this is old description',
            ... )
            >>> fv = fs.register_feature_view(feature_view=fv, version='v1')
            >>> fs.list_feature_views().select("name", "version", "desc").show()
            ------------------------------------------------
            |"NAME"  |"VERSION"  |"DESC"                   |
            ------------------------------------------------
            |FOO     |v1         |this is old description  |
            ------------------------------------------------
            <BLANKLINE>
            >>> # update_feature_view will apply new arguments to the registered feature view.
            >>> new_fv = fs.update_feature_view(
            ...     name='foo',
            ...     version='v1',
            ...     desc='that is new descption',
            ... )
            >>> fs.list_feature_views().select("name", "version", "desc").show()
            ------------------------------------------------
            |"NAME"  |"VERSION"  |"DESC"                   |
            ------------------------------------------------
            |FOO     |v1         |THAT IS NEW DESCRIPTION  |
            ------------------------------------------------

        Raises:
            SnowflakeMLException: [RuntimeError] If FeatureView is not managed and refresh_freq is defined.
            SnowflakeMLException: [RuntimeError] Failed to update feature view.
        """
        feature_view = self._validate_feature_view_name_and_version_input(name, version)
        new_desc = desc if desc is not None else feature_view.desc

        if feature_view.status == FeatureViewStatus.STATIC:
            if refresh_freq is not None or warehouse is not None:
                full_name = f"{feature_view.name}/{feature_view.version}"
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=RuntimeError(
                        f"Static feature view '{full_name}' does not support refresh_freq and warehouse."
                    ),
                )
            new_query = f"""
                ALTER VIEW {feature_view.fully_qualified_name()} SET
                COMMENT = '{new_desc}'
            """
        else:
            warehouse = SqlIdentifier(warehouse) if warehouse else feature_view.warehouse
            # TODO(@wezhou): we need to properly handle cron expr
            new_query = f"""
                ALTER DYNAMIC TABLE {feature_view.fully_qualified_name()} SET
                TARGET_LAG = '{refresh_freq or feature_view.refresh_freq}'
                WAREHOUSE = {warehouse}
                COMMENT = '{new_desc}'
            """

        try:
            self._session.sql(new_query).collect(statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(
                    f"Update feature view {feature_view.name}/{feature_view.version} failed: {e}"
                ),
            ) from e
        return self.get_feature_view(name=feature_view.name, version=str(feature_view.version))

    @overload
    def read_feature_view(self, feature_view: str, version: str) -> DataFrame:
        ...

    @overload
    def read_feature_view(self, feature_view: FeatureView) -> DataFrame:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def read_feature_view(self, feature_view: Union[FeatureView, str], version: Optional[str] = None) -> DataFrame:
        """
        Read values from a FeatureView.

        Args:
            feature_view: A FeatureView object to read from, or the name of feature view.
                If name is provided then version also must be provided.
            version: Optional version of feature view. Must set when argument feature_view is a str.

        Returns:
            Snowpark DataFrame(lazy mode) containing the FeatureView data.

        Raises:
            SnowflakeMLException: [ValueError] version argument is missing when argument feature_view is a str.
            SnowflakeMLException: [ValueError] FeatureView is not registered.

        Example::

            >>> fs = FeatureStore(...)
            >>> # Read from feature view name and version.
            >>> fs.read_feature_view('foo', 'v1').show()
            ------------------------------------------
            |"NAME"  |"ID"  |"TITLE"  |"AGE"  |"TS"  |
            ------------------------------------------
            |jonh    |1     |boss     |20     |100   |
            |porter  |2     |manager  |30     |200   |
            ------------------------------------------
            <BLANKLINE>
            >>> # Read from feature view object.
            >>> fv = fs.get_feature_view('foo', 'v1')
            >>> fs.read_feature_view(fv).show()
            ------------------------------------------
            |"NAME"  |"ID"  |"TITLE"  |"AGE"  |"TS"  |
            ------------------------------------------
            |jonh    |1     |boss     |20     |100   |
            |porter  |2     |manager  |30     |200   |
            ------------------------------------------

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)

        if feature_view.status == FeatureViewStatus.DRAFT or feature_view.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"FeatureView {feature_view.name} has not been registered."),
            )

        return self._session.sql(f"SELECT * FROM {feature_view.fully_qualified_name()}")

    @dispatch_decorator()
    def list_feature_views(
        self,
        *,
        entity_name: Optional[str] = None,
        feature_view_name: Optional[str] = None,
    ) -> DataFrame:
        """
        List FeatureViews in the FeatureStore.
        If entity_name is specified, FeatureViews associated with that Entity will be listed.
        If feature_view_name is specified, further reducing the results to only match the specified name.

        Args:
            entity_name: Entity name.
            feature_view_name: FeatureView name.

        Returns:
            FeatureViews information as a Snowpark DataFrame.

        Example::

            >>> fs = FeatureStore(...)
            >>> draft_fv = FeatureView(
            ...     name='foo',
            ...     entities=[e1, e2],
            ...     feature_df=session.sql('...'),
            ...     desc='this is description',
            ... )
            >>> fs.register_feature_view(feature_view=draft_fv, version='v1')
            >>> fs.list_feature_views().select("name", "version", "desc").show()
            --------------------------------------------
            |"NAME"  |"VERSION"  |"DESC"               |
            --------------------------------------------
            |FOO     |v1         |this is description  |
            --------------------------------------------

        """
        if feature_view_name is not None:
            feature_view_name = SqlIdentifier(feature_view_name)

        if entity_name is not None:
            entity_name = SqlIdentifier(entity_name)
            return self._optimized_find_feature_views(entity_name, feature_view_name)
        else:
            output_values: List[List[Any]] = []
            for row, _ in self._get_fv_backend_representations(feature_view_name, prefix_match=True):
                self._extract_feature_view_info(row, output_values)
            return self._session.create_dataframe(output_values, schema=_LIST_FEATURE_VIEW_SCHEMA)

    @dispatch_decorator()
    def get_feature_view(self, name: str, version: str) -> FeatureView:
        """
        Retrieve previously registered FeatureView.

        Args:
            name: FeatureView name.
            version: FeatureView version.

        Returns:
            FeatureView object.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView with name and version is not found,
                or incurred exception when reconstructing the FeatureView object.

        Example::

            >>> fs = FeatureStore(...)
            >>> # draft_fv is a local object that hasn't materiaized to Snowflake backend yet.
            >>> draft_fv = FeatureView(
            ...     name='foo',
            ...     entities=[e1],
            ...     feature_df=session.sql('...'),
            ...     desc='this is description',
            ... )
            >>> fs.register_feature_view(feature_view=draft_fv, version='v1')
            <BLANKLINE>
            >>> # fv is a local object that maps to a Snowflake backend object.
            >>> fv = fs.get_feature_view('foo', 'v1')
            >>> print(f"name: {fv.name}")
            >>> print(f"version:{fv.version}")
            >>> print(f"desc:{fv.desc}")
            name: FOO
            version:v1
            desc:this is description

        """
        name = SqlIdentifier(name)
        version = FeatureViewVersion(version)

        fv_name = FeatureView._get_physical_name(name, version)
        results = self._get_fv_backend_representations(fv_name)
        if len(results) != 1:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Failed to find FeatureView {name}/{version}: {results}"),
            )

        return self._compose_feature_view(
            results[0][0], results[0][1], self.list_entities().collect(statement_params=self._telemetry_stmp)
        )

    @overload
    def refresh_feature_view(self, feature_view: FeatureView) -> None:
        ...

    @overload
    def refresh_feature_view(self, feature_view: str, version: str) -> None:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def refresh_feature_view(self, feature_view: Union[FeatureView, str], version: Optional[str] = None) -> None:
        """Manually refresh a feature view.

        Args:
            feature_view: A registered feature view object, or the name of feature view.
            version: Optional version of feature view. Must set when argument feature_view is a str.

        Example::

            >>> fs = FeatureStore(...)
            >>> fv = fs.get_feature_view(name='MY_FV', version='v1')
            <BLANKLINE>
            >>> # refresh with name and version
            >>> fs.refresh_feature_view('MY_FV', 'v1')
            >>> fs.get_refresh_history('MY_FV', 'v1').show()
            -----------------------------------------------------------------------------------------------------
            |"NAME"    |"STATE"    |"REFRESH_START_TIME"        |"REFRESH_END_TIME"          |"REFRESH_ACTION"  |
            -----------------------------------------------------------------------------------------------------
            |MY_FV$v1  |SUCCEEDED  |2024-07-10 14:53:58.504000  |2024-07-10 14:53:59.088000  |INCREMENTAL       |
            -----------------------------------------------------------------------------------------------------
            <BLANKLINE>
            >>> # refresh with feature view object
            >>> fs.refresh_feature_view(fv)
            >>> fs.get_refresh_history(fv).show()
            -----------------------------------------------------------------------------------------------------
            |"NAME"    |"STATE"    |"REFRESH_START_TIME"        |"REFRESH_END_TIME"          |"REFRESH_ACTION"  |
            -----------------------------------------------------------------------------------------------------
            |MY_FV$v1  |SUCCEEDED  |2024-07-10 14:54:06.680000  |2024-07-10 14:54:07.226000  |INCREMENTAL       |
            |MY_FV$v1  |SUCCEEDED  |2024-07-10 14:53:58.504000  |2024-07-10 14:53:59.088000  |INCREMENTAL       |
            -----------------------------------------------------------------------------------------------------

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)

        if feature_view.status == FeatureViewStatus.STATIC:
            warnings.warn(
                "Static feature view can't be refreshed. You must set refresh_freq when register_feature_view().",
                stacklevel=2,
                category=UserWarning,
            )
            return
        self._update_feature_view_status(feature_view, "REFRESH")

    @overload
    def get_refresh_history(
        self, feature_view: FeatureView, version: Optional[str] = None, *, verbose: bool = False
    ) -> DataFrame:
        ...

    @overload
    def get_refresh_history(self, feature_view: str, version: str, *, verbose: bool = False) -> DataFrame:
        ...

    def get_refresh_history(
        self, feature_view: Union[FeatureView, str], version: Optional[str] = None, *, verbose: bool = False
    ) -> DataFrame:
        """Get refresh hisotry statistics about a feature view.

        Args:
            feature_view: A registered feature view object, or the name of feature view.
            version: Optional version of feature view. Must set when argument feature_view is a str.
            verbose: Return more detailed history when set true.

        Returns:
            A dataframe contains the refresh history information.

        Example::

            >>> fs = FeatureStore(...)
            >>> fv = fs.get_feature_view(name='MY_FV', version='v1')
            >>> # refresh with name and version
            >>> fs.refresh_feature_view('MY_FV', 'v1')
            >>> fs.get_refresh_history('MY_FV', 'v1').show()
            -----------------------------------------------------------------------------------------------------
            |"NAME"    |"STATE"    |"REFRESH_START_TIME"        |"REFRESH_END_TIME"          |"REFRESH_ACTION"  |
            -----------------------------------------------------------------------------------------------------
            |MY_FV$v1  |SUCCEEDED  |2024-07-10 14:53:58.504000  |2024-07-10 14:53:59.088000  |INCREMENTAL       |
            -----------------------------------------------------------------------------------------------------
            <BLANKLINE>
            >>> # refresh with feature view object
            >>> fs.refresh_feature_view(fv)
            >>> fs.get_refresh_history(fv).show()
            -----------------------------------------------------------------------------------------------------
            |"NAME"    |"STATE"    |"REFRESH_START_TIME"        |"REFRESH_END_TIME"          |"REFRESH_ACTION"  |
            -----------------------------------------------------------------------------------------------------
            |MY_FV$v1  |SUCCEEDED  |2024-07-10 14:54:06.680000  |2024-07-10 14:54:07.226000  |INCREMENTAL       |
            |MY_FV$v1  |SUCCEEDED  |2024-07-10 14:53:58.504000  |2024-07-10 14:53:59.088000  |INCREMENTAL       |
            -----------------------------------------------------------------------------------------------------

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)

        if feature_view.status == FeatureViewStatus.STATIC:
            warnings.warn(
                "Static feature view never refreshes.",
                stacklevel=2,
                category=UserWarning,
            )
            return self._session.create_dataframe([Row()])

        if feature_view.status == FeatureViewStatus.DRAFT:
            warnings.warn(
                "This feature view has not been registered thus has no refresh history.",
                stacklevel=2,
                category=UserWarning,
            )
            return self._session.create_dataframe([Row()])

        fv_resolved_name = FeatureView._get_physical_name(
            feature_view.name,
            feature_view.version,  # type: ignore[arg-type]
        ).resolved()
        select_cols = "*" if verbose else "name, state, refresh_start_time, refresh_end_time, refresh_action"
        return self._session.sql(
            f"""
            SELECT
                {select_cols}
            FROM TABLE (
                {self._config.database}.INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY ()
            )
            WHERE NAME = '{fv_resolved_name}'
            AND SCHEMA_NAME = '{self._config.schema}'
            """
        )

    @overload
    def resume_feature_view(self, feature_view: FeatureView) -> FeatureView:
        ...

    @overload
    def resume_feature_view(self, feature_view: str, version: str) -> FeatureView:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def resume_feature_view(self, feature_view: Union[FeatureView, str], version: Optional[str] = None) -> FeatureView:
        """
        Resume a previously suspended FeatureView.

        Args:
            feature_view: FeatureView object or name to resume.
            version: Optional version of feature view. Must set when argument feature_view is a str.

        Returns:
            A new feature view with updated status.

        Example::

            >>> fs = FeatureStore(...)
            >>> # you must already have feature views registered
            >>> fv = fs.get_feature_view(name='MY_FV', version='v1')
            >>> fs.suspend_feature_view('MY_FV', 'v1')
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |MY_FV   |v1         |SUSPENDED           |
            -------------------------------------------
            <BLANKLINE>
            >>> fs.resume_feature_view('MY_FV', 'v1')
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |MY_FV   |v1         |ACTIVE              |
            -------------------------------------------

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)
        return self._update_feature_view_status(feature_view, "RESUME")

    @overload
    def suspend_feature_view(self, feature_view: FeatureView) -> FeatureView:
        ...

    @overload
    def suspend_feature_view(self, feature_view: str, version: str) -> FeatureView:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def suspend_feature_view(self, feature_view: Union[FeatureView, str], version: Optional[str] = None) -> FeatureView:
        """
        Suspend an active FeatureView.

        Args:
            feature_view: FeatureView object or name to suspend.
            version: Optional version of feature view. Must set when argument feature_view is a str.

        Returns:
            A new feature view with updated status.

        Example::

            >>> fs = FeatureStore(...)
            >>> # assume you already have feature views registered
            >>> fv = fs.get_feature_view(name='MY_FV', version='v1')
            >>> fs.suspend_feature_view('MY_FV', 'v1')
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |MY_FV   |v1         |SUSPENDED           |
            -------------------------------------------
            <BLANKLINE>
            >>> fs.resume_feature_view('MY_FV', 'v1')
            >>> fs.list_feature_views().select("NAME", "VERSION", "SCHEDULING_STATE").show()
            -------------------------------------------
            |"NAME"  |"VERSION"  |"SCHEDULING_STATE"  |
            -------------------------------------------
            |MY_FV   |v1         |ACTIVE              |
            -------------------------------------------

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)
        return self._update_feature_view_status(feature_view, "SUSPEND")

    @overload
    def delete_feature_view(self, feature_view: FeatureView) -> None:
        ...

    @overload
    def delete_feature_view(self, feature_view: str, version: str) -> None:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def delete_feature_view(self, feature_view: Union[FeatureView, str], version: Optional[str] = None) -> None:
        """
        Delete a FeatureView.

        Args:
            feature_view: FeatureView object or name to delete.
            version: Optional version of feature view. Must set when argument feature_view is a str.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView is not registered.

        Example::

            >>> fs = FeatureStore(...)
            >>> fv = FeatureView('FV0', ...)
            >>> fv1 = fs.register_feature_view(fv, 'FIRST')
            >>> fv2 = fs.register_feature_view(fv, 'SECOND')
            >>> fs.list_feature_views().select('NAME', 'VERSION').show()
            ----------------------
            |"NAME"  |"VERSION"  |
            ----------------------
            |FV0     |SECOND     |
            |FV0     |FIRST      |
            ----------------------
            <BLANKLINE>
            >>> # delete with name and version
            >>> fs.delete_feature_view('FV0', 'FIRST')
            >>> fs.list_feature_views().select('NAME', 'VERSION').show()
            ----------------------
            |"NAME"  |"VERSION"  |
            ----------------------
            |FV0     |SECOND     |
            ----------------------
            <BLANKLINE>
            >>> # delete with feature view object
            >>> fs.delete_feature_view(fv2)
            >>> fs.list_feature_views().select('NAME', 'VERSION').show()
            ----------------------
            |"NAME"  |"VERSION"  |
            ----------------------
            |        |           |
            ----------------------

        """
        feature_view = self._validate_feature_view_name_and_version_input(feature_view, version)

        # TODO: we should leverage lineage graph to check downstream deps, and block the deletion
        # if there're other FVs depending on this
        if feature_view.status == FeatureViewStatus.DRAFT or feature_view.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"FeatureView {feature_view.name} has not been registered."),
            )

        fully_qualified_name = feature_view.fully_qualified_name()
        if feature_view.status == FeatureViewStatus.STATIC:
            self._session.sql(f"DROP VIEW IF EXISTS {fully_qualified_name}").collect(
                statement_params=self._telemetry_stmp
            )
        else:
            self._session.sql(f"DROP DYNAMIC TABLE IF EXISTS {fully_qualified_name}").collect(
                statement_params=self._telemetry_stmp
            )
            if feature_view.refresh_freq == "DOWNSTREAM":
                self._session.sql(f"DROP TASK IF EXISTS {fully_qualified_name}").collect(
                    statement_params=self._telemetry_stmp
                )

        logger.info(f"Deleted FeatureView {feature_view.name}/{feature_view.version}.")

    @dispatch_decorator()
    def list_entities(self) -> DataFrame:
        """
        List all Entities in the FeatureStore.

        Returns:
            Snowpark DataFrame containing the results.

        Example::

            >>> fs = FeatureStore(...)
            >>> e_1 = Entity("my_entity", ['col_1'], desc='My first entity.')
            >>> fs.register_entity(e_1)
            >>> fs.list_entities().show()
            -----------------------------------------------------------
            |"NAME"     |"JOIN_KEYS"  |"DESC"            |"OWNER"     |
            -----------------------------------------------------------
            |MY_ENTITY  |["COL_1"]    |My first entity.  |REGTEST_RL  |
            -----------------------------------------------------------

        """
        prefix_len = len(_ENTITY_TAG_PREFIX) + 1
        return cast(
            DataFrame,
            self._session.sql(
                f"SHOW TAGS LIKE '{_ENTITY_TAG_PREFIX}%' IN SCHEMA {self._config.full_schema_path}"
            ).select(
                F.col('"name"').substr(prefix_len, _ENTITY_NAME_LENGTH_LIMIT).alias("NAME"),
                F.col('"allowed_values"').alias("JOIN_KEYS"),
                F.col('"comment"').alias("DESC"),
                F.col('"owner"').alias("OWNER"),
            ),
        )

    @dispatch_decorator()
    def get_entity(self, name: str) -> Entity:
        """
        Retrieve previously registered Entity object.

        Args:
            name: Entity name.

        Returns:
            Entity object.

        Raises:
            SnowflakeMLException: [ValueError] Entity is not found.
            SnowflakeMLException: [RuntimeError] Failed to retrieve tag reference information.
            SnowflakeMLException: [RuntimeError] Failed to find resources.

        Example::

            >>> fs = FeatureStore(...)
            >>> # e_1 is a local object that hasn't registered to Snowflake backend yet.
            >>> e_1 = Entity("my_entity", ['col_1'], desc='My first entity.')
            >>> fs.register_entity(e_1)
            <BLANKLINE>
            >>> # e_2 is a local object that points a backend object in Snowflake.
            >>> e_2 = fs.get_entity("my_entity")
            >>> print(e_2)
            Entity(name=MY_ENTITY, join_keys=['COL_1'], owner=REGTEST_RL, desc=My first entity.)

        """
        name = SqlIdentifier(name)
        try:
            result = (
                self.list_entities()
                .filter(F.col("NAME") == name.resolved())
                .collect(statement_params=self._telemetry_stmp)
            )
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to list entities: {e}"),
            ) from e
        if len(result) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Cannot find Entity with name: {name}."),
            )

        join_keys = self._recompose_join_keys(result[0]["JOIN_KEYS"])

        return Entity._construct_entity(
            name=SqlIdentifier(result[0]["NAME"], case_sensitive=True).identifier(),
            join_keys=join_keys,
            desc=result[0]["DESC"],
            owner=result[0]["OWNER"],
        )

    @dispatch_decorator()
    def delete_entity(self, name: str) -> None:
        """
        Delete a previously registered Entity.

        Args:
            name: Name of entity to be deleted.

        Raises:
            SnowflakeMLException: [ValueError] Entity with given name not exists.
            SnowflakeMLException: [RuntimeError] Failed to alter schema or drop tag.
            SnowflakeMLException: [RuntimeError] Failed to find resources.

        Example::

            >>> fs = FeatureStore(...)
            >>> e_1 = Entity("my_entity", ['col_1'], desc='My first entity.')
            >>> fs.register_entity(e_1)
            >>> fs.list_entities().show()
            -----------------------------------------------------------
            |"NAME"     |"JOIN_KEYS"  |"DESC"            |"OWNER"     |
            -----------------------------------------------------------
            |MY_ENTITY  |["COL_1"]    |My first entity.  |REGTEST_RL  |
            -----------------------------------------------------------
            <BLANKLINE>
            >>> fs.delete_entity("my_entity")
            >>> fs.list_entities().show()
            -------------------------------------------
            |"NAME"  |"JOIN_KEYS"  |"DESC"  |"OWNER"  |
            -------------------------------------------
            |        |             |        |         |
            -------------------------------------------

        """
        name = SqlIdentifier(name)

        if not self._validate_entity_exists(name):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Entity {name} does not exist."),
            )

        active_feature_views = self.list_feature_views(entity_name=name).collect(statement_params=self._telemetry_stmp)

        if len(active_feature_views) > 0:
            active_fvs = [r["NAME"] for r in active_feature_views]
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_DELETE_FAILED,
                original_exception=ValueError(f"Cannot delete Entity {name} due to active FeatureViews: {active_fvs}."),
            )

        tag_name = self._get_fully_qualified_name(self._get_entity_name(name))
        try:
            self._session.sql(f"DROP TAG IF EXISTS {tag_name}").collect(statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to delete entity: {e}."),
            ) from e
        logger.info(f"Deleted Entity {name}.")

    @dispatch_decorator()
    def retrieve_feature_values(
        self,
        spine_df: DataFrame,
        features: Union[List[Union[FeatureView, FeatureViewSlice]], List[str]],
        *,
        spine_timestamp_col: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None,
        include_feature_view_timestamp_col: bool = False,
    ) -> DataFrame:
        """
        Enrich spine dataframe with feature values. Mainly used to generate inference data input.
        If spine_timestamp_col is specified, point-in-time feature values will be fetched.

        Args:
            spine_df: Snowpark DataFrame to join features into.
            features: List of features to join into the spine_df. Can be a list of FeatureView or FeatureViewSlice,
                or a list of serialized feature objects from Dataset.
            spine_timestamp_col: Timestamp column in spine_df for point-in-time feature value lookup.
            exclude_columns: Column names to exclude from the result dataframe.
            include_feature_view_timestamp_col: Generated dataset will include timestamp column of feature view
                (if feature view has timestamp column) if set true. Default to false.

        Returns:
            Snowpark DataFrame containing the joined results.

        Raises:
            ValueError: if features is empty.

        Example::

            >>> fs = FeatureStore(...)
            >>> # Assume you already have feature view registered.
            >>> fv = fs.get_feature_view('my_fv', 'v1')
            >>> # Spine dataframe has same join keys as the entity of fv.
            >>> spine_df = session.create_dataframe(["1", "2"], schema=["id"])
            >>> fs.retrieve_feature_values(spine_df, [fv]).show()
            --------------------
            |"END_STATION_ID"  |
            --------------------
            |505               |
            |347               |
            |466               |
            --------------------

        """
        if spine_timestamp_col is not None:
            spine_timestamp_col = SqlIdentifier(spine_timestamp_col)

        if len(features) == 0:
            raise ValueError("features cannot be empty")
        if isinstance(features[0], str):
            features = self._load_serialized_feature_views(cast(List[str], features))

        df, _ = self._join_features(
            spine_df,
            cast(List[Union[FeatureView, FeatureViewSlice]], features),
            spine_timestamp_col,
            include_feature_view_timestamp_col,
        )

        if exclude_columns is not None:
            df = self._exclude_columns(df, exclude_columns)

        return df

    @dispatch_decorator()
    def generate_training_set(
        self,
        spine_df: DataFrame,
        features: List[Union[FeatureView, FeatureViewSlice]],
        *,
        save_as: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        include_feature_view_timestamp_col: bool = False,
    ) -> DataFrame:
        """
        Generate a training set from the specified Spine DataFrame and Feature Views. Result is
        materialized to a Snowflake Table if `save_as` is specified.

        Args:
            spine_df: Snowpark DataFrame to join features into.
            features: A list of FeatureView or FeatureViewSlice which contains features to be joined.
            save_as: If specified, a new table containing the produced result will be created. Name can be a fully
                qualified name or an unqualified name. If unqualified, defaults to the Feature Store database and schema
            spine_timestamp_col: Name of timestamp column in spine_df that will be used to join
                time-series features. If spine_timestamp_col is not none, the input features also must have
                timestamp_col.
            spine_label_cols: Name of column(s) in spine_df that contains labels.
            exclude_columns: Name of column(s) to exclude from the resulting training set.
            include_feature_view_timestamp_col: Generated dataset will include timestamp column of feature view
                (if feature view has timestamp column) if set true. Default to false.

        Returns:
            Returns a Snowpark DataFrame representing the training set.

        Raises:
            SnowflakeMLException: [RuntimeError] Materialized table name already exists
            SnowflakeMLException: [RuntimeError] Failed to create materialized table.

        Example::

            >>> fs = FeatureStore(session, ...)
            >>> # Assume you already have feature view registered.
            >>> fv = fs.get_feature_view("MY_FV", "1")
            >>> # Spine dataframe has same join keys as the entity of fv.
            >>> spine_df = session.create_dataframe(["1", "2"], schema=["id"])
            >>> training_set = fs.generate_training_set(
            ...     spine_df,
            ...     [fv],
            ...     save_as="my_training_set",
            ... )
            >>> print(type(training_set))
            <class 'snowflake.snowpark.table.Table'>
            <BLANKLINE>
            >>> print(training_set.queries)
            {'queries': ['SELECT  *  FROM (my_training_set)'], 'post_actions': []}

        """
        if spine_timestamp_col is not None:
            spine_timestamp_col = SqlIdentifier(spine_timestamp_col)
        if spine_label_cols is not None:
            spine_label_cols = to_sql_identifiers(spine_label_cols)  # type: ignore[assignment]

        result_df, join_keys = self._join_features(
            spine_df, features, spine_timestamp_col, include_feature_view_timestamp_col
        )

        if exclude_columns is not None:
            result_df = self._exclude_columns(result_df, exclude_columns)

        if save_as is not None:
            try:
                save_as = self._get_fully_qualified_name(save_as)
                result_df.write.mode("errorifexists").save_as_table(save_as, statement_params=self._telemetry_stmp)

                # Add tag
                task_obj_info = _FeatureStoreObjInfo(_FeatureStoreObjTypes.TRAINING_DATA, snowml_version.VERSION)
                self._session.sql(
                    f"""
                    ALTER TABLE {save_as}
                    SET TAG {self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)}='{task_obj_info.to_json()}'
                    """
                ).collect(statement_params=self._telemetry_stmp)

                return self._session.table(save_as)

            except SnowparkSQLException as e:
                if e.sql_error_code == sql_error_codes.OBJECT_ALREADY_EXISTS:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.OBJECT_ALREADY_EXISTS,
                        original_exception=RuntimeError(str(e)),
                    ) from e
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(f"An error occurred during training set materialization: {e}."),
                ) from e
        return result_df

    @overload
    def generate_dataset(
        self,
        name: str,
        spine_df: DataFrame,
        features: List[Union[FeatureView, FeatureViewSlice]],
        *,
        version: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        include_feature_view_timestamp_col: bool = False,
        desc: str = "",
        output_type: Literal["dataset"] = "dataset",
    ) -> dataset.Dataset:
        ...

    @overload
    def generate_dataset(
        self,
        name: str,
        spine_df: DataFrame,
        features: List[Union[FeatureView, FeatureViewSlice]],
        *,
        output_type: Literal["table"],
        version: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        include_feature_view_timestamp_col: bool = False,
        desc: str = "",
    ) -> DataFrame:
        ...

    @dispatch_decorator()  # type: ignore[misc]
    def generate_dataset(
        self,
        name: str,
        spine_df: DataFrame,
        features: List[Union[FeatureView, FeatureViewSlice]],
        *,
        version: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        include_feature_view_timestamp_col: bool = False,
        desc: str = "",
        output_type: Literal["dataset", "table"] = "dataset",
    ) -> Union[dataset.Dataset, DataFrame]:
        """
        Generate dataset by given source table and feature views.

        Args:
            name: The name of the Dataset to be generated. Datasets are uniquely identified within a schema
                by their name and version.
            spine_df: Snowpark DataFrame to join features into.
            features: A list of FeatureView or FeatureViewSlice which contains features to be joined.
            version: The version of the Dataset to be generated. If none specified, the current timestamp
                will be used instead.
            spine_timestamp_col: Name of timestamp column in spine_df that will be used to join
                time-series features. If spine_timestamp_col is not none, the input features also must have
                timestamp_col.
            spine_label_cols: Name of column(s) in spine_df that contains labels.
            exclude_columns: Name of column(s) to exclude from the resulting training set.
            include_feature_view_timestamp_col: Generated dataset will include timestamp column of feature view
                (if feature view has timestamp column) if set true. Default to false.
            desc: A description about this dataset.
            output_type: (Deprecated) The type of Snowflake storage to use for the generated training data.

        Returns:
            If output_type is "dataset" (default), returns a Dataset object.
            If output_type is "table", returns a Snowpark DataFrame representing the table.

        Raises:
            SnowflakeMLException: [ValueError] Invalid output_type specified.
            SnowflakeMLException: [RuntimeError] Dataset name/version already exists.
            SnowflakeMLException: [RuntimeError] Failed to find resources.

        Example::

            >>> fs = FeatureStore(session, ...)
            >>> # Assume you already have feature view registered.
            >>> fv = fs.get_feature_view("MY_FV", "1")
            >>> # Spine dataframe has same join keys as the entity of fv.
            >>> spine_df = session.create_dataframe(["1", "2"], schema=["id"])
            >>> my_dataset = fs.generate_dataset(
            ...     "my_dataset"
            ...     spine_df,
            ...     [fv],
            ... )
            >>> # Current timestamp will be used as default version name.
            >>> # You can explicitly overwrite by setting a version.
            >>> my_dataset.list_versions()
            ['2024_07_12_11_26_22']
            <BLANKLINE>
            >>> my_dataset.read.to_snowpark_dataframe().show(n=3)
            -------------------------------------------------------
            |"QUALITY"  |"FIXED_ACIDITY"     |"VOLATILE_ACIDITY"  |
            -------------------------------------------------------
            |3          |11.600000381469727  |0.5799999833106995  |
            |3          |8.300000190734863   |1.0199999809265137  |
            |3          |7.400000095367432   |1.184999942779541   |
            -------------------------------------------------------

        """
        if output_type not in {"table", "dataset"}:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(f"Invalid output_type: {output_type}."),
            )

        # Convert name to fully qualified name if not already fully qualified
        name = self._get_fully_qualified_name(name)
        version = version or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        fs_meta = FeatureStoreMetadata(
            spine_query=spine_df.queries["queries"][-1],
            compact_feature_views=[fv._get_compact_repr().to_json() for fv in features],
            spine_timestamp_col=spine_timestamp_col,
        )

        # Only set a save_as name if output_type is table
        table_name = f"{name}_{version}" if output_type == "table" else None
        result_df = self.generate_training_set(
            spine_df,
            features,
            spine_timestamp_col=spine_timestamp_col,
            spine_label_cols=spine_label_cols,
            exclude_columns=exclude_columns,
            include_feature_view_timestamp_col=include_feature_view_timestamp_col,
            save_as=table_name,
        )
        if output_type == "table":
            warnings.warn(
                "Generating a table from generate_dataset() is deprecated and will be removed in a future release,"
                " use generate_training_set() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return result_df

        try:
            assert output_type == "dataset"
            if not self._is_dataset_enabled():
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.SNOWML_CREATE_FAILED,
                    original_exception=RuntimeError(
                        "Dataset is not enabled in your account. Ask your account admin to set"
                        " FEATURE_DATASET=ENABLED or use generate_training_set() instead"
                        " to generate the data as a Snowflake Table."
                    ),
                )
            # TODO: Add feature store tag once Dataset (version) supports tags
            ds: dataset.Dataset = dataset.create_from_dataframe(
                self._session,
                name,
                version,
                input_dataframe=result_df,
                exclude_cols=[spine_timestamp_col] if spine_timestamp_col is not None else [],
                label_cols=spine_label_cols,
                properties=fs_meta,
                comment=desc,
            )
            return ds

        except dataset_errors.DatasetExistError as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.OBJECT_ALREADY_EXISTS,
                original_exception=RuntimeError(str(e)),
            ) from e
        except SnowparkSQLException as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"An error occurred during dataset generation: {e}."),
            ) from e

    @dispatch_decorator()
    def load_feature_views_from_dataset(self, ds: dataset.Dataset) -> List[Union[FeatureView, FeatureViewSlice]]:
        """
        Retrieve FeatureViews used during Dataset construction.

        Args:
            ds: Dataset object created from feature store.

        Returns:
            List of FeatureViews used during Dataset construction.

        Raises:
            ValueError: if dataset object is not generated from feature store.

        Example::

            >>> fs = FeatureStore(session, ...)
            >>> # Assume you already have feature view registered.
            >>> fv = fs.get_feature_view("MY_FV", "1.0")
            >>> # Spine dataframe has same join keys as the entity of fv.
            >>> spine_df = session.create_dataframe(["1", "2"], schema=["id"])
            >>> my_dataset = fs.generate_dataset(
            ...     "my_dataset"
            ...     spine_df,
            ...     [fv],
            ... )
            >>> fvs = fs.load_feature_views_from_dataset(my_dataset)
            >>> print(len(fvs))
            1
            <BLANKLINE>
            >>> print(type(fvs[0]))
            <class 'snowflake.ml.feature_store.feature_view.FeatureView'>
            <BLANKLINE>
            >>> print(fvs[0].name)
            MY_FV
            <BLANKLINE>
            >>> print(fvs[0].version)
            1.0

        """
        assert ds.selected_version is not None
        source_meta = ds.selected_version._get_metadata()
        if (
            source_meta is None
            or not isinstance(source_meta.properties, FeatureStoreMetadata)
            or (
                source_meta.properties.serialized_feature_views is None
                and source_meta.properties.compact_feature_views is None
            )
        ):
            raise ValueError(f"Dataset {ds} does not contain valid feature view information.")

        properties = source_meta.properties
        if properties.serialized_feature_views:
            return self._load_serialized_feature_views(properties.serialized_feature_views)
        else:
            return self._load_compact_feature_views(properties.compact_feature_views)  # type: ignore[arg-type]

    @dispatch_decorator()
    def _clear(self, dryrun: bool = True) -> None:
        """
        Clear all feature views and entities. Note Feature Store schema and metadata will NOT be purged
        together. Use SQL to delete schema and metadata instead.

        Args:
            dryrun: Print a list of objects will be deleted but not actually perform the deletion when true.
        """
        warnings.warn(
            "It will clear ALL feature views and entities in this Feature Store. Make sure your role"
            " has sufficient access to all feature views and entities. Insufficient access to some feature"
            " views or entities will leave Feature Store in an incomplete state.",
            stacklevel=2,
            category=UserWarning,
        )

        all_fvs_df = self.list_feature_views()
        all_entities_df = self.list_entities()
        all_fvs_rows = all_fvs_df.collect(statement_params=self._telemetry_stmp)
        all_entities_rows = all_entities_df.collect(statement_params=self._telemetry_stmp)

        if dryrun:
            logger.info(
                "Following feature views and entities will be deleted."
                + " Set 'dryrun=False' to perform the actual deletion.",
            )
            logger.info(f"Total {len(all_fvs_rows)} Feature views to be deleted:")
            all_fvs_df.show(n=len(all_fvs_rows))
            logger.info(f"\nTotal {len(all_entities_rows)} Entities to be deleted:")
            all_entities_df.show(n=len(all_entities_rows))
            return

        for fv_row in all_fvs_rows:
            fv = self.get_feature_view(
                SqlIdentifier(fv_row["NAME"], case_sensitive=True).identifier(), fv_row["VERSION"]
            )
            self.delete_feature_view(fv)

        for entity_row in all_entities_rows:
            self.delete_entity(SqlIdentifier(entity_row["NAME"], case_sensitive=True).identifier())

        logger.info(f"Feature store {self._config.full_schema_path} has been cleared.")

    def _get_feature_view_if_exists(self, name: str, version: str) -> FeatureView:
        existing_fv = self.get_feature_view(name, version)
        warnings.warn(
            f"FeatureView {name}/{version} already exists. Skip registration."
            + " Set `overwrite` to True if you want to replace existing FeatureView.",
            stacklevel=2,
            category=UserWarning,
        )
        return existing_fv

    def _recompose_join_keys(self, join_key: str) -> List[str]:
        # ALLOWED_VALUES in TAG will follow format ["key_1,key2,..."]
        # since keys are already resolved following the SQL identifier rule on the write path,
        # we simply parse the keys back and wrap them with quotes to preserve cases
        # Example join_key repr from TAG value: "[key1,key2,key3]"
        join_keys = join_key[2:-2].split(",")
        res = []
        for k in join_keys:
            res.append(f'"{k}"')
        return res

    def _create_dynamic_table(
        self,
        feature_view_name: SqlIdentifier,
        feature_view: FeatureView,
        fully_qualified_name: str,
        column_descs: str,
        tagging_clause: str,
        schedule_task: bool,
        warehouse: SqlIdentifier,
        block: bool,
        override: bool,
    ) -> None:
        # TODO: cluster by join keys once DT supports that
        try:
            override_clause = " OR REPLACE" if override else ""
            query = f"""CREATE{override_clause} DYNAMIC TABLE {fully_qualified_name} ({column_descs})
                TARGET_LAG = '{'DOWNSTREAM' if schedule_task else feature_view.refresh_freq}'
                COMMENT = '{feature_view.desc}'
                TAG (
                    {tagging_clause}
                )
                WAREHOUSE = {warehouse}
                REFRESH_MODE = {feature_view.refresh_mode}
                INITIALIZE = {feature_view.initialize}
            """
            if feature_view.cluster_by:
                cluster_by_clause = f"CLUSTER BY ({', '.join(feature_view.cluster_by)})"
                query += f"{cluster_by_clause}"

            query += f"""
                AS {feature_view.query}
            """
            self._session.sql(query).collect(block=block, statement_params=self._telemetry_stmp)

            if schedule_task:
                task_obj_info = _FeatureStoreObjInfo(
                    _FeatureStoreObjTypes.FEATURE_VIEW_REFRESH_TASK, snowml_version.VERSION
                )
                try:
                    self._session.sql(
                        f"""CREATE{override_clause} TASK {fully_qualified_name}
                            WAREHOUSE = {warehouse}
                            SCHEDULE = 'USING CRON {feature_view.refresh_freq}'
                            AS ALTER DYNAMIC TABLE {fully_qualified_name} REFRESH
                        """
                    ).collect(statement_params=self._telemetry_stmp)
                    self._session.sql(
                        f"""
                        ALTER TASK {fully_qualified_name}
                        SET TAG {self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)}='{task_obj_info.to_json()}'
                    """
                    ).collect(statement_params=self._telemetry_stmp)
                    self._session.sql(f"ALTER TASK {fully_qualified_name} RESUME").collect(
                        statement_params=self._telemetry_stmp
                    )
                except Exception:
                    self._session.sql(f"DROP DYNAMIC TABLE IF EXISTS {fully_qualified_name}").collect(
                        statement_params=self._telemetry_stmp
                    )
                    self._session.sql(f"DROP TASK IF EXISTS {fully_qualified_name}").collect(
                        statement_params=self._telemetry_stmp
                    )
                    raise
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(
                    f"Create dynamic table [\n{query}\n] or task {fully_qualified_name} failed: {e}."
                ),
            ) from e

        if block:
            self._check_dynamic_table_refresh_mode(feature_view_name)

    def _check_dynamic_table_refresh_mode(self, feature_view_name: SqlIdentifier) -> None:
        found_dts = self._find_object("DYNAMIC TABLES", feature_view_name)
        if len(found_dts) != 1:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Can not find dynamic table: `{feature_view_name}`."),
            )
        if found_dts[0]["refresh_mode"] != "INCREMENTAL":
            warnings.warn(
                "Your pipeline won't be incrementally refreshed due to: "
                + f"\"{found_dts[0]['refresh_mode_reason']}\".",
                stacklevel=2,
                category=UserWarning,
            )

    def _validate_entity_exists(self, name: SqlIdentifier) -> bool:
        full_entity_tag_name = self._get_entity_name(name)
        found_rows = self._find_object("TAGS", full_entity_tag_name)
        return len(found_rows) == 1

    def _join_features(
        self,
        spine_df: DataFrame,
        features: List[Union[FeatureView, FeatureViewSlice]],
        spine_timestamp_col: Optional[SqlIdentifier],
        include_feature_view_timestamp_col: bool,
    ) -> Tuple[DataFrame, List[SqlIdentifier]]:
        for f in features:
            f = f.feature_view_ref if isinstance(f, FeatureViewSlice) else f
            if f.status == FeatureViewStatus.DRAFT:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(f"FeatureView {f.name} has not been registered."),
                )
            for e in f.entities:
                for k in e.join_keys:
                    if k not in to_sql_identifiers(spine_df.columns):
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_ARGUMENT,
                            original_exception=ValueError(
                                f"join_key {k} from Entity {e.name} in FeatureView {f.name} is not found in spine_df."
                            ),
                        )

        if self._asof_join_enabled is None:
            self._asof_join_enabled = self._is_asof_join_enabled()

        # TODO: leverage Snowpark dataframe for more concise syntax once it supports AsOfJoin
        query = spine_df.queries["queries"][-1]
        layer = 0
        for f in features:
            if isinstance(f, FeatureViewSlice):
                cols = f.names
                f = f.feature_view_ref
            else:
                cols = f.feature_names

            join_keys = list({k for e in f.entities for k in e.join_keys})
            join_keys_str = ", ".join(join_keys)
            assert f.version is not None
            join_table_name = f.fully_qualified_name()

            if spine_timestamp_col is not None and f.timestamp_col is not None:
                if self._asof_join_enabled:
                    if include_feature_view_timestamp_col:
                        f_ts_col_alias = identifier.concat_names([f.name, "_", f.version, "_", f.timestamp_col])
                        f_ts_col_str = f"r_{layer}.{f.timestamp_col} AS {f_ts_col_alias},"
                    else:
                        f_ts_col_str = ""
                    query = f"""
                        SELECT
                            l_{layer}.*,
                            {f_ts_col_str}
                            r_{layer}.* EXCLUDE ({join_keys_str}, {f.timestamp_col})
                        FROM ({query}) l_{layer}
                        ASOF JOIN (
                            SELECT {join_keys_str}, {f.timestamp_col}, {', '.join(cols)}
                            FROM {join_table_name}
                        ) r_{layer}
                        MATCH_CONDITION (l_{layer}.{spine_timestamp_col} >= r_{layer}.{f.timestamp_col})
                        ON {' AND '.join([f'l_{layer}.{k} = r_{layer}.{k}' for k in join_keys])}
                    """
                else:
                    query = self._composed_union_window_join_query(
                        layer=layer,
                        s_query=query,
                        s_ts_col=spine_timestamp_col,
                        f_df=f.feature_df,
                        f_table_name=join_table_name,
                        f_ts_col=f.timestamp_col,
                        join_keys=join_keys,
                    )
            else:
                query = f"""
                    SELECT
                        l_{layer}.*,
                        r_{layer}.* EXCLUDE ({join_keys_str})
                    FROM ({query}) l_{layer}
                    LEFT JOIN (
                        SELECT {join_keys_str}, {', '.join(cols)}
                        FROM {join_table_name}
                    ) r_{layer}
                    ON {' AND '.join([f'l_{layer}.{k} = r_{layer}.{k}' for k in join_keys])}
                """
            layer += 1

        # TODO: construct result dataframe with datframe APIs once ASOF join is supported natively.
        # Below code manually construct result dataframe from private members of spine dataframe, which
        # likely will cause unintentional issues. This setp is needed because spine_df might contains
        # prerequisite queries and post actions that must be carried over to result dataframe.
        result_df = self._session.sql(query)
        result_df._plan.queries = spine_df._plan.queries[:-1] + result_df._plan.queries
        result_df._plan.post_actions = spine_df._plan.post_actions

        return result_df, join_keys

    def _check_database_exists_or_throw(self) -> None:
        resolved_db_name = self._config.database.resolved()
        dbs = self._session.sql(
            f"""
            SHOW DATABASES LIKE '{resolved_db_name}' STARTS WITH '{resolved_db_name}'
        """
        ).collect(statement_params=self._telemetry_stmp)
        if len(dbs) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Database {resolved_db_name} does not exist."),
            )

    def _check_internal_objects_exist_or_throw(self) -> None:
        schema_result = self._find_object("SCHEMAS", self._config.schema)
        if len(schema_result) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(
                    f"Feature store schema {self._config.schema} does not exist. "
                    "Use CreationMode.CREATE_IF_NOT_EXIST mode instead if you want to create one."
                ),
            )
        for tag_name in to_sql_identifiers(
            [
                _FEATURE_STORE_OBJECT_TAG,
                _FEATURE_VIEW_METADATA_TAG,
            ]
        ):
            tag_result = self._find_object("TAGS", tag_name)
            if len(tag_result) == 0:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(
                        f"Feature store internal tag {tag_name} does not exist. "
                        "Use CreationMode.CREATE_IF_NOT_EXIST mode instead if you want to create one."
                    ),
                )

    def _is_asof_join_enabled(self) -> bool:
        result = None
        try:
            result = self._session.sql(
                """
                WITH
                spine AS (
                    SELECT "ID", "TS" FROM ( SELECT $1 AS "ID", $2 AS "TS" FROM VALUES (1 :: INT, 100 :: INT))
                ),
                feature AS (
                    SELECT "ID", "TS" FROM ( SELECT $1 AS "ID", $2 AS "TS" FROM VALUES (1 :: INT, 100 :: INT))
                )
                SELECT * FROM spine
                ASOF JOIN feature
                MATCH_CONDITION ( spine.ts >= feature.ts )
                ON spine.id = feature.id;
            """
            ).collect(statement_params=self._telemetry_stmp)
        except SnowparkSQLException:
            return False
        return result is not None and len(result) == 1

    # Visualize how the query works:
    #   https://docs.google.com/presentation/d/15fT2F34OFp5RPv2-hZirHw6wliPRVRlPHvoCMIB00oY/edit#slide=id.g25ab53e6c8d_0_32
    def _composed_union_window_join_query(
        self,
        layer: int,
        s_query: str,
        s_ts_col: SqlIdentifier,
        f_df: DataFrame,
        f_table_name: str,
        f_ts_col: SqlIdentifier,
        join_keys: List[SqlIdentifier],
    ) -> str:
        s_df = self._session.sql(s_query)
        s_only_cols = [col for col in to_sql_identifiers(s_df.columns) if col not in [*join_keys, s_ts_col]]
        f_only_cols = [col for col in to_sql_identifiers(f_df.columns) if col not in [*join_keys, f_ts_col]]
        join_keys_str = ", ".join(join_keys)
        temp_prefix = "_FS_TEMP_"

        def join_cols(cols: List[SqlIdentifier], end_comma: bool, rename: bool, prefix: str = "") -> str:
            if not cols:
                return ""
            cols = [f"{prefix}{col}" for col in cols]  # type: ignore[misc]
            if rename:
                cols = [f"{col} AS {col.replace(temp_prefix, '')}" for col in cols]  # type: ignore[misc]
            line_end = "," if end_comma else ""
            return ", ".join(cols) + line_end

        # Part 1: CTE of spine query
        spine_cte = f"""
            WITH spine_{layer} AS (
                {s_query}
            ),"""

        # Part 2: create union of spine table and feature tables
        s_select = f"""
            SELECT
                'SPINE' {temp_prefix}src,
                {s_ts_col},
                {join_keys_str},
                {join_cols(s_only_cols, end_comma=True, rename=False)}
                {join_cols(f_only_cols, end_comma=False, rename=False, prefix='null AS ')}
            FROM ({s_query})"""
        f_select = f"""
            SELECT
                'FEATURE' {temp_prefix}src,
                {f_ts_col} {s_ts_col},
                {join_keys_str},
                {join_cols(s_only_cols, end_comma=True, rename=False, prefix='null AS ')}
                {join_cols(f_only_cols, end_comma=False, rename=False)}
            FROM {f_table_name}"""
        union_cte = f"""
            unioned_{layer} AS (
                {s_select}
                UNION ALL
                {f_select}
            ),"""

        # Part 3: create window cte and add window column
        window_select = f"SELECT {temp_prefix}src, {s_ts_col}, {join_keys_str}"
        for f_col in f_only_cols:
            window_select = (
                window_select
                + f"""
                ,last_value({f_col}) IGNORE NULLS OVER (
                    PARTITION BY {join_keys_str}
                    ORDER BY {s_ts_col} ASC, {temp_prefix}src ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS {temp_prefix}{f_col}"""
            )
        window_select = window_select + f" FROM unioned_{layer}"
        window_cte = f"""
            windowed_{layer} AS (
                {window_select}
            )"""

        # Part 4: join original spine table with window table
        prefix_f_only_cols = to_sql_identifiers(
            [f"{temp_prefix}{name.resolved()}" for name in f_only_cols],
            case_sensitive=True,
        )
        last_select = f"""
            SELECT
                {join_keys_str},
                {s_ts_col},
                {join_cols(s_only_cols, end_comma=True, rename=False)}
                {join_cols(prefix_f_only_cols, end_comma=False, rename=True)}
            FROM spine_{layer}
            JOIN windowed_{layer}
            USING ({join_keys_str}, {s_ts_col})
            WHERE windowed_{layer}.{temp_prefix}src = 'SPINE'"""

        # Part 5: complete query
        complete_query = spine_cte + union_cte + window_cte + last_select

        return complete_query

    def _get_entity_name(self, raw_name: SqlIdentifier) -> SqlIdentifier:
        return SqlIdentifier(identifier.concat_names([_ENTITY_TAG_PREFIX, raw_name]))

    def _get_fully_qualified_name(self, name: Union[SqlIdentifier, str]) -> str:
        # Do a quick check to see if we can skip regex operations
        if "." not in name:
            return f"{self._config.full_schema_path}.{name}"

        db_name, schema_name, object_name = identifier.parse_schema_level_object_identifier(name)
        return "{}.{}.{}".format(
            db_name or self._config.database,
            schema_name or self._config.schema,
            object_name,
        )

    # TODO: SHOW DYNAMIC TABLES is very slow while other show objects are fast, investigate with DT in SNOW-902804.
    def _get_fv_backend_representations(
        self, object_name: Optional[SqlIdentifier], prefix_match: bool = False
    ) -> List[Tuple[Row, _FeatureStoreObjTypes]]:
        dynamic_table_results = [
            (d, _FeatureStoreObjTypes.MANAGED_FEATURE_VIEW)
            for d in self._find_object("DYNAMIC TABLES", object_name, prefix_match)
        ]
        view_results = [
            (d, _FeatureStoreObjTypes.EXTERNAL_FEATURE_VIEW)
            for d in self._find_object("VIEWS", object_name, prefix_match)
        ]
        return dynamic_table_results + view_results

    def _update_feature_view_status(self, feature_view: FeatureView, operation: str) -> FeatureView:
        assert operation in [
            "RESUME",
            "SUSPEND",
            "REFRESH",
        ], f"Operation: {operation} not supported"
        if feature_view.status == FeatureViewStatus.DRAFT or feature_view.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"FeatureView {feature_view.name} has not been registered."),
            )

        fully_qualified_name = feature_view.fully_qualified_name()
        try:
            self._session.sql(f"ALTER DYNAMIC TABLE {fully_qualified_name} {operation}").collect(
                statement_params=self._telemetry_stmp
            )
            if operation != "REFRESH":
                self._session.sql(f"ALTER TASK IF EXISTS {fully_qualified_name} {operation}").collect(
                    statement_params=self._telemetry_stmp
                )
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to update feature view {fully_qualified_name}'s status: {e}"),
            ) from e

        logger.info(f"Successfully {operation} FeatureView {feature_view.name}/{feature_view.version}.")
        return self.get_feature_view(feature_view.name, feature_view.version)

    def _optimized_find_feature_views(
        self, entity_name: SqlIdentifier, feature_view_name: Optional[SqlIdentifier]
    ) -> DataFrame:
        if not self._validate_entity_exists(entity_name):
            return self._session.create_dataframe([], schema=_LIST_FEATURE_VIEW_SCHEMA)

        # TODO: this can be optimized further by directly getting all possible FVs and filter by tag
        # it's easier to rewrite the code once we can remove the tag_reference path
        all_fvs = self._get_fv_backend_representations(object_name=None)
        fv_maps = {SqlIdentifier(r["name"], case_sensitive=True): r for r, _ in all_fvs}

        if len(fv_maps.keys()) == 0:
            return self._session.create_dataframe([], schema=_LIST_FEATURE_VIEW_SCHEMA)

        filters = [lambda d: d["entityName"].startswith(feature_view_name.resolved())] if feature_view_name else None
        res = self._lookup_tagged_objects(self._get_entity_name(entity_name), filters)

        output_values: List[List[Any]] = []
        for r in res:
            row = fv_maps[SqlIdentifier(r["entityName"], case_sensitive=True)]
            self._extract_feature_view_info(row, output_values)

        return self._session.create_dataframe(output_values, schema=_LIST_FEATURE_VIEW_SCHEMA)

    def _extract_feature_view_info(self, row: Row, output_values: List[List[Any]]) -> None:
        name, version = row["name"].split(_FEATURE_VIEW_NAME_DELIMITER)
        fv_metadata, _ = self._lookup_feature_view_metadata(row, FeatureView._get_physical_name(name, version))

        values: List[Any] = []
        values.append(name)
        values.append(version)
        values.append(row["database_name"])
        values.append(row["schema_name"])
        values.append(row["created_on"])
        values.append(row["owner"])
        values.append(row["comment"])
        values.append(fv_metadata.entities)
        values.append(row["target_lag"] if "target_lag" in row else None)
        values.append(row["refresh_mode"] if "refresh_mode" in row else None)
        values.append(row["scheduling_state"] if "scheduling_state" in row else None)
        values.append(row["warehouse"] if "warehouse" in row else None)
        values.append(json.dumps(self._extract_cluster_by_columns(row["cluster_by"])) if "cluster_by" in row else None)
        output_values.append(values)

    def _lookup_feature_view_metadata(self, row: Row, fv_name: str) -> Tuple[_FeatureViewMetadata, str]:
        if len(row["text"]) == 0:
            # NOTE: if this is a shared feature view, then text column will be empty due to privacy constraints.
            # So instead of looking at original query text, we will obtain metadata by querying the tag value.
            # For query body, we will just use a simple select instead of original DDL query since shared feature views
            # are read-only.
            try:
                res = self._lookup_tags(
                    domain="table", obj_name=fv_name, filter_fns=[lambda d: d["tagName"] == _FEATURE_VIEW_METADATA_TAG]
                )
                fv_metadata = _FeatureViewMetadata.from_json(res[0]["tagValue"])
                query = f"SELECT * FROM {self._get_fully_qualified_name(fv_name)}"
                return (fv_metadata, query)
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWML_ERROR,
                    original_exception=RuntimeError(f"Failed to extract feature_view metadata for {fv_name}: {e}."),
                )
        else:
            m = re.match(_DT_OR_VIEW_QUERY_PATTERN, row["text"])
            if m is None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWML_ERROR,
                    original_exception=RuntimeError(f"Failed to parse query text for FeatureView {fv_name}: {row}."),
                )
            fv_metadata = _FeatureViewMetadata.from_json(m.group("fv_metadata"))
            query = m.group("query")
            return (fv_metadata, query)

    def _compose_feature_view(self, row: Row, obj_type: _FeatureStoreObjTypes, entity_list: List[Row]) -> FeatureView:
        def find_and_compose_entity(name: str) -> Entity:
            name = SqlIdentifier(name).resolved()
            for e in entity_list:
                if e["NAME"] == name:
                    return Entity(
                        name=SqlIdentifier(e["NAME"], case_sensitive=True).identifier(),
                        join_keys=self._recompose_join_keys(e["JOIN_KEYS"]),
                        desc=e["DESC"],
                    )
            raise RuntimeError(f"Cannot find entity {name} from retrieved entity list: {entity_list}")

        name, version = row["name"].split(_FEATURE_VIEW_NAME_DELIMITER)
        name = SqlIdentifier(name, case_sensitive=True)
        fv_name = FeatureView._get_physical_name(name, version)
        fv_metadata, query = self._lookup_feature_view_metadata(row, fv_name)

        infer_schema_df = self._session.sql(f"SELECT * FROM {self._get_fully_qualified_name(fv_name)}")
        desc = row["comment"]

        if obj_type == _FeatureStoreObjTypes.MANAGED_FEATURE_VIEW:
            df = self._session.sql(query)
            entities = [find_and_compose_entity(n) for n in fv_metadata.entities]
            ts_col = fv_metadata.timestamp_col
            timestamp_col = ts_col if ts_col not in _LEGACY_TIMESTAMP_COL_PLACEHOLDER_VALS else None
            re_initialize = re.match(_DT_INITIALIZE_PATTERN, row["text"])
            initialize = re_initialize.group("initialize") if re_initialize is not None else "ON_CREATE"

            fv = FeatureView._construct_feature_view(
                name=name,
                entities=entities,
                feature_df=df,
                timestamp_col=timestamp_col,
                desc=desc,
                version=version,
                status=(
                    FeatureViewStatus(row["scheduling_state"])
                    if len(row["scheduling_state"]) > 0
                    else FeatureViewStatus.MASKED
                ),
                feature_descs=self._fetch_column_descs("DYNAMIC TABLE", fv_name),
                refresh_freq=row["target_lag"],
                database=self._config.database.identifier(),
                schema=self._config.schema.identifier(),
                warehouse=(
                    SqlIdentifier(row["warehouse"], case_sensitive=True).identifier()
                    if len(row["warehouse"]) > 0
                    else None
                ),
                refresh_mode=row["refresh_mode"],
                refresh_mode_reason=row["refresh_mode_reason"],
                initialize=initialize,
                owner=row["owner"],
                infer_schema_df=infer_schema_df,
                session=self._session,
                cluster_by=self._extract_cluster_by_columns(row["cluster_by"]),
            )
            return fv
        else:
            df = self._session.sql(query)
            entities = [find_and_compose_entity(n) for n in fv_metadata.entities]
            ts_col = fv_metadata.timestamp_col
            timestamp_col = ts_col if ts_col not in _LEGACY_TIMESTAMP_COL_PLACEHOLDER_VALS else None

            fv = FeatureView._construct_feature_view(
                name=name,
                entities=entities,
                feature_df=df,
                timestamp_col=timestamp_col,
                desc=desc,
                version=version,
                status=FeatureViewStatus.STATIC,
                feature_descs=self._fetch_column_descs("VIEW", fv_name),
                refresh_freq=None,
                database=self._config.database.identifier(),
                schema=self._config.schema.identifier(),
                warehouse=None,
                refresh_mode=None,
                refresh_mode_reason=None,
                initialize="ON_CREATE",
                owner=row["owner"],
                infer_schema_df=infer_schema_df,
                session=self._session,
            )
            return fv

    def _fetch_column_descs(self, obj_type: str, obj_name: SqlIdentifier) -> Dict[str, str]:
        res = self._session.sql(f"DESC {obj_type} {self._get_fully_qualified_name(obj_name)}").collect(
            statement_params=self._telemetry_stmp
        )

        descs = {}
        for r in res:
            if r["comment"] is not None:
                descs[SqlIdentifier(r["name"], case_sensitive=True).identifier()] = r["comment"]
        return descs

    def _find_object(
        self,
        object_type: str,
        object_name: Optional[SqlIdentifier],
        prefix_match: bool = False,
    ) -> List[Row]:
        """Try to find an object by given type and name pattern.

        Args:
            object_type: Type of the object. Could be TABLES, TAGS etc.
            object_name: Name of object. It will match everything of object_type is object_name is None.
            prefix_match: Will search all objects with object_name as prefix if set True. Otherwise
                will do exact on object_name. Default to false. If object_name is empty and prefix_match is
                True, then it will match everything of object_type.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to find resource.

        Returns:
            Return a list of rows round.
        """
        if object_name is None:
            match_name = "%"
        elif prefix_match:
            match_name = object_name.resolved() + "%"
        else:
            match_name = object_name.resolved()

        search_space, obj_domain = self._obj_search_spaces[object_type]
        all_rows = []
        fs_tag_objects = []
        tag_free_object_types = ["TAGS", "SCHEMAS", "WAREHOUSES", "DATASETS"]
        try:
            search_scope = f"IN {search_space}" if search_space is not None else ""
            all_rows = self._session.sql(f"SHOW {object_type} LIKE '{match_name}' {search_scope}").collect(
                statement_params=self._telemetry_stmp
            )
            # There could be none-FS objects under FS schema, thus filter on objects with FS special tag.
            if object_type not in tag_free_object_types and len(all_rows) > 0:
                fs_obj_rows = self._lookup_tagged_objects(
                    _FEATURE_STORE_OBJECT_TAG, [lambda d: d["domain"] == obj_domain]
                )
                fs_tag_objects = [row["entityName"] for row in fs_obj_rows]
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to find object : {e}"),
            ) from e

        result = []
        for row in all_rows:
            found_name = row["name"]
            prefix = object_name.resolved() if object_name is not None else ""
            if found_name.startswith(prefix) and (object_type in tag_free_object_types or found_name in fs_tag_objects):
                result.append(row)
        return result

    def _load_serialized_feature_views(
        self, serialized_feature_views: List[str]
    ) -> List[Union[FeatureView, FeatureViewSlice]]:
        results: List[Union[FeatureView, FeatureViewSlice]] = []
        for obj in serialized_feature_views:
            try:
                obj_type = json.loads(obj)[_FEATURE_OBJ_TYPE]
            except Exception as e:
                raise ValueError(f"Malformed serialized feature object: {obj}") from e

            if obj_type == FeatureView.__name__:
                results.append(FeatureView.from_json(obj, self._session))
            elif obj_type == FeatureViewSlice.__name__:
                results.append(FeatureViewSlice.from_json(obj, self._session))
            else:
                raise ValueError(f"Unsupported feature object type: {obj_type}")
        return results

    def _load_compact_feature_views(
        self, compact_feature_views: List[str]
    ) -> List[Union[FeatureView, FeatureViewSlice]]:
        results: List[Union[FeatureView, FeatureViewSlice]] = []
        for obj in compact_feature_views:
            results.append(FeatureView._load_from_compact_repr(self._session, obj))
        return results

    def _exclude_columns(self, df: DataFrame, exclude_columns: List[str]) -> DataFrame:
        exclude_columns = to_sql_identifiers(exclude_columns)  # type: ignore[assignment]
        df_cols = to_sql_identifiers(df.columns)
        for col in exclude_columns:
            if col not in df_cols:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"{col} in exclude_columns not exists in dataframe columns: {df_cols}"
                    ),
                )
        return cast(DataFrame, df.drop(exclude_columns))

    def _is_dataset_enabled(self) -> bool:
        try:
            self._session.sql(f"SHOW DATASETS IN SCHEMA {self._config.full_schema_path}").collect(
                statement_params=self._telemetry_stmp
            )
            return True
        except SnowparkSQLException:
            return False

    def _check_feature_store_object_versions(self) -> None:
        versions = self._collapse_object_versions()
        if len(versions) > 0 and pkg_version.parse(snowml_version.VERSION) < versions[0]:
            warnings.warn(
                "The current snowflake-ml-python version out of date, package upgrade recommended "
                + f"(current={snowml_version.VERSION}, recommended>={str(versions[0])})",
                stacklevel=2,
                category=UserWarning,
            )

    def _filter_results(
        self, results: List[Dict[str, str]], filter_fns: Optional[List[Callable[[Dict[str, str]], bool]]] = None
    ) -> List[Dict[str, str]]:
        if filter_fns is None:
            return results

        filtered_results = []
        for r in results:
            if all([fn(r) for fn in filter_fns]):
                filtered_results.append(r)
        return filtered_results

    def _lookup_tags(
        self, domain: str, obj_name: str, filter_fns: Optional[List[Callable[[Dict[str, str]], bool]]] = None
    ) -> List[Dict[str, str]]:
        """
        Lookup tag values for a given object, optionally apply filters on the results.

        Args:
            domain: Domain of the obj to look for tag. E.g. table
            obj_name: Name of the obj.
            filter_fns: List of filter functions applied on the results.

        Returns:
            List of tag values in dictionary format.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to lookup tags.

        Example::

            self._lookup_tags("TABLE", "MY_FV", [lambda d: d["tagName"] == "TARGET_TAG_NAME"])

        """
        # NOTE: use ENTITY_DETAIL system fn to query tags for given object for it to work in
        # processes using owner's right. e.g. Streamlit, or stored procedure
        try:
            res = self._session.sql(
                f"""
                SELECT ENTITY_DETAIL('{domain}','{self._get_fully_qualified_name(obj_name)}', '["TAG_REFERENCES"]');
            """
            ).collect(statement_params=self._telemetry_stmp)
            entity_detail = json.loads(res[0][0])
            results = entity_detail["tagReferencesInfo"]["tagReferenceList"]
            return self._filter_results(results, filter_fns)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to lookup tags for object for {obj_name}: {e}"),
            ) from e

    def _lookup_tagged_objects(
        self, tag_name: str, filter_fns: Optional[List[Callable[[Dict[str, str]], bool]]] = None
    ) -> List[Dict[str, str]]:
        """
        Lookup objects based on specified tag name, optionally apply filters on the results.

        Args:
            tag_name: Name of the tag.
            filter_fns: List of filter functions applied on the results.

        Returns:
            List of objects in dictionary format.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to lookup tagged objects.

        Example::

            self._lookup_tagged_objects("TARGET_TAG_NAME", [lambda d: d["entityName"] == "MY_FV"])

        """
        # NOTE: use ENTITY_DETAIL system fn to query objects from tag for it to work in
        # processes using owner's right. e.g. Streamlit, or stored procedure
        try:
            res = self._session.sql(
                f"""
                SELECT ENTITY_DETAIL('TAG','{self._get_fully_qualified_name(tag_name)}', '["TAG_REFERENCES_INTERNAL"]');
            """
            ).collect(statement_params=self._telemetry_stmp)
            entity_detail = json.loads(res[0][0])
            results = entity_detail["referencedEntities"]["tagReferenceList"]
            return self._filter_results(results, filter_fns)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to lookup tagged objects for {tag_name}: {e}"),
            ) from e

    def _collapse_object_versions(self) -> List[pkg_version.Version]:
        try:
            res = self._lookup_tagged_objects(_FEATURE_STORE_OBJECT_TAG)
        except Exception:
            # since this is a best effort user warning to upgrade pkg versions
            # we are treating failures as benign error
            return []
        versions = set()
        compatibility_breakage_detected = False
        for r in res:
            info = _FeatureStoreObjInfo.from_json(r["tagValue"])
            if info.type == _FeatureStoreObjTypes.UNKNOWN:
                compatibility_breakage_detected = True
            versions.add(pkg_version.parse(info.pkg_version))

        sorted_versions = sorted(versions, reverse=True)
        if compatibility_breakage_detected:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_PACKAGE_OUTDATED,
                original_exception=RuntimeError(
                    f"The current snowflake-ml-python version {snowml_version.VERSION} is out of date, "
                    + f"please upgrade to at least {sorted_versions[0]}."
                ),
            )
        return sorted_versions

    def _validate_feature_view_name_and_version_input(
        self, feature_view: Union[FeatureView, str], version: Optional[str] = None
    ) -> FeatureView:
        if isinstance(feature_view, str):
            if version is None:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError("Version must be provided when argument feature_view is a str."),
                )
            feature_view = self.get_feature_view(feature_view, version)
        elif not isinstance(feature_view, FeatureView):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    "Invalid type of argument feature_view. It must be either str or FeatureView type."
                ),
            )

        return feature_view

    @staticmethod
    def _extract_cluster_by_columns(cluster_by_clause: str) -> List[str]:
        # Use regex to extract elements inside the parentheses.
        match = re.search(r"\((.*?)\)", cluster_by_clause)
        if match:
            # Handle both quoted and unquoted column names.
            return re.findall(identifier.SF_IDENTIFIER_RE, match.group(1))
        return []
