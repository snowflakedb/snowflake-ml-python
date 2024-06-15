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


# TODO: remove "" after dataset is updated
class _FeatureStoreObjTypes(Enum):
    UNKNOWN = "UNKNOWN"  # for forward compatibility
    MANAGED_FEATURE_VIEW = "MANAGED_FEATURE_VIEW"
    EXTERNAL_FEATURE_VIEW = "EXTERNAL_FEATURE_VIEW"
    FEATURE_VIEW_REFRESH_TASK = "FEATURE_VIEW_REFRESH_TASK"
    TRAINING_DATA = ""

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
    ]
)


class CreationMode(Enum):
    FAIL_IF_NOT_EXIST = 1
    CREATE_IF_NOT_EXIST = 2


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
                )

        # TODO: remove this after tag_ref_internal rollout
        self._use_optimized_tag_ref = self._tag_ref_internal_enabled()
        self._check_feature_store_object_versions()
        logger.info(f"Successfully connected to feature store: {self._config.full_schema_path}.")

    @telemetry.send_api_usage_telemetry(project=_PROJECT)
    def update_default_warehouse(self, warehouse_name: str) -> None:
        """Update default warehouse for feature store.

        Args:
            warehouse_name: Name of warehouse.

        Raises:
            SnowflakeMLException: If warehouse does not exists.
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
            entity: Entity object to register.

        Returns:
            A registered entity object.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to find resources.
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

    # TODO: add support to update column desc once SNOW-894249 is fixed
    @dispatch_decorator()
    def register_feature_view(
        self,
        feature_view: FeatureView,
        version: str,
        block: bool = True,
        overwrite: bool = False,
    ) -> FeatureView:
        """
        Materialize a FeatureView to Snowflake backend.
        Incremental maintenance for updates on the source data will be automated if refresh_freq is set.
        NOTE: Each new materialization will trigger a full FeatureView history refresh for the data included in the
              FeatureView.

        Examples:
            ...
            draft_fv = FeatureView(name="my_fv", entities=[entities], feature_df)
            registered_fv = fs.register_feature_view(feature_view=draft_fv, version="v1")
            ...

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
            desc = feature_view.feature_descs.get(SqlIdentifier(col.name), None)
            desc = "" if desc is None else f"COMMENT '{desc}'"
            return f"{col.name} {desc}"

        column_descs = ", ".join([f"{create_col_desc(col)}" for col in feature_view.output_schema.fields])

        if refresh_freq is not None:
            schedule_task = refresh_freq != "DOWNSTREAM" and timeparse(refresh_freq) is None
            self._create_dynamic_table(
                feature_view_name,
                feature_view,
                fully_qualified_name,
                column_descs,
                tagging_clause_str,
                schedule_task,
                self._default_warehouse,
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

    @dispatch_decorator()
    def update_feature_view(
        self, name: str, version: str, refresh_freq: Optional[str] = None, warehouse: Optional[str] = None
    ) -> FeatureView:
        """Update a registered feature view.
            Check feature_view.py for which fields are allowed to be updated after registration.

        Args:
            name: name of the FeatureView to be updated.
            version: version of the FeatureView to be updated.
            refresh_freq: updated refresh frequency.
            warehouse: updated warehouse.

        Returns:
            Updated FeatureView.

        Raises:
            SnowflakeMLException: [RuntimeError] If FeatureView is not managed and refresh_freq is defined.
            SnowflakeMLException: [RuntimeError] Failed to update feature view.
        """
        feature_view = self.get_feature_view(name=name, version=version)
        if refresh_freq is not None and feature_view.status == FeatureViewStatus.STATIC:
            full_name = f"{feature_view.name}/{feature_view.version}"
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=RuntimeError(f"Feature view {full_name} must be non-static so that can be updated."),
            )

        warehouse = SqlIdentifier(warehouse) if warehouse else feature_view.warehouse

        # TODO(@wezhou): we need to properly handle cron expr
        try:
            self._session.sql(
                f"""ALTER DYNAMIC TABLE {feature_view.fully_qualified_name()} SET
                    TARGET_LAG = '{refresh_freq or feature_view.refresh_freq}'
                    WAREHOUSE = {warehouse}
                """
            ).collect(statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(
                    f"Update feature view {feature_view.name}/{feature_view.version} failed: {e}"
                ),
            ) from e
        return self.get_feature_view(name=name, version=version)

    @dispatch_decorator()
    def read_feature_view(self, feature_view: FeatureView) -> DataFrame:
        """
        Read FeatureView data.

        Args:
            feature_view: FeatureView to retrieve data from.

        Returns:
            Snowpark DataFrame(lazy mode) containing the FeatureView data.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView is not registered.
        """
        if feature_view.status == FeatureViewStatus.DRAFT or feature_view.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"FeatureView {feature_view.name} has not been registered."),
            )

        return self._session.sql(f"SELECT * FROM {feature_view.fully_qualified_name()}")

    @dispatch_decorator()
    def list_feature_views(
        self,
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
        """
        if feature_view_name is not None:
            feature_view_name = SqlIdentifier(feature_view_name)

        if entity_name is not None:
            entity_name = SqlIdentifier(entity_name)
            if self._use_optimized_tag_ref:
                return self._optimized_find_feature_views(entity_name, feature_view_name)
            else:
                return self._find_feature_views(entity_name, feature_view_name)
        else:
            output_values: List[List[Any]] = []
            for row in self._get_fv_backend_representations(feature_view_name, prefix_match=True):
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

        return self._compose_feature_view(results[0], self.list_entities().collect())

    @dispatch_decorator()
    def resume_feature_view(self, feature_view: FeatureView) -> FeatureView:
        """
        Resume a previously suspended FeatureView.

        Args:
            feature_view: FeatureView to resume.

        Returns:
            A new feature view with updated status.
        """
        return self._update_feature_view_status(feature_view, "RESUME")

    @dispatch_decorator()
    def suspend_feature_view(self, feature_view: FeatureView) -> FeatureView:
        """
        Suspend an active FeatureView.

        Args:
            feature_view: FeatureView to suspend.

        Returns:
            A new feature view with updated status.
        """
        return self._update_feature_view_status(feature_view, "SUSPEND")

    @dispatch_decorator()
    def delete_feature_view(self, feature_view: FeatureView) -> None:
        """
        Delete a FeatureView.

        Args:
            feature_view: FeatureView to delete.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView is not registered.
        """
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
        """
        name = SqlIdentifier(name)
        try:
            result = self.list_entities().filter(F.col("NAME") == name.resolved()).collect()
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
            name: Entity name.

        Raises:
            SnowflakeMLException: [ValueError] Entity with given name not exists.
            SnowflakeMLException: [RuntimeError] Failed to alter schema or drop tag.
            SnowflakeMLException: [RuntimeError] Failed to find resources.
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
        """
        if spine_timestamp_col is not None:
            spine_timestamp_col = SqlIdentifier(spine_timestamp_col)

        if len(features) == 0:
            raise ValueError("features cannot be empty")
        if isinstance(features[0], str):
            features = self._load_serialized_feature_objects(cast(List[str], features))

        df, _ = self._join_features(
            spine_df,
            cast(List[Union[FeatureView, FeatureViewSlice]], features),
            spine_timestamp_col,
            include_feature_view_timestamp_col,
        )

        if exclude_columns is not None:
            df = self._exclude_columns(df, exclude_columns)

        return df

    @overload
    def generate_dataset(
        self,
        name: str,
        spine_df: DataFrame,
        features: List[Union[FeatureView, FeatureViewSlice]],
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
            spine_df: The fact table contains the raw dataset.
            features: A list of FeatureView or FeatureViewSlice which contains features to be joined.
            version: The version of the Dataset to be generated. If none specified, the current timestamp
                will be used instead.
            spine_timestamp_col: Name of timestamp column in spine_df that will be used to join
                time-series features. If spine_timestamp_col is not none, the input features also must have
                timestamp_col.
            spine_label_cols: Name of column(s) in spine_df that contains labels.
            exclude_columns: Column names to exclude from the result dataframe.
                The underlying storage will still contain the columns.
            include_feature_view_timestamp_col: Generated dataset will include timestamp column of feature view
                (if feature view has timestamp column) if set true. Default to false.
            desc: A description about this dataset.
            output_type: The type of Snowflake storage to use for the generated training data.

        Returns:
            If output_type is "dataset" (default), returns a Dataset object.
            If output_type is "table", returns a Snowpark DataFrame representing the table.

        Raises:
            SnowflakeMLException: [ValueError] Dataset name/version already exists
            SnowflakeMLException: [ValueError] Snapshot creation failed.
            SnowflakeMLException: [ValueError] Invalid output_type specified.
            SnowflakeMLException: [RuntimeError] Failed to create clone from table.
            SnowflakeMLException: [RuntimeError] Failed to find resources.
        """
        if output_type not in {"table", "dataset"}:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(f"Invalid output_type: {output_type}."),
            )
        if spine_timestamp_col is not None:
            spine_timestamp_col = SqlIdentifier(spine_timestamp_col)
        if spine_label_cols is not None:
            spine_label_cols = to_sql_identifiers(spine_label_cols)  # type: ignore[assignment]

        result_df, join_keys = self._join_features(
            spine_df, features, spine_timestamp_col, include_feature_view_timestamp_col
        )

        # Convert name to fully qualified name if not already fully qualified
        db_name, schema_name, object_name, _ = identifier.parse_schema_level_object_identifier(name)
        name = "{}.{}.{}".format(
            db_name or self._config.database,
            schema_name or self._config.schema,
            object_name,
        )
        version = version or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        if exclude_columns is not None:
            result_df = self._exclude_columns(result_df, exclude_columns)

        fs_meta = FeatureStoreMetadata(
            spine_query=spine_df.queries["queries"][-1],
            serialized_feature_views=[fv.to_json() for fv in features],
            spine_timestamp_col=spine_timestamp_col,
        )

        try:
            if output_type == "table":
                table_name = f"{name}_{version}"
                result_df.write.mode("errorifexists").save_as_table(table_name)
                ds_df = self._session.table(table_name)
                return ds_df
            else:
                assert output_type == "dataset"
                if not self._is_dataset_enabled():
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.SNOWML_CREATE_FAILED,
                        original_exception=RuntimeError(
                            "Dataset is not enabled in your account. Ask your account admin to set"
                            ' FEATURE_DATASET=ENABLED or set output_type="table" to generate the data'
                            " as a Snowflake Table instead."
                        ),
                    )
                ds: dataset.Dataset = dataset.create_from_dataframe(
                    self._session,
                    name,
                    version,
                    input_dataframe=result_df,
                    exclude_cols=[spine_timestamp_col],
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
        """
        assert ds.selected_version is not None
        source_meta = ds.selected_version._get_metadata()
        if (
            source_meta is None
            or not isinstance(source_meta.properties, FeatureStoreMetadata)
            or source_meta.properties.serialized_feature_views is None
        ):
            raise ValueError(f"Dataset {ds} does not contain valid feature view information.")

        return self._load_serialized_feature_objects(source_meta.properties.serialized_feature_views)

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
        all_fvs_rows = all_fvs_df.collect()
        all_entities_rows = all_entities_df.collect()

        if dryrun:
            logger.info(
                "Following feature views and entities will be deleted."
                + " Set 'dryrun=False' to perform the actual deletion."
            )
            logger.info(f"Total {len(all_fvs_rows)} Feature views to be deleted:")
            all_fvs_df.show(n=len(all_fvs_rows))
            logger.info(f"\nTotal {len(all_entities_rows)} entities to be deleted:")
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
                + f"\"{found_dts[0]['refresh_mode_reason']}\". "
                + "It will likely incurr higher cost.",
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
            ).collect()
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
        return f"{self._config.full_schema_path}.{name}"

    # TODO: SHOW DYNAMIC TABLES is very slow while other show objects are fast, investigate with DT in SNOW-902804.
    def _get_fv_backend_representations(
        self, object_name: Optional[SqlIdentifier], prefix_match: bool = False
    ) -> List[Row]:
        dynamic_table_results = self._find_object("DYNAMIC TABLES", object_name, prefix_match)
        view_results = self._find_object("VIEWS", object_name, prefix_match)
        return dynamic_table_results + view_results

    def _update_feature_view_status(self, feature_view: FeatureView, operation: str) -> FeatureView:
        assert operation in [
            "RESUME",
            "SUSPEND",
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
        fv_maps = {SqlIdentifier(r["name"], case_sensitive=True): r for r in all_fvs}

        if len(fv_maps.keys()) == 0:
            return self._session.create_dataframe([], schema=_LIST_FEATURE_VIEW_SCHEMA)

        filter_clause = f"WHERE OBJECT_NAME LIKE '{feature_view_name.resolved()}%'" if feature_view_name else ""
        try:
            res = self._session.sql(
                f"""
                    SELECT
                        OBJECT_NAME
                    FROM TABLE(
                        {self._config.database}.INFORMATION_SCHEMA.TAG_REFERENCES_INTERNAL(
                            TAG_NAME => '{self._get_fully_qualified_name(self._get_entity_name(entity_name))}'
                        )
                    ) {filter_clause}"""
            ).collect(statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to find feature views' by entity {entity_name}: {e}"),
            ) from e

        output_values: List[List[Any]] = []
        for r in res:
            row = fv_maps[SqlIdentifier(r["OBJECT_NAME"], case_sensitive=True)]
            self._extract_feature_view_info(row, output_values)

        return self._session.create_dataframe(output_values, schema=_LIST_FEATURE_VIEW_SCHEMA)

    def _extract_feature_view_info(self, row: Row, output_values: List[List[Any]]) -> None:
        name, version = row["name"].split(_FEATURE_VIEW_NAME_DELIMITER)
        m = re.match(_DT_OR_VIEW_QUERY_PATTERN, row["text"])
        if m is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError(f"Failed to parse query text for FeatureView {name}/{version}: {row}."),
            )

        fv_metadata = _FeatureViewMetadata.from_json(m.group("fv_metadata"))

        values: List[Any] = []
        values.append(name)
        values.append(version)
        values.append(row["database_name"])
        values.append(row["schema_name"])
        values.append(row["created_on"])
        values.append(row["owner"])
        values.append(row["comment"])
        values.append(fv_metadata.entities)
        output_values.append(values)

    def _find_feature_views(self, entity_name: SqlIdentifier, feature_view_name: Optional[SqlIdentifier]) -> DataFrame:
        if not self._validate_entity_exists(entity_name):
            return self._session.create_dataframe([], schema=_LIST_FEATURE_VIEW_SCHEMA)

        all_fvs = self._get_fv_backend_representations(object_name=None)
        fv_maps = {SqlIdentifier(r["name"], case_sensitive=True): r for r in all_fvs}

        if len(fv_maps.keys()) == 0:
            return self._session.create_dataframe([], schema=_LIST_FEATURE_VIEW_SCHEMA)

        # NOTE: querying INFORMATION_SCHEMA for Entity lineage can be expensive depending on how many active
        # FeatureViews there are. If this ever become an issue, consider exploring improvements.
        try:
            queries = [
                f"""
                    SELECT
                        TAG_VALUE,
                        OBJECT_NAME
                    FROM TABLE(
                        {self._config.database}.INFORMATION_SCHEMA.TAG_REFERENCES(
                            '{self._get_fully_qualified_name(fv_name)}',
                            'table'
                        )
                    )
                    WHERE LEVEL = 'TABLE'
                    AND TAG_NAME = '{_FEATURE_VIEW_METADATA_TAG}'
                """
                for fv_name in fv_maps.keys()
            ]

            results = self._session.sql("\nUNION\n".join(queries)).collect(statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to retrieve feature views' information: {e}"),
            ) from e

        output_values: List[List[Any]] = []
        for r in results:
            fv_metadata = _FeatureViewMetadata.from_json(r["TAG_VALUE"])
            for retrieved_entity in fv_metadata.entities:
                if entity_name == SqlIdentifier(retrieved_entity, case_sensitive=True):
                    fv_name, _ = r["OBJECT_NAME"].split(_FEATURE_VIEW_NAME_DELIMITER)
                    fv_name = SqlIdentifier(fv_name, case_sensitive=True)
                    obj_name = SqlIdentifier(r["OBJECT_NAME"], case_sensitive=True)
                    if feature_view_name is not None:
                        if fv_name == feature_view_name:
                            self._extract_feature_view_info(fv_maps[obj_name], output_values)
                        else:
                            continue
                    else:
                        self._extract_feature_view_info(fv_maps[obj_name], output_values)
        return self._session.create_dataframe(output_values, schema=_LIST_FEATURE_VIEW_SCHEMA)

    def _compose_feature_view(self, row: Row, entity_list: List[Row]) -> FeatureView:
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
        m = re.match(_DT_OR_VIEW_QUERY_PATTERN, row["text"])
        if m is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWML_ERROR,
                original_exception=RuntimeError(f"Failed to parse query text for FeatureView {name}/{version}: {row}."),
            )

        fv_name = FeatureView._get_physical_name(name, version)
        infer_schema_df = self._session.sql(f"SELECT * FROM {self._get_fully_qualified_name(fv_name)}")

        if m.group("obj_type") == "DYNAMIC TABLE":
            query = m.group("query")
            df = self._session.sql(query)
            desc = m.group("comment")
            fv_metadata = _FeatureViewMetadata.from_json(m.group("fv_metadata"))
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
                status=FeatureViewStatus(row["scheduling_state"]),
                feature_descs=self._fetch_column_descs("DYNAMIC TABLE", fv_name),
                refresh_freq=row["target_lag"],
                database=self._config.database.identifier(),
                schema=self._config.schema.identifier(),
                warehouse=SqlIdentifier(row["warehouse"], case_sensitive=True).identifier(),
                refresh_mode=row["refresh_mode"],
                refresh_mode_reason=row["refresh_mode_reason"],
                owner=row["owner"],
                infer_schema_df=infer_schema_df,
            )
            return fv
        else:
            query = m.group("query")
            df = self._session.sql(query)
            desc = m.group("comment")
            fv_metadata = _FeatureViewMetadata.from_json(m.group("fv_metadata"))
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
                owner=row["owner"],
                infer_schema_df=infer_schema_df,
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
                if self._use_optimized_tag_ref:
                    fs_obj_rows = self._session.sql(
                        f"""
                            SELECT
                                OBJECT_NAME
                            FROM TABLE(
                                {self._config.database}.INFORMATION_SCHEMA.TAG_REFERENCES_INTERNAL(
                                    TAG_NAME => '{self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)}'
                                )
                            )
                            WHERE DOMAIN='{obj_domain}'
                        """
                    ).collect(statement_params=self._telemetry_stmp)
                else:
                    # TODO: remove this after tag_ref_internal rollout
                    # Note: <object_name> in TAG_REFERENCES(<object_name>) is case insensitive,
                    # use double quotes to make it case-sensitive.
                    queries = [
                        f"""
                            SELECT OBJECT_NAME
                            FROM TABLE(
                                {self._config.database}.INFORMATION_SCHEMA.TAG_REFERENCES(
                                    '{self._get_fully_qualified_name(SqlIdentifier(row['name'], case_sensitive=True))}',
                                    '{obj_domain}'
                                )
                            )
                            WHERE TAG_NAME = '{_FEATURE_STORE_OBJECT_TAG}'
                            AND TAG_SCHEMA = '{self._config.schema.resolved()}'
                        """
                        for row in all_rows
                    ]
                    fs_obj_rows = self._session.sql("\nUNION\n".join(queries)).collect(
                        statement_params=self._telemetry_stmp
                    )

                fs_tag_objects = [row["OBJECT_NAME"] for row in fs_obj_rows]
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

    def _load_serialized_feature_objects(
        self, serialized_feature_objs: List[str]
    ) -> List[Union[FeatureView, FeatureViewSlice]]:
        results: List[Union[FeatureView, FeatureViewSlice]] = []
        for obj in serialized_feature_objs:
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

    def _tag_ref_internal_enabled(self) -> bool:
        try:
            self._session.sql(
                f"""
                    SELECT * FROM TABLE(
                        {self._config.database}.INFORMATION_SCHEMA.TAG_REFERENCES_INTERNAL(
                            TAG_NAME => '{self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)}'
                        )
                    ) LIMIT 1;
                """
            ).collect()
            return True
        except Exception:
            return False

    def _is_dataset_enabled(self) -> bool:
        try:
            self._session.sql(f"SHOW DATASETS IN SCHEMA {self._config.full_schema_path}").collect()
            return True
        except SnowparkSQLException as e:
            if "'DATASETS' does not exist" in e.message:
                return False
            raise

    def _check_feature_store_object_versions(self) -> None:
        versions = self._collapse_object_versions()
        if len(versions) > 0 and pkg_version.parse(snowml_version.VERSION) < versions[0]:
            warnings.warn(
                "The current snowflake-ml-python version out of date, package upgrade recommended "
                + f"(current={snowml_version.VERSION}, recommended>={str(versions[0])})",
                stacklevel=2,
                category=UserWarning,
            )

    def _collapse_object_versions(self) -> List[pkg_version.Version]:
        if not self._use_optimized_tag_ref:
            return []

        query = f"""
            SELECT
                TAG_VALUE
            FROM TABLE(
                {self._config.database}.INFORMATION_SCHEMA.TAG_REFERENCES_INTERNAL(
                    TAG_NAME => '{self._get_fully_qualified_name(_FEATURE_STORE_OBJECT_TAG)}'
                )
            )
        """
        try:
            res = self._session.sql(query).collect(statement_params=self._telemetry_stmp)
        except Exception:
            # since this is a best effort user warning to upgrade pkg versions
            # we are treating failures as benign error
            return []
        versions = set()
        compatibility_breakage_detected = False
        for r in res:
            info = _FeatureStoreObjInfo.from_json(r["TAG_VALUE"])
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
