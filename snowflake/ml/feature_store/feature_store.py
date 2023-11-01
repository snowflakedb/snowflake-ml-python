import datetime
import json
import logging
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, cast

from pytimeparse.timeparse import timeparse

from snowflake import connector
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import identifier, query_result_checker as qrc
from snowflake.ml.dataset.dataset import Dataset, FeatureStoreMetadata
from snowflake.ml.feature_store.entity import (
    ENTITY_JOIN_KEY_DELIMITER,
    ENTITY_NAME_LENGTH_LIMIT,
    FEATURE_VIEW_ENTITY_TAG_DELIMITER,
    Entity,
)
from snowflake.ml.feature_store.feature_view import (
    FEATURE_OBJ_TYPE,
    FEATURE_VIEW_NAME_DELIMITER,
    TIMESTAMP_COL_PLACEHOLDER,
    FeatureView,
    FeatureViewSlice,
    FeatureViewStatus,
)
from snowflake.snowpark import DataFrame, Row, Session, functions as F
from snowflake.snowpark._internal import type_utils, utils as snowpark_utils
from snowflake.snowpark.types import StructField

logger = logging.getLogger(__name__)

ENTITY_TAG_PREFIX = "SNOWML_FEATURE_STORE_ENTITY_"
FEATURE_VIEW_ENTITY_TAG = "SNOWML_FEATURE_STORE_FV_ENTITIES"
FEATURE_VIEW_TS_COL_TAG = "SNOWML_FEATURE_STORE_FV_TS_COL"
FEATURE_STORE_OBJECT_TAG = "SNOWML_FEATURE_STORE_OBJECT"
PROJECT = "FeatureStore"

# TODO: Enable when ASOF join is released. https://snowflakecomputing.atlassian.net/browse/SNOW-780702
_ENABLE_ASOF_JOIN = False

DT_QUERY_PATTERN = re.compile(
    r""".*COMMENT\ =\ '(?P<comment>.*)'\s*
        TAG.*?{entity_tag}\ =\ '(?P<entities>.*?)',\n
           .*?{ts_col_tag}\ =\ '(?P<ts_col>.*?)',?\n.*
        lag\ =\ '(?P<refresh_freq>.*?)'\ warehouse\ =\ (?P<warehouse>.*?)\s*AS\ (?P<query>.*)
    """.format(
        entity_tag=FEATURE_VIEW_ENTITY_TAG, ts_col_tag=FEATURE_VIEW_TS_COL_TAG
    ),
    flags=re.DOTALL | re.IGNORECASE | re.X,
)

VIEW_QUERY_PATTERN = re.compile(
    r""".*COMMENT\ =\ '(?P<comment>.*)'\s*
        TAG.*?{entity_tag}\ =\ '(?P<entities>.*?)',\n
           .*?{ts_col_tag}\ =\ '(?P<ts_col>.*?)',?.*?
        AS\ (?P<query>.*)
    """.format(
        entity_tag=FEATURE_VIEW_ENTITY_TAG, ts_col_tag=FEATURE_VIEW_TS_COL_TAG
    ),
    flags=re.DOTALL | re.IGNORECASE | re.X,
)


class CreationMode(Enum):
    FAIL_IF_NOT_EXIST = 1
    CREATE_IF_NOT_EXIST = 2


@dataclass(frozen=True)
class _FeatureStoreConfig:
    database: str
    schema: str
    default_warehouse: str

    @property
    def full_schema_path(self) -> str:
        return f"{self.database}.{self.schema}"

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str: str) -> "_FeatureStoreConfig":
        json_dict = json.loads(json_str)
        return cls(**json_dict)


class FeatureStore:
    """
    FeatureStore provides APIs to create, materialize, retrieve and manage feature pipelines.
    """

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
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
            default_warehouse: Default warehouse setting to materialize feature pipelines.
            creation_mode: Create new backend or fail if not exist upon feature store creation.

        Raises:
            SnowflakeMLException: [ValueError] Default_warehouse does not exist.
            SnowflakeMLException: [ValueError] FAIL_IF_NOT_EXIST is set and feature store not exists.
            SnowflakeMLException: [RuntimeError] Failed to find resources.
            SnowflakeMLException: [RuntimeError] Failed to create feature store.
        """
        self._telemetry_stmp = telemetry.get_function_usage_statement_params(PROJECT)
        self._session: Session = session
        self._config = _FeatureStoreConfig(
            database=identifier.resolve_identifier(database),
            schema=identifier.resolve_identifier(name),
            default_warehouse=identifier.resolve_identifier(default_warehouse),
        )
        # A dict from object name to tuple of search space and object domain.
        # search space used in query "SHOW <object_TYPE> LIKE <object_name> IN <search_space>"
        # object domain used in query "TAG_REFERENCE(<object_name>, <object_domain>)"
        self._obj_search_spaces = {
            "TABLES": (self._config.full_schema_path, "TABLE"),
            "DYNAMIC TABLES": (self._config.full_schema_path, "TABLE"),
            "VIEWS": (self._config.full_schema_path, "TABLE"),
            "SCHEMAS": (f"DATABASE {self._config.database}", "SCHEMA"),
            "TAGS": (self._config.full_schema_path, None),
            "TASKS": (self._config.full_schema_path, "TASK"),
            "WAREHOUSES": (None, None),
        }

        # DESC WAREHOUSE requires MONITOR privilege on the warehouse which is a high privilege
        # some users not usually have.
        warehouse_result = self._find_object("WAREHOUSES", self._config.default_warehouse)
        if len(warehouse_result) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Cannot find warehouse {self._config.default_warehouse}"),
            )

        if creation_mode == CreationMode.FAIL_IF_NOT_EXIST:
            schema_result = self._find_object("SCHEMAS", self._config.schema)
            if len(schema_result) == 0:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(f"Feature store {name} does not exist."),
                )
        else:
            try:
                self._session.sql(f"CREATE DATABASE IF NOT EXISTS {self._config.database}").collect(
                    statement_params=self._telemetry_stmp
                )
                self._session.sql(f"CREATE SCHEMA IF NOT EXISTS {self._config.full_schema_path}").collect(
                    statement_params=self._telemetry_stmp
                )
                for tag in [
                    FEATURE_VIEW_ENTITY_TAG,
                    FEATURE_VIEW_TS_COL_TAG,
                    FEATURE_STORE_OBJECT_TAG,
                ]:
                    self._session.sql(f"CREATE TAG IF NOT EXISTS {self._get_fully_qualified_name(tag)}").collect(
                        statement_params=self._telemetry_stmp
                    )
            except Exception as e:
                self.clear()
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(f"Failed to create feature store {name}: {e}."),
                )

        logger.info(f"Successfully connected to feature store: {self._config.full_schema_path}.")

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def register_entity(self, entity: Entity) -> None:
        """
        Register Entity in the FeatureStore.

        Args:
            entity: Entity object to register.

        Raises:
            SnowflakeMLException: [ValueError] Entity with same name is already registered.
            SnowflakeMLException: [RuntimeError] Failed to find resources.
        """
        tag_name = self._get_entity_name(entity.name)
        found_rows = self._find_object("TAGS", tag_name)
        if len(found_rows) > 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.OBJECT_ALREADY_EXISTS,
                original_exception=ValueError(f"Entity {entity.name} already exists."),
                suppress_source_trace=True,
            )

        join_keys_str = ENTITY_JOIN_KEY_DELIMITER.join(entity.join_keys)
        tag_name = self._get_fully_qualified_name(tag_name)
        self._session.sql(f"CREATE TAG IF NOT EXISTS {tag_name} COMMENT = '{entity.desc}'").collect(
            statement_params=self._telemetry_stmp
        )
        self._session.sql(
            f"ALTER SCHEMA {self._config.full_schema_path} SET TAG {tag_name} = '{join_keys_str}'"
        ).collect(statement_params=self._telemetry_stmp)
        logger.info(f"Registered Entity {entity}.")

    # TODO: add support to update column desc once SNOW-894249 is fixed
    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def register_feature_view(
        self,
        feature_view: FeatureView,
        version: str,
        refresh_freq: Optional[str] = None,
        warehouse: Optional[str] = None,
        block: bool = False,
    ) -> FeatureView:
        """
        Materialize a FeatureView to Snowflake backend.
        Incremental maintenance for updates on the source data will be automated if refresh_freq is set.

        NOTE: Each new materialization will trigger a full FeatureView history refresh for the data included in the
              FeatureView.

        Args:
            feature_view: FeatureView instance to materialize.
            version: version of the registered FeatureView.
                NOTE: `$` is not a valid char for the version identifier. Also version will be capitalized.
            refresh_freq: Time unit defining how often the new feature data should be generated.
                Valid args are { <num> { seconds | minutes | hours | days } | DOWNSTREAM | <cron expr> <time zone>}.
                NOTE: Currently minimum refresh frequency is 1 minute.
                NOTE: If refresh_freq is in cron expression format, there must be a valid time zone as well.
                    E.g. * * * * * UTC
                NOTE: If refresh_freq is not provided, then FeatureView will be registered as View on Snowflake backend
                    and there won't be extra storage cost.
            warehouse: warehouse to run the compute for the registered FeatureView, if not provided default_warehouse
                specified in the FeatureStore will be used.
            block: Specify whether the FeatureView backend materialization should be blocking or not. If blocking then
                the API will wait until the initial FeatureView data is generated.

        Returns:
            FeatureView object with version and status populated.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView is already registered, or duplicate name and version
                are detected.
            SnowflakeMLException: [ValueError] FeatureView entity has not been registered.
            SnowflakeMLException: [RuntimeError] Failed to create dynamic table, task, or view.
            SnowflakeMLException: [RuntimeError] Failed to find resources.
        """
        if feature_view.status != FeatureViewStatus.DRAFT:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.OBJECT_ALREADY_EXISTS,
                original_exception=ValueError(
                    f"FeatureView {feature_view.name} with version {feature_view.version} has already been registered."
                ),
            )
        self._validate_version_identifier(version)

        # TODO: ideally we should move this to FeatureView creation time
        for e in feature_view.entities:
            if not self._validate_entity_exists(e.name):
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(f"Entity {e.name} has not been registered."),
                )

        feature_view_name = self._get_feature_view_name(feature_view.name, version)
        dynamic_table_results = self._find_object("DYNAMIC TABLES", feature_view_name)
        view_results = self._find_object("VIEWS", feature_view_name)
        if len(dynamic_table_results) > 0 or len(view_results) > 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.OBJECT_ALREADY_EXISTS,
                original_exception=ValueError(
                    f"FeatureView {feature_view.name} with version {version} already exists."
                ),
                suppress_source_trace=True,
            )

        fully_qualified_name = self._get_fully_qualified_name(feature_view_name)
        entities = FEATURE_VIEW_ENTITY_TAG_DELIMITER.join(
            [identifier.get_unescaped_names(e.name) for e in feature_view.entities]
        )
        timestamp_col = (
            feature_view.timestamp_col if feature_view.timestamp_col is not None else TIMESTAMP_COL_PLACEHOLDER
        )

        def create_col_desc(col: StructField) -> str:
            desc = feature_view.feature_descs.get(col.name, None)
            desc = "" if desc is None else f"COMMENT '{desc}'"
            return f"{col.name} {desc}"

        column_descs = ", ".join([f"{create_col_desc(col)}" for col in feature_view.output_schema.fields])
        new_fv = feature_view
        new_fv._version = version
        new_fv._database = self._config.database
        new_fv._schema = self._config.schema

        if refresh_freq is not None:
            schedule_task = refresh_freq != "DOWNSTREAM" and timeparse(refresh_freq) is None
            target_warehouse = self._config.default_warehouse if warehouse is None else warehouse
            new_fv._warehouse = identifier.strip_wrapping_quotes(identifier.resolve_identifier(target_warehouse))
            new_fv._refresh_freq = "DOWNSTREAM" if schedule_task else refresh_freq
            self._create_dynamic_table(
                feature_view_name,
                fully_qualified_name,
                column_descs,
                new_fv,
                entities,
                schedule_task,
                refresh_freq,
                timestamp_col,
                block,
            )
            new_fv._status = self._get_feature_view_status(new_fv)
        else:
            try:
                query = f"""CREATE VIEW {fully_qualified_name} ({column_descs})
                    COMMENT = '{new_fv.desc}'
                    TAG (
                        {FEATURE_VIEW_ENTITY_TAG} = '{entities}',
                        {FEATURE_VIEW_TS_COL_TAG} = '{timestamp_col}',
                        {FEATURE_STORE_OBJECT_TAG} = ''
                    )
                    AS {new_fv.query}
                """
                self._session.sql(query).collect(statement_params=self._telemetry_stmp)
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(f"Create view {fully_qualified_name} [\n{query}\n] failed: {e}"),
                ) from e

            new_fv._status = FeatureViewStatus.STATIC

        logger.info(f"Registered FeatureView {new_fv.name} with version {new_fv.version}.")
        return new_fv

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
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

        fv_name = self._get_feature_view_name(feature_view.name, feature_view.version)
        return self._session.sql(f"SELECT * FROM {self._get_fully_qualified_name(fv_name)}")

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def list_feature_views(
        self,
        entity_name: Optional[str] = None,
        feature_view_name: Optional[str] = None,
        as_dataframe: bool = True,
    ) -> Union[DataFrame, List[FeatureView]]:
        """
        List FeatureViews in the FeatureStore.
        If entity_name is specified, FeatureViews associated with that Entity will be listed.
        If feature_view_name is specified, further reducing the results to only match the specified name.

        Args:
            entity_name: Entity name.
            feature_view_name: FeatureView name.
            as_dataframe: whether the return type should be a DataFrame.

        Returns:
            List of FeatureViews or in a DataFrame representation.
        """
        if entity_name is not None:
            fvs = self._find_feature_views(entity_name, feature_view_name)
        else:
            fv_name = "" if feature_view_name is None else feature_view_name
            fvs = []
            for row in self._get_backend_representations(f"{fv_name}%"):
                fvs.append(self._compose_feature_view(row))

        if as_dataframe:
            values = []
            schema = None
            for fv in fvs:
                values.append(list(fv._to_dict().values()))
                schema = [x.lstrip("_") for x in list(fv._to_dict().keys())] if schema is None else schema
            return self._session.create_dataframe(values, schema=schema)
        else:
            return fvs

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
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
        fv_name = self._get_feature_view_name(name, version)
        results = self._get_backend_representations(fv_name)
        if len(results) != 1:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Failed to find FeatureView {name} with version {version}: {results}"),
            )

        return self._compose_feature_view(results[0])

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def merge_features(
        self,
        features: List[Union[FeatureView, FeatureViewSlice]],
        name: str,
        desc: str = "",
    ) -> FeatureView:
        """
        Merge multiple registered FeatureView or FeatureViewSlice to form a new FeatureView.
        This is typically used to add new features to existing FeatureViews since registered FeatureView is immutable.
        The FeatureViews or FeatureViewSlices to merge should have same Entity and timestamp column setup.

        Args:
            features: List of FeatureViews or FeatureViewSlices to merge
            name: name of the new constructed FeatureView
            desc: description of the new constructed FeatureView

        Returns:
            a new FeatureView with features merged.

        Raises:
            SnowflakeMLException: [ValueError] Features length is not valid or if Entitis and timestamp_col is
                inconsistent.
            SnowflakeMLException: [ValueError] FeatureView has not been registered.
            SnowflakeMLException: [ValueError] FeatureView merge failed.
        """
        if len(features) < 2:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError("features should have at least two entries"),
            )

        left = features[0]
        left_columns = None
        if isinstance(left, FeatureViewSlice):
            left_columns = ", ".join(left.names)
            left = left.feature_view_ref

        if left.status == FeatureViewStatus.DRAFT:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"FeatureView {left.name} has not been registered."),
            )

        join_keys = [k for e in left.entities for k in e.join_keys]

        ts_col_expr = "" if left.timestamp_col is None else f" , {left.timestamp_col}"
        left_columns = "*" if left_columns is None else f"{', '.join(join_keys)}, {left_columns}{ts_col_expr}"
        left_df = self._session.sql(f"SELECT {left_columns} FROM {left.fully_qualified_name()}")

        for right in features[1:]:
            right_columns = None
            if isinstance(right, FeatureViewSlice):
                right_columns = ", ".join(right.names)
                right = right.feature_view_ref

            if left.entities != right.entities:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"Cannot merge FeatureView {left.name} and {right.name} with different Entities: "
                        f"{left.entities} vs {right.entities}"  # noqa: E501
                    ),
                )
            if left.timestamp_col != right.timestamp_col:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(
                        f"Cannot merge FeatureView {left.name} and {right.name} with different timestamp_col: "
                        f"{left.timestamp_col} vs {right.timestamp_col}"  # noqa: E501
                    ),
                )
            if right.status == FeatureViewStatus.DRAFT:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(f"FeatureView {right.name} has not been registered."),
                )

            right_columns = "*" if right_columns is None else f"{', '.join(join_keys)}, {right_columns}"
            exclude_ts_expr = (
                "" if right.timestamp_col is None or right_columns != "*" else f"EXCLUDE {right.timestamp_col}"
            )
            right_df = self._session.sql(
                f"SELECT {right_columns} {exclude_ts_expr} FROM {right.fully_qualified_name()}"
            )

            left_df = left_df.join(right=right_df, on=join_keys)

        return FeatureView(
            name=name,
            entities=left.entities,
            feature_df=left_df,
            timestamp_col=left.timestamp_col,
            desc=desc,
        )

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def resume_feature_view(self, feature_view: FeatureView) -> FeatureView:
        """
        Resume a previously suspended FeatureView.

        Args:
            feature_view: FeatureView to resume.

        Returns:
            FeatureView with updated status.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView is not in suspended status.
            SnowflakeMLException: [RuntimeError] Failed to update feature view status.
        """
        if feature_view.status != FeatureViewStatus.SUSPENDED:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_UPDATE_FAILED,
                original_exception=ValueError(
                    f"FeatureView {feature_view.name} is not in suspended status. Actual status: {feature_view.status}"
                ),
            )

        return self._update_feature_view_status(feature_view, "RESUME")

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def suspend_feature_view(self, feature_view: FeatureView) -> FeatureView:
        """
        Suspend a running FeatureView.

        Args:
            feature_view: FeatureView to suspend.

        Returns:
            FeatureView with updated status.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView is not in running status.
            SnowflakeMLException: [RuntimeError] Failed to update feature view status.
        """
        if feature_view.status != FeatureViewStatus.RUNNING:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_UPDATE_FAILED,
                original_exception=ValueError(
                    f"FeatureView {feature_view.name} is not in running status. Actual status: {feature_view.status}"
                ),
            )
        return self._update_feature_view_status(feature_view, "SUSPEND")

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def delete_feature_view(self, feature_view: FeatureView) -> None:
        """
        Delete a FeatureView.

        Args:
            feature_view: FeatureView to delete.

        Raises:
            SnowflakeMLException: [ValueError] FeatureView is not registered.
        """
        if feature_view.status == FeatureViewStatus.DRAFT or feature_view.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"FeatureView {feature_view.name} has not been registered."),
            )

        fully_qualified_name = self._get_fully_qualified_name(
            self._get_feature_view_name(feature_view.name, feature_view.version)
        )
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

        logger.info(f"Deleted FeatureView {feature_view.name} with version {feature_view.version}.")

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def list_entities(self) -> DataFrame:
        """
        List all Entities in the FeatureStore.

        Returns:
            Snowpark DataFrame containing the results.
        """
        prefix_len = len(ENTITY_TAG_PREFIX) + 1
        tag_values_df = self._session.sql(
            f"""
            SELECT SUBSTR(TAG_NAME,{prefix_len},{ENTITY_NAME_LENGTH_LIMIT}) AS NAME,
            TAG_VALUE AS JOIN_KEYS
            FROM TABLE(
                {self._config.database}.INFORMATION_SCHEMA.TAG_REFERENCES(
                '{self._config.full_schema_path}',
                'SCHEMA'
            )
            )
            WHERE TAG_NAME LIKE '{ENTITY_TAG_PREFIX}%'
        """
        )
        tag_metadata_df = self._session.sql(
            f"SHOW TAGS LIKE '{ENTITY_TAG_PREFIX}%' IN SCHEMA {self._config.full_schema_path}"
        )
        return cast(
            DataFrame,
            tag_values_df.join(
                right=tag_metadata_df.with_column("NAME", F.substr('"name"', prefix_len, ENTITY_NAME_LENGTH_LIMIT))
                .with_column_renamed('"comment"', "DESC")
                .select("NAME", "DESC"),
                on=["NAME"],
                how="left",
            ),
        )

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
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
        full_entity_tag_name = self._get_entity_name(name)
        prefix_len = len(ENTITY_TAG_PREFIX) + 1

        found_tags = self._find_object("TAGS", full_entity_tag_name)
        if len(found_tags) == 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Cannot find Entity with name {name}."),
            )

        try:
            name = self._get_entity_name(name)
            unesc_full_entity_tag_name = identifier.get_unescaped_names(name)

            tag_values = (
                qrc.SqlResultValidator(
                    self._session,
                    f"""
                        SELECT SUBSTR(TAG_NAME,{prefix_len},{ENTITY_NAME_LENGTH_LIMIT}) AS NAME,
                        TAG_VALUE AS JOIN_KEYS
                        FROM TABLE(
                            {self._config.database}.INFORMATION_SCHEMA.TAG_REFERENCES(
                            '{self._config.full_schema_path}',
                            'SCHEMA'
                        )
                        )
                        WHERE TAG_NAME LIKE '{unesc_full_entity_tag_name}'
                        AND TAG_DATABASE = '{identifier.get_unescaped_names(self._config.database)}'
                    """,
                    self._telemetry_stmp,
                )
                .has_dimensions(expected_rows=1)
                .validate()
            )
        except connector.DataError as e:  # raised by SqlResultValidator
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Cannot find Entity with name {name}."),
            ) from e
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to retrieve tag reference information: {e}"),
            ) from e

        return Entity(
            name=tag_values[0]["NAME"],
            join_keys=tag_values[0]["JOIN_KEYS"].split(ENTITY_JOIN_KEY_DELIMITER),
            desc=found_tags[0]["comment"],
        )

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
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
        if not self._validate_entity_exists(name):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Entity {name} does not exist."),
            )

        active_feature_views = list(self.list_feature_views(entity_name=name, as_dataframe=False))
        if len(active_feature_views) > 0:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.SNOWML_DELETE_FAILED,
                original_exception=ValueError(
                    f"Cannot delete Entity {name} due to active FeatureViews: {[f.name for f in active_feature_views]}."
                ),
            )

        tag_name = self._get_fully_qualified_name(self._get_entity_name(name))
        try:
            self._session.sql(f"ALTER SCHEMA {self._config.full_schema_path} UNSET TAG {tag_name}").collect(
                statement_params=self._telemetry_stmp
            )
            self._session.sql(f"DROP TAG IF EXISTS {tag_name}").collect(statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to alter schema or drop tag: {e}."),
            ) from e
        logger.info(f"Deleted Entity {name}.")

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def retrieve_feature_values(
        self,
        spine_df: DataFrame,
        features: Union[List[Union[FeatureView, FeatureViewSlice]], List[str]],
        spine_timestamp_col: Optional[str] = None,
    ) -> DataFrame:
        """
        Enrich spine dataframe with feature values. Mainly used to generate inference data input.
        If spine_timestamp_col is specified, point-in-time feature values will be fetched.

        Args:
            spine_df: Snowpark DataFrame to join features into.
            features: List of features to join into the spine_df. Can be a list of FeatureView or FeatureViewSlice,
                or a list of serialized feature objects from Dataset.
            spine_timestamp_col: Timestamp column in spine_df for point-in-time feature value lookup.

        Returns:
            Snowpark DataFrame containing the joined results.

        Raises:
            ValueError: if features is empty.
        """
        if len(features) == 0:
            raise ValueError("features cannot be empty")
        if isinstance(features[0], str):
            features = self._load_serialized_feature_objects(cast(List[str], features))

        df, _ = self._join_features(
            spine_df,
            cast(List[Union[FeatureView, FeatureViewSlice]], features),
            spine_timestamp_col,
        )
        return df

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def generate_dataset(
        self,
        spine_df: DataFrame,
        features: List[Union[FeatureView, FeatureViewSlice]],
        materialized_table: Optional[str] = None,
        spine_timestamp_col: Optional[str] = None,
        spine_label_cols: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        save_mode: str = "errorifexists",
        desc: str = "",
    ) -> Dataset:
        """
        Generate dataset by given source table and feature views.

        Args:
            spine_df: The fact table contains the raw dataset.
            features: A list of FeatureView or FeatureViewSlice which contains features to be joined.
            materialized_table: The destination table where produced result will be stored. If it's none, then result
                won't be registered. If materialized_table is provided, then produced result will be written into
                the provided table. Note result dataset will be a snowflake clone of registered table.
                New data can append on same registered table and previously generated dataset won't be affected.
                Default result table name will be a concatenation of materialized_table name and current timestamp.
            spine_timestamp_col: Name of timestamp column in spine_df that will be used to join time-series features.
                If spine_timestamp_col is not none, the input features also must have timestamp_col.
            spine_label_cols: Name of column(s) in spine_df that contains labels.
            exclude_columns: Column names to exclude from the result dataframe.
                The underlying storage will still contain the columns.
            save_mode: How new data is saved. currently support:
                errorifexists: Raise error if registered table already exists.
                merge: Merge new data if registered table already exists.
            desc: A description about this dataset.

        Returns:
            A Dataset object.

        Raises:
            SnowflakeMLException: [ValueError] save_mode is invalid.
            SnowflakeMLException: [ValueError] spine_df contains more than one query.
            SnowflakeMLException: [ValueError] Materialized_table contains invalid char `.`.
            SnowflakeMLException: [ValueError] Materialized_table already exists with save_mode `errorifexists`.
            SnowflakeMLException: [ValueError] Snapshot creation failed.
            SnowflakeMLException: [RuntimeError] Failed to create clone from table.
            SnowflakeMLException: [RuntimeError] Failed to find resources.
        """

        allowed_save_mode = {"errorifexists", "merge"}
        if save_mode.lower() not in allowed_save_mode:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"'{save_mode}' is not supported. Current supported save modes: {','.join(allowed_save_mode)}"
                ),
            )

        if len(spine_df.queries["queries"]) != 1:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"spine_df must contain only one query. Got: {spine_df.queries['queries']}"
                ),
            )

        result_df, join_keys = self._join_features(spine_df, features, spine_timestamp_col)

        snapshot_table = None
        if materialized_table is not None:
            if "." in materialized_table:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ARGUMENT,
                    original_exception=ValueError(f"materialized_table {materialized_table} contains invalid char `.`"),
                )

            found_rows = self._find_object("TABLES", materialized_table)
            if save_mode.lower() == "errorifexists" and len(found_rows) > 0:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.OBJECT_ALREADY_EXISTS,
                    original_exception=ValueError(f"Dataset table {materialized_table} already exists."),
                )

            self._dump_dataset(result_df, materialized_table, join_keys, spine_timestamp_col)

            snapshot_table = f"{materialized_table}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
            snapshot_table = self._get_fully_qualified_name(snapshot_table)
            materialized_table = self._get_fully_qualified_name(materialized_table)

            try:
                self._session.sql(f"CREATE TABLE {snapshot_table} CLONE {materialized_table}").collect(
                    statement_params=self._telemetry_stmp
                )
            except Exception as e:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                    original_exception=RuntimeError(
                        f"Failed to create clone {materialized_table} from table {snapshot_table}: {e}."
                    ),
                ) from e

            result_df = self._session.sql(f"SELECT * FROM {snapshot_table}")

        if exclude_columns is not None:
            dataset_cols = identifier.get_unescaped_names(result_df.columns)
            for col in exclude_columns:
                if identifier.get_unescaped_names(col) not in dataset_cols:
                    raise snowml_exceptions.SnowflakeMLException(
                        error_code=error_codes.INVALID_ARGUMENT,
                        original_exception=ValueError(
                            f"{col} in exclude_columns not exists in generated dataset columns: {dataset_cols}"
                        ),
                    )
            result_df = result_df.drop(exclude_columns)

        fs_meta = FeatureStoreMetadata(
            spine_query=spine_df.queries["queries"][0],
            connection_params=vars(self._config),
            features=[fv.to_json() for fv in features],
        )

        dataset = Dataset(
            self._session,
            df=result_df,
            materialized_table=materialized_table,
            snapshot_table=snapshot_table,
            timestamp_col=spine_timestamp_col,
            label_cols=spine_label_cols,
            feature_store_metadata=fs_meta,
            desc=desc,
        )
        return dataset

    @telemetry.send_api_usage_telemetry(project=PROJECT)
    @snowpark_utils.private_preview(version="1.0.8")
    def clear(self) -> None:
        """
        Clear all feature store internal objects including feature views, entities etc. Note feature store
        instance (snowflake schema) won't be deleted. Use snowflake to delete feature store instance.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to clear feature store.
        """
        try:
            result = self._session.sql(
                f"""
                SELECT *
                FROM {self._config.database}.INFORMATION_SCHEMA.SCHEMATA
                WHERE SCHEMA_NAME = '{self._config.schema}'
            """
            ).collect()
            if len(result) == 0:
                return

            object_types = ["DYNAMIC TABLES", "TABLES", "VIEWS", "TASKS"]
            for obj_type in object_types:
                all_object_rows = self._find_object(obj_type, "%")
                for row in all_object_rows:
                    obj_name = self._get_fully_qualified_name(identifier.get_inferred_name(row["name"]))
                    self._session.sql(f"DROP {obj_type[:-1]} {obj_name}").collect()
                    logger.info(f"Deleted {obj_type[:-1]}: {obj_name}.")

            entity_tags = self._find_object("TAGS", f"{ENTITY_TAG_PREFIX}%")
            all_tags = [
                FEATURE_VIEW_ENTITY_TAG,
                FEATURE_VIEW_TS_COL_TAG,
                FEATURE_STORE_OBJECT_TAG,
            ] + [row["name"] for row in entity_tags]
            for tag_name in all_tags:
                obj_name = self._get_fully_qualified_name(identifier.get_inferred_name(tag_name))
                self._session.sql(f"DROP TAG IF EXISTS {obj_name}").collect()
                logger.info(f"Deleted TAG: {obj_name}.")

        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to clear feature store {self._config.full_schema_path}: {e}."),
            ) from e
        logger.info(f"Feature store {self._config.full_schema_path} has been cleared.")

    def _create_dynamic_table(
        self,
        feature_view_name: str,
        fully_qualified_name: str,
        column_descs: str,
        feature_view: FeatureView,
        entities: str,
        schedule_task: bool,
        refresh_freq: str,
        timestamp_col: str,
        block: bool,
    ) -> None:
        # TODO: cluster by join keys once DT supports that
        try:
            query = f"""CREATE DYNAMIC TABLE {fully_qualified_name} ({column_descs})
                TARGET_LAG = '{feature_view._refresh_freq}'
                COMMENT = '{feature_view.desc}'
                TAG (
                    {self._get_fully_qualified_name(FEATURE_VIEW_ENTITY_TAG)} = '{entities}',
                    {self._get_fully_qualified_name(FEATURE_VIEW_TS_COL_TAG)} = '{timestamp_col}',
                    {self._get_fully_qualified_name(FEATURE_STORE_OBJECT_TAG)} = ''
                )
                WAREHOUSE = "{feature_view._warehouse}"
                AS {feature_view.query}
            """
            self._session.sql(query).collect(statement_params=self._telemetry_stmp)
            self._session.sql(f"ALTER DYNAMIC TABLE {fully_qualified_name} REFRESH").collect(
                block=block, statement_params=self._telemetry_stmp
            )

            if schedule_task:
                self._session.sql(
                    f"""CREATE TASK {fully_qualified_name}
                        WAREHOUSE = "{feature_view._warehouse}"
                        SCHEDULE = 'USING CRON {refresh_freq}'
                        AS ALTER DYNAMIC TABLE {fully_qualified_name} REFRESH
                    """
                ).collect(statement_params=self._telemetry_stmp)
                self._session.sql(
                    f"""
                    ALTER TASK {fully_qualified_name}
                    SET TAG {self._get_fully_qualified_name(FEATURE_STORE_OBJECT_TAG)} = ''
                """
                ).collect(statement_params=self._telemetry_stmp)
                self._session.sql(f"ALTER TASK {fully_qualified_name} RESUME").collect(
                    statement_params=self._telemetry_stmp
                )
        except Exception as e:
            self._session.sql(f"DROP DYNAMIC TABLE IF EXISTS {fully_qualified_name}").collect(
                statement_params=self._telemetry_stmp
            )
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(
                    f"Create dynamic table [\n{query}\n] or task {fully_qualified_name} failed: {e}."
                ),
            ) from e

        found_dts = self._find_object("DYNAMIC TABLES", feature_view_name)
        if len(found_dts) != 1:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"Can not find dynamic table: `{feature_view_name}`."),
            )
        if found_dts[0]["refresh_mode"] != "INCREMENTAL":
            warnings.warn(
                f"Dynamic table: `{fully_qualified_name}` will not refresh in INCREMENTAL mode. "
                + "It will likely incurr bigger computation cost. "
                + f"The reason is: {found_dts[0]['refresh_mode_reason']}",
                category=UserWarning,
            )

    def _dump_dataset(
        self,
        df: DataFrame,
        table_name: str,
        join_keys: List[str],
        spine_timestamp_col: Optional[str] = None,
    ) -> None:
        if len(df.queries["queries"]) != 1:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(f"Dataset df must contain only one query. Got: {df.queries['queries']}"),
            )

        schema = ", ".join([f"{c.name} {type_utils.convert_sp_to_sf_type(c.datatype)}" for c in df.schema.fields])
        fully_qualified_name = self._get_fully_qualified_name(table_name)

        try:
            self._session.sql(
                f"""CREATE TABLE IF NOT EXISTS {fully_qualified_name} ({schema})
                    CLUSTER BY ({', '.join(join_keys)})
                    TAG ({self._get_fully_qualified_name(FEATURE_STORE_OBJECT_TAG)} = '')
                """
            ).collect(block=True, statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to create table {fully_qualified_name}: {e}."),
            ) from e

        source_query = df.queries["queries"][0]

        if spine_timestamp_col is not None:
            join_keys.append(spine_timestamp_col)

        _, _, dest_alias, _ = identifier.parse_schema_level_object_identifier(fully_qualified_name)
        source_alias = f"{dest_alias}_source"
        join_cond = " AND ".join([f"{dest_alias}.{k} = {source_alias}.{k}" for k in join_keys])
        update_clause = ", ".join([f"{dest_alias}.{c} = {source_alias}.{c}" for c in df.columns])
        insert_clause = ", ".join([f"{source_alias}.{c}" for c in df.columns])
        query = f"""
            MERGE INTO {fully_qualified_name} USING ({source_query}) {source_alias}  ON {join_cond}
            WHEN MATCHED THEN UPDATE SET {update_clause}
            WHEN NOT MATCHED THEN INSERT ({', '.join(df.columns)}) VALUES ({insert_clause})
        """
        try:
            self._session.sql(query).collect(block=True, statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to create dataset {fully_qualified_name} with merge: {e}."),
            ) from e

    def _validate_version_identifier(self, version: str) -> None:
        if FEATURE_VIEW_NAME_DELIMITER in version:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"Version identifier `{version}` contains invalid character `{FEATURE_VIEW_NAME_DELIMITER}`."
                ),
            )

    def _validate_entity_exists(self, name: str) -> bool:
        full_entity_tag_name = self._get_entity_name(name)
        found_rows = self._find_object("TAGS", full_entity_tag_name)
        return len(found_rows) > 0

    def _join_features(
        self,
        spine_df: DataFrame,
        features: List[Union[FeatureView, FeatureViewSlice]],
        spine_timestamp_col: Optional[str],
    ) -> Tuple[DataFrame, List[str]]:
        if len(spine_df.queries["queries"]) != 1:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"spine_df must contain only one query. Got: {spine_df.queries['queries']}"
                ),
            )

        for f in features:
            f = f.feature_view_ref if isinstance(f, FeatureViewSlice) else f
            if f.status == FeatureViewStatus.DRAFT:
                raise snowml_exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=ValueError(f"FeatureView {f.name} has not been registered."),
                )
            for e in f.entities:
                for k in e.join_keys:
                    k = identifier.get_unescaped_names(k)
                    if k not in identifier.get_unescaped_names(spine_df.columns):
                        raise snowml_exceptions.SnowflakeMLException(
                            error_code=error_codes.INVALID_ARGUMENT,
                            original_exception=ValueError(
                                f"join_key {k} from Entity {e.name} in FeatureView {f.name} is not found in spine_df."
                            ),
                        )

        # TODO: leverage Snowpark dataframe for more concise syntax once it supports AsOfJoin
        query = spine_df.queries["queries"][0]
        layer = 0
        for f in features:
            if isinstance(f, FeatureViewSlice):
                cols = f.names
                f = f.feature_view_ref
            else:
                cols = f.feature_names

            join_keys = [k for e in f.entities for k in e.join_keys]
            join_keys_str = ", ".join(join_keys)
            assert f.version is not None
            join_table_name = self._get_fully_qualified_name(self._get_feature_view_name(f.name, f.version))

            if spine_timestamp_col is not None and f.timestamp_col is not None:
                if _ENABLE_ASOF_JOIN:
                    query = f"""
                        SELECT
                            l_{layer}.*,
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
                        f_cols=cols,
                        f_ts_col=f.timestamp_col,
                        join_keys=join_keys,
                    )
            else:
                query = f"""
                    SELECT
                        l_{layer}.*,
                        r_{layer}.* EXCLUDE {join_keys_str}
                    FROM ({query}) l_{layer}
                    LEFT JOIN (
                        SELECT {join_keys_str}, {', '.join(cols)}
                        FROM {join_table_name}
                    ) r_{layer}
                    ON {' AND '.join([f'l_{layer}.{k} = r_{layer}.{k}' for k in join_keys])}
                """
            layer += 1

        return self._session.sql(query), join_keys

    def _composed_union_window_join_query(
        self,
        layer: int,
        s_query: str,
        s_ts_col: str,
        f_df: DataFrame,
        f_table_name: str,
        f_cols: List[str],
        f_ts_col: str,
        join_keys: List[str],
    ) -> str:
        s_df = self._session.sql(s_query)
        s_only_cols = [col for col in s_df.columns if col not in identifier.get_unescaped_names([*join_keys, s_ts_col])]
        f_only_cols = [col for col in f_df.columns if col not in identifier.get_unescaped_names([*join_keys, f_ts_col])]
        join_keys_str = ", ".join(join_keys)
        temp_prefix = "_fs_temp_"

        def join_cols(cols: List[str], end_comma: bool, rename: bool, prefix: str = "") -> str:
            if not cols:
                return ""
            cols = [f"{prefix}{col}" for col in cols]
            if rename:
                cols = [f"{col} AS {col[len(temp_prefix):]}" for col in cols]
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
                {join_cols(f_only_cols,end_comma=False, rename=False)}
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
                    ORDER BY {s_ts_col}
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS {temp_prefix}{f_col}"""
            )
        window_select = window_select + f" FROM unioned_{layer}"
        window_cte = f"""
            windowed_{layer} AS (
                {window_select}
            )"""

        # Part 4: join original spine table with window table
        prefix_f_only_cols = [f"{temp_prefix}{name}" for name in f_only_cols]
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

    def _get_feature_view_name(self, raw_name: str, version: str) -> str:
        return identifier.concat_names([raw_name, FEATURE_VIEW_NAME_DELIMITER, version])

    def _get_entity_name(self, raw_name: str) -> str:
        return identifier.concat_names([ENTITY_TAG_PREFIX, raw_name])

    def _get_fully_qualified_name(self, name: str) -> str:
        return f"{self._config.full_schema_path}.{name}"

    # TODO: SHOW DYNAMIC TABLES is very slow while other show objects are fast, investigate with DT in SNOW-902804.
    def _get_backend_representations(self, object_name_pattern: str) -> List[Row]:
        dynamic_table_results = self._find_object("DYNAMIC TABLES", object_name_pattern)
        view_results = self._find_object("VIEWS", object_name_pattern)
        return dynamic_table_results + view_results

    def _update_feature_view_status(self, feature_view: FeatureView, operation: str) -> FeatureView:
        if feature_view.status == FeatureViewStatus.DRAFT or feature_view.version is None:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(f"FeatureView {feature_view.name} has not been registered."),
            )

        fully_qualified_name = self._get_fully_qualified_name(
            self._get_feature_view_name(feature_view.name, feature_view.version)
        )
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

        feature_view._status = self._get_feature_view_status(feature_view)
        logger.info(f"Successfully {operation} FeatureView {feature_view.name} with version {feature_view.version}.")
        return feature_view

    def _get_feature_view_status(self, feature_view: FeatureView) -> FeatureViewStatus:
        fv_name = self._get_feature_view_name(
            feature_view.name,
            feature_view.version if feature_view.version is not None else "",
        )
        results = self._get_backend_representations(fv_name)
        if len(results) != 1:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(
                    f"Failed to get status for {feature_view.name} with version {feature_view.version}: {results}"
                ),
            )

        return FeatureViewStatus(results[0]["scheduling_state"])

    def _find_feature_views(self, entity_name: str, feature_view_name: Optional[str]) -> List[FeatureView]:
        if not self._validate_entity_exists(entity_name):
            return []

        all_fv_names = [r["name"] for r in self._get_backend_representations("%")]
        if len(all_fv_names) == 0:
            return []

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
                    AND TAG_NAME = '{FEATURE_VIEW_ENTITY_TAG}'
                """
                for fv_name in identifier.get_escaped_names(all_fv_names)
            ]

            results = self._session.sql("\nUNION\n".join(queries)).collect(statement_params=self._telemetry_stmp)
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to retrieve feature views' information: {e}"),
            ) from e
        outputs = []
        for r in results:
            if identifier.get_unescaped_names(entity_name) in r["TAG_VALUE"]:
                fv_name, version = r["OBJECT_NAME"].split(FEATURE_VIEW_NAME_DELIMITER)
                if feature_view_name is not None:
                    if fv_name == identifier.get_unescaped_names(feature_view_name):
                        outputs.append(self.get_feature_view(fv_name, version))
                    else:
                        continue
                else:
                    outputs.append(self.get_feature_view(fv_name, version))
        return outputs

    def _compose_feature_view(self, row: Row) -> FeatureView:
        name, version = row["name"].split(FEATURE_VIEW_NAME_DELIMITER)

        m = re.match(DT_QUERY_PATTERN, row["text"])
        if m is not None:
            query = m.group("query")
            df = self._session.sql(query)
            desc = m.group("comment")
            entity_names = m.group("entities")
            entities = [self.get_entity(n) for n in entity_names.split(FEATURE_VIEW_ENTITY_TAG_DELIMITER)]
            ts_col = m.group("ts_col")
            timestamp_col = ts_col if ts_col != TIMESTAMP_COL_PLACEHOLDER else None

            fv = FeatureView._construct_feature_view(
                name=name,
                entities=entities,
                feature_df=df,
                timestamp_col=timestamp_col,
                desc=desc,
                version=version,
                status=FeatureViewStatus(row["scheduling_state"]),
                feature_descs=self._fetch_column_descs("DYNAMIC TABLE", row["name"]),
                refresh_freq=m.group("refresh_freq"),
                database=self._config.database,
                schema=self._config.schema,
                warehouse=m.group("warehouse"),
            )
            return fv

        m = re.match(VIEW_QUERY_PATTERN, row["text"])
        if m is not None:
            query = m.group("query")
            df = self._session.sql(query)
            desc = m.group("comment")
            entity_names = m.group("entities")
            entities = [self.get_entity(n) for n in entity_names.split(FEATURE_VIEW_ENTITY_TAG_DELIMITER)]
            ts_col = m.group("ts_col")
            timestamp_col = ts_col if ts_col != TIMESTAMP_COL_PLACEHOLDER else None

            fv = FeatureView._construct_feature_view(
                name=name,
                entities=entities,
                feature_df=df,
                timestamp_col=timestamp_col,
                desc=desc,
                version=version,
                status=FeatureViewStatus.STATIC,
                feature_descs=self._fetch_column_descs("VIEW", row["name"]),
                refresh_freq=None,
                database=self._config.database,
                schema=self._config.schema,
                warehouse=None,
            )
            return fv

        raise snowml_exceptions.SnowflakeMLException(
            error_code=error_codes.INTERNAL_SNOWML_ERROR,
            original_exception=RuntimeError(
                f"Failed to parse query text for FeatureView {name} with version {version}: {row}."
            ),
        )

    def _fetch_column_descs(self, obj_type: str, obj_name: str) -> Dict[str, str]:
        res = self._session.sql(f"DESC {obj_type} {self._get_fully_qualified_name(obj_name)}").collect(
            statement_params=self._telemetry_stmp
        )

        descs = {}
        for r in res:
            if r["comment"] is not None:
                descs[r["name"]] = r["comment"]
        return descs

    def _find_object(self, object_type: str, object_name_pattern: str) -> List[Row]:
        """Try to find an object by given type and name pattern.

        Args:
            object_type: Type of the object. Could be TABLES, TAGS etc.
            object_name_pattern: Name match pattern of object. It obeys snowflake identifier requirements.
                and can be used with SQL wildcard character '%'.
                Examples:
                1. object_name_pattern="bar" will return objects with lowercase name: bar.
                2. object_name_pattern=BAR will return objects with case-insensitive name: bar.
                3. object_name_pattern=BAR% will return objects with name starts with case-insensitive: bar.

        Raises:
            SnowflakeMLException: [RuntimeError] Failed to find resource.

        Returns:
            Return a list of rows round.
        """
        if object_name_pattern == "":
            return []

        if object_name_pattern == "%":
            unesc_object_name = object_name_pattern
            object_name = ""
        elif object_name_pattern[-1] == "%":
            assert '"' not in object_name_pattern, "wildcard search doesn't support double quotes"
            unesc_object_name = object_name_pattern
            object_name = unesc_object_name[:-1]
        else:
            unesc_object_name = identifier.get_unescaped_names(object_name_pattern)
            object_name = unesc_object_name

        search_space, obj_domain = self._obj_search_spaces[object_type]
        all_rows = []
        fs_objects = []
        tag_free_object_types = ["TAGS", "SCHEMAS"]
        try:
            search_scope = f"IN {search_space}" if search_space is not None else ""
            all_rows = self._session.sql(f"SHOW {object_type} LIKE '{unesc_object_name}' {search_scope}").collect(
                statement_params=self._telemetry_stmp
            )
            if object_name_pattern == "%" and object_type not in tag_free_object_types and len(all_rows) > 0:
                # Note: <object_name> in TAG_REFERENCES(<object_name>) is case insensitive,
                # use double quotes to make it case-sensitive.
                queries = [
                    f"""
                        SELECT OBJECT_NAME
                        FROM TABLE(
                            {self._config.database}.INFORMATION_SCHEMA.TAG_REFERENCES(
                                '{self._get_fully_qualified_name(identifier.get_inferred_name(row['name']))}',
                                '{obj_domain}'
                            )
                        )
                        WHERE TAG_NAME = '{FEATURE_STORE_OBJECT_TAG}'
                        AND TAG_SCHEMA = '{identifier.get_unescaped_names(self._config.schema)}'
                    """
                    for row in all_rows
                ]
                fs_obj_rows = self._session.sql("\nUNION\n".join(queries)).collect(
                    statement_params=self._telemetry_stmp
                )
                fs_objects = [row["OBJECT_NAME"] for row in fs_obj_rows]
        except Exception as e:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INTERNAL_SNOWPARK_ERROR,
                original_exception=RuntimeError(f"Failed to find object {object_name_pattern}: {e}"),
            ) from e

        result = []
        for row in all_rows:
            found_name = row["name"]
            if found_name.startswith(object_name) and (
                object_name_pattern != "%" or object_type in tag_free_object_types or found_name in fs_objects
            ):
                result.append(row)
        return result

    def _load_serialized_feature_objects(
        self, serialized_feature_objs: List[str]
    ) -> List[Union[FeatureView, FeatureViewSlice]]:
        results: List[Union[FeatureView, FeatureViewSlice]] = []
        for obj in serialized_feature_objs:
            try:
                obj_type = json.loads(obj)[FEATURE_OBJ_TYPE]
            except Exception as e:
                raise ValueError(f"Malformed serialized feature object: {obj}") from e

            if obj_type == FeatureView.__name__:
                results.append(FeatureView.from_json(obj, self._session))
            elif obj_type == FeatureViewSlice.__name__:
                results.append(FeatureViewSlice.from_json(obj, self._session))
            else:
                raise ValueError(f"Unsupported feature object type: {obj_type}")
        return results
