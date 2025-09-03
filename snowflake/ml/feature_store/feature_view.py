from __future__ import annotations

import json
import logging
import re
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional, Union

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils import identifier
from snowflake.ml._internal.utils.identifier import concat_names
from snowflake.ml._internal.utils.sql_identifier import (
    SqlIdentifier,
    to_sql_identifiers,
)
from snowflake.ml.feature_store import feature_store
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.lineage import lineage_node
from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.types import (
    DateType,
    StructType,
    TimestampType,
    TimeType,
    _NumericType,
)

_DEFAULT_TARGET_LAG = "10 seconds"
_FEATURE_VIEW_NAME_DELIMITER = "$"
_LEGACY_TIMESTAMP_COL_PLACEHOLDER_VALS = ["FS_TIMESTAMP_COL_PLACEHOLDER_VAL", "NULL"]
_TIMESTAMP_COL_PLACEHOLDER = "NULL"
_FEATURE_OBJ_TYPE = "FEATURE_OBJ_TYPE"
_ONLINE_TABLE_SUFFIX = "$ONLINE"
# Feature view version rule is aligned with dataset version rule in SQL.
_FEATURE_VIEW_VERSION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]*$")
_FEATURE_VIEW_VERSION_MAX_LENGTH = 128

_RESULT_SCAN_QUERY_PATTERN = re.compile(
    r".*FROM\s*TABLE\s*\(\s*RESULT_SCAN\s*\(.*",
    flags=re.DOTALL | re.IGNORECASE | re.X,
)


@dataclass(frozen=True)
class OnlineConfig:
    """Configuration for online feature storage."""

    enable: bool = False
    target_lag: Optional[str] = None

    def __post_init__(self) -> None:
        if self.target_lag is None:
            return
        if not isinstance(self.target_lag, str) or not self.target_lag.strip():
            raise ValueError("target_lag must be a non-empty string")

        object.__setattr__(self, "target_lag", self.target_lag.strip())

    def to_json(self) -> str:
        data: dict[str, Any] = asdict(self)
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> OnlineConfig:
        data = json.loads(json_str)
        return cls(**data)


class StoreType(Enum):
    """
    Enumeration for specifying the storage type when reading from or refreshing feature views.

    The Feature View supports two storage modes:
    - OFFLINE: Traditional batch storage for historical feature data and training
    - ONLINE: Low-latency storage optimized for real-time feature serving
    """

    ONLINE = "online"
    OFFLINE = "offline"


@dataclass(frozen=True)
class _FeatureViewMetadata:
    """Represent metadata tracked on top of FV backend object"""

    entities: list[str]
    timestamp_col: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> _FeatureViewMetadata:
        state_dict = json.loads(json_str)
        return cls(**state_dict)


@dataclass(frozen=True)
class _CompactRepresentation:
    """
    A compact representation for FeatureView and FeatureViewSlice, which contains fully qualified name
    and optionally a list of feature indices (None means all features will be included).
    This is to make the metadata much smaller when generating dataset.
    """

    db: str
    sch: str
    name: str
    version: str
    feature_indices: Optional[list[int]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> _CompactRepresentation:
        state_dict = json.loads(json_str)
        return cls(**state_dict)


class FeatureViewVersion(str):
    def __new__(cls, version: str) -> FeatureViewVersion:
        if not _FEATURE_VIEW_VERSION_RE.match(version) or len(version) > _FEATURE_VIEW_VERSION_MAX_LENGTH:
            raise ValueError(
                f"`{version}` is not a valid feature view version. "
                "It must start with letter or digit, and followed by letter, digit, '_', '-' or '.'. "
                f"The length limit is {_FEATURE_VIEW_VERSION_MAX_LENGTH}."
            )
        return super().__new__(cls, version)

    def __init__(self, version: str) -> None:
        super().__init__()


class FeatureViewStatus(Enum):
    MASKED = "MASKED"  # for shared feature views where scheduling state is not available
    DRAFT = "DRAFT"
    STATIC = "STATIC"
    RUNNING = "RUNNING"  # This can be deprecated after BCR 2024_02 gets fully deployed
    SUSPENDED = "SUSPENDED"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class FeatureViewSlice:
    feature_view_ref: FeatureView
    names: list[SqlIdentifier]

    def __repr__(self) -> str:
        states = (f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({', '.join(states)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureViewSlice):
            return False

        return self.names == other.names and self.feature_view_ref == other.feature_view_ref

    def to_json(self) -> str:
        fvs_dict = {
            "feature_view_ref": self.feature_view_ref.to_json(),
            "names": self.names,
            _FEATURE_OBJ_TYPE: self.__class__.__name__,
        }
        return json.dumps(fvs_dict)

    @classmethod
    def from_json(cls, json_str: str, session: Session) -> FeatureViewSlice:
        json_dict = json.loads(json_str)
        if _FEATURE_OBJ_TYPE not in json_dict or json_dict[_FEATURE_OBJ_TYPE] != cls.__name__:
            raise ValueError(f"Invalid json str for {cls.__name__}: {json_str}")
        del json_dict[_FEATURE_OBJ_TYPE]
        json_dict["feature_view_ref"] = FeatureView.from_json(json_dict["feature_view_ref"], session)
        return cls(**json_dict)

    def _get_compact_repr(self) -> _CompactRepresentation:
        return _CompactRepresentation(
            db=self.feature_view_ref.database.identifier(),  # type: ignore[union-attr]
            sch=self.feature_view_ref.schema.identifier(),  # type: ignore[union-attr]
            name=self.feature_view_ref.name.identifier(),
            version=self.feature_view_ref.version,  # type: ignore[arg-type]
            feature_indices=self._feature_names_to_indices(),
        )

    def _feature_names_to_indices(self) -> list[int]:
        name_to_indices_map = {name: idx for idx, name in enumerate(self.feature_view_ref.feature_names)}
        return [name_to_indices_map[n] for n in self.names]


class FeatureView(lineage_node.LineageNode):
    """
    A FeatureView instance encapsulates a logical group of features.
    """

    def __init__(
        self,
        name: str,
        entities: list[Entity],
        feature_df: DataFrame,
        *,
        timestamp_col: Optional[str] = None,
        refresh_freq: Optional[str] = None,
        desc: str = "",
        warehouse: Optional[str] = None,
        initialize: str = "ON_CREATE",
        refresh_mode: str = "AUTO",
        cluster_by: Optional[list[str]] = None,
        online_config: Optional[OnlineConfig] = None,
        **_kwargs: Any,
    ) -> None:
        """
        Create a FeatureView instance.

        Args:
            name: name of the FeatureView. NOTE: following Snowflake identifier rule
            entities: entities that the FeatureView is associated with.
            feature_df: Snowpark DataFrame containing data source and all feature feature_df logics.
                Final projection of the DataFrame should contain feature names, join keys and timestamp(if applicable).
            timestamp_col: name of the timestamp column for point-in-time lookup when consuming the
                feature values.
            refresh_freq: Time unit defining how often the new feature data should be generated.
                Valid args are { <num> { seconds | minutes | hours | days } | DOWNSTREAM | <cron expr> <time zone>}.
                NOTE: Currently minimum refresh frequency is 1 minute.
                NOTE: If refresh_freq is in cron expression format, there must be a valid time zone as well.
                    E.g. * * * * * UTC
                NOTE: If refresh_freq is not provided, then FeatureView will be registered as View on Snowflake backend
                    and there won't be extra storage cost.
            desc: description of the FeatureView.
            warehouse: warehouse to refresh feature view. Not needed for static feature view (refresh_freq is None).
                For managed feature view, this warehouse will overwrite the default warehouse of Feature Store if it is
                specified, otherwise the default warehouse will be used.
            initialize: Specifies the behavior of the initial refresh of feature view. This property cannot be altered
                after you register the feature view. It supports ON_CREATE (default) or ON_SCHEDULE. ON_CREATE refreshes
                the feature view synchronously at creation. ON_SCHEDULE refreshes the feature view at the next scheduled
                refresh. It is only effective when refresh_freq is not None.
            refresh_mode: The refresh mode of managed feature view. The value can be 'AUTO', 'FULL' or 'INCREMENTAL'.
                For managed feature view, the default value is 'AUTO'. For static feature view it has no effect.
                Check https://docs.snowflake.com/en/sql-reference/sql/create-dynamic-table for for details.
            cluster_by: Columns to cluster the feature view by.
                - Defaults to the join keys from entities.
                - If `timestamp_col` is provided, it is added to the default clustering keys.
            online_config: Optional configuration for online storage. If provided with enable=True,
                online storage will be enabled. Defaults to None (no online storage).
            _kwargs: reserved kwargs for system generated args. NOTE: DO NOT USE.

        Example::

            >>> fs = FeatureStore(...)
            >>> # draft_fv is a local object that hasn't materialized to Snowflake backend yet.
            >>> feature_df = session.sql("select f_1, f_2 from source_table")
            >>> draft_fv = FeatureView(
            ...     name="my_fv",
            ...     entities=[e1, e2],
            ...     feature_df=feature_df,
            ...     timestamp_col='TS', # optional
            ...     refresh_freq='1d',  # optional
            ...     desc='A line about this feature view',  # optional
            ...     warehouse='WH'      # optional, the warehouse used to refresh (managed) feature view
            ... )
            >>> print(draft_fv.status)
            FeatureViewStatus.DRAFT
            <BLANKLINE>
            >>> # registered_fv is a local object that maps to a Snowflake backend object.
            >>> registered_fv = fs.register_feature_view(draft_fv, "v1")
            >>> print(registered_fv.status)
            FeatureViewStatus.ACTIVE
            <BLANKLINE>
            >>> # Example with online configuration for online feature storage
            >>> config = OnlineConfig(enable=True, target_lag='15s')
            >>> online_fv = FeatureView(
            ...     name="my_online_fv",
            ...     entities=[e1, e2],
            ...     feature_df=feature_df,
            ...     timestamp_col='TS',
            ...     refresh_freq='1d',
            ...     desc='Feature view with online storage',
            ...     online_config=config  # optional, enables online feature storage
            ... )
            >>> registered_online_fv = fs.register_feature_view(online_fv, "v1")
            >>> print(registered_online_fv.online)
            True

        # noqa: DAR401
        """
        if online_config is not None:
            logging.warning("'online_config' is in private preview since 1.12.0. Do not use it in production.")

        self._name: SqlIdentifier = SqlIdentifier(name)
        self._entities: list[Entity] = entities
        self._feature_df: DataFrame = feature_df
        self._timestamp_col: Optional[SqlIdentifier] = (
            SqlIdentifier(timestamp_col) if timestamp_col is not None else None
        )
        self._desc: str = desc
        self._infer_schema_df: DataFrame = _kwargs.pop("_infer_schema_df", self._feature_df)
        self._query: str = self._get_query()
        self._version: Optional[FeatureViewVersion] = None
        self._status: FeatureViewStatus = FeatureViewStatus.DRAFT
        feature_names = self._get_feature_names()
        self._feature_desc: Optional[OrderedDict[SqlIdentifier, str]] = (
            OrderedDict((f, "") for f in feature_names) if feature_names is not None else None
        )
        self._refresh_freq: Optional[str] = refresh_freq
        self._database: Optional[SqlIdentifier] = None
        self._schema: Optional[SqlIdentifier] = None
        self._initialize: str = initialize
        self._warehouse: Optional[SqlIdentifier] = SqlIdentifier(warehouse) if warehouse is not None else None
        self._refresh_mode: Optional[str] = refresh_mode
        self._refresh_mode_reason: Optional[str] = None
        self._owner: Optional[str] = None
        self._cluster_by: list[SqlIdentifier] = (
            [SqlIdentifier(col) for col in cluster_by] if cluster_by is not None else self._get_default_cluster_by()
        )
        self._online_config: Optional[OnlineConfig] = online_config

        # Validate kwargs
        if _kwargs:
            raise TypeError(f"FeatureView.__init__ got an unexpected keyword argument: '{next(iter(_kwargs.keys()))}'")

        self._validate()

    def slice(self, names: list[str]) -> FeatureViewSlice:
        """
        Select a subset of features within the FeatureView.

        Args:
            names: feature names to select.

        Returns:
            FeatureViewSlice instance containing selected features.

        Raises:
            ValueError: if selected feature names is not found in the FeatureView.

        Example::

            >>> fs = FeatureStore(...)
            >>> e = fs.get_entity('TRIP_ID')
            >>> # feature_df contains 3 features and 1 entity
            >>> feature_df = session.table(source_table).select(
            ...     'TRIPDURATION',
            ...     'START_STATION_LATITUDE',
            ...     'END_STATION_LONGITUDE',
            ...     'TRIP_ID'
            ... )
            >>> darft_fv = FeatureView(name='F_TRIP', entities=[e], feature_df=feature_df)
            >>> fv = fs.register_feature_view(darft_fv, version='1.0')
            >>> # shows all 3 features
            >>> fv.feature_names
            ['TRIPDURATION', 'START_STATION_LATITUDE', 'END_STATION_LONGITUDE']
            <BLANKLINE>
            >>> # slice a subset of features
            >>> fv_slice = fv.slice(['TRIPDURATION', 'START_STATION_LATITUDE'])
            >>> fv_slice.names
            ['TRIPDURATION', 'START_STATION_LATITUDE']
            <BLANKLINE>
            >>> # query the full set of features in original feature view
            >>> fv_slice.feature_view_ref.feature_names
            ['TRIPDURATION', 'START_STATION_LATITUDE', 'END_STATION_LONGITUDE']

        """

        res = []
        for name in names:
            name = SqlIdentifier(name)
            if name not in self.feature_names:
                raise ValueError(f"Feature name {name} not found in FeatureView {self.name}.")
            res.append(name)
        return FeatureViewSlice(self, res)

    def fully_qualified_name(self) -> str:
        """
        Returns the fully qualified name (<database_name>.<schema_name>.<feature_view_name>) for the
        FeatureView in Snowflake.

        Returns:
            fully qualified name string.

        Raises:
            RuntimeError: if the FeatureView is not registered.

        Example::

            >>> fs = FeatureStore(...)
            >>> e = fs.get_entity('TRIP_ID')
            >>> feature_df = session.table(source_table).select(
            ...     'TRIPDURATION',
            ...     'START_STATION_LATITUDE',
            ...     'TRIP_ID'
            ... )
            >>> darft_fv = FeatureView(name='F_TRIP', entities=[e], feature_df=feature_df)
            >>> registered_fv = fs.register_feature_view(darft_fv, version='1.0')
            >>> registered_fv.fully_qualified_name()
            'MY_DB.MY_SCHEMA."F_TRIP$1.0"'

        """
        if self.status == FeatureViewStatus.DRAFT or self.version is None:
            raise RuntimeError(f"FeatureView {self.name} has not been registered.")
        return f"{self._database}.{self._schema}.{FeatureView._get_physical_name(self.name, self.version)}"

    def attach_feature_desc(self, descs: dict[str, str]) -> FeatureView:
        """
        Associate feature level descriptions to the FeatureView.

        Args:
            descs: Dictionary contains feature name and corresponding descriptions.

        Returns:
            FeatureView with feature level desc attached.

        Raises:
            ValueError: if feature name is not found in the FeatureView.

        Example::

            >>> fs = FeatureStore(...)
            >>> e = fs.get_entity('TRIP_ID')
            >>> feature_df = session.table(source_table).select('TRIPDURATION', 'START_STATION_LATITUDE', 'TRIP_ID')
            >>> draft_fv = FeatureView(name='F_TRIP', entities=[e], feature_df=feature_df)
            >>> draft_fv = draft_fv.attach_feature_desc({
            ...     "TRIPDURATION": "Duration of a trip.",
            ...     "START_STATION_LATITUDE": "Latitude of the start station."
            ... })
            >>> registered_fv = fs.register_feature_view(draft_fv, version='1.0')
            >>> registered_fv.feature_descs
            OrderedDict([('TRIPDURATION', 'Duration of a trip.'),
                ('START_STATION_LATITUDE', 'Latitude of the start station.')])

        """
        if self._feature_desc is None:
            warnings.warn(
                "Failed to read feature view schema. Probably feature view is not refreshed yet. "
                "Schema will be available after initial refresh.",
                stacklevel=2,
                category=UserWarning,
            )
            return self

        for f, d in descs.items():
            f = SqlIdentifier(f)
            if f not in self._feature_desc:
                raise ValueError(
                    f"Feature name {f} is not found in FeatureView {self.name}, "
                    f"valid feature names are: {self.feature_names}"
                )
            self._feature_desc[f] = d
        return self

    @property
    def name(self) -> SqlIdentifier:
        return self._name

    @property
    def entities(self) -> list[Entity]:
        return self._entities

    @property
    def feature_df(self) -> DataFrame:
        return self._feature_df

    @property
    def timestamp_col(self) -> Optional[SqlIdentifier]:
        return self._timestamp_col

    @property
    def cluster_by(self) -> Optional[list[SqlIdentifier]]:
        return self._cluster_by

    @property
    def desc(self) -> str:
        return self._desc

    @desc.setter
    def desc(self, new_value: str) -> None:
        """Set the description of feature view.

        Args:
            new_value: new value of description.

        Example::

            >>> fs = FeatureStore(...)
            >>> e = fs.get_entity('TRIP_ID')
            >>> darft_fv = FeatureView(
            ...     name='F_TRIP',
            ...     entities=[e],
            ...     feature_df=feature_df,
            ...     desc='old desc'
            ... )
            >>> fv_1 = fs.register_feature_view(darft_fv, version='1.0')
            >>> print(fv_1.desc)
            old desc
            <BLANKLINE>
            >>> darft_fv.desc = 'NEW DESC'
            >>> fv_2 = fs.register_feature_view(darft_fv, version='2.0')
            >>> print(fv_2.desc)
            NEW DESC

        """
        warnings.warn(
            "You must call register_feature_view() to make it effective. "
            "Or use update_feature_view(desc=<new_value>).",
            stacklevel=2,
            category=UserWarning,
        )
        self._desc = new_value

    @property
    def query(self) -> str:
        return self._query

    @property
    def version(self) -> Optional[FeatureViewVersion]:
        return self._version

    @property
    def status(self) -> FeatureViewStatus:
        return self._status

    @property
    def feature_names(self) -> list[SqlIdentifier]:
        return list(self._feature_desc.keys()) if self._feature_desc is not None else []

    @property
    def feature_descs(self) -> Optional[dict[SqlIdentifier, str]]:
        return self._feature_desc

    @property
    def online(self) -> bool:
        return self._online_config.enable if self._online_config else False

    @property
    def online_config(self) -> Optional[OnlineConfig]:
        return self._online_config

    def fully_qualified_online_table_name(self) -> str:
        """Get the fully qualified name for the online feature table.

        Returns:
            The fully qualified name (<database_name>.<schema_name>.<online_table_name>) for the
            online feature table in Snowflake.

        Raises:
            RuntimeError: if the FeatureView is not registered or not configured for online storage.
        """
        if self.status == FeatureViewStatus.DRAFT or self.version is None:
            raise RuntimeError(f"FeatureView {self.name} has not been registered.")
        if not self.online:
            raise RuntimeError(f"FeatureView {self.name} is not configured for online storage.")
        online_table_name = self._get_online_table_name(self.name, self.version)
        return f"{self._database}.{self._schema}.{online_table_name}"

    def list_columns(self) -> DataFrame:
        """List all columns and their information.

        Returns:
            A Snowpark DataFrame contains feature information.

        Example::

            >>> fs = FeatureStore(...)
            >>> e = Entity("foo", ["id"], desc='my entity')
            >>> fs.register_entity(e)
            <BLANKLINE>
            >>> draft_fv = FeatureView(
            ...     name="fv",
            ...     entities=[e],
            ...     feature_df=self._session.table(<source_table>).select(["NAME", "ID", "TITLE", "AGE", "TS"]),
            ...     timestamp_col="ts",
            >>> ).attach_feature_desc({"AGE": "my age", "TITLE": '"my title"'})
            >>> fv = fs.register_feature_view(draft_fv, '1.0')
            <BLANKLINE>
            >>> fv.list_columns().show()
            --------------------------------------------------
            |"NAME"  |"CATEGORY"  |"DTYPE"      |"DESC"      |
            --------------------------------------------------
            |NAME    |FEATURE     |string(64)   |            |
            |ID      |ENTITY      |bigint       |my entity   |
            |TITLE   |FEATURE     |string(128)  |"my title"  |
            |AGE     |FEATURE     |bigint       |my age      |
            |TS      |TIMESTAMP   |bigint       |NULL        |
            --------------------------------------------------

        """
        session = self._feature_df.session
        rows = []  # type: ignore[var-annotated]

        if self.feature_descs is None:
            warnings.warn(
                "Failed to read feature view schema. Probably feature view is not refreshed yet. "
                "Schema will be available after initial refresh.",
                stacklevel=2,
                category=UserWarning,
            )
            return session.create_dataframe(rows, schema=["name", "category", "dtype", "desc"])

        for name, type in self._feature_df.dtypes:
            if SqlIdentifier(name) in self.feature_descs:
                desc = self.feature_descs[SqlIdentifier(name)]
                rows.append((name, "FEATURE", type, desc))
            elif SqlIdentifier(name) == self._timestamp_col:
                rows.append((name, "TIMESTAMP", type, None))  # type: ignore[arg-type]
            else:
                for e in self._entities:
                    if SqlIdentifier(name) in e.join_keys:
                        rows.append((name, "ENTITY", type, e.desc))
                        break

        return session.create_dataframe(rows, schema=["name", "category", "dtype", "desc"])

    @property
    def refresh_freq(self) -> Optional[str]:
        return self._refresh_freq

    @refresh_freq.setter
    def refresh_freq(self, new_value: str) -> None:
        """Set refresh frequency of feature view.

        Args:
            new_value: The new value of refresh frequency.

        Example::

            >>> fs = FeatureStore(...)
            >>> e = fs.get_entity('TRIP_ID')
            >>> darft_fv = FeatureView(
            ...     name='F_TRIP',
            ...     entities=[e],
            ...     feature_df=feature_df,
            ...     refresh_freq='1d'
            ... )
            >>> fv_1 = fs.register_feature_view(darft_fv, version='1.0')
            >>> print(fv_1.refresh_freq)
            1 day
            <BLANKLINE>
            >>> darft_fv.refresh_freq = '12h'
            >>> fv_2 = fs.register_feature_view(darft_fv, version='2.0')
            >>> print(fv_2.refresh_freq)
            12 hours

        """
        warnings.warn(
            "You must call register_feature_view() to make it effective. "
            "Or use update_feature_view(refresh_freq=<new_value>).",
            stacklevel=2,
            category=UserWarning,
        )
        self._refresh_freq = new_value

    @property
    def database(self) -> Optional[SqlIdentifier]:
        return self._database

    @property
    def schema(self) -> Optional[SqlIdentifier]:
        return self._schema

    @property
    def warehouse(self) -> Optional[SqlIdentifier]:
        return self._warehouse

    @warehouse.setter
    def warehouse(self, new_value: str) -> None:
        """Set warehouse of feature view.

        Args:
            new_value: The new value of warehouse.

        Example::

            >>> fs = FeatureStore(...)
            >>> e = fs.get_entity('TRIP_ID')
            >>> darft_fv = FeatureView(
            ...     name='F_TRIP',
            ...     entities=[e],
            ...     feature_df=feature_df,
            ...     refresh_freq='1d',
            ...     warehouse='WH1',
            ... )
            >>> fv_1 = fs.register_feature_view(darft_fv, version='1.0')
            >>> print(fv_1.warehouse)
            WH1
            <BLANKLINE>
            >>> darft_fv.warehouse = 'WH2'
            >>> fv_2 = fs.register_feature_view(darft_fv, version='2.0')
            >>> print(fv_2.warehouse)
            WH2

        """
        warnings.warn(
            "You must call register_feature_view() to make it effective. "
            "Or use update_feature_view(warehouse=<new_value>).",
            stacklevel=2,
            category=UserWarning,
        )
        self._warehouse = SqlIdentifier(new_value)

    @property
    def initialize(self) -> str:
        return self._initialize

    @property
    def output_schema(self) -> StructType:
        return self._infer_schema_df.schema

    @property
    def refresh_mode(self) -> Optional[str]:
        return self._refresh_mode

    @property
    def refresh_mode_reason(self) -> Optional[str]:
        return self._refresh_mode_reason

    @property
    def owner(self) -> Optional[str]:
        return self._owner

    def _metadata(self) -> _FeatureViewMetadata:
        entity_names = [e.name.identifier() for e in self.entities]
        ts_col = self.timestamp_col.identifier() if self.timestamp_col is not None else _TIMESTAMP_COL_PLACEHOLDER
        return _FeatureViewMetadata(entity_names, ts_col)

    def _get_query(self) -> str:
        if len(self._feature_df.queries["queries"]) != 1:
            raise ValueError(
                f"""feature_df dataframe must contain only 1 query.
Got {len(self._feature_df.queries['queries'])}: {self._feature_df.queries['queries']}
"""
            )
        return str(self._feature_df.queries["queries"][0])

    def _validate(self) -> None:
        if _FEATURE_VIEW_NAME_DELIMITER in self._name:
            raise ValueError(
                f"FeatureView name `{self._name}` contains invalid character `{_FEATURE_VIEW_NAME_DELIMITER}`."
            )

        df_cols = self._get_column_names()
        if df_cols is not None:
            for e in self._entities:
                for k in e.join_keys:
                    if k not in df_cols:
                        raise ValueError(f"join_key {k} in Entity {e.name} is not found in input dataframe: {df_cols}")

            if self._timestamp_col is not None:
                ts_col = self._timestamp_col
                if ts_col == SqlIdentifier(_TIMESTAMP_COL_PLACEHOLDER):
                    raise ValueError(f"Invalid timestamp_col name, cannot be {_TIMESTAMP_COL_PLACEHOLDER}.")
                if ts_col not in df_cols:
                    raise ValueError(f"timestamp_col {ts_col} is not found in input dataframe.")

                col_type = self._infer_schema_df.schema[ts_col].datatype
                if not isinstance(col_type, (DateType, TimeType, TimestampType, _NumericType)):
                    raise ValueError(f"Invalid data type for timestamp_col {ts_col}: {col_type}.")

            if self.cluster_by is not None:
                for column in self.cluster_by:
                    if column not in df_cols:
                        raise ValueError(
                            f"Column '{column}' in `cluster_by` is not in the feature DataFrame schema. "
                            f"{df_cols}, {self.cluster_by}"
                        )

        if re.match(_RESULT_SCAN_QUERY_PATTERN, self._query) is not None:
            raise ValueError(f"feature_df should not be reading from RESULT_SCAN. Invalid query: {self._query}")

        if self._initialize not in ["ON_CREATE", "ON_SCHEDULE"]:
            raise ValueError("'initialize' only supports ON_CREATE or ON_SCHEDULE.")

    def _get_column_names(self) -> Optional[list[SqlIdentifier]]:
        try:
            return to_sql_identifiers(self._infer_schema_df.columns)
        except SnowparkSQLException as e:
            warnings.warn(
                "Failed to read feature view schema. Probably feature view is not refreshed yet. "
                f"Schema will be available after initial refresh. Original exception: {e}",
                stacklevel=2,
                category=UserWarning,
            )
            return None

    def _get_feature_names(self) -> Optional[list[SqlIdentifier]]:
        join_keys = [k for e in self._entities for k in e.join_keys]
        ts_col = [self._timestamp_col] if self._timestamp_col is not None else []
        feature_names = self._get_column_names()
        if feature_names is not None:
            return [c for c in feature_names if c not in join_keys + ts_col]
        return None

    def __repr__(self) -> str:
        states = (f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({', '.join(states)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureView):
            return False

        return (
            self.name == other.name
            and self.version == other.version
            and self.timestamp_col == other.timestamp_col
            and self.entities == other.entities
            and self.desc == other.desc
            and self.feature_descs == other.feature_descs
            and self.feature_names == other.feature_names
            and self.query == other.query
            and self.refresh_freq == other.refresh_freq
            and str(self.status) == str(other.status)
            and self.database == other.database
            and self.warehouse == other.warehouse
            and self.refresh_mode == other.refresh_mode
            and self.refresh_mode_reason == other.refresh_mode_reason
            and self._owner == other._owner
        )

    def _to_dict(self) -> dict[str, str]:
        fv_dict = self.__dict__.copy()
        if "_feature_df" in fv_dict:
            fv_dict.pop("_feature_df")
        if "_infer_schema_df" in fv_dict:
            infer_schema_df = fv_dict.pop("_infer_schema_df")
            fv_dict["_infer_schema_query"] = infer_schema_df.queries["queries"][0]
        fv_dict["_entities"] = [e._to_dict() for e in self._entities]
        fv_dict["_status"] = str(self._status)
        fv_dict["_name"] = str(self._name) if self._name is not None else None
        fv_dict["_version"] = str(self._version) if self._version is not None else None
        fv_dict["_database"] = str(self._database) if self._database is not None else None
        fv_dict["_schema"] = str(self._schema) if self._schema is not None else None
        fv_dict["_warehouse"] = str(self._warehouse) if self._warehouse is not None else None
        fv_dict["_timestamp_col"] = str(self._timestamp_col) if self._timestamp_col is not None else None
        fv_dict["_initialize"] = str(self._initialize)

        feature_desc_dict = {}
        if self._feature_desc is not None:
            for k, v in self._feature_desc.items():
                feature_desc_dict[k.identifier()] = v
            fv_dict["_feature_desc"] = feature_desc_dict

        fv_dict["_online_config"] = self._online_config.to_json() if self._online_config is not None else None

        lineage_node_keys = [key for key in fv_dict if key.startswith("_node") or key == "_session"]

        for key in lineage_node_keys:
            fv_dict.pop(key)

        return fv_dict

    def to_df(self, session: Optional[Session] = None) -> DataFrame:
        """Convert feature view to a Snowpark DataFrame object.

        Args:
            session: [deprecated] This argument has no effect. No need to pass a session object.

        Returns:
            A Snowpark Dataframe object contains the information about feature view.

        Example::

            >>> fs = FeatureStore(...)
            >>> e = Entity("foo", ["id"], desc='my entity')
            >>> fs.register_entity(e)
            <BLANKLINE>
            >>> draft_fv = FeatureView(
            ...     name="fv",
            ...     entities=[e],
            ...     feature_df=self._session.table(<source_table>).select(["NAME", "ID", "TITLE", "AGE", "TS"]),
            ...     timestamp_col="ts",
            >>> ).attach_feature_desc({"AGE": "my age", "TITLE": '"my title"'})
            >>> fv = fs.register_feature_view(draft_fv, '1.0')
            <BLANKLINE>
            >>> fv.to_df().show()
            ----------------------------------------------------------------...
            |"NAME"  |"ENTITIES"                |"TIMESTAMP_COL"  |"DESC"  |
            ----------------------------------------------------------------...
            |FV      |[                         |TS               |foobar  |
            |        |  {                       |                 |        |
            |        |    "desc": "my entity",  |                 |        |
            |        |    "join_keys": [        |                 |        |
            |        |      "ID"                |                 |        |
            |        |    ],                    |                 |        |
            |        |    "name": "FOO",        |                 |        |
            |        |    "owner": null         |                 |        |
            |        |  }                       |                 |        |
            |        |]                         |                 |        |
            ----------------------------------------------------------------...
        """
        values = list(self._to_dict().values())
        schema = [x.lstrip("_") for x in list(self._to_dict().keys())]
        values.append(str(FeatureView._get_physical_name(self._name, self._version)))  # type: ignore[arg-type]
        schema.append("physical_name")
        return self._feature_df.session.create_dataframe([values], schema=schema)

    def to_json(self) -> str:
        state_dict = self._to_dict()
        state_dict[_FEATURE_OBJ_TYPE] = self.__class__.__name__
        return json.dumps(state_dict)

    @classmethod
    def from_json(cls, json_str: str, session: Session) -> FeatureView:
        json_dict = json.loads(json_str)
        if _FEATURE_OBJ_TYPE not in json_dict or json_dict[_FEATURE_OBJ_TYPE] != cls.__name__:
            raise ValueError(f"Invalid json str for {cls.__name__}: {json_str}")

        entities = []
        for e_json in json_dict["_entities"]:
            e = Entity(e_json["name"], e_json["join_keys"], desc=e_json["desc"])
            e.owner = e_json["owner"]
            entities.append(e)

        return FeatureView._construct_feature_view(
            name=json_dict["_name"],
            entities=entities,
            feature_df=session.sql(json_dict["_query"]),
            timestamp_col=json_dict["_timestamp_col"],
            desc=json_dict["_desc"],
            version=json_dict["_version"],
            status=json_dict["_status"],
            feature_descs=json_dict["_feature_desc"],
            refresh_freq=json_dict["_refresh_freq"],
            database=json_dict["_database"],
            schema=json_dict["_schema"],
            warehouse=json_dict["_warehouse"],
            refresh_mode=json_dict["_refresh_mode"],
            refresh_mode_reason=json_dict["_refresh_mode_reason"],
            initialize=json_dict["_initialize"],
            owner=json_dict["_owner"],
            infer_schema_df=session.sql(json_dict.get("_infer_schema_query", None)),
            session=session,
            online_config=OnlineConfig.from_json(json_dict["_online_config"])
            if json_dict.get("_online_config")
            else None,
        )

    def _get_compact_repr(self) -> _CompactRepresentation:
        return _CompactRepresentation(
            db=self.database.identifier(),  # type: ignore[union-attr]
            sch=self.schema.identifier(),  # type: ignore[union-attr]
            name=self.name.identifier(),
            version=self.version,  # type: ignore[arg-type]
        )

    @staticmethod
    def _get_physical_name(fv_name: SqlIdentifier, fv_version: FeatureViewVersion) -> SqlIdentifier:
        return SqlIdentifier(
            concat_names(
                [
                    str(fv_name),
                    _FEATURE_VIEW_NAME_DELIMITER,
                    str(fv_version),
                ]
            )
        )

    @staticmethod
    def _load_from_compact_repr(session: Session, serialized_repr: str) -> Union[FeatureView, FeatureViewSlice]:
        compact_repr = _CompactRepresentation.from_json(serialized_repr)

        fs = feature_store.FeatureStore(
            session, compact_repr.db, compact_repr.sch, default_warehouse=session.get_current_warehouse()
        )
        fv = fs.get_feature_view(compact_repr.name, compact_repr.version)

        if compact_repr.feature_indices is not None:
            feature_names = [fv.feature_names[i] for i in compact_repr.feature_indices]
            return fv.slice(feature_names)  # type: ignore[no-any-return]
        return fv  # type: ignore[no-any-return]

    @staticmethod
    def _load_from_lineage_node(session: Session, name: str, version: str) -> FeatureView:
        db_name, feature_store_name, feature_view_name = identifier.parse_schema_level_object_identifier(name)

        session_warehouse = session.get_current_warehouse()

        if not session_warehouse:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError("No active warehouse selected in the current session"),
            )

        fs = feature_store.FeatureStore(session, db_name, feature_store_name, default_warehouse=session_warehouse)
        return fs.get_feature_view(feature_view_name, version)  # type: ignore[no-any-return]

    @staticmethod
    def _construct_feature_view(
        name: str,
        entities: list[Entity],
        feature_df: DataFrame,
        timestamp_col: Optional[str],
        desc: str,
        version: str,
        status: FeatureViewStatus,
        feature_descs: dict[str, str],
        refresh_freq: Optional[str],
        database: str,
        schema: str,
        warehouse: Optional[str],
        refresh_mode: Optional[str],
        refresh_mode_reason: Optional[str],
        initialize: str,
        owner: Optional[str],
        infer_schema_df: Optional[DataFrame],
        session: Session,
        cluster_by: Optional[list[str]] = None,
        online_config: Optional[OnlineConfig] = None,
    ) -> FeatureView:
        fv = FeatureView(
            name=name,
            entities=entities,
            feature_df=feature_df,
            timestamp_col=timestamp_col,
            desc=desc,
            _infer_schema_df=infer_schema_df,
            cluster_by=cluster_by,
            online_config=online_config,
        )
        fv._version = FeatureViewVersion(version) if version is not None else None
        fv._status = status
        fv._refresh_freq = refresh_freq
        fv._database = SqlIdentifier(database) if database is not None else None
        fv._schema = SqlIdentifier(schema) if schema is not None else None
        fv._warehouse = SqlIdentifier(warehouse) if warehouse is not None else None
        fv._refresh_mode = refresh_mode
        fv._refresh_mode_reason = refresh_mode_reason
        fv._initialize = initialize
        fv._owner = owner
        fv.attach_feature_desc(feature_descs)

        lineage_node.LineageNode.__init__(
            fv, session=session, name=f"{fv.database}.{fv._schema}.{name}", domain="feature_view", version=version
        )
        return fv

    #
    def _get_default_cluster_by(self) -> list[SqlIdentifier]:
        """
        Get default columns to cluster the feature view by.
        Default cluster_by columns are join keys from entities and timestamp_col if it exists

        Returns:
            List of SqlIdentifiers representing the default columns to cluster the feature view by.
        """
        # We don't focus on the order of entities here, as users can define a custom 'cluster_by'
        # if a specific order is required.
        default_cluster_by_cols = [key for entity in self.entities if entity.join_keys for key in entity.join_keys]

        if self.timestamp_col:
            default_cluster_by_cols.append(self.timestamp_col)

        return default_cluster_by_cols

    @staticmethod
    def _get_online_table_name(
        feature_view_name: Union[SqlIdentifier, str], version: Optional[Union[FeatureViewVersion, str]] = None
    ) -> SqlIdentifier:
        """Get the online feature table name without qualification.

        Args:
            feature_view_name: Offline feature view name.
            version: Feature view version. If not provided, feature_view_name must be a SqlIdentifier.

        Returns:
            The online table name SqlIdentifier
        """
        if version is None:
            assert isinstance(feature_view_name, SqlIdentifier), "Single argument must be SqlIdentifier"
            online_name = f"{feature_view_name.resolved()}{_ONLINE_TABLE_SUFFIX}"
            return SqlIdentifier(online_name, case_sensitive=True)
        else:
            fv_name = (
                feature_view_name
                if isinstance(feature_view_name, SqlIdentifier)
                else SqlIdentifier(feature_view_name, case_sensitive=True)
            )
            fv_version = version if isinstance(version, FeatureViewVersion) else FeatureViewVersion(version)
            physical_name = FeatureView._get_physical_name(fv_name, fv_version).resolved()
            online_name = f"{physical_name}{_ONLINE_TABLE_SUFFIX}"
            return SqlIdentifier(online_name, case_sensitive=True)


lineage_node.DOMAIN_LINEAGE_REGISTRY["feature_view"] = FeatureView
