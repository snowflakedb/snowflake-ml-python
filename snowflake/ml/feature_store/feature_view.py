from __future__ import annotations

import json
import re
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

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
from snowflake.snowpark.types import (
    DateType,
    StructType,
    TimestampType,
    TimeType,
    _NumericType,
)

_FEATURE_VIEW_NAME_DELIMITER = "$"
_LEGACY_TIMESTAMP_COL_PLACEHOLDER_VALS = ["FS_TIMESTAMP_COL_PLACEHOLDER_VAL", "NULL"]
_TIMESTAMP_COL_PLACEHOLDER = "NULL"
_FEATURE_OBJ_TYPE = "FEATURE_OBJ_TYPE"
# Feature view version rule is aligned with dataset version rule in SQL.
_FEATURE_VIEW_VERSION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.\-]*$")
_FEATURE_VIEW_VERSION_MAX_LENGTH = 128

_RESULT_SCAN_QUERY_PATTERN = re.compile(
    r".*FROM\s*TABLE\s*\(\s*RESULT_SCAN\s*\(.*",
    flags=re.DOTALL | re.IGNORECASE | re.X,
)


@dataclass(frozen=True)
class _FeatureViewMetadata:
    """Represent metadata tracked on top of FV backend object"""

    entities: List[str]
    timestamp_col: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> _FeatureViewMetadata:
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
    names: List[SqlIdentifier]

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


class FeatureView(lineage_node.LineageNode):
    """
    A FeatureView instance encapsulates a logical group of features.
    """

    def __init__(
        self,
        name: str,
        entities: List[Entity],
        feature_df: DataFrame,
        *,
        timestamp_col: Optional[str] = None,
        refresh_freq: Optional[str] = None,
        desc: str = "",
        warehouse: Optional[str] = None,
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
            _kwargs: reserved kwargs for system generated args. NOTE: DO NOT USE.

        Example::

            >>> fs = FeatureStore(...)
            >>> # draft_fv is a local object that hasn't materiaized to Snowflake backend yet.
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

        """

        self._name: SqlIdentifier = SqlIdentifier(name)
        self._entities: List[Entity] = entities
        self._feature_df: DataFrame = feature_df
        self._timestamp_col: Optional[SqlIdentifier] = (
            SqlIdentifier(timestamp_col) if timestamp_col is not None else None
        )
        self._desc: str = desc
        self._infer_schema_df: DataFrame = _kwargs.get("_infer_schema_df", self._feature_df)
        self._query: str = self._get_query()
        self._version: Optional[FeatureViewVersion] = None
        self._status: FeatureViewStatus = FeatureViewStatus.DRAFT
        self._feature_desc: OrderedDict[SqlIdentifier, str] = OrderedDict((f, "") for f in self._get_feature_names())
        self._refresh_freq: Optional[str] = refresh_freq
        self._database: Optional[SqlIdentifier] = None
        self._schema: Optional[SqlIdentifier] = None
        self._warehouse: Optional[SqlIdentifier] = SqlIdentifier(warehouse) if warehouse is not None else None
        self._refresh_mode: Optional[str] = None
        self._refresh_mode_reason: Optional[str] = None
        self._owner: Optional[str] = None
        self._validate()

    def slice(self, names: List[str]) -> FeatureViewSlice:
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

    def attach_feature_desc(self, descs: Dict[str, str]) -> FeatureView:
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
    def entities(self) -> List[Entity]:
        return self._entities

    @property
    def feature_df(self) -> DataFrame:
        return self._feature_df

    @property
    def timestamp_col(self) -> Optional[SqlIdentifier]:
        return self._timestamp_col

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
    def feature_names(self) -> List[SqlIdentifier]:
        return list(self._feature_desc.keys())

    @property
    def feature_descs(self) -> Dict[SqlIdentifier, str]:
        return self._feature_desc

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

        unescaped_df_cols = to_sql_identifiers(self._infer_schema_df.columns)
        for e in self._entities:
            for k in e.join_keys:
                if k not in unescaped_df_cols:
                    raise ValueError(
                        f"join_key {k} in Entity {e.name} is not found in input dataframe: {unescaped_df_cols}"
                    )

        if self._timestamp_col is not None:
            ts_col = self._timestamp_col
            if ts_col == SqlIdentifier(_TIMESTAMP_COL_PLACEHOLDER):
                raise ValueError(f"Invalid timestamp_col name, cannot be {_TIMESTAMP_COL_PLACEHOLDER}.")
            if ts_col not in to_sql_identifiers(self._infer_schema_df.columns):
                raise ValueError(f"timestamp_col {ts_col} is not found in input dataframe.")

            col_type = self._infer_schema_df.schema[ts_col].datatype
            if not isinstance(col_type, (DateType, TimeType, TimestampType, _NumericType)):
                raise ValueError(f"Invalid data type for timestamp_col {ts_col}: {col_type}.")

        if re.match(_RESULT_SCAN_QUERY_PATTERN, self._query) is not None:
            raise ValueError(f"feature_df should not be reading from RESULT_SCAN. Invalid query: {self._query}")

    def _get_feature_names(self) -> List[SqlIdentifier]:
        join_keys = [k for e in self._entities for k in e.join_keys]
        ts_col = [self._timestamp_col] if self._timestamp_col is not None else []
        feature_names = to_sql_identifiers(self._infer_schema_df.columns, case_sensitive=False)
        return [c for c in feature_names if c not in join_keys + ts_col]

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

    def _to_dict(self) -> Dict[str, str]:
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

        feature_desc_dict = {}
        for k, v in self._feature_desc.items():
            feature_desc_dict[k.identifier()] = v
        fv_dict["_feature_desc"] = feature_desc_dict

        lineage_node_keys = [key for key in fv_dict if key.startswith("_node") or key == "_session"]

        for key in lineage_node_keys:
            fv_dict.pop(key)

        return fv_dict

    def to_df(self, session: Session) -> DataFrame:
        values = list(self._to_dict().values())
        schema = [x.lstrip("_") for x in list(self._to_dict().keys())]
        values.append(str(FeatureView._get_physical_name(self._name, self._version)))  # type: ignore[arg-type]
        schema.append("physical_name")
        return session.create_dataframe([values], schema=schema)

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
            owner=json_dict["_owner"],
            infer_schema_df=session.sql(json_dict.get("_infer_schema_query", None)),
            session=session,
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
    def _load_from_lineage_node(session: Session, name: str, version: str) -> FeatureView:
        db_name, feature_store_name, feature_view_name, _ = identifier.parse_schema_level_object_identifier(name)

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
        entities: List[Entity],
        feature_df: DataFrame,
        timestamp_col: Optional[str],
        desc: str,
        version: str,
        status: FeatureViewStatus,
        feature_descs: Dict[str, str],
        refresh_freq: Optional[str],
        database: str,
        schema: str,
        warehouse: Optional[str],
        refresh_mode: Optional[str],
        refresh_mode_reason: Optional[str],
        owner: Optional[str],
        infer_schema_df: Optional[DataFrame],
        session: Session,
    ) -> FeatureView:
        fv = FeatureView(
            name=name,
            entities=entities,
            feature_df=feature_df,
            timestamp_col=timestamp_col,
            desc=desc,
            _infer_schema_df=infer_schema_df,
        )
        fv._version = FeatureViewVersion(version) if version is not None else None
        fv._status = status
        fv._refresh_freq = refresh_freq
        fv._database = SqlIdentifier(database) if database is not None else None
        fv._schema = SqlIdentifier(schema) if schema is not None else None
        fv._warehouse = SqlIdentifier(warehouse) if warehouse is not None else None
        fv._refresh_mode = refresh_mode
        fv._refresh_mode_reason = refresh_mode_reason
        fv._owner = owner
        fv.attach_feature_desc(feature_descs)

        lineage_node.LineageNode.__init__(
            fv, session=session, name=f"{fv.database}.{fv._schema}.{name}", domain="feature_view", version=version
        )
        return fv


lineage_node.DOMAIN_LINEAGE_REGISTRY["feature_view"] = FeatureView
