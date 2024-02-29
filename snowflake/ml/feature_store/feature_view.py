from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml._internal.utils.identifier import concat_names
from snowflake.ml._internal.utils.sql_identifier import (
    SqlIdentifier,
    to_sql_identifiers,
)
from snowflake.ml.feature_store.entity import Entity
from snowflake.snowpark import DataFrame, Session
from snowflake.snowpark.types import (
    DateType,
    StructType,
    TimestampType,
    TimeType,
    _NumericType,
)

_FEATURE_VIEW_NAME_DELIMITER = "$"
_TIMESTAMP_COL_PLACEHOLDER = "FS_TIMESTAMP_COL_PLACEHOLDER_VAL"
_FEATURE_OBJ_TYPE = "FEATURE_OBJ_TYPE"
_FEATURE_VIEW_VERSION_RE = re.compile("^([A-Za-z0-9_]*)$")


class FeatureViewVersion(str):
    def __new__(cls, version: str) -> FeatureViewVersion:
        if not _FEATURE_VIEW_VERSION_RE.match(version):
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(
                    f"`{version}` is not a valid feature view version. Only letter, number and underscore is allowed."
                ),
            )
        return super().__new__(cls, version.upper())

    def __init__(self, version: str) -> None:
        return super().__init__()


class FeatureViewStatus(Enum):
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


class FeatureView:
    """
    A FeatureView instance encapsulates a logical group of features.
    """

    def __init__(
        self,
        name: str,
        entities: List[Entity],
        feature_df: DataFrame,
        timestamp_col: Optional[str] = None,
        refresh_freq: Optional[str] = None,
        desc: str = "",
    ) -> None:
        """
        Create a FeatureView instance.

        Args:
            name: name of the FeatureView. NOTE: FeatureView name will be capitalized.
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
        """

        self._name: SqlIdentifier = SqlIdentifier(name)
        self._entities: List[Entity] = entities
        self._feature_df: DataFrame = feature_df
        self._timestamp_col: Optional[SqlIdentifier] = (
            SqlIdentifier(timestamp_col) if timestamp_col is not None else None
        )
        self._desc: str = desc
        self._query: str = self._get_query()
        self._version: Optional[FeatureViewVersion] = None
        self._status: FeatureViewStatus = FeatureViewStatus.DRAFT
        self._feature_desc: OrderedDict[SqlIdentifier, str] = OrderedDict((f, "") for f in self._get_feature_names())
        self._refresh_freq: Optional[str] = refresh_freq
        self._database: Optional[SqlIdentifier] = None
        self._schema: Optional[SqlIdentifier] = None
        self._warehouse: Optional[SqlIdentifier] = None
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
        """

        res = []
        for name in names:
            name = SqlIdentifier(name)
            if name not in self.feature_names:
                raise ValueError(f"Feature name {name} not found in FeatureView {self.name}.")
            res.append(name)
        return FeatureViewSlice(self, res)

    def physical_name(self) -> SqlIdentifier:
        """Returns the physical name for this feature in Snowflake.

        Returns:
            Physical name string.

        Raises:
            RuntimeError: if the FeatureView is not materialized.
        """
        if self.status == FeatureViewStatus.DRAFT or self.version is None:
            raise RuntimeError(f"FeatureView {self.name} has not been materialized.")
        return FeatureView._get_physical_name(self.name, self.version)

    def fully_qualified_name(self) -> str:
        """Returns the fully qualified name (<database_name>.<schema_name>.<feature_view_name>) for the
            FeatureView in Snowflake.

        Returns:
            fully qualified name string.
        """
        return f"{self._database}.{self._schema}.{self.physical_name()}"

    def attach_feature_desc(self, descs: Dict[str, str]) -> FeatureView:
        """
        Associate feature level descriptions to the FeatureView.

        Args:
            descs: Dictionary contains feature name and corresponding descriptions.

        Returns:
            FeatureView with feature level desc attached.

        Raises:
            ValueError: if feature name is not found in the FeatureView.
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
        if self.status == FeatureViewStatus.DRAFT or self.status == FeatureViewStatus.STATIC:
            raise RuntimeError(
                f"Feature view {self.name}/{self.version} must be registered and non-static to update refresh_freq."
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
        if self.status == FeatureViewStatus.DRAFT or self.status == FeatureViewStatus.STATIC:
            raise RuntimeError(
                f"Feature view {self.name}/{self.version} must be registered and non-static to update warehouse."
            )
        self._warehouse = SqlIdentifier(new_value)

    @property
    def output_schema(self) -> StructType:
        return self._feature_df.schema

    @property
    def refresh_mode(self) -> Optional[str]:
        return self._refresh_mode

    @property
    def refresh_mode_reason(self) -> Optional[str]:
        return self._refresh_mode_reason

    @property
    def owner(self) -> Optional[str]:
        return self._owner

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

        unescaped_df_cols = to_sql_identifiers(self._feature_df.columns)
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
            if ts_col not in to_sql_identifiers(self._feature_df.columns):
                raise ValueError(f"timestamp_col {ts_col} is not found in input dataframe.")

            col_type = self._feature_df.schema[ts_col].datatype
            if not isinstance(col_type, (DateType, TimeType, TimestampType, _NumericType)):
                raise ValueError(f"Invalid data type for timestamp_col {ts_col}: {col_type}.")

    def _get_feature_names(self) -> List[SqlIdentifier]:
        join_keys = [k for e in self._entities for k in e.join_keys]
        ts_col = [self._timestamp_col] if self._timestamp_col is not None else []
        feature_names = to_sql_identifiers(self._feature_df.columns, case_sensitive=True)
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

        return fv_dict

    def to_df(self, session: Session) -> DataFrame:
        values = list(self._to_dict().values())
        schema = [x.lstrip("_") for x in list(self._to_dict().keys())]
        values.append(str(self.physical_name()))
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
            e = Entity(e_json["name"], e_json["join_keys"], e_json["desc"])
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
        database: Optional[str],
        schema: Optional[str],
        warehouse: Optional[str],
        refresh_mode: Optional[str],
        refresh_mode_reason: Optional[str],
        owner: Optional[str],
    ) -> FeatureView:
        fv = FeatureView(
            name=name,
            entities=entities,
            feature_df=feature_df,
            timestamp_col=timestamp_col,
            desc=desc,
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
        return fv
