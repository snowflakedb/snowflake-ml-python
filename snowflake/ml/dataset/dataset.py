import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from snowflake.snowpark import DataFrame, Session


def _get_val_or_null(val: Any) -> Any:
    return val if val is not None else "null"


def _wrap_embedded_str(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    return s


DATASET_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class FeatureStoreMetadata:
    """
    Feature store metadata.

    Properties:
        spine_query: The input query on source table which will be joined with features.
        connection_params: a config contains feature store metadata.
        features: A list of feature serialized object in the feature store.

    """

    spine_query: str
    connection_params: Dict[str, str]
    features: List[str]

    def to_json(self) -> str:
        state_dict = {
            # TODO(zhe): Additional wrap is needed because ml_.artifact.ad_artifact takes a dict
            # but we retrieve it as an object. Snowpark serialization is inconsistent with
            # our deserialization. A fix is let artifact table stores string and callers
            # handles both serialization and deserialization.
            "spine_query": _wrap_embedded_str(self.spine_query),
            "connection_params": _wrap_embedded_str(json.dumps(self.connection_params)),
            "features": _wrap_embedded_str(json.dumps(self.features)),
        }
        return json.dumps(state_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "FeatureStoreMetadata":
        json_dict = json.loads(json_str)
        return cls(
            spine_query=json_dict["spine_query"],
            connection_params=json.loads(json_dict["connection_params"]),
            features=json.loads(json_dict["features"]),
        )


class Dataset:
    """Metadata of dataset."""

    def __init__(
        self,
        session: Session,
        df: DataFrame,
        generation_timestamp: Optional[float] = None,
        materialized_table: Optional[str] = None,
        snapshot_table: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        label_cols: Optional[List[str]] = None,
        feature_store_metadata: Optional[FeatureStoreMetadata] = None,
        desc: str = "",
    ) -> None:
        """Initialize dataset object.

        Args:
            session: An active snowpark session.
            df: A dataframe object representing the dataset generation.
            generation_timestamp: The timestamp when this dataset is generated. It will use current time if
                not provided.
            materialized_table: The destination table name which data will writes into.
            snapshot_table: A snapshot table name on the materialized table.
            timestamp_col: Timestamp column which was used for point-in-time correct feature lookup.
            label_cols: Name of column(s) in materialized_table that contains labels.
            feature_store_metadata: A feature store metadata object.
            desc: A description about this dataset.
        """
        self.df = df
        self.generation_timestamp = generation_timestamp if generation_timestamp is not None else time.time()
        self.materialized_table = materialized_table
        self.snapshot_table = snapshot_table
        self.timestamp_col = timestamp_col
        self.label_cols = label_cols
        self.feature_store_metadata = feature_store_metadata
        self.desc = desc

        self.id = uuid4().hex.upper()
        self.owner = session.sql("SELECT CURRENT_USER()").collect()[0]["CURRENT_USER()"]
        self.version = DATASET_SCHEMA_VERSION

    @property
    def name(self) -> str:
        """Get name of this dataset. It returns snapshot table name if it exists. Otherwise returns empty string.

        Returns:
            A string name.
        """
        return self.snapshot_table if self.snapshot_table is not None else ""

    def load_features(self) -> Optional[List[str]]:
        if self.feature_store_metadata is not None:
            return self.feature_store_metadata.features
        else:
            return None

    def to_json(self) -> str:
        if len(self.df.queries["queries"]) != 1:
            raise ValueError(
                f"""df dataframe must contain only 1 query.
Got {len(self.df.queries['queries'])}: {self.df.queries['queries']}
"""
            )

        state_dict = {
            "df_query": self.df.queries["queries"][0],
            "id": self.id,
            "generation_timestamp": self.generation_timestamp,
            "owner": self.owner,
            "materialized_table": _get_val_or_null(self.materialized_table),
            "snapshot_table": _get_val_or_null(self.snapshot_table),
            "timestamp_col": _get_val_or_null(self.timestamp_col),
            "label_cols": _get_val_or_null(self.label_cols),
            "feature_store_metadata": self.feature_store_metadata.to_json()
            if self.feature_store_metadata is not None
            else "null",
            "version": self.version,
            "desc": self.desc,
        }
        return json.dumps(state_dict)

    @classmethod
    def from_json(cls, json_str: str, session: Session) -> "Dataset":
        json_dict = json.loads(json_str)
        json_dict["df"] = session.sql(json_dict.pop("df_query"))

        fs_meta_json = json_dict["feature_store_metadata"]
        json_dict["feature_store_metadata"] = (
            FeatureStoreMetadata.from_json(fs_meta_json) if fs_meta_json != "null" else None
        )

        uid = json_dict.pop("id")
        version = json_dict.pop("version")
        owner = json_dict.pop("owner")

        result = cls(session, **json_dict)
        result.id = uid
        result.version = version
        result.owner = owner

        return result

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Dataset) and self.to_json() == other.to_json()
