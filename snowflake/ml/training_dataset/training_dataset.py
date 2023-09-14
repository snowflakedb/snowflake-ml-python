import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from snowflake.snowpark import DataFrame, Session


def _get_val_or_null(val: Any) -> Any:
    return val if val is not None else "null"


def _wrap_embedded_str(s: str) -> str:
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    return s


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


@dataclass(frozen=True)
class TrainingDataset:
    """
    Training dataset object contains the metadata and async job object if training task is still running.

    Properties:
        df: A dataframe object representing the training dataset generation.
        materialized_table: The destination table name which training data will writes into.
        snapshot_table: A snapshot table name on the materialized table.
        timestamp_col: Name of timestamp column in spine_df that will be used to join time-series features.
            If spine_timestamp_col is not none, the input features also must have timestamp_col.
        label_cols: Name of column(s) in materialized_table that contains training labels.
        feature_store_metadata: A feature store metadata object.
        desc: A description about this training dataset.
    """

    df: DataFrame
    materialized_table: Optional[str]
    snapshot_table: Optional[str]
    timestamp_col: Optional[str]
    label_cols: Optional[List[str]]
    feature_store_metadata: Optional[FeatureStoreMetadata]
    desc: str

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
            "materialized_table": _get_val_or_null(self.materialized_table),
            "snapshot_table": _get_val_or_null(self.snapshot_table),
            "timestamp_col": _get_val_or_null(self.timestamp_col),
            "label_cols": _get_val_or_null(self.label_cols),
            "feature_store_metadata": self.feature_store_metadata.to_json()
            if self.feature_store_metadata is not None
            else "null",
            "desc": self.desc,
        }
        return json.dumps(state_dict)

    @classmethod
    def from_json(cls, json_str: str, session: Session) -> "TrainingDataset":
        json_dict = json.loads(json_str)
        json_dict["df"] = session.sql(json_dict["df_query"])
        json_dict.pop("df_query")

        fs_meta_json = json_dict["feature_store_metadata"]
        json_dict["feature_store_metadata"] = FeatureStoreMetadata.from_json(fs_meta_json)
        return cls(**json_dict)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TrainingDataset) and self.to_json() == other.to_json()

    def id(self) -> str:
        """Return a unique identifier of this training dataset.

        Raises:
            ValueError: when snapshot_table is None.

        Returns:
            A unique identifier string.
        """
        if self.snapshot_table is None:
            raise ValueError("snapshot_table is required to generate id.")
        return self.snapshot_table
