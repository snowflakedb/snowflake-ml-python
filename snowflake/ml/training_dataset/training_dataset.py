from dataclasses import dataclass
from typing import Dict, List, Optional

from snowflake.snowpark import DataFrame


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


@dataclass(frozen=True)
class TrainingDataset:
    """
    Training dataset object contains the metadata and async job object if training task is still running.

    Properties:
        df: A dataframe object representing the training dataset generation.
        materialized_table: The destination table name which training data will writes into.
        timestamp_col: Name of timestamp column in spine_df that will be used to join time-series features.
            If spine_timestamp_col is not none, the input features also must have timestamp_col.
        label_cols: Name of colum(s) in materialized_table that contains training labels.
        feature_store_metadata: A feature store metadata object.
        desc: A description about this training dataset.
    """

    df: DataFrame
    materialized_table: Optional[str]
    timestamp_col: Optional[str]
    label_cols: Optional[List[str]]
    feature_store_metadata: Optional[FeatureStoreMetadata]
    desc: str
