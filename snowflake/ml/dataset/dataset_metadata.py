import dataclasses
import json
import typing
from typing import Any, Optional, Union

_PROPERTY_TYPE_KEY = "$proptype$"
DATASET_SCHEMA_VERSION = "1"


@dataclasses.dataclass(frozen=True)
class FeatureStoreMetadata:
    """
    Feature store metadata.

    Properties:
        spine_query: The input query on source table which will be joined with features.
        serialized_feature_views: A list of serialized feature objects in the feature store.
        compact_feature_views: A compact representation of a FeatureView or FeatureViewSlice.
        spine_timestamp_col: Timestamp column which was used for point-in-time correct feature lookup.
    """

    spine_query: str
    serialized_feature_views: Optional[list[str]] = None
    compact_feature_views: Optional[list[str]] = None
    spine_timestamp_col: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, input_json: Union[dict[str, Any], str, bytes]) -> "FeatureStoreMetadata":
        if isinstance(input_json, dict):
            return cls(**input_json)
        return cls(**json.loads(input_json))


DatasetPropertiesType = Union[
    FeatureStoreMetadata,
]

# Union[T] gets automatically squashed to T, so default to [T] if get_args() returns empty
_DatasetPropTypes = typing.get_args(DatasetPropertiesType) or [DatasetPropertiesType]
_DatasetPropTypeDict = {t.__name__: t for t in _DatasetPropTypes}


@dataclasses.dataclass(frozen=True)
class DatasetMetadata:
    """
    Dataset metadata.

    Properties:
        source_query: The query string used to produce the Dataset.
        owner: The owner of the Dataset.
        generation_timestamp: The timestamp when this dataset was generated.
        exclude_cols: Name of column(s) in dataset to be excluded during training/testing.
            These are typically columns for human inspection such as timestamp or other meta-information.
            Columns included in `label_cols` do not need to be included here.
        label_cols: Name of column(s) in dataset that contains labels.
        properties: Additional metadata properties.
    """

    source_query: str
    owner: str
    exclude_cols: Optional[list[str]] = None
    label_cols: Optional[list[str]] = None
    properties: Optional[DatasetPropertiesType] = None
    schema_version: str = dataclasses.field(default=DATASET_SCHEMA_VERSION, init=False)

    def to_json(self) -> str:
        state_dict = dataclasses.asdict(self)
        if self.properties:
            prop_type = type(self.properties).__name__
            if prop_type not in _DatasetPropTypeDict:
                raise ValueError(
                    f"Unsupported `properties` type={prop_type} (supported={','.join(_DatasetPropTypeDict.keys())})"
                )
            state_dict[_PROPERTY_TYPE_KEY] = prop_type
        return json.dumps(state_dict)

    @classmethod
    def from_json(cls, input_json: Union[dict[str, Any], str, bytes]) -> "DatasetMetadata":
        if not input_json:
            raise ValueError("json_str was empty or None")
        try:
            state_dict: dict[str, Any] = (
                input_json if isinstance(input_json, dict) else json.loads(input_json, strict=False)
            )

            # TODO: Validate schema version
            _ = state_dict.pop("schema_version", DATASET_SCHEMA_VERSION)

            prop_type = state_dict.pop(_PROPERTY_TYPE_KEY, None)
            prop_values = state_dict.get("properties", {})
            if prop_type:
                prop_cls = _DatasetPropTypeDict.get(prop_type, None)
                if prop_cls is None:
                    raise TypeError(
                        f"Unsupported `properties` type={prop_type} (supported={','.join(_DatasetPropTypeDict.keys())})"
                    )
                state_dict["properties"] = prop_cls(**prop_values)
            elif prop_values:
                raise TypeError(f"`properties` provided but missing `{_PROPERTY_TYPE_KEY}`")
            return cls(**state_dict)
        except TypeError as e:
            raise ValueError("Invalid input schema") from e
