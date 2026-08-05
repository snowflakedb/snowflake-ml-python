"""Enum surface for the declarative feature store library.

"""

from enum import Enum

from snowflake.ml.feature_store.spec.enums import (
    TYPE_ALIASES,
    FeatureAggregationMethod,
    FeatureViewKind,
    FSBaseType,
    SourceType,
    normalize_type,
)


class OpKind(str, Enum):
    """The kind of plan operation.

    Decl-local — plan operations have no equivalent on the spec /
    imperative side.
    """

    CREATE_ENTITY = "CREATE_ENTITY"
    CREATE_SOURCE = "CREATE_SOURCE"
    CREATE_FV = "CREATE_FV"
    CREATE_FG = "CREATE_FG"
    UPDATE_SOURCE = "UPDATE_SOURCE"
    UPDATE_ENTITY = "UPDATE_ENTITY"
    UPDATE_FV = "UPDATE_FV"
    RECREATE_FV = "RECREATE_FV"
    RECREATE_SOURCE = "RECREATE_SOURCE"
    DROP_ENTITY = "DROP_ENTITY"
    DROP_SOURCE = "DROP_SOURCE"
    DROP_FV = "DROP_FV"
    DROP_FG = "DROP_FG"
    NO_CHANGE = "NO_CHANGE"


__all__ = [
    "FSBaseType",
    "FeatureAggregationMethod",
    "FeatureViewKind",
    "OpKind",
    "SourceType",
    "TYPE_ALIASES",
    "normalize_type",
]
