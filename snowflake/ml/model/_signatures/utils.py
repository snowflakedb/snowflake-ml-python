import warnings
from typing import Any, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from snowflake.ml.model._signatures import core


def convert_list_to_ndarray(data: List[Any]) -> npt.NDArray[Any]:
    """Create a numpy array from list or nested list. Avoid ragged list and unaligned types.

    Args:
        data: List or nested list.

    Raises:
        ValueError: Raised when ragged nested list or list containing non-basic type confronted.
        ValueError: Raised when ragged nested list or list containing non-basic type confronted.

    Returns:
        The converted numpy array.
    """
    warnings.filterwarnings("error", category=np.VisibleDeprecationWarning)
    try:
        arr = np.array(data)
    except np.VisibleDeprecationWarning:
        # In recent version of numpy, this warning should be raised when bad list provided.
        raise ValueError(
            f"Unable to construct signature: Ragged nested or Unsupported list-like data {data} confronted."
        )
    warnings.filterwarnings("default", category=np.VisibleDeprecationWarning)
    if arr.dtype == object:
        # If not raised, then a array of object would be created.
        raise ValueError(
            f"Unable to construct signature: Ragged nested or Unsupported list-like data {data} confronted."
        )
    return arr


def rename_features(
    features: Sequence[core.BaseFeatureSpec], feature_names: Optional[List[str]] = None
) -> Sequence[core.BaseFeatureSpec]:
    """It renames the feature in features provided optional feature names.

    Args:
        features: A sequence of feature specifications and feature group specifications.
        feature_names: A list of names to assign to features and feature groups. Defaults to None.

    Raises:
        ValueError: Raised when provided feature_names does not match the data shape.

    Returns:
        A sequence of feature specifications and feature group specifications being renamed if names provided.
    """
    if feature_names:
        if len(feature_names) == len(features):
            for ft, ft_name in zip(features, feature_names):
                ft._name = ft_name
        else:
            raise ValueError(
                f"{len(feature_names)} feature names are provided, while there are {len(features)} features."
            )
    return features


def rename_pandas_df(data: pd.DataFrame, features: Sequence[core.BaseFeatureSpec]) -> pd.DataFrame:
    """It renames pandas dataframe that has non-object column index with provided features.

    Args:
        data: A pandas dataframe to be renamed.
        features: A sequence of feature specifications and feature group specifications to rename the dataframe.

    Raises:
        ValueError: Raised when the data does not have the same number of features as signature.

    Returns:
        A pandas dataframe with columns renamed.
    """
    df_cols = data.columns
    if df_cols.dtype in [np.int64, np.uint64, np.float64]:
        if len(features) != len(data.columns):
            raise ValueError(
                "Data does not have the same number of features as signature. "
                + f"Signature requires {len(features)} features, but have {len(data.columns)} in input data."
            )
        data.columns = pd.Index([feature.name for feature in features])
    return data
