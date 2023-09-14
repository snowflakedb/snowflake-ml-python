#!/usr/bin/env python3
# TODO(hayu): [SNOW-750748] Rename/split framework/utils to constants.py, extract_args.py, etc.
import inspect
import warnings
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Optional, Union

import numpy as np
import sklearn
from packaging import version

from snowflake import snowpark
from snowflake.ml._internal.exceptions import (
    error_codes,
    error_messages,
    exceptions,
    exceptions as snowml_exceptions,
    modeling_error_messages,
)
from snowflake.snowpark import exceptions as snowpark_exceptions, functions as F
from snowflake.snowpark._internal import utils

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

# numeric states to the corresponding Snowpark functions
NUMERIC_STATE_TO_FUNC_DICT = {
    "count": F.count,
    "max": F.max,
    "mean": F.avg,
    "median": F.median,
    "min": F.min,
    "stddev": F.stddev,
    "stddev_pop": F.stddev_pop,
    "variance": F.variance,
    "var_pop": F.var_pop,
}

# basic states to the corresponding Snowpark functions
BASIC_STATE_TO_FUNC_DICT = {
    "mode": F.mode,
}

# states, as the combination of numeric and basic states,
# to the corresponding Snowpark functions,
# used to convert state strings to utility and SQL functions
STATE_TO_FUNC_DICT = {**NUMERIC_STATE_TO_FUNC_DICT, **BASIC_STATE_TO_FUNC_DICT}


class NumericStatistics(str, Enum):
    COUNT = "count"
    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    STDDEV = "stddev"
    STDDEV_POP = "stddev_pop"
    VARIANCE = "variance"
    VAR_POP = "var_pop"


class BasicStatistics(str, Enum):
    MODE = "mode"


def get_default_args(func: Callable[..., None]) -> Dict[str, Any]:
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def generate_value_with_prefix(prefix: str) -> str:
    return f"{prefix}{utils.generate_random_alphanumeric()}"


def get_filtered_valid_sklearn_args(
    args: Dict[str, Any],
    default_sklearn_args: Dict[str, Any],
    sklearn_initial_keywords: Optional[Union[str, Iterable[str]]] = None,
    sklearn_unused_keywords: Optional[Union[str, Iterable[str]]] = None,
    snowml_only_keywords: Optional[Union[str, Iterable[str]]] = None,
    sklearn_added_keyword_to_version_dict: Optional[Dict[str, str]] = None,
    sklearn_added_kwarg_value_to_version_dict: Optional[Dict[str, Dict[str, str]]] = None,
    sklearn_deprecated_keyword_to_version_dict: Optional[Dict[str, str]] = None,
    sklearn_removed_keyword_to_version_dict: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Get valid sklearn keyword arguments with non-default values.

    Validate if the scikit-learn package version meets the requirements.
    This method validates both keywords and values (e.g. OneHotEncoder
    handle_unknown='infrequent_if_exist').

    Args:
        args: Keyword arguments to validate.
        default_sklearn_args: Default sklearn keyword arguments.
        sklearn_initial_keywords: Initial keywords in sklearn.
        sklearn_unused_keywords: Sklearn keywords that are unused in snowml.
        snowml_only_keywords: snowml only keywords not present in sklearn.
        sklearn_added_keyword_to_version_dict: Added keywords mapped to the sklearn versions in which they were added.
        sklearn_added_kwarg_value_to_version_dict: Added keyword argument values mapped to the sklearn versions
            in which they were added.
        sklearn_deprecated_keyword_to_version_dict: Deprecated keywords mapped to the sklearn versions in which
            they were deprecated.
        sklearn_removed_keyword_to_version_dict: Removed keywords mapped to the sklearn versions in which they
            were removed.

    Returns:
        Sklearn keyword arguments.

    Raises:
        SnowflakeMLException: If the input args contains an invalid key.
        SnowflakeMLException: If the scikit-learn package version does not meet the requirements
            for the keyword arguments.
    """
    # get args to be passed to sklearn
    sklearn_args = {}
    for key, val in args.items():
        # initial sklearn keyword
        if sklearn_initial_keywords and (key in sklearn_initial_keywords):
            sklearn_args[key] = val
        # deprecated sklearn keyword
        elif sklearn_deprecated_keyword_to_version_dict and (key in sklearn_deprecated_keyword_to_version_dict):
            sklearn_args[key] = val
        # removed sklearn keyword
        elif sklearn_removed_keyword_to_version_dict and (key in sklearn_removed_keyword_to_version_dict):
            sklearn_args[key] = val
        # added sklearn keyword
        elif sklearn_added_keyword_to_version_dict and (key in sklearn_added_keyword_to_version_dict):
            # If the keyword is not in sklearn, then there is likely a
            # version mismatch. The arg is still added to `sklearn_args`
            # for sklearn version validation.
            if key not in default_sklearn_args:
                sklearn_args[key] = val
            # pass non-default values to sklearn to avoid raising an
            # unexpected argument error even if the argument is not
            # passed to the snowml transformer
            elif val != default_sklearn_args[key]:
                # check if both values are numpy.nan since numpy.nan != numpy.nan
                if isinstance(val, float) and (np.isnan(val) and np.isnan(default_sklearn_args[key])):
                    continue
                sklearn_args[key] = val
        # unused sklearn keyword in snowml
        elif sklearn_unused_keywords and (key in sklearn_unused_keywords):
            continue
        # snowml only keyword
        elif snowml_only_keywords and (key in snowml_only_keywords):
            continue
        # unknown keyword
        else:
            raise snowml_exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=ValueError(error_messages.UNEXPECTED_KEYWORD.format(key)),
            )

    # validate sklearn version
    sklearn_version = sklearn.__version__
    for key, val in sklearn_args.items():
        # added sklearn keyword
        if (
            sklearn_added_keyword_to_version_dict
            and (key in sklearn_added_keyword_to_version_dict)
            and (version.parse(sklearn_version) < version.parse(sklearn_added_keyword_to_version_dict[key]))
        ):
            required_version = sklearn_added_keyword_to_version_dict[key]
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.DEPENDENCY_VERSION_ERROR,
                original_exception=ImportError(
                    modeling_error_messages.INCOMPATIBLE_NEW_SKLEARN_PARAM.format(
                        key, required_version, sklearn_version
                    )
                ),
            )

        # added keyword argument value
        if (
            sklearn_added_kwarg_value_to_version_dict
            and (key in sklearn_added_kwarg_value_to_version_dict)
            and (isinstance(val, str) and val in sklearn_added_kwarg_value_to_version_dict[key])
            and (version.parse(sklearn_version) < version.parse(sklearn_added_kwarg_value_to_version_dict[key][val]))
        ):
            required_version = sklearn_added_kwarg_value_to_version_dict[key][val]
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.DEPENDENCY_VERSION_ERROR,
                original_exception=ImportError(
                    modeling_error_messages.INCOMPATIBLE_NEW_SKLEARN_PARAM.format(
                        f"{key}={val}", required_version, sklearn_version
                    )
                ),
            )

        # deprecated sklearn keyword
        if (
            sklearn_deprecated_keyword_to_version_dict
            and (key in sklearn_deprecated_keyword_to_version_dict)
            and (version.parse(sklearn_version) >= version.parse(sklearn_deprecated_keyword_to_version_dict[key]))
        ):
            deprecated_version = sklearn_deprecated_keyword_to_version_dict[key]
            msg = f"Incompatible scikit-learn version: '{key}' deprecated since scikit-learn={deprecated_version}.."
            warnings.warn(msg, DeprecationWarning)

        # removed sklearn keyword
        if (
            sklearn_removed_keyword_to_version_dict
            and (key in sklearn_removed_keyword_to_version_dict)
            and (version.parse(sklearn_version) >= version.parse(sklearn_removed_keyword_to_version_dict[key]))
        ):
            removed_version = sklearn_removed_keyword_to_version_dict[key]
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.DEPENDENCY_VERSION_ERROR,
                original_exception=ImportError(
                    modeling_error_messages.REMOVED_SKLEARN_PARAM.format(key, removed_version, sklearn_version)
                ),
            )

    return sklearn_args


def str_to_bool(value: str) -> Union[bool, None]:
    if value is None:
        return None
    elif value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    return None


def _handle_str_bool_type() -> np.vectorize:
    return np.vectorize(str_to_bool)


def to_native_format(obj: Any) -> Any:
    if "XGB" in obj.__class__.__name__:
        return obj.to_xgboost()
    elif "LGBM" in obj.__class__.__name__:
        return obj.to_lightgbm()
    return obj.to_sklearn()


def table_exists(session: snowpark.Session, table_name: str, statement_params: Dict[str, Any]) -> bool:
    try:
        session.table(table_name).limit(0).collect(statement_params=statement_params)
        return True
    except snowpark_exceptions.SnowparkSQLException:
        return False
