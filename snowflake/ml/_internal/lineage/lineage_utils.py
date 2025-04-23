import copy
import functools
from typing import Any, Callable, Optional, get_args

from snowflake import snowpark
from snowflake.ml.data import data_source

_DATA_SOURCES_ATTR = "_data_sources"


def _wrap_func(
    fn: Callable[..., snowpark.DataFrame], data_sources: list[data_source.DataSource]
) -> Callable[..., snowpark.DataFrame]:
    """Wrap a DataFrame transform function to propagate data_sources to derived DataFrames."""

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> snowpark.DataFrame:
        df = fn(*args, **kwargs)
        patch_dataframe(df, data_sources=data_sources, inplace=True)
        return df

    return wrapped


def _wrap_class_func(fn: Callable[..., snowpark.DataFrame]) -> Callable[..., snowpark.DataFrame]:
    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> snowpark.DataFrame:
        df = fn(*args, **kwargs)
        data_sources = get_data_sources(*args, *kwargs.values())
        if data_sources:
            patch_dataframe(df, data_sources, inplace=True)
        return df

    return wrapped


def get_data_sources(*args: Any) -> Optional[list[data_source.DataSource]]:
    """Helper method for extracting data sources attribute from DataFrames in an argument list"""
    result: Optional[list[data_source.DataSource]] = None
    for arg in args:
        srcs = getattr(arg, _DATA_SOURCES_ATTR, None)
        if isinstance(srcs, list) and all(isinstance(s, get_args(data_source.DataSource)) for s in srcs):
            if result is None:
                result = []
            result += srcs
    return result


def set_data_sources(obj: Any, data_sources: Optional[list[data_source.DataSource]]) -> None:
    """Helper method for attaching data sources to an object"""
    if data_sources:
        assert all(isinstance(ds, get_args(data_source.DataSource)) for ds in data_sources)
    setattr(obj, _DATA_SOURCES_ATTR, data_sources)


def patch_dataframe(
    df: snowpark.DataFrame, data_sources: list[data_source.DataSource], inplace: bool = False
) -> snowpark.DataFrame:
    """
    Monkey patch a DataFrame to add attach the provided data_sources as an attribute of the DataFrame.
    Also patches the DataFrame's transformation functions to propagate the new data sources attribute to
    derived DataFrames.

    Args:
        df: DataFrame to be patched
        data_sources: List of data sources for the DataFrame
        inplace: If True, patches to DataFrame in-place. If False, creates a shallow copy of the DataFrame.

    Returns:
        Patched DataFrame
    """
    # Instance-level monkey-patches
    funcs = [
        "_with_plan",
        "_lateral",
        "group_by",
        "group_by_grouping_sets",
        "cube",
        "pivot",
        "rollup",
        "cache_result",
        "_to_df",  # RelationalGroupedDataFrame
    ]
    if not inplace:
        df = copy.copy(df)
    set_data_sources(df, data_sources)
    for func in funcs:
        fn = getattr(df, func, None)
        if fn is not None:
            setattr(df, func, _wrap_func(fn, data_sources=data_sources))
    return df


# Class-level monkey-patches
for klass, func_list in {
    snowpark.DataFrame: [
        "__copy__",
    ],
    snowpark.RelationalGroupedDataFrame: [],
}.items():
    assert isinstance(func_list, list)  # mypy
    for func in func_list:
        fn = getattr(klass, func)
        setattr(klass, func, _wrap_class_func(fn))
