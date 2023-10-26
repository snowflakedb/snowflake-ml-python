from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas._typing import ArrayLike

from snowflake.snowpark import DataFrame, Session

_EqualityFunc = Callable[[Any, Any], bool]

DATA = [
    ["1", "c", "g1ehQlL80t", -1.0, 0.0],
    ["2", "a", "zOyDvcyZ2s", 8.3, 124.6],
    ["3", "b", "zOyDvcyZ2s", 2.0, 2253463.5342],
    ["4", "A", "TuDOcLxToB", 3.5, -1350.6407],
    ["5", "d", "g1ehQlL80t", 2.5, -1.0],
    ["6", "b", "g1ehQlL80t", 4.0, 457946.23462],
    ["7", "b", "g1ehQlL80t", -10.0, -12.564],
]

DATA_NONE_NAN: List[List[Any]] = [
    ["1", "a", "g1ehQlL80t", -1.0, 0.0],
    ["2", "a", "zOyDvcyZ2s", None, np.nan],
    ["3", "b", None, 2.0, 2253463.5342],
    ["4", "A", "TuDOcLxToB", None, -1350.6407],
    ["5", None, "g1ehQlL80t", 2.5, -1.0],
    ["6", None, "g1ehQlL80t", 4.0, None],
    ["7", "b", "TuDOcLxToB", -10.0, -12.564],
]

DATA_NAN = [
    ["1", "a", "g1ehQlL80t", -1.0, 0.0],
    ["2", "a", "zOyDvcyZ2s", np.nan, np.nan],
    ["3", "b", np.nan, 2.0, 2253463.5342],
    ["4", "A", "TuDOcLxToB", np.nan, -1350.6407],
    ["5", np.nan, "g1ehQlL80t", 2.5, -1.0],
    ["6", np.nan, "g1ehQlL80t", 4.0, np.nan],
    ["7", "b", "TuDOcLxToB", -10.0, -12.564],
]

DATA_ALL_NONE = [
    ["1", None, None, None, 0],
    ["2", None, None, None, 455],
    ["3", None, None, None, 2],
    ["4", None, None, None, -13],
    ["5", None, None, None, -1],
    ["6", None, None, None, 0],
    ["7", None, None, None, -13],
]


DATA_BOOLEAN: List[List[Any]] = [
    ["1", "c", "g1ehQlL80t", True, True],
    ["2", "a", "zOyDvcyZ2s", None, False],
    ["3", "b", "zOyDvcyZ2s", False, None],
    ["4", "A", "TuDOcLxToB", True, False],
    ["5", "d", "g1ehQlL80t", False, True],
    ["6", "b", "g1ehQlL80t", True, False],
    ["7", "b", "g1ehQlL80t", True, True],
]

DATA_CLIP = [
    ["1", "c", "g1ehQlL80t", -1.0, 0.0],
    ["2", "a", "zOyDvcyZ2s", 8.3, 124.6],
    ["3", "b", "zOyDvcyZ2s", 2.0, 230385.5342],
    ["4", "A", "TuDOcLxToB", 3.5, -1350.6407],
    ["5", "d", "g1ehQlL80t", 2.5, -1.0],
    ["6", "b", "g1ehQlL80t", 4.0, 5304.23920462],
    ["7", "b", "g1ehQlL80t", -10.0, -12.564],
]

SCHEMA = ["ID", "STR1", "STR2", "FLOAT1", "FLOAT2"]

SCHEMA_BOOLEAN = ["ID", "STR1", "STR2", "BOOL1", "BOOL2"]

VALUES = [0.0, 5.0, 10.2, 0.00001, -0.6234608, 109253.13059, -50453.20395]

VALUES_CLIP = [
    0.0,
    5.0,
    10.2,
    0.00000001,
    -0.6234608,
    402120653.13059,
    -60235453.20395,
]

CATEGORICAL_VALUES_LIST = [
    ["d", "b", "c", "d", "a", "A", "b"],
    [
        "zOyDvcyZ2s",
        "g1ehQlL80t",
        "TuDOcLxToB",
        "TuDOcLxToB",
        "zOyDvcyZ2s",
        "TuDOcLxToB",
        "g1ehQlL80t",
    ],
]

UNKNOWN_CATEGORICAL_VALUES_LIST = [
    ["z", "a", "b"],
    ["zOyDvcyZ2s", "g1ehQlL80t", "g1ehQlL80t"],
]

NONE_CATEGORICAL_VALUES_LIST = [
    ["A", "b", "a", None, "a", "A", "b"],
    [
        None,
        "g1ehQlL80t",
        "TuDOcLxToB",
        "TuDOcLxToB",
        "zOyDvcyZ2s",
        "TuDOcLxToB",
        "g1ehQlL80t",
    ],
]

BOOLEAN_VALUES = [True, True, None, False, True, False, False]

NUMERIC_COLS = ["FLOAT1", "FLOAT2"]

CATEGORICAL_COLS = ["STR1", "STR2"]

BOOLEAN_COLS = ["BOOL1", "BOOL2"]

OUTPUT_COLS = ["OUTPUT1", "OUTPUT2"]

ID_COL = "ID"

MIN_INT = np.iinfo(np.int32).min
MAX_INT = np.iinfo(np.int32).max


class DataType(Enum):
    INTEGER = 1
    FLOAT = 2


def gen_fuzz_data(
    rows: int, types: List[DataType], low: Union[int, List[int]] = MIN_INT, high: Union[int, List[int]] = MAX_INT
) -> Tuple[List[Any], List[str], List[str]]:
    """
    Generate random data based on input column types and row count.
    First column in the result data will be an ID column for indexing.

    Args:
        rows: num of rows to generate
        types: type per column
        low: lower bound(s) of the output interval (inclusive)
        high: upper bound(s) of the output interval (exclusive)

    Returns:
        A tuple of generated data and column names

    Raises:
        ValueError: if data type is not supported
    """
    data: List[npt.NDArray[Any]] = [np.arange(1, rows + 1, 1)]
    names = ["ID"]
    snowflake_identifiers = ["ID"]

    for idx, t in enumerate(types):
        _low = low if isinstance(low, int) else low[idx]
        _high = high if isinstance(high, int) else high[idx]
        if t == DataType.INTEGER:
            data.append(np.random.randint(_low, _high, rows))
        elif t == DataType.FLOAT:
            data.append(np.random.uniform(_low, _high, rows))
        else:
            raise ValueError(f"Unsupported data type {t}")
        names.append(f"col_{idx}")
        snowflake_identifiers.append(f'"col_{idx}"')
    data = np.core.records.fromarrays(data, names=names).tolist()  # type: ignore[call-overload]

    return data, names, snowflake_identifiers


def get_df(
    session: Session,
    data: List[List[Any]],
    schema: List[str],
    fillna: Optional[Union[object, ArrayLike]] = None,
) -> Tuple[pd.DataFrame, DataFrame]:
    """Create pandas dataframe and Snowpark dataframes from input data. The schema passed should be
    a pandas schema, which will be converted to a schema using snowflake identifiers when `session.create_dataframe`
    is called.

    Args:
        session: Snowpark session object.
        data: List of input data to convert to dataframe.
        schema: The pandas schema for dataframe to be created.
        fillna: Value to fill for NA values in the input data.

    Returns:
        A tuple containing a pandas dataframe and a snowpark dataframe.
    """
    df_pandas = pd.DataFrame(data, columns=schema)
    if fillna is not None:
        df_pandas.fillna(value=fillna, inplace=True)
    df = session.create_dataframe(df_pandas)
    df_pandas.columns = df.columns

    return df_pandas, df


def sort_by_columns(array: npt.NDArray[Any], num_col: int = 1) -> npt.NDArray[Any]:
    keys = [array[:, idx] for idx in range(num_col)]
    return array[np.lexsort(keys)]  # type: ignore[no-any-return]


def get_pandas_feature(X: Union[npt.NDArray[Any], pd.DataFrame], feature_idx: int) -> npt.NDArray[Any]:
    if hasattr(X, "iloc"):
        # pandas dataframes
        return X.iloc[:, feature_idx]  # type: ignore[no-any-return]
    # numpy arrays, sparse arrays
    return X[:, feature_idx]


def create_columns_metadata(session: Session) -> None:
    session.sql(
        """
    create or replace table columns_metadata
        (
            version varchar,
            column_name varchar,
            datatype varchar,
            tags varchar,
            is_array boolean,
            basic_statistics object,
            numeric_statistics object
        );
    """
    ).collect()


def create_transformer_state(session: Session) -> None:
    session.sql(
        """
    create or replace table transformer_state
        (
            id varchar,
            version varchar,
            definition_id varchar,
            start_time datetime,
            file_names array,
            custom_state object
        );
    """
    ).collect()


def create_transformer_definition(session: Session) -> None:
    session.sql(
        """
    create or replace table transformer_definition
        (
            id varchar,
            name varchar,
            settings object,
            input_columns array,
            output_columns array
        );
    """
    ).collect()


def create_dictionary_state(session: Session) -> None:
    session.sql(
        """
    create or replace table dictionary_state
        (
            version varchar,
            column_name varchar,
            column_value varchar,
            count bigint,
            index bigint
        );
    """
    ).collect()


def drop_columns_metadata(session: Session) -> None:
    session.sql(
        """
    drop table if exists columns_metadata;
    """
    ).collect()


def drop_transformer_state(session: Session) -> None:
    session.sql(
        """
    drop table if exists transformer_state;
    """
    ).collect()


def drop_transformer_definition(session: Session) -> None:
    session.sql(
        """
    drop table if exists transformer_definition;
    """
    ).collect()


def drop_dictionary_state(session: Session) -> None:
    session.sql(
        """
    drop table if exists dictionary_state;
    """
    ).collect()


def drop_all_tables(session: Session) -> None:
    drop_columns_metadata(session)
    drop_transformer_state(session)
    drop_transformer_definition(session)
    drop_dictionary_state(session)


def equal_default(x1: Any, x2: Any) -> bool:
    return bool(x1 == x2)


def equal_np_array(x1: npt.NDArray[Any], x2: npt.NDArray[Any]) -> bool:
    return bool(np.array_equal(x1, x2))


def equal_pandas_df(x1: pd.DataFrame, x2: pd.DataFrame) -> bool:
    return bool(x1.equals(x2))


def equal_pandas_df_ignore_row_order(x1: pd.DataFrame, x2: pd.DataFrame) -> bool:
    y1 = x1.sort_values(by=x1.columns.to_list()).reset_index(drop=True)
    y2 = x2.sort_values(by=x2.columns.to_list()).reset_index(drop=True)
    return equal_pandas_df(y1, y2)


def equal_optional_of(equality_func: _EqualityFunc) -> _EqualityFunc:
    def f(x1: Optional[Any], x2: Optional[Any]) -> bool:
        if x1 is not None and x2 is not None:
            return equality_func(x1, x2)
        return x1 is None and x2 is None

    return f


def equal_list_of(equality_func: _EqualityFunc) -> _EqualityFunc:
    def f(x1: List[Any], x2: List[Any]) -> bool:
        return len(x1) == len(x2) and all([equality_func(y1, y2) for (y1, y2) in zip(x1, x2)])

    return f
