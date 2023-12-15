from typing import Literal, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from snowflake.ml.model._signatures import snowpark_handler
from snowflake.snowpark import DataFrame as SnowparkDataFrame


def check_sp_df_res(
    res_sp_df: SnowparkDataFrame,
    expected_pd_df: pd.DataFrame,
    *,
    check_dtype: bool = True,
    check_index_type: Union[bool, Literal["equiv"]] = "equiv",
    check_column_type: Union[bool, Literal["equiv"]] = "equiv",
    check_frame_type: bool = True,
    check_names: bool = True,
) -> None:
    res_pd_df = snowpark_handler.SnowparkDataFrameHandler.convert_to_df(res_sp_df)

    def totuple(a: Union[npt.ArrayLike, Tuple[object], object]) -> Union[Tuple[object], object]:
        try:
            return tuple(totuple(i) for i in a)  # type: ignore[union-attr]
        except TypeError:
            return a

    for df in [res_pd_df, expected_pd_df]:
        for col in df.columns:
            if isinstance(df[col][0], list):
                df[col] = df[col].apply(tuple)
            elif isinstance(df[col][0], np.ndarray):
                df[col] = df[col].apply(totuple)

    pd.testing.assert_frame_equal(
        res_pd_df.sort_values(by=res_pd_df.columns.tolist()).reset_index(drop=True),
        expected_pd_df.sort_values(by=expected_pd_df.columns.tolist()).reset_index(drop=True),
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_column_type=check_column_type,
        check_frame_type=check_frame_type,
        check_names=check_names,
    )
