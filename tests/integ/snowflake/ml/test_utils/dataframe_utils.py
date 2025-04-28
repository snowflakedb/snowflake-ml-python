import json
from typing import Any, Literal, Optional, Union

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
    check_exact: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    deserialize_json: Optional[bool] = False,
) -> None:
    res_pd_df = snowpark_handler.SnowparkDataFrameHandler.convert_to_df(res_sp_df)

    def totuple(a: Union[npt.ArrayLike, tuple[object], object]) -> Union[tuple[object], object]:
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

    kwargs: dict[str, Any] = {
        "check_dtype": check_dtype,
        "check_index_type": check_index_type,
        "check_column_type": check_column_type,
        "check_frame_type": check_frame_type,
        "check_names": check_names,
        "check_exact": check_exact,
    }

    if check_names:
        res_pd_df = res_pd_df[expected_pd_df.columns]

    if check_exact is False:
        if atol is not None:
            kwargs["atol"] = atol
        if rtol is not None:
            kwargs["rtol"] = rtol

    res_pd_df = res_pd_df.sort_values(by=res_pd_df.columns.tolist()).reset_index(drop=True)
    expected_pd_df = expected_pd_df.sort_values(by=expected_pd_df.columns.tolist()).reset_index(drop=True)

    if deserialize_json:
        for df in [res_pd_df, expected_pd_df]:
            for col in df.columns:
                if isinstance(df[col][0], str):
                    df[col] = df[col].apply(json.loads)

    pd.testing.assert_frame_equal(
        res_pd_df,
        expected_pd_df,
        **kwargs,
    )


def convert2D_json_to_3D(array: npt.NDArray[Any]) -> list[list[list[Any]]]:
    final_array = []
    for i in range(array.shape[0]):
        tmp = []
        for j in range(array.shape[1]):
            json_to_dict = json.loads(array[i][j])
            num_keys = len(json_to_dict.keys())
            tmp.append([float(json_to_dict[str(k)]) for k in range(num_keys)])
        final_array.append(tmp)
    return final_array
