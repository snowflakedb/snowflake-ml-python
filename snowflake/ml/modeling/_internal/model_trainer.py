from typing import Optional, Protocol, Union

import pandas as pd

from snowflake.snowpark import DataFrame


class ModelTrainer(Protocol):
    """
    Interface for model trainer implementations.

    There are multiple flavors of training like training with pandas datasets, training with
    Snowpark datasets using sprocs, and out of core training with Snowpark datasets etc.
    """

    def train(self) -> object:
        raise NotImplementedError

    def train_fit_predict(
        self,
        expected_output_cols_list: list[str],
        drop_input_cols: Optional[bool] = False,
        example_output_pd_df: Optional[pd.DataFrame] = None,
    ) -> tuple[Union[DataFrame, pd.DataFrame], object]:
        raise NotImplementedError

    def train_fit_transform(
        self,
        expected_output_cols_list: list[str],
        drop_input_cols: Optional[bool] = False,
    ) -> tuple[Union[DataFrame, pd.DataFrame], object]:
        raise NotImplementedError
