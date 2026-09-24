from typing import Protocol

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
        drop_input_cols: bool | None = False,
        example_output_pd_df: pd.DataFrame | None = None,
    ) -> tuple[DataFrame | pd.DataFrame, object]:
        raise NotImplementedError

    def train_fit_transform(
        self,
        expected_output_cols_list: list[str],
        drop_input_cols: bool | None = False,
    ) -> tuple[DataFrame | pd.DataFrame, object]:
        raise NotImplementedError
