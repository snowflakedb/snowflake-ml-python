from typing import List, Optional, Protocol, Tuple, Union

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
        expected_output_cols_list: List[str],
        drop_input_cols: Optional[bool] = False,
        example_output_pd_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Union[DataFrame, pd.DataFrame], object]:
        raise NotImplementedError

    def train_fit_transform(
        self,
        expected_output_cols_list: List[str],
        drop_input_cols: Optional[bool] = False,
    ) -> Tuple[Union[DataFrame, pd.DataFrame], object]:
        raise NotImplementedError
