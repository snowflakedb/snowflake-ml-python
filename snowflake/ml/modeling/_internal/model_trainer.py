from typing import List, Protocol, Tuple, Union

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
        pass_through_columns: List[str],
        expected_output_cols_list: List[str],
    ) -> Tuple[Union[DataFrame, pd.DataFrame], object]:
        raise NotImplementedError
