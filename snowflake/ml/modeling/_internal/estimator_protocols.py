from typing import List, Optional, Protocol

import pandas as pd

from snowflake.snowpark import DataFrame, Session


# TODO: Add more specific entities to type hint estimators instead of using `object`.
class FitPredictHandlers(Protocol):
    def batch_inference(
        self,
        dataset: DataFrame,
        session: Session,
        estimator: object,
        dependencies: List[str],
        inference_method: str,
        input_cols: List[str],
        pass_through_columns: List[str],
        expected_output_cols_list: List[str],
        expected_output_cols_type: str = "",
    ) -> DataFrame:
        raise NotImplementedError

    def score_pandas(
        self,
        dataset: pd.DataFrame,
        estimator: object,
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> float:
        raise NotImplementedError

    def score_snowpark(
        self,
        dataset: DataFrame,
        session: Session,
        estimator: object,
        dependencies: List[str],
        score_sproc_imports: List[str],
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> float:
        raise NotImplementedError


# TODO: Add more specific entities to type hint estimators instead of using `object`.
class CVHandlers(Protocol):
    def batch_inference(
        self,
        dataset: DataFrame,
        session: Session,
        estimator: object,
        dependencies: List[str],
        inference_method: str,
        input_cols: List[str],
        pass_through_columns: List[str],
        expected_output_cols_list: List[str],
        expected_output_cols_type: str = "",
    ) -> DataFrame:
        raise NotImplementedError

    def score_pandas(
        self,
        dataset: pd.DataFrame,
        estimator: object,
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> float:
        raise NotImplementedError

    def score_snowpark(
        self,
        dataset: DataFrame,
        session: Session,
        estimator: object,
        dependencies: List[str],
        score_sproc_imports: List[str],
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> float:
        raise NotImplementedError
