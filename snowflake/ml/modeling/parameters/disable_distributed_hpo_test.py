from typing import List, Optional, Union
from unittest import mock

import pandas as pd
from absl.testing import absltest
from sklearn import model_selection
from snowflake.ml.modeling.xgboost.xgb_classifier import XGBClassifier

from snowflake.ml.modeling.model_selection._internal._grid_search_cv import GridSearchCV
from snowflake.ml.modeling.model_selection._internal._randomized_search_cv import (
    RandomizedSearchCV,
)
from snowflake.snowpark import DataFrame, Session


class MockHandlers:
    def fit_pandas(
        self,
        dataset: pd.DataFrame,
        estimator: object,
        input_cols: List[str],
        label_cols: Optional[List[str]],
        sample_weight_col: Optional[str],
    ) -> object:
        raise NotImplementedError

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

    def fit_snowpark(
        self,
        dataset: DataFrame,
        session: Session,
        estimator: object,
        dependencies: List[str],
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> object:
        response_obj = mock.Mock(spec=model_selection.GridSearchCV)
        response_obj.function = "FIT_SNOWPARK"
        return response_obj

    def fit_search_snowpark(
        self,
        param_grid: Union[model_selection.ParameterGrid, model_selection.ParameterSampler],
        dataset: DataFrame,
        session: Session,
        estimator: Union[model_selection.GridSearchCV, model_selection.RandomizedSearchCV],
        dependencies: List[str],
        udf_imports: List[str],
        input_cols: List[str],
        label_cols: List[str],
        sample_weight_col: Optional[str],
    ) -> Union[model_selection.GridSearchCV, model_selection.RandomizedSearchCV]:
        response_obj = mock.Mock(spec=model_selection.GridSearchCV)
        response_obj.function = "FIT_SEARCH"
        return response_obj


class DisableDistributedHPOTest(absltest.TestCase):
    @mock.patch(
        "snowflake.ml.modeling.model_selection._internal._grid_search_cv.pkg_version_utils"
        ".get_valid_pkg_versions_supported_in_snowflake_conda_channel"
    )
    @mock.patch("snowflake.ml.modeling.model_selection._internal._grid_search_cv.is_single_node")
    def test_disable_distributed_hpo(self, is_single_node_mock: mock.Mock, pkg_version_mock: mock.Mock) -> None:
        is_single_node_mock.return_value = False
        pkg_version_mock.return_value = []
        mock_session = mock.MagicMock(spec=Session)
        mock_dataframe = mock.MagicMock(spec=DataFrame)
        mock_dataframe._session = mock_session

        estimator = XGBClassifier()
        grid_search_cv = GridSearchCV(estimator=estimator, param_grid=dict(fake=[1, 2]))
        grid_search_cv._handlers = MockHandlers()

        randomized_search_cv = RandomizedSearchCV(estimator=estimator, param_distributions=dict(fake=[1, 2]))
        randomized_search_cv._handlers = MockHandlers()

        grid_search_cv._fit_snowpark(mock_dataframe)
        randomized_search_cv._fit_snowpark(mock_dataframe)

        assert grid_search_cv._sklearn_object is not None
        assert randomized_search_cv._sklearn_object is not None
        self.assertTrue(grid_search_cv._sklearn_object.function, "FIT_SEARCH")
        self.assertEqual(randomized_search_cv._sklearn_object.function, "FIT_SEARCH")

        # Disable distributed HPO
        import snowflake.ml.modeling.parameters.disable_distributed_hpo  # noqa: F401

        self.assertFalse(GridSearchCV._ENABLE_DISTRIBUTED)
        self.assertFalse(RandomizedSearchCV._ENABLE_DISTRIBUTED)

        grid_search_cv = GridSearchCV(estimator=estimator, param_grid=dict(fake=[1, 2]))
        grid_search_cv._handlers = MockHandlers()
        randomized_search_cv = RandomizedSearchCV(estimator=estimator, param_distributions=dict(fake=[1, 2]))
        randomized_search_cv._handlers = MockHandlers()

        grid_search_cv._fit_snowpark(mock_dataframe)
        randomized_search_cv._fit_snowpark(mock_dataframe)

        assert grid_search_cv._sklearn_object is not None
        assert randomized_search_cv._sklearn_object is not None
        self.assertTrue(grid_search_cv._sklearn_object.function, "FIT_SNOWPARK")
        self.assertEqual(randomized_search_cv._sklearn_object.function, "FIT_SNOWPARK")


if __name__ == "__main__":
    absltest.main()
