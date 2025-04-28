import io
from typing import Any
from unittest import mock

import cloudpickle as cp
import numpy as np
from absl.testing import absltest, parameterized
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from snowflake.ml.modeling._internal.model_specifications import (
    ModelSpecificationsBuilder,
)
from snowflake.ml.modeling._internal.snowpark_implementations.distributed_hpo_trainer import (
    construct_cv_results,
)
from snowflake.snowpark import Row

each_cv_result_basic_sample = [
    {
        "mean_fit_time": np.array([0.00315547]),
        "std_fit_time": np.array([0.0]),
        "mean_score_time": np.array([0.00176454]),
        "std_score_time": np.array([0.0]),
        "param_n_components": np.ma.array(
            data=[2], mask=[False], fill_value="?", dtype=object
        ),  # type: ignore[no-untyped-call]
        "params": [{"n_components": 2}],
        "split0_test_score": np.array([-13.61564833]),
        "mean_test_score": np.array([-13.61564833]),
        "std_test_score": np.array([0.0]),
        "rank_test_score": np.array([1], dtype=np.int32),
    },
    {
        "mean_fit_time": np.array([0.00257707]),
        "std_fit_time": np.array([0.0]),
        "mean_score_time": np.array([0.00151849]),
        "std_score_time": np.array([0.0]),
        "param_n_components": np.ma.array(
            data=[2], mask=[False], fill_value="?", dtype=object
        ),  # type: ignore[no-untyped-call]
        "params": [{"n_components": 2}],
        "split0_test_score": np.array([-8.57012999]),
        "mean_test_score": np.array([-8.57012999]),
        "std_test_score": np.array([0.0]),
        "rank_test_score": np.array([1], dtype=np.int32),
    },
    {
        "mean_fit_time": np.array([0.00270677]),
        "std_fit_time": np.array([0.0]),
        "mean_score_time": np.array([0.00146675]),
        "std_score_time": np.array([0.0]),
        "param_n_components": np.ma.array(
            data=[1], mask=[False], fill_value="?", dtype=object
        ),  # type: ignore[no-untyped-call]
        "params": [{"n_components": 1}],
        "split0_test_score": np.array([-12.50893109]),
        "mean_test_score": np.array([-12.50893109]),
        "std_test_score": np.array([0.0]),
        "rank_test_score": np.array([1], dtype=np.int32),
    },
    {
        "mean_fit_time": np.array([0.00293922]),
        "std_fit_time": np.array([0.0]),
        "mean_score_time": np.array([0.00342846]),
        "std_score_time": np.array([0.0]),
        "param_n_components": np.ma.array(
            data=[1], mask=[False], fill_value="?", dtype=object
        ),  # type: ignore[no-untyped-call]
        "params": [{"n_components": 1}],
        "split0_test_score": np.array([-21.4394793]),
        "mean_test_score": np.array([-21.4394793]),
        "std_test_score": np.array([0.0]),
        "rank_test_score": np.array([1], dtype=np.int32),
    },
    {
        "mean_fit_time": np.array([0.00297642]),
        "std_fit_time": np.array([0.0]),
        "mean_score_time": np.array([0.00161123]),
        "std_score_time": np.array([0.0]),
        "param_n_components": np.ma.array(
            data=[1], mask=[False], fill_value="?", dtype=object
        ),  # type: ignore[no-untyped-call]
        "params": [{"n_components": 1}],
        "split0_test_score": np.array([-9.62685757]),
        "mean_test_score": np.array([-9.62685757]),
        "std_test_score": np.array([0.0]),
        "rank_test_score": np.array([1], dtype=np.int32),
    },
    {
        "mean_fit_time": np.array([0.00596809]),
        "std_fit_time": np.array([0.0]),
        "mean_score_time": np.array([0.00264239]),
        "std_score_time": np.array([0.0]),
        "param_n_components": np.ma.array(
            data=[2], mask=[False], fill_value="?", dtype=object
        ),  # type: ignore[no-untyped-call]
        "params": [{"n_components": 2}],
        "split0_test_score": np.array([-29.95119419]),
        "mean_test_score": np.array([-29.95119419]),
        "std_test_score": np.array([0.0]),
        "rank_test_score": np.array([1], dtype=np.int32),
    },
]

each_cv_result_return_train = [
    {
        "mean_fit_time": np.array([0.00315547]),
        "std_fit_time": np.array([0.0]),
        "mean_score_time": np.array([0.00176454]),
        "std_score_time": np.array([0.0]),
        "param_n_components": np.ma.array(
            data=[2], mask=[False], fill_value="?", dtype=object
        ),  # type: ignore[no-untyped-call]
        "params": [{"n_components": 2}],
        "split0_train_score": np.array([-13.61564833]),
        "split0_test_score": np.array([-13.61564833]),
        "mean_train_score": np.array([-13.61564833]),
        "std_train_score": np.array([0.0]),
        "mean_test_score": np.array([-13.61564833]),
        "std_test_score": np.array([0.0]),
        "rank_test_score": np.array([1], dtype=np.int32),
    },
    {
        "mean_fit_time": np.array([0.00257707]),
        "std_fit_time": np.array([0.0]),
        "mean_score_time": np.array([0.00151849]),
        "std_score_time": np.array([0.0]),
        "param_n_components": np.ma.array(
            data=[2], mask=[False], fill_value="?", dtype=object
        ),  # type: ignore[no-untyped-call]
        "params": [{"n_components": 2}],
        "split0_train_score": np.array([-8.57012999]),
        "split0_test_score": np.array([-8.57012999]),
        "mean_train_score": np.array([-8.57012999]),
        "std_train_score": np.array([0.0]),
        "mean_test_score": np.array([-8.57012999]),
        "std_test_score": np.array([0.0]),
        "rank_test_score": np.array([1], dtype=np.int32),
    },
]

SAMPLES: dict[str, dict[str, Any]] = {
    "basic": {
        "estimator": GridSearchCV(estimator=PCA(), param_grid={"n_components": range(1, 3)}, cv=3),
        "n_splits": 3,
        "param_grid": [{"n_components": 1}, {"n_components": 2}],
        "each_cv_result": each_cv_result_basic_sample,
        "IDX_LENGTH": 3,
        "PARAM_LENGTH": 2,
        "CV_RESULT_": {
            "mean_fit_time": np.array([0.00770839, 0.00551335]),
            "std_fit_time": np.array([0.00061078, 0.00179875]),
            "mean_score_time": np.array([0.00173187, 0.00182652]),
            "std_score_time": np.array([0.00016869, 0.00014979]),
            "param_n_components": np.ma.masked_array(
                data=[1, 2], mask=False, fill_value="?", dtype=object
            ),  # type: ignore[no-untyped-call]
            "params": np.array([{"n_components": 1}, {"n_components": 2}], dtype=object),
            "split0_test_score": np.array([-21.4394793, -29.95119419]),
            "mean_test_score": np.array([-14.52508932, -17.37899084]),
            "std_test_score": np.array([5.02879565, 9.12540544]),
            "rank_test_score": np.array([1, 2]),
            "split1_test_score": np.array([-9.62685757, -8.57012999]),
            "split2_test_score": np.array([-12.50893109, -13.61564833]),
        },
    },
    "return_train_score": {
        "estimator": GridSearchCV(estimator=PCA(), param_grid={"n_components": range(1, 2)}, cv=2),
        "n_splits": 2,
        "param_grid": [{"n_components": 2}],
        "each_cv_result": each_cv_result_return_train,
        "IDX_LENGTH": 2,
        "PARAM_LENGTH": 1,
        "CV_RESULT_": {
            "mean_fit_time": np.array([0.00286627]),
            "std_fit_time": np.array([0.0002892]),
            "mean_score_time": np.array([0.00164152]),
            "std_score_time": np.array([0.00012303]),
            "param_n_components": np.ma.masked_array(
                data=[2], mask=[False], fill_value="?", dtype=object
            ),  # type: ignore[no-untyped-call]
            "params": [{"n_components": 2}],
            "split0_test_score": np.array([-13.61564833]),
            "split1_test_score": np.array([-8.57012999]),
            "mean_test_score": np.array([-11.09288916]),
            "std_test_score": np.array([2.52275917]),
            "rank_test_score": np.array([1], dtype=np.int32),
        },
    },
}

for key, val in SAMPLES.items():
    combine_hex_cv_result = []
    for each_array in val["each_cv_result"]:
        with io.BytesIO() as f:
            cp.dump(each_array, f)
            f.seek(0)
            binary_cv_results = f.getvalue().hex()
            combine_hex_cv_result.append(binary_cv_results)
    SAMPLES[key]["combine_hex_cv_result"] = combine_hex_cv_result


class SnowparkHandlersUnitTest(parameterized.TestCase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        zipped = sorted(zip([5, 4, 2, 0, 1, 3], SAMPLES["basic"]["combine_hex_cv_result"]), key=lambda x: x[0])
        self.RAW_DATA_SP = [Row(val) for _, val in zipped]

    def test_sklearn_model_selection_wrapper_provider_lightgbm_installed(self) -> None:
        orig_import = __import__

        def import_mock(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "lightgbm":
                lightgbm_mock = mock.MagicMock()
                lightgbm_mock.__version__ = "1"
                return lightgbm_mock
            return orig_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_mock):
            model = GridSearchCV(estimator=XGBRegressor(), param_grid={"max_depth": [10, 100]})
            provider = ModelSpecificationsBuilder.build(model=model)

            self.assertEqual(provider.imports, ["sklearn", "xgboost", "lightgbm"])

    def test_sklearn_model_selection_wrapper_provider_lightgbm_not_installed(self) -> None:
        orig_import = __import__

        def import_mock(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "lightgbm":
                raise ModuleNotFoundError
            return orig_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_mock):
            model = GridSearchCV(estimator=XGBRegressor(), param_grid={"max_depth": [10, 100]})
            provider = ModelSpecificationsBuilder.build(model=model)

            self.assertEqual(provider.imports, ["sklearn", "xgboost"])

    def test_xgboost_wrapper_provider(self) -> None:
        provider = ModelSpecificationsBuilder.build(model=XGBRegressor())
        self.assertEqual(provider.imports, ["xgboost"])

    def test_sklearn_wrapper_provider(self) -> None:
        provider = ModelSpecificationsBuilder.build(model=LinearRegression())
        self.assertEqual(provider.imports, ["sklearn"])

    def test_lightgbm_wrapper_provider(self) -> None:
        orig_import = __import__

        def import_mock(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "lightgbm":
                lightgbm_mock = mock.MagicMock()
                lightgbm_mock.__version__ = "1"
                return lightgbm_mock
            return orig_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_mock):
            provider = ModelSpecificationsBuilder.build(model=LGBMRegressor())
            self.assertEqual(provider.imports, ["lightgbm"])

    def _compare_cv_results(self, cv_result_1: dict[str, Any], cv_result_2: dict[str, Any]) -> None:
        # compare the keys
        self.assertEqual(sorted(cv_result_1.keys()), sorted(cv_result_2.keys()))
        # compare the values
        for k, v in cv_result_1.items():
            if isinstance(v, np.ndarray):
                if k.startswith("param_"):  # compare the masked array
                    np.ma.allequal(v, cv_result_2[k])  # type: ignore[no-untyped-call]
                elif k == "params":  # compare the parameter combination
                    self.assertEqual(v.tolist(), cv_result_2[k].tolist())
                elif k.endswith("test_score"):  # compare the test score
                    np.testing.assert_allclose(v, cv_result_2[k], rtol=1.0e-1, atol=1.0e-2)
                # Do not compare the fit time

    def test_cv_result(self) -> None:
        multimetric, cv_results_ = construct_cv_results(
            SAMPLES["basic"]["estimator"],
            SAMPLES["basic"]["n_splits"],
            SAMPLES["basic"]["param_grid"],
            self.RAW_DATA_SP,
            SAMPLES["basic"]["IDX_LENGTH"],
            SAMPLES["basic"]["PARAM_LENGTH"],
        )
        self.assertEqual(multimetric, False)
        self._compare_cv_results(cv_results_, SAMPLES["basic"]["CV_RESULT_"])

    def test_cv_result_return_train_score(self) -> None:
        multimetric, cv_results_ = construct_cv_results(
            SAMPLES["return_train_score"]["estimator"],
            SAMPLES["return_train_score"]["n_splits"],
            SAMPLES["return_train_score"]["param_grid"],
            [Row(val) for val in SAMPLES["return_train_score"]["combine_hex_cv_result"]],
            SAMPLES["return_train_score"]["IDX_LENGTH"],
            SAMPLES["return_train_score"]["PARAM_LENGTH"],
        )
        self.assertEqual(multimetric, False)
        self._compare_cv_results(cv_results_, SAMPLES["return_train_score"]["CV_RESULT_"])

    def test_cv_result_incorrect_param_length(self) -> None:
        with self.assertRaises(ValueError):
            construct_cv_results(
                SAMPLES["basic"]["estimator"],
                SAMPLES["basic"]["n_splits"],
                SAMPLES["basic"]["param_grid"],
                self.RAW_DATA_SP,
                SAMPLES["basic"]["IDX_LENGTH"],
                1,
            )

    def test_cv_result_nan(self) -> None:
        # corner cases with nan values
        with self.assertRaises(ValueError):
            construct_cv_results(
                SAMPLES["basic"]["estimator"],
                SAMPLES["basic"]["n_splits"],
                SAMPLES["basic"]["param_grid"],
                self.RAW_DATA_SP,
                0,
                SAMPLES["basic"]["PARAM_LENGTH"],
            )
        # empty list
        with self.assertRaises(ValueError):
            construct_cv_results(
                SAMPLES["basic"]["estimator"],
                SAMPLES["basic"]["n_splits"],
                SAMPLES["basic"]["param_grid"],
                [],
                SAMPLES["basic"]["IDX_LENGTH"],
                SAMPLES["basic"]["PARAM_LENGTH"],
            )


if __name__ == "__main__":
    absltest.main()
