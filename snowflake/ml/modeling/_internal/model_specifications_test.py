from typing import Any
from unittest import mock

from absl.testing import absltest, parameterized
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from snowflake.ml.modeling._internal.model_specifications import (
    ModelSpecificationsBuilder,
)


class SnowparkHandlersUnitTest(parameterized.TestCase):
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


if __name__ == "__main__":
    absltest.main()
