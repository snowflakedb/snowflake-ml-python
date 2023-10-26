from typing import Any
from unittest import mock

from absl.testing import absltest, parameterized

from snowflake.ml.modeling._internal.snowpark_handlers import (
    LightGBMWrapperProvider,
    SklearnModelSelectionWrapperProvider,
    SklearnWrapperProvider,
    XGBoostWrapperProvider,
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
            provider = SklearnModelSelectionWrapperProvider()

            self.assertEqual(provider.imports, ["sklearn", "xgboost", "lightgbm"])

    def test_sklearn_model_selection_wrapper_provider_lightgbm_not_installed(self) -> None:
        orig_import = __import__

        def import_mock(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "lightgbm":
                raise ModuleNotFoundError
            return orig_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=import_mock):
            provider = SklearnModelSelectionWrapperProvider()

            self.assertEqual(provider.imports, ["sklearn", "xgboost"])

    def test_xgboost_wrapper_provider(self) -> None:
        provider = XGBoostWrapperProvider()
        self.assertEqual(provider.imports, ["xgboost"])

    def test_sklearn_wrapper_provider(self) -> None:
        provider = SklearnWrapperProvider()
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
            provider = LightGBMWrapperProvider()
            self.assertEqual(provider.imports, ["lightgbm"])


if __name__ == "__main__":
    absltest.main()
