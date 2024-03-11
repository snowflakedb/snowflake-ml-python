import sys

from absl.testing import absltest
from sklearn.linear_model import LinearRegression

from snowflake.ml.modeling._internal.ml_runtime_implementations.ml_runtime_handlers import (
    MLRuntimeTransformHandlers,
)
from snowflake.snowpark import DataFrame


class MLRuntimeTransformHandlersTest(absltest.TestCase):
    def setUp(self) -> None:
        self.dataset = absltest.mock.MagicMock(spec=DataFrame)
        self.estimator = absltest.mock.MagicMock(spec=LinearRegression)

    def test_exception_client_package_available(self) -> None:
        with absltest.mock.patch.dict(sys.modules, {"snowflake.ml.runtime": absltest.mock.Mock()}):
            MLRuntimeTransformHandlers(
                dataset=self.dataset,
                estimator=self.estimator,
                class_name="",
                subproject="",
            )

    def test_exception_client_package_unavailable(self) -> None:

        with absltest.mock.patch.dict(
            sys.modules, {key: value for key, value in sys.modules.items() if key != "snowflake.ml.runtime"}
        ):
            with self.assertRaises(ModuleNotFoundError):
                MLRuntimeTransformHandlers(
                    dataset=self.dataset,
                    estimator=self.estimator,
                    class_name="",
                    subproject="",
                )


if __name__ == "__main__":
    absltest.main()
