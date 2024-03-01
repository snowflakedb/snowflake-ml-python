import sys

from absl.testing import absltest
from sklearn.linear_model import LinearRegression

from snowflake.ml.modeling._internal.ml_runtime_implementations.ml_runtime_trainer import (
    MlRuntimeModelTrainer,
)
from snowflake.snowpark import DataFrame


class MlRuntimeModelTrainerTest(absltest.TestCase):
    def setUp(self) -> None:
        self.dataset = absltest.mock.MagicMock(spec=DataFrame)
        self.dataset._session = absltest.mock.Mock()
        self.estimator = absltest.mock.MagicMock(spec=LinearRegression)

    def test_exception_client_package_available(self) -> None:
        with absltest.mock.patch.dict(sys.modules, {"snowflake.ml.runtime": absltest.mock.Mock()}):
            MlRuntimeModelTrainer(
                estimator=self.estimator,
                dataset=self.dataset,
                session=self.dataset._session,
                input_cols=["col_1", "col_2"],
                label_cols=["col_1"],
                sample_weight_col=None,
            )

    def test_exception_client_package_unavailable(self) -> None:

        with absltest.mock.patch.dict(
            sys.modules, {key: value for key, value in sys.modules.items() if key != "snowflake.ml.runtime"}
        ):
            with self.assertRaises(ModuleNotFoundError):
                MlRuntimeModelTrainer(
                    estimator=self.estimator,
                    dataset=self.dataset,
                    session=self.dataset._session,
                    input_cols=["col_1", "col_2"],
                    label_cols=["col_1"],
                    sample_weight_col=None,
                )


if __name__ == "__main__":
    absltest.main()
