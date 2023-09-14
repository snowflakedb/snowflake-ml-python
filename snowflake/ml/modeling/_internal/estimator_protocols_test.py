from typing import Protocol

from absl.testing import absltest, parameterized

from snowflake.ml.modeling._internal.estimator_protocols import (
    CVHandlers,
    FitPredictHandlers,
)


class EstimatorProtocolsTest(parameterized.TestCase):
    def test_fit_predict_handlers(self) -> None:
        self.assertIsInstance(FitPredictHandlers, Protocol)

    def test_cv_handlers(self) -> None:
        self.assertIsInstance(CVHandlers, Protocol)


if __name__ == "__main__":
    absltest.main()
