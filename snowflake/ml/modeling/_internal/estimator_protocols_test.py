from typing import Protocol

from absl.testing import absltest, parameterized

from snowflake.ml.modeling._internal.estimator_protocols import TransformerHandlers


class EstimatorProtocolsTest(parameterized.TestCase):
    def test_fit_predict_handlers(self) -> None:
        self.assertIsInstance(TransformerHandlers, Protocol)

    def test_cv_handlers(self) -> None:
        self.assertIsInstance(TransformerHandlers, Protocol)


if __name__ == "__main__":
    absltest.main()
