from absl.testing import absltest, parameterized

from tests.integ.snowflake.ml.jobs.job_test_base import ModelingJobTestBase

"""
this integration test is only for classic models, like XGBoost and lightgbm.
"""


class ClassicalModelTest(ModelingJobTestBase):
    @parameterized.parameters("xgboost", "lightgbm", "sklearn")
    def test_classic_models(self, model_name: str) -> None:
        self.train_models(model_name, "model_scripts/classical_models.py")


if __name__ == "__main__":
    absltest.main()
