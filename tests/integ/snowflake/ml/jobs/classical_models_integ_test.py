from absl.testing import absltest, parameterized
from packaging import version

from snowflake.ml._internal import env
from tests.integ.snowflake.ml.jobs import modeling_job_test_base

"""
this integration test is only for classic models, like XGBoost and lightgbm.
"""


class ClassicalModelTest(modeling_job_test_base.BaseModelTest):
    @parameterized.parameters("xgboost", "lightgbm", "sklearn")
    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_classic_models(self, model_name: str) -> None:
        self.train_models(model_name, "model_scripts/classical_models.py")


if __name__ == "__main__":
    absltest.main()
