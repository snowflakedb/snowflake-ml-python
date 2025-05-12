from absl.testing import absltest
from packaging import version

from snowflake.ml._internal import env
from tests.integ.snowflake.ml.jobs import modeling_job_test_base

"""
this integration test is only for pytorch.
"""


class PytorchModelTest(modeling_job_test_base.BaseModelTest):
    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_pytorch_models(self) -> None:
        self.train_models("pytorch", "model_scripts/pytorch_models.py")


if __name__ == "__main__":
    absltest.main()
