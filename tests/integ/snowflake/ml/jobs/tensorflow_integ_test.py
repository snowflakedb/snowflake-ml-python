from absl.testing import absltest
from packaging import version

from snowflake.ml._internal import env
from tests.integ.snowflake.ml.jobs.job_test_base import ModelingJobTestBase

"""
this integration test is only for tensorflow.
"""


class TensorflowModelTest(ModelingJobTestBase):
    @absltest.skipIf(
        version.Version(env.PYTHON_VERSION) >= version.Version("3.11"),
        "only works for Python 3.10 and below due to pickle compatibility",
    )
    def test_tensorflow_models(self) -> None:
        rows = self.session.sql("SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'PYPI%'").collect()
        if not rows:
            self.fail("No PyPI EAI found in environment.")
        pypi_eais = [r["name"] for r in rows]
        self.train_models("tensorflow", "model_scripts/tensorflow_models.py", ["tensorflow"], pypi_eais)


if __name__ == "__main__":
    absltest.main()
