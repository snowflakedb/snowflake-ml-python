from absl.testing import absltest

from tests.integ.snowflake.ml.jobs.job_test_base import ModelingJobTestBase

"""
this integration test is only for pytorch.
"""


class PytorchModelTest(ModelingJobTestBase):
    def test_pytorch_models(self) -> None:
        self.train_models("pytorch", "model_scripts/pytorch_models.py")


if __name__ == "__main__":
    absltest.main()
