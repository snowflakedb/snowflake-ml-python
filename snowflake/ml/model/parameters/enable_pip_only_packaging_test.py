from absl.testing import absltest

from snowflake.ml.model._packager.model_env import model_env


class EnablePipOnlyPackagingTest(absltest.TestCase):
    def test_enable_pip_only_packaging(self) -> None:
        self.assertFalse(model_env._ENABLE_PIP_ONLY_PACKAGING)

        # Enable pip-only packaging
        import snowflake.ml.model.parameters.enable_pip_only_packaging  # noqa: F401

        self.assertTrue(model_env._ENABLE_PIP_ONLY_PACKAGING)


if __name__ == "__main__":
    absltest.main()
