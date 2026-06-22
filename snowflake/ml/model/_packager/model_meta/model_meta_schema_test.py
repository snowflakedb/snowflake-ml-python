import datetime

from absl.testing import absltest

from snowflake.ml import version as snowml_version
from snowflake.ml.model._packager.model_meta import model_meta_schema
from snowflake.ml.test_utils import test_env_utils


class ModelMetaSchemaTest(absltest.TestCase):
    def test_model_meta_schema_version(self) -> None:
        datetime.datetime.strptime(model_meta_schema.MODEL_METADATA_VERSION, "%Y-%m-%d")
        if model_meta_schema.MODEL_METADATA_MIN_SNOWPARK_ML_VERSION != snowml_version.VERSION:
            self.assertIn(
                model_meta_schema.MODEL_METADATA_MIN_SNOWPARK_ML_VERSION,
                test_env_utils.get_snowpark_ml_released_versions(),
            )

    def test_env_dict_is_packaged_pip_only(self) -> None:
        pip_only_env: model_meta_schema.ModelEnvDict = {
            "pip": "env/requirements.txt",
            "python_version": "3.10",
            "snowpark_ml_version": "1.0.0",
        }
        conda_env: model_meta_schema.ModelEnvDict = {
            **pip_only_env,
            "conda": "env/conda.yml",
        }
        self.assertTrue(model_meta_schema.env_dict_is_packaged_pip_only(pip_only_env))
        self.assertFalse(model_meta_schema.env_dict_is_packaged_pip_only(conda_env))


if __name__ == "__main__":
    absltest.main()
