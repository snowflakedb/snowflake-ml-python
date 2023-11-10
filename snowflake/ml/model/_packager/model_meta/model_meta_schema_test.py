import datetime

from absl.testing import absltest

from snowflake.ml._internal import env as snowml_env
from snowflake.ml.model._packager.model_meta import model_meta_schema
from snowflake.ml.test_utils import test_env_utils


class ModelMetaSchemaTest(absltest.TestCase):
    def test_model_meta_schema_version(self) -> None:
        datetime.datetime.strptime(model_meta_schema.MODEL_METADATA_VERSION, "%Y-%m-%d")
        if model_meta_schema.MODEL_METADATA_MIN_SNOWPARK_ML_VERSION != snowml_env.VERSION:
            self.assertIn(
                model_meta_schema.MODEL_METADATA_MIN_SNOWPARK_ML_VERSION,
                test_env_utils.get_snowpark_ml_released_versions(),
            )


if __name__ == "__main__":
    absltest.main()
