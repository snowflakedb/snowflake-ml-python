import os
import tempfile
import uuid

import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml import registry
from snowflake.ml.model import custom_model
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.test_utils import db_manager


class DemoModelWithArtifacts(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        with open(context.path("bias"), encoding="utf-8") as f:
            v = int(f.read())
        self.bias = v

    @custom_model.inference_api
    def predict(self, input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"output": input["c1"] + self.bias})


class ModelWithAdditionalImportTest(absltest.TestCase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        login_options = connection_params.SnowflakeLoginOptions()

        self._run_id = uuid.uuid4().hex
        self._test_db = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(self._run_id, "db").upper()
        self._test_schema = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self._run_id, "schema"
        ).upper()

        self._session = Session.builder.configs(
            {
                **login_options,
                **{"database": self._test_db, "schema": self._test_schema},
            }
        ).create()

        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.create_database(self._test_db)
        self._db_manager.create_schema(self._test_schema)
        self._db_manager.cleanup_databases(expire_hours=6)
        self.registry = registry.Registry(self._session)

    def tearDown(self) -> None:
        self._db_manager.drop_database(self._test_db)
        self._session.close()

    def test_multiple_model(self) -> None:
        version = "v1"
        arr = np.array([[1], [4]])
        pd_df = pd.DataFrame(arr, columns=["c1"])

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                f.write("10")
            lm_1 = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            name_1 = f"model_{self._run_id}_1"

            self.registry.log_model(lm_1, model_name=name_1, version_name=version, sample_input_data=pd_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "bias"), "w", encoding="utf-8") as f:
                f.write("20")
            lm_2 = DemoModelWithArtifacts(
                custom_model.ModelContext(models={}, artifacts={"bias": os.path.join(tmpdir, "bias")})
            )
            name_2 = f"model_{self._run_id}_2"

            self.registry.log_model(lm_2, model_name=name_2, version_name=version, sample_input_data=pd_df)

        res = (
            self._session.sql(f"SELECT {name_1}!predict(1):output as A, {name_2}!predict(1):output as B")
            .collect()[0]
            .as_dict()
        )

        self.assertDictEqual(res, {"A": "11", "B": "21"})


if __name__ == "__main__":
    absltest.main()
