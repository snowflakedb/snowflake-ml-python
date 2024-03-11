import importlib
import uuid
from functools import partial
from typing import Literal

import importlib_resources
import numpy as np
import pandas as pd
import xgboost as xgb
from absl.testing import absltest, parameterized
from sklearn import compose, datasets, impute, pipeline, preprocessing

from snowflake.ml import registry
from snowflake.ml.model import custom_model, model_signature
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session
from tests.integ.snowflake.ml.registry.model.my_module.utils import column_labeller
from tests.integ.snowflake.ml.test_utils import db_manager


class ModelWithAdditionalImportTest(parameterized.TestCase):
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

    @parameterized.parameters([{"import_method": "ext_modules"}, {"import_method": "code_paths"}])
    def test_additional_import(self, import_method: Literal["ext_modules", "code_paths"]) -> None:
        name = f"model_{self._run_id}"
        version = f"ver_{import_method}"

        X, y = datasets.make_classification()
        X = pd.DataFrame(X, columns=["X" + str(i) for i in range(20)])
        log_trans = pipeline.Pipeline(
            [
                ("impute", impute.SimpleImputer()),
                ("scaler", preprocessing.MinMaxScaler()),
                (
                    "logger",
                    preprocessing.FunctionTransformer(
                        np.log1p,
                        feature_names_out=partial(column_labeller, "LOG"),
                    ),
                ),
            ]
        )
        preproc_pipe = compose.ColumnTransformer(
            [("log", log_trans, ["X0", "X1"])],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        preproc_pipe.set_output(transform="pandas")
        preproc_pipe.fit(X, y)

        xgb_data = xgb.DMatrix(preproc_pipe.transform(X), y)
        booster = xgb.train(dict(max_depth=5), xgb_data, num_boost_round=10)

        class MyModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext) -> None:
                super().__init__(context)

            @custom_model.inference_api
            def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                xgb_data = xgb.DMatrix(self.context.model_ref("pipeline").transform(X))
                preds = self.context.model_ref("model").predict(xgb_data)
                res_df = pd.DataFrame({"output": preds})
                return res_df

        my_model = MyModel(
            custom_model.ModelContext(
                models={
                    "pipeline": preproc_pipe,
                    "model": booster,
                },
                artifacts={},
            )
        )

        sig = model_signature.ModelSignature(
            inputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name=f"X{i}") for i in range(20)],
            outputs=[model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="output")],
        )

        if import_method == "ext_modules":
            my_module = importlib.import_module("tests.integ.snowflake.ml.registry.model.my_module")
            mv = self.registry.log_model(
                my_model,
                model_name=name,
                version_name=version,
                signatures={"predict": sig},
                ext_modules=[my_module],
            )
        else:
            code_path = importlib_resources.files("tests")._paths[0]
            mv = self.registry.log_model(
                my_model,
                model_name=name,
                version_name=version,
                signatures={"predict": sig},
                code_paths=[code_path],
            )

        mv.run(X, function_name="predict")


if __name__ == "__main__":
    absltest.main()
