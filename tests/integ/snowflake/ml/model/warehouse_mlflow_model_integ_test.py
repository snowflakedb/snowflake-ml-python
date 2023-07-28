#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, ensemble, model_selection

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.model._signatures import numpy_handler
from snowflake.ml.utils import connection_params
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session
from tests.integ.snowflake.ml.model import warehouse_model_integ_test_utils
from tests.integ.snowflake.ml.test_utils import db_manager


class TestWarehouseMLFlowModelInteg(parameterized.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

        self._db_manager = db_manager.DBManager(self._session)
        self._db_manager.cleanup_schemas()
        self._db_manager.cleanup_stages()
        self._db_manager.cleanup_user_functions()

        # To create different UDF names among different runs
        self.run_id = uuid.uuid4().hex
        self._test_schema_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "model_deployment_mlflow_model_test_schema"
        )
        self._db_manager.create_schema(self._test_schema_name)
        self._db_manager.use_schema(self._test_schema_name)

        self.deploy_stage_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "deployment_stage"
        )
        self.full_qual_stage = self._db_manager.create_stage(
            self.deploy_stage_name, schema_name=self._test_schema_name, sse_encrypted=False
        )

    @classmethod
    def tearDownClass(self) -> None:
        self._db_manager.drop_stage(self.deploy_stage_name, schema_name=self._test_schema_name)
        self._db_manager.drop_schema(self._test_schema_name)
        self._session.close()

    def base_test_case(
        self,
        name: str,
        model: model_types.SupportedModelType,
        sample_input: model_types.SupportedDataType,
        test_input: model_types.SupportedDataType,
        deploy_params: Dict[str, Tuple[Dict[str, Any], Callable[[Union[pd.DataFrame, SnowparkDataFrame]], Any]]],
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        warehouse_model_integ_test_utils.base_test_case(
            self._db_manager,
            run_id=self.run_id,
            full_qual_stage=self.full_qual_stage,
            name=name,
            model=model,
            sample_input=sample_input,
            test_input=test_input,
            deploy_params=deploy_params,
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True, "test_released_library": False},
        {"model_in_stage": False, "permanent_deploy": False, "test_released_library": False},
        # {"model_in_stage": True, "permanent_deploy": False, "test_released_library": True},
        # {"model_in_stage": False, "permanent_deploy": True, "test_released_library": True},
    )
    def test_mlflow_model_deploy_sklearn_df(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        db = datasets.load_diabetes(as_frame=True)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        with mlflow.start_run() as run:
            rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)

            # Use the model to make predictions on the test dataset.
            predictions = rf.predict(X_test)
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                rf,
                "model",
                signature=signature,
                metadata={"author": "halu", "version": "1"},
                conda_env={
                    "dependencies": [
                        "python=3.8.13",
                        "mlflow==2.3.1",
                        "cloudpickle==2.0.0",
                        "numpy==1.23.4",
                        "psutil==5.9.0",
                        "scikit-learn==1.2.2",
                        "scipy==1.9.3",
                        "typing-extensions==4.5.0",
                    ],
                    "name": "mlflow-env",
                },
            )

            run_id = run.info.run_id

        self.base_test_case(
            name="mlflow_model_sklearn_df",
            model=mlflow.pyfunc.load_model(f"runs:/{run_id}/model"),
            sample_input=None,
            test_input=X_test,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(np.expand_dims(predictions, axis=1), res.to_numpy()),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )

    @parameterized.parameters(  # type: ignore[misc]
        {"model_in_stage": True, "permanent_deploy": True, "test_released_library": False},
        {"model_in_stage": False, "permanent_deploy": False, "test_released_library": False},
        # {"model_in_stage": True, "permanent_deploy": False, "test_released_library": True},
        # {"model_in_stage": False, "permanent_deploy": True, "test_released_library": True},
    )
    def test_mlflow_model_deploy_sklearn(
        self,
        model_in_stage: Optional[bool] = False,
        permanent_deploy: Optional[bool] = False,
        test_released_library: Optional[bool] = False,
    ) -> None:
        db = datasets.load_diabetes()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(db.data, db.target)
        with mlflow.start_run() as run:
            rf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
            rf.fit(X_train, y_train)

            # Use the model to make predictions on the test dataset.
            predictions = rf.predict(X_test)
            signature = mlflow.models.signature.infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(
                rf,
                "model",
                signature=signature,
                metadata={"author": "halu", "version": "1"},
                conda_env={
                    "dependencies": [
                        "python=3.8.13",
                        "mlflow==2.3.1",
                        "cloudpickle==2.0.0",
                        "numpy==1.23.4",
                        "psutil==5.9.0",
                        "scikit-learn==1.2.2",
                        "scipy==1.9.3",
                        "typing-extensions==4.5.0",
                    ],
                    "name": "mlflow-env",
                },
            )

            run_id = run.info.run_id

        X_test_df = numpy_handler.SeqOfNumpyArrayHandler.convert_to_df([X_test])

        self.base_test_case(
            name="mlflow_model_sklearn",
            model=mlflow.pyfunc.load_model(f"runs:/{run_id}/model"),
            sample_input=None,
            test_input=X_test_df,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(np.expand_dims(predictions, axis=1), res.to_numpy()),
                ),
            },
            model_in_stage=model_in_stage,
            permanent_deploy=permanent_deploy,
            test_released_library=test_released_library,
        )


if __name__ == "__main__":
    absltest.main()
