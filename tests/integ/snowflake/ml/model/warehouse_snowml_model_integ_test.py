#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#

import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from absl.testing import absltest, parameterized
from sklearn import datasets

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.modeling.lightgbm import LGBMRegressor
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.utils import connection_params
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session
from tests.integ.snowflake.ml.model import warehouse_model_integ_test_utils
from tests.integ.snowflake.ml.test_utils import db_manager


@pytest.mark.pip_incompatible
class TestWarehouseSnowMLModelInteg(parameterized.TestCase):
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
            self.run_id, "model_deployment_snowml_model_test_schema"
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
        permanent_deploy: Optional[bool] = False,
        test_released_version: Optional[str] = None,
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
            permanent_deploy=permanent_deploy,
            test_released_version=test_released_version,
        )

    @parameterized.product(permanent_deploy=[True, False], test_released_version=[None, "1.0.5"])  # type: ignore[misc]
    def test_snowml_model_deploy_snowml_sklearn(
        self,
        permanent_deploy: Optional[bool] = False,
        test_released_version: Optional[str] = None,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        self.base_test_case(
            name="snowml_model_sklearn",
            model=regr,
            sample_input=None,
            test_input=test_features,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
            test_released_version=test_released_version,
        )

    @parameterized.product(permanent_deploy=[True, False], test_released_version=[None, "1.0.5"])  # type: ignore[misc]
    def test_snowml_model_deploy_xgboost(
        self,
        permanent_deploy: Optional[bool] = False,
        test_released_version: Optional[str] = None,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        self.base_test_case(
            name="snowml_model_xgb",
            model=regr,
            sample_input=None,
            test_input=test_features,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
            test_released_version=test_released_version,
        )

    @parameterized.product(permanent_deploy=[True, False], test_released_version=[None, "1.0.5"])  # type: ignore[misc]
    def test_snowml_model_deploy_lightgbm(
        self,
        permanent_deploy: Optional[bool] = False,
        test_released_version: Optional[str] = None,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LGBMRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        self.base_test_case(
            name="snowml_model_lightgbm",
            model=regr,
            sample_input=None,
            test_input=test_features,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
            test_released_version=test_released_version,
        )


if __name__ == "__main__":
    absltest.main()
