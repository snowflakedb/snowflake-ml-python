import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Union

import inflection
import lightgbm
import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized
from sklearn import datasets, model_selection

from snowflake.ml.model import type_hints as model_types
from snowflake.ml.utils import connection_params
from snowflake.snowpark import DataFrame as SnowparkDataFrame, Session
from tests.integ.snowflake.ml.model import warehouse_model_integ_test_utils
from tests.integ.snowflake.ml.test_utils import dataframe_utils, db_manager


class TestWarehouseLightGBMModelInteg(parameterized.TestCase):
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
            self.run_id, "model_deployment_lightgbm_model_test_schema"
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
        sample_input_data: model_types.SupportedDataType,
        test_input: model_types.SupportedDataType,
        deploy_params: Dict[str, Tuple[Dict[str, Any], Callable[[Union[pd.DataFrame, SnowparkDataFrame]], Any]]],
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        warehouse_model_integ_test_utils.base_test_case(
            self._db_manager,
            run_id=self.run_id,
            full_qual_stage=self.full_qual_stage,
            name=name,
            model=model,
            sample_input_data=sample_input_data,
            test_input=test_input,
            deploy_params=deploy_params,
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_lightgbm_classifier(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = lightgbm.LGBMClassifier()
        classifier.fit(cal_X_train, cal_y_train)

        self.base_test_case(
            name="lightgbm_model",
            model=classifier,
            sample_input_data=cal_X_test,
            test_input=cal_X_test,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        res.values, np.expand_dims(classifier.predict(cal_X_test), axis=1)
                    ),
                ),
                "predict_proba": (
                    {},
                    lambda res: np.testing.assert_allclose(
                        res.values,
                        classifier.predict_proba(cal_X_test),
                    ),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_lightgbm_classifier_sp(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        classifier = lightgbm.LGBMClassifier()
        classifier.fit(cal_X_train, cal_y_train)

        y_df_expected = pd.concat(
            [
                cal_X_test.reset_index(drop=True),
                pd.DataFrame(classifier.predict(cal_X_test), columns=["output_feature_0"]),
            ],
            axis=1,
        )
        y_df_expected_proba = pd.concat(
            [
                cal_X_test.reset_index(drop=True),
                pd.DataFrame(classifier.predict_proba(cal_X_test), columns=["output_feature_0", "output_feature_1"]),
            ],
            axis=1,
        )

        cal_data_sp_df_train = self._session.create_dataframe(cal_X_train)
        cal_data_sp_df_test = self._session.create_dataframe(cal_X_test)
        self.base_test_case(
            name="lightgbm_model_sp",
            model=classifier,
            sample_input_data=cal_data_sp_df_train,
            test_input=cal_data_sp_df_test,
            deploy_params={
                "predict": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
                "predict_proba": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected_proba, check_dtype=False),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_lightgbm_booster(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        regressor = lightgbm.train({"objective": "regression"}, lightgbm.Dataset(cal_X_train, label=cal_y_train))
        y_pred = regressor.predict(cal_X_test)

        self.base_test_case(
            name="lightgbm_booster",
            model=regressor,
            sample_input_data=cal_X_test,
            test_input=cal_X_test,
            deploy_params={
                "predict": (
                    {},
                    lambda res: np.testing.assert_allclose(res.values, np.expand_dims(y_pred, axis=1), rtol=1e-6),
                ),
            },
            permanent_deploy=permanent_deploy,
        )

    @parameterized.product(permanent_deploy=[True, False])  # type: ignore[misc]
    def test_lightgbm_booster_sp(
        self,
        permanent_deploy: Optional[bool] = False,
    ) -> None:
        cal_data = datasets.load_breast_cancer(as_frame=True)
        cal_X = cal_data.data
        cal_y = cal_data.target
        cal_X.columns = [inflection.parameterize(c, "_") for c in cal_X.columns]
        cal_X_train, cal_X_test, cal_y_train, cal_y_test = model_selection.train_test_split(cal_X, cal_y)

        regressor = lightgbm.train({"objective": "regression"}, lightgbm.Dataset(cal_X_train, label=cal_y_train))
        y_df_expected = pd.concat(
            [
                cal_X_test.reset_index(drop=True),
                pd.DataFrame(regressor.predict(cal_X_test), columns=["output_feature_0"]),
            ],
            axis=1,
        )

        cal_data_sp_df_train = self._session.create_dataframe(cal_X_train)
        cal_data_sp_df_test = self._session.create_dataframe(cal_X_test)
        self.base_test_case(
            name="lightgbm_booster_sp",
            model=regressor,
            sample_input_data=cal_data_sp_df_train,
            test_input=cal_data_sp_df_test,
            deploy_params={
                "predict": (
                    {},
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, check_dtype=False),
                ),
            },
            permanent_deploy=permanent_deploy,
        )


if __name__ == "__main__":
    absltest.main()
