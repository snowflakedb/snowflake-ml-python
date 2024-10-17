import os
import posixpath

import numpy as np
import shap
import yaml
from absl.testing import absltest, parameterized
from sklearn import datasets

from snowflake.ml import dataset
from snowflake.ml._internal.utils import identifier
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from snowflake.ml.modeling.lightgbm import LGBMRegressor
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.snowpark import types as T
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils, test_env_utils


class TestRegistryModelingModelInteg(registry_model_test_base.RegistryModelTestBase):
    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_snowml_model_deploy_snowml_sklearn_explain_disabled(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        getattr(self, registry_test_fn)(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_snowml_model_deploy_snowml_sklearn_explain_default(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        EXPLAIN_OUTPUT_COLUMNS = [identifier.concat_names([feature, "_explanation"]) for feature in INPUT_COLUMNS]
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        test_data = test_features[INPUT_COLUMNS]
        expected_explanations = shap.Explainer(regr.to_sklearn(), masker=test_data)(test_data).values

        getattr(self, registry_test_fn)(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
                "explain": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        dataframe_utils.convert2D_json_to_3D(res[EXPLAIN_OUTPUT_COLUMNS].values),
                        expected_explanations,
                        rtol=1e-4,
                    ),
                ),
            },
            sample_input_data=test_data,
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_snowml_model_deploy_snowml_sklearn_explain_enabled(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        EXPLAIN_OUTPUT_COLUMNS = [identifier.concat_names([feature, "_explanation"]) for feature in INPUT_COLUMNS]
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        test_data = test_features[INPUT_COLUMNS]
        expected_explanations = shap.Explainer(regr.to_sklearn(), masker=test_data)(test_data).values
        getattr(self, registry_test_fn)(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
                "explain": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        dataframe_utils.convert2D_json_to_3D(res[EXPLAIN_OUTPUT_COLUMNS].values),
                        expected_explanations,
                        rtol=1e-4,
                    ),
                ),
            },
            sample_input_data=test_data,
            options={"enable_explainability": True},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_snowml_model_deploy_xgboost_explain_disabled(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        getattr(self, registry_test_fn)(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_snowml_model_deploy_xgboost_explain_default(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        PRED_OUTPUT_COLUMNS = "PREDICTED_TARGET"
        EXPLAIN_OUTPUT_COLUMNS = [identifier.concat_names([feature, "_explanation"]) for feature in INPUT_COLUMNS]

        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=PRED_OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        expected_explanations = shap.Explainer(regr.to_xgboost())(test_features[INPUT_COLUMNS]).values

        getattr(self, registry_test_fn)(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[PRED_OUTPUT_COLUMNS].values, regr.predict(test_features)[PRED_OUTPUT_COLUMNS].values
                    ),
                ),
                "explain": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[EXPLAIN_OUTPUT_COLUMNS].values, expected_explanations, rtol=1e-4
                    ),
                ),
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_snowml_model_deploy_xgboost_explain_enabled(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        PRED_OUTPUT_COLUMNS = "PREDICTED_TARGET"
        EXPLAIN_OUTPUT_COLUMNS = [identifier.concat_names([feature, "_explanation"]) for feature in INPUT_COLUMNS]

        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=PRED_OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        expected_explanations = shap.Explainer(regr.to_xgboost())(test_features[INPUT_COLUMNS]).values

        getattr(self, registry_test_fn)(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[PRED_OUTPUT_COLUMNS].values, regr.predict(test_features)[PRED_OUTPUT_COLUMNS].values
                    ),
                ),
                "explain": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[EXPLAIN_OUTPUT_COLUMNS].values, expected_explanations, rtol=1e-4
                    ),
                ),
            },
            options={"enable_explainability": True},
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_snowml_model_deploy_lightgbm_explain_disabled(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LGBMRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        getattr(self, registry_test_fn)(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            options={"enable_explainability": False},
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_snowml_model_deploy_lightgbm_explain_default(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        PRED_OUTPUT_COLUMNS = "PREDICTED_TARGET"
        EXPLAIN_OUTPUT_COLUMNS = [identifier.concat_names([feature, "_explanation"]) for feature in INPUT_COLUMNS]
        regr = LGBMRegressor(input_cols=INPUT_COLUMNS, output_cols=PRED_OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        expected_explanations = shap.Explainer(regr.to_lightgbm())(test_features[INPUT_COLUMNS]).values

        getattr(self, registry_test_fn)(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[PRED_OUTPUT_COLUMNS].values, regr.predict(test_features)[PRED_OUTPUT_COLUMNS].values
                    ),
                ),
                "explain": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[EXPLAIN_OUTPUT_COLUMNS].values,
                        expected_explanations,
                        rtol=1e-5,
                    ),
                ),
            },
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
    )
    def test_snowml_model_deploy_lightgbm_explain_enabled(
        self,
        registry_test_fn: str,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        PRED_OUTPUT_COLUMNS = "PREDICTED_TARGET"
        EXPLAIN_OUTPUT_COLUMNS = [identifier.concat_names([feature, "_explanation"]) for feature in INPUT_COLUMNS]
        regr = LGBMRegressor(input_cols=INPUT_COLUMNS, output_cols=PRED_OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        expected_explanations = shap.Explainer(regr.to_lightgbm())(test_features[INPUT_COLUMNS]).values

        getattr(self, registry_test_fn)(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[PRED_OUTPUT_COLUMNS].values, regr.predict(test_features)[PRED_OUTPUT_COLUMNS].values
                    ),
                ),
                "explain": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[EXPLAIN_OUTPUT_COLUMNS].values,
                        expected_explanations,
                        rtol=1e-5,
                    ),
                ),
            },
            options={"enable_explainability": True},
            function_type_assert={
                "explain": model_manifest_schema.ModelMethodFunctionTypes.TABLE_FUNCTION,
                "predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION,
            },
        )

    @parameterized.product(  # type: ignore[misc]
        registry_test_fn=registry_model_test_base.RegistryModelTestBase.REGISTRY_TEST_FN_LIST,
        use_pipeline=[False, True],
    )
    def test_dataset_to_model_lineage(
        self,
        registry_test_fn: str,
        use_pipeline: bool = False,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        if use_pipeline:
            regr = Pipeline([("regr", regr)])
        schema = T.StructType(
            [
                T.StructField("SEPALLENGTH", T.DoubleType()),
                T.StructField("SEPALWIDTH", T.DoubleType()),
                T.StructField("PETALLENGTH", T.DoubleType()),
                T.StructField("PETALWIDTH", T.DoubleType()),
                T.StructField("TARGET", T.StringType()),
                T.StructField("PREDICTED_TARGET", T.StringType()),
            ]
        )
        test_features_df = self.session.create_dataframe(iris_X, schema=schema)

        test_features_df.write.mode("overwrite").save_as_table("testTable")

        test_features_dataset = dataset.create_from_dataframe(
            session=self.session,
            name="trainDataset",
            version="v1",
            input_dataframe=test_features_df,
        )

        # Case 1 : Capture Lineage via fit() API of MANIFEST.yml file
        test_df = test_features_dataset.read.to_snowpark_dataframe()

        regr.fit(test_df)
        self._check_lineage_in_manifest_file(regr, test_features_dataset)

        # Case 2 : test remaining life cycle.
        getattr(self, registry_test_fn)(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    iris_X,
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(iris_X)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            additional_dependencies=["fsspec", "aiohttp", "cryptography"],
        )

        # Case 3 : Capture Lineage via sample_input of log_model of MANIFEST.yml file
        pandas_df = test_features_dataset.read.to_pandas()

        regr.fit(pandas_df)
        self._check_lineage_in_manifest_file(
            regr, test_features_dataset, sample_input_data=test_features_dataset.read.to_snowpark_dataframe()
        )

        # Case 4 : Dont capture lineage of if its not passed via with fit() API or sample_input
        pandas_df = test_features_dataset.read.to_pandas()

        regr.fit(pandas_df)
        self._check_lineage_in_manifest_file(
            regr, test_features_dataset, sample_input_data=pandas_df, lineage_should_exist=False
        )

        # Case 5 : Capture Lineage via fit() API of MANIFEST.yml file
        table_backed_dataframe = self.session.table("testTable")
        regr.fit(table_backed_dataframe)
        self._check_lineage_in_manifest_file(regr, "testTable", is_dataset=False)

        # Case 6 : Capture Lineage via sample_input of log_model of MANIFEST.yml file
        regr.fit(table_backed_dataframe.to_pandas())
        self._check_lineage_in_manifest_file(
            regr, "testTable", is_dataset=False, sample_input_data=table_backed_dataframe
        )

    def _check_lineage_in_manifest_file(
        self, model, data_source, is_dataset=True, sample_input_data=None, lineage_should_exist=True
    ):
        model_name = "some_name"
        tmp_stage_path = posixpath.join(self.session.get_session_stage(), f"{model_name}_{1}")
        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self.session, "snowflake-snowpark-python!=1.12.0")
        ]
        mc = model_composer.ModelComposer(self.session, stage_path=tmp_stage_path)

        mc.save(
            name=model_name,
            model=model,
            signatures=None,
            sample_input_data=sample_input_data,
            conda_dependencies=conda_dependencies,
            metadata={"author": "rsureshbabu", "version": "2"},
            options={"relax_version": False},
        )

        with open(os.path.join(tmp_stage_path, mc._workspace.name, "MANIFEST.yml"), encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
            if lineage_should_exist:
                assert "lineage_sources" in yaml_content
                assert isinstance(yaml_content["lineage_sources"], list)
                assert len(yaml_content["lineage_sources"]) == 1

                source = yaml_content["lineage_sources"][0]
                assert isinstance(source, dict)
                if is_dataset:
                    assert source.get("type") == "DATASET"
                    assert source.get("entity") == f"{data_source.fully_qualified_name}"
                    assert source.get("version") == f"{data_source._version.name}"
                else:
                    assert source.get("type") == "QUERY"
                    assert data_source in source.get("entity")
            else:
                assert "lineage_sources" not in yaml_content


if __name__ == "__main__":
    absltest.main()
