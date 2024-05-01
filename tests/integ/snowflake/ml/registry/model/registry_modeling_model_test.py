import os
import posixpath

import numpy as np
import yaml
from absl.testing import absltest
from sklearn import datasets

from snowflake.ml import dataset
from snowflake.ml.model._model_composer import model_composer
from snowflake.ml.modeling.lightgbm import LGBMRegressor
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.snowpark import types as T
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import test_env_utils


class TestRegistryModelingModelInteg(registry_model_test_base.RegistryModelTestBase):
    def test_snowml_model_deploy_snowml_sklearn(
        self,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X
        regr.fit(test_features)

        self._test_registry_model(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
        )

    def test_snowml_model_deploy_xgboost(
        self,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = XGBRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        self._test_registry_model(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
        )

    def test_snowml_model_deploy_lightgbm(
        self,
    ) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LGBMRegressor(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        test_features = iris_X[:10]
        regr.fit(test_features)

        self._test_registry_model(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    test_features,
                    lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(test_features)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
        )

    def test_dataset_to_model_lineage(self) -> None:
        iris_X = datasets.load_iris(as_frame=True).frame
        iris_X.columns = [s.replace(" (CM)", "").replace(" ", "") for s in iris_X.columns.str.upper()]

        INPUT_COLUMNS = ["SEPALLENGTH", "SEPALWIDTH", "PETALLENGTH", "PETALWIDTH"]
        LABEL_COLUMNS = "TARGET"
        OUTPUT_COLUMNS = "PREDICTED_TARGET"
        regr = LogisticRegression(input_cols=INPUT_COLUMNS, output_cols=OUTPUT_COLUMNS, label_cols=LABEL_COLUMNS)
        schema = [
            T.StructField("SEPALLENGTH", T.DoubleType()),
            T.StructField("SEPALWIDTH", T.DoubleType()),
            T.StructField("PETALLENGTH", T.DoubleType()),
            T.StructField("PETALWIDTH", T.DoubleType()),
            T.StructField("TARGET", T.StringType()),
            T.StructField("PREDICTED_TARGET", T.StringType()),
        ]
        test_features_df = self._session.create_dataframe(iris_X, schema=schema)

        test_features_dataset = dataset.create_from_dataframe(
            session=self._session,
            name="trainDataset",
            version="v1",
            input_dataframe=test_features_df,
        )

        test_df = test_features_dataset.read.to_snowpark_dataframe()

        regr.fit(test_df)

        # Case 1 : test generation of MANIFEST.yml file

        model_name = "some_name"
        tmp_stage_path = posixpath.join(self._session.get_session_stage(), f"{model_name}_{1}")
        conda_dependencies = [
            test_env_utils.get_latest_package_version_spec_in_server(self._session, "snowflake-snowpark-python!=1.12.0")
        ]
        mc = model_composer.ModelComposer(self._session, stage_path=tmp_stage_path)

        mc.save(
            name=model_name,
            model=regr,
            signatures=None,
            sample_input_data=None,
            conda_dependencies=conda_dependencies,
            metadata={"author": "rsureshbabu", "version": "1"},
            options={"relax_version": False},
        )

        with open(os.path.join(tmp_stage_path, mc._workspace.name, "MANIFEST.yml"), encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
            assert "lineage_sources" in yaml_content
            assert isinstance(yaml_content["lineage_sources"], list)
            assert len(yaml_content["lineage_sources"]) == 1

            source = yaml_content["lineage_sources"][0]
            assert isinstance(source, dict)
            assert source.get("type") == "DATASET"
            assert source.get("entity") == f"{test_features_dataset.fully_qualified_name}"
            assert source.get("version") == f"{test_features_dataset._version.name}"

        # Case 2 : test remaining life cycle.
        self._test_registry_model(
            model=regr,
            prediction_assert_fns={
                "predict": (
                    iris_X,
                    lambda res: lambda res: np.testing.assert_allclose(
                        res[OUTPUT_COLUMNS].values, regr.predict(iris_X)[OUTPUT_COLUMNS].values
                    ),
                ),
            },
            additional_dependencies=["fsspec", "aiohttp", "cryptography"],
        )


if __name__ == "__main__":
    absltest.main()
