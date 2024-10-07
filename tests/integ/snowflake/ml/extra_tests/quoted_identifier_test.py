import os
from unittest import skipIf

import numpy as np
from absl.testing import absltest, parameterized
from scipy.stats import randint

from snowflake.ml._internal.env_utils import SNOWML_SPROC_ENV
from snowflake.ml.modeling.compose import ColumnTransformer
from snowflake.ml.modeling.ensemble import RandomForestClassifier
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.model_selection import RandomizedSearchCV
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from snowflake.ml.registry import Registry
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


class QuotedIdentifierTest(parameterized.TestCase):
    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()

    def tearDown(self):
        self._session.close()

    @skipIf(
        os.getenv("IN_SPCS_ML_RUNTIME") == "True",
        "Skipping this test on Container Runtimes. See: https://snowflakecomputing.atlassian.net/browse/SNOW-1633651",
    )
    @parameterized.parameters(False, True)
    def test_sp_quoted_identifier_modeling(self, test_within_sproc) -> None:
        if test_within_sproc:
            os.environ[SNOWML_SPROC_ENV] = "True"

        # dataset's columns names are intentionally set as double quotes to maintain lower cased characters
        CATEGORICAL_COLUMNS = ['"cut"', '"color"', '"clarity"']
        NUMERICAL_COLUMNS = ['"carat"', '"depth"', '"x"', '"y"', '"z"']
        LABEL_COLUMNS = ['"price"']
        sp_df = self._session.create_dataframe(
            np.concatenate(
                [
                    np.random.randint(0, 10, size=(1000, len(CATEGORICAL_COLUMNS))),
                    np.random.random(size=(1000, len(NUMERICAL_COLUMNS))),
                    np.random.randint(0, 1, size=(1000, 1)),
                ],
                axis=1,
            ).tolist(),
            schema=CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + LABEL_COLUMNS,
        )
        self.assertListEqual(sp_df.columns, CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS + LABEL_COLUMNS)

        feature_columns = [
            {"name": '"carat"', "encoding_rescaling": "Standard rescaling", "missingness_impute": "Average"},
            {"name": '"cut"', "encoding_rescaling": "Dummy encoding", "missingness_impute": "Most frequent value"},
            {"name": '"color"', "encoding_rescaling": "Ordinal encoding", "missingness_impute": "Most frequent value"},
            {"name": '"depth"', "encoding_rescaling": "Standard rescaling", "missingness_impute": "Median"},
            {"name": '"x"', "encoding_rescaling": "Standard rescaling", "missingness_impute": "Average"},
            {"name": '"y"', "encoding_rescaling": "Standard rescaling", "missingness_impute": "Median"},
            {"name": '"z"', "encoding_rescaling": "Standard rescaling", "missingness_impute": "Median"},
            {"name": '"clarity"', "encoding_rescaling": "Dummy encoding", "missingness_impute": "Most frequent value"},
        ]

        ALL_FEATURE_COLUMNS = [feature["name"] for feature in feature_columns]

        diamonds_train_df, test_df = sp_df.random_split(weights=[0.9, 0.1], seed=0)

        col_transformer_list = []

        for feature in feature_columns:
            feature_name = feature["name"]
            transformer_name = feature_name[1:-1] + "_tform"

            feature_transformers = []

            if feature["missingness_impute"] == "Average":
                feature_transformers.append(("imputer", SimpleImputer(strategy="mean")))
            if feature["missingness_impute"] == "Median":
                feature_transformers.append(("imputer", SimpleImputer(strategy="median")))
            if feature["missingness_impute"] == "Constant":
                if "constant_impute" in feature:
                    feature_transformers.append(
                        ("imputer", SimpleImputer(strategy="constant", fill_value=feature["constant_impute"]))
                    )
                else:
                    feature_transformers.append(("imputer", SimpleImputer(strategy="constant")))
            if feature["missingness_impute"] == "Most frequent value":
                feature_transformers.append(("imputer", SimpleImputer(strategy="most_frequent")))
            if feature["encoding_rescaling"] == "Standard rescaling":
                feature_transformers.append(("enc", StandardScaler()))
            if feature["encoding_rescaling"] == "Min-max rescaling":
                feature_transformers.append(("enc", MinMaxScaler()))
            if feature["encoding_rescaling"] == "Dummy encoding":
                feature_transformers.append(
                    ("enc", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=10))
                )
            if feature["encoding_rescaling"] == "Ordinal encoding":
                feature_transformers.append(
                    (
                        "enc",
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
                    )
                )
            col_transformer_list.append((transformer_name, Pipeline(feature_transformers), [feature_name]))

        preprocessor = ColumnTransformer(transformers=col_transformer_list)

        rf_algo = {
            "algorithm": "random_forest_classification",
            "sklearn_obj": RandomForestClassifier(),
            "gs_params": {
                "clf__n_estimators": randint(50, 250),
                "clf__max_depth": randint(3, 7),
                "clf__min_samples_leaf": randint(2, 5),
            },
        }

        def train_model(algo, prepr, score_met, col_lab, feat_names, train_sp_df, cross_val, num_iter):
            pipe = Pipeline(steps=[("preprocessor", prepr), ("clf", algo["sklearn_obj"])])

            random_search_clf = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=algo["gs_params"],
                n_iter=num_iter,
                cv=cross_val,
                scoring=score_met,
                n_jobs=-1,
                verbose=1,
                input_cols=feat_names,
                label_cols=col_lab,
                output_cols="PREDICTION",
            )

            random_search_clf.fit(train_sp_df)

            return {"algorithm": algo["algorithm"], "trained_model": random_search_clf}

        trained_rf_model = train_model(
            rf_algo, preprocessor, "roc_auc", LABEL_COLUMNS, ALL_FEATURE_COLUMNS, diamonds_train_df, 3, 4
        )

        trained_model = trained_rf_model["trained_model"]

        trained_model.predict(test_df)

        registry = Registry(session=self._session)
        snowflake_registry_model_description = "Example on model registry"
        snowflake_model_name = f"Registry_Model_{os.urandom(9).hex().upper()}"

        registry.log_model(
            model=trained_model,
            model_name=snowflake_model_name,
            version_name="v1",
            comment=snowflake_registry_model_description,
        )

        # Need to set tags at the parent model level
        parent_model = registry.get_model(snowflake_model_name)

        # Need to create the tag object in Snowflake if it doesn't exist
        self._session.sql("CREATE TAG IF NOT EXISTS APPLICATION;").collect()
        self._session.sql("CREATE TAG IF NOT EXISTS DATAIKU_PROJECT_KEY;").collect()
        self._session.sql("CREATE TAG IF NOT EXISTS DATAIKU_SAVED_MODEL_ID;").collect()

        parent_model.set_tag("application", "Dataiku")

        model_for_scoring = registry.get_model(snowflake_model_name).version("v1")

        predictions = model_for_scoring.run(test_df, function_name="predict_proba")

        predictions.to_pandas()

        registry.delete_model(snowflake_model_name)


if __name__ == "__main__":
    absltest.main()
