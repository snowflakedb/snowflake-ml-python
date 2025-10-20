import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from xgboost import XGBClassifier as SkXGBClassifier

from snowflake.ml.modeling.ensemble import RandomForestClassifier
from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session


class MultiLabelTargetTest(TestCase):
    def _load_data(self):
        X, y = make_multilabel_classification(n_samples=32, n_classes=5, n_labels=3, random_state=0)
        self.feature_cols = [f"FEATURE_{i+1}" for i in range(len(X[0]))]
        self.target_cols = [f"TARGET_{i+1}" for i in range(len(y[0]))]
        self.mult_cl_data = np.concatenate((X, y), axis=1)

    def setUp(self):
        """Creates Snowpark and Snowflake environments for testing."""
        self._session = Session.builder.configs(SnowflakeLoginOptions()).create()
        self._load_data()

    def tearDown(self):
        self._session.close()

    def test_random_forest_regressor_with_five_label_cols(self) -> None:
        # Build a pandas DF with a stable row id, then create Snowpark DF from it
        pd_df = pd.DataFrame(self.mult_cl_data, columns=self.feature_cols + self.target_cols)
        pd_df.insert(0, "ROW_ID", np.arange(len(pd_df), dtype=np.int64))

        snf_df = self._session.create_dataframe(pd_df)
        snf_df.write.save_as_table("multi_target_cl", mode="overwrite")
        multi_target_cl_df = self._session.table("multi_target_cl").sort("ROW_ID")
        multi_target_cl_df_pd = multi_target_cl_df.to_pandas()

        cl_multi_model = RandomForestClassifier(
            input_cols=self.feature_cols, label_cols=self.target_cols, random_state=0, n_jobs=1
        )
        native_classifier = SkRandomForestClassifier(random_state=0, n_jobs=1)
        cl_multi_model.fit(multi_target_cl_df)
        native_classifier.fit(multi_target_cl_df_pd[self.feature_cols], multi_target_cl_df_pd[self.target_cols])

        assert cl_multi_model.get_output_cols() == [
            "OUTPUT_TARGET_1",
            "OUTPUT_TARGET_2",
            "OUTPUT_TARGET_3",
            "OUTPUT_TARGET_4",
            "OUTPUT_TARGET_5",
        ]

        training_predictions_df = cl_multi_model.predict_proba(multi_target_cl_df).to_pandas().sort_values("ROW_ID")
        PREDICT_PROBA_COLS_SP = []
        for c in training_predictions_df.columns:
            if "PREDICT_PROBA" in c:
                PREDICT_PROBA_COLS_SP.append(c)
        # training_predictions shape: (32, 10)
        training_predictions = training_predictions_df[PREDICT_PROBA_COLS_SP].to_numpy()
        # native_predictions shape(5, 32, 2)
        native_predictions = np.array(native_classifier.predict_proba(multi_target_cl_df_pd[self.feature_cols]))
        # native_predictions: Concatenate along the last axis (axis=2)
        # Since we want to join every (32, 2) across the first axis, resulting in (32, 10)
        concatenated_array = np.concatenate([native_predictions[i] for i in range(native_predictions.shape[0])], axis=1)

        np.testing.assert_allclose(
            training_predictions.flatten(), concatenated_array.flatten(), rtol=1.0e-1, atol=1.0e-2
        )

        training_predictions_log_proba_df = (
            cl_multi_model.predict_log_proba(multi_target_cl_df).to_pandas().sort_values("ROW_ID")
        )
        PREDICT_LOG_PROBA_COLS_SP = []
        for c in training_predictions_log_proba_df.columns:
            if "PREDICT_LOG_PROBA" in c:
                PREDICT_LOG_PROBA_COLS_SP.append(c)
        # training_predictions shape: (32, 10)
        training_predictions_log_proba = training_predictions_log_proba_df[PREDICT_LOG_PROBA_COLS_SP].to_numpy()
        # native_predictions shape: (5, 32, 2)
        native_predictions_log_proba = np.array(
            native_classifier.predict_log_proba(multi_target_cl_df_pd[self.feature_cols])
        )
        # native_predictions_log_proba: Concatenate along the last axis (axis=2)
        # Since we want to join every (32, 2) across the first axis, resulting in (32, 10)
        concatenated_array_log_proba = np.concatenate(
            [native_predictions_log_proba[i] for i in range(native_predictions_log_proba.shape[0])], axis=1
        )

        np.testing.assert_allclose(
            training_predictions_log_proba.flatten(), concatenated_array_log_proba.flatten(), rtol=1.0e-1, atol=1.0e-2
        )

    def test_xgb_regressor_with_five_label_cols(self) -> None:
        # Build a pandas DF with a stable row id, then create Snowpark DF from it
        pd_df = pd.DataFrame(self.mult_cl_data, columns=self.feature_cols + self.target_cols)
        pd_df.insert(0, "ROW_ID", np.arange(len(pd_df), dtype=np.int64))

        snf_df = self._session.create_dataframe(pd_df)
        snf_df.write.save_as_table("multi_target_cl", mode="overwrite")
        multi_target_cl_df = self._session.table("multi_target_cl").sort("ROW_ID")
        multi_target_cl_df_pd = multi_target_cl_df.to_pandas()

        cl_multi_model = XGBClassifier(
            input_cols=self.feature_cols, label_cols=self.target_cols, tree_method="hist", random_state=42
        )
        cl_multi_model.fit(multi_target_cl_df)
        assert cl_multi_model.get_output_cols() == [
            "OUTPUT_TARGET_1",
            "OUTPUT_TARGET_2",
            "OUTPUT_TARGET_3",
            "OUTPUT_TARGET_4",
            "OUTPUT_TARGET_5",
        ]

        native_classifier = SkXGBClassifier(tree_method="hist", random_state=42)
        native_classifier.fit(multi_target_cl_df_pd[self.feature_cols], multi_target_cl_df_pd[self.target_cols])

        training_predictions_sp_input = cl_multi_model.predict_proba(multi_target_cl_df_pd)
        training_predictions_pd_input = (
            cl_multi_model.predict_proba(multi_target_cl_df).to_pandas().sort_values("ROW_ID")
        )
        PREDICT_PROBA_COLS_SP = []
        for c in training_predictions_sp_input.columns:
            if "PREDICT_PROBA" in c:
                PREDICT_PROBA_COLS_SP.append(c)
        # training_predictions shape: (32, 5)
        training_predictions = training_predictions_sp_input.sort_values("ROW_ID")[PREDICT_PROBA_COLS_SP].to_numpy()
        training_predictions_pd_input = training_predictions_pd_input[PREDICT_PROBA_COLS_SP].to_numpy()
        # native_predictions shape: (32, 5)
        native_predictions = native_classifier.predict_proba(multi_target_cl_df_pd[self.feature_cols])
        # Compute native predictions using the same row order as Snowpark -> pandas path
        native_predictions_sp_order = native_classifier.predict_proba(
            multi_target_cl_df.to_pandas().sort_values("ROW_ID")[self.feature_cols]
        )

        np.testing.assert_allclose(
            training_predictions.flatten(), native_predictions.flatten(), rtol=1.0e-1, atol=1.0e-2
        )
        np.testing.assert_allclose(
            training_predictions_pd_input.flatten(), native_predictions_sp_order.flatten(), rtol=1.0e-1, atol=1.0e-2
        )


if __name__ == "__main__":
    main()
