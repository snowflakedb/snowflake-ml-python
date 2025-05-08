import uuid
from typing import cast

import altair as alt
import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.monitoring.explain_visualize import (
    plot_force,
    plot_influence_sensitivity,
    plot_violin,
)
from tests.integ.snowflake.ml.test_utils import common_test_base, db_manager


class ExplainVisualizeTest(common_test_base.CommonTestBase):
    def _setup_data(self):
        np.random.seed(42)
        n_samples = 10

        # Generate feature values
        feature_1 = np.random.uniform(-5, 5, n_samples)
        feature_2 = np.random.uniform(-3, 7, n_samples)
        feature_3 = np.random.uniform(0, 10, n_samples)

        feat_df = pd.DataFrame(
            {
                "Feature_1": feature_1,
                "Feature_2": feature_2,
                "Feature_3": feature_3,
            }
        )

        # Generate influence values
        feature_1_influence = feature_1 * 0.5 + np.random.normal(0, 0.5, n_samples)
        feature_2_influence = feature_2 * -0.3 + np.random.normal(0, 0.4, n_samples)
        feature_3_influence = feature_3 * 0.2 + np.random.normal(0, 0.3, n_samples)

        shap_df = pd.DataFrame(
            {
                "Feature_1_influence": feature_1_influence,
                "Feature_2_influence": feature_2_influence,
                "Feature_3_influence": feature_3_influence,
            }
        )

        self.shap_df = shap_df.round(3)
        self.feat_df = feat_df.round(3)

        self.shap_df_snowpark = self.session.create_dataframe(self.shap_df)
        self.feat_df_snowpark = self.session.create_dataframe(self.feat_df)

        self.shap_df_numpy = self.shap_df.to_numpy()
        self.feat_df_numpy = self.feat_df.to_numpy()

    def setUp(self) -> None:
        super().setUp()

        self.run_id = uuid.uuid4().hex
        self._db_manager = db_manager.DBManager(self.session)
        self._schema_name = "PUBLIC"
        self._db_name = db_manager.TestObjectNameGenerator.get_snowml_test_object_name(
            self.run_id, "TEST_EXPLAIN_VISUALIZE"
        ).upper()
        self._db_manager.create_database(self._db_name)

        self._db_manager.cleanup_databases(expire_hours=6)
        self._setup_data()

    def _assert_plot_columns(self, plot: alt.LayerChart, expected_columns: list[str]) -> None:
        assert plot is not None
        plot_data: pd.DataFrame = cast(pd.DataFrame, plot.data)
        self.assertEqual(
            expected_columns,
            list(plot_data.columns),
        )

    def test_plot_force(self) -> None:
        plot: alt.LayerChart = plot_force(self.shap_df.iloc[0], self.feat_df.iloc[0])
        self._assert_plot_columns(
            plot,
            [
                "start",
                "end",
                "avg",
                "influence_value",
                "influence_annotated",
                "feature_value",
                "feature_annotated",
                "bar_direction",
            ],
        )

    def test_plot_force_snowpark(self) -> None:
        shap_row = self.shap_df_snowpark.collect()[0]
        feat_row = self.feat_df_snowpark.collect()[0]

        plot: alt.LayerChart = plot_force(shap_row, feat_row)
        self._assert_plot_columns(
            plot,
            [
                "start",
                "end",
                "avg",
                "influence_value",
                "influence_annotated",
                "feature_value",
                "feature_annotated",
                "bar_direction",
            ],
        )

    def test_plot_influence_sensitivity(self) -> None:
        plot = plot_influence_sensitivity(self.shap_df.iloc[0], self.feat_df.iloc[0])
        self._assert_plot_columns(
            plot,
            ["feature_value", "shap_value"],
        )

    def test_plot_violin(self) -> None:
        for test_shap_df, test_feat_df in [
            (self.shap_df, self.feat_df),
            (self.shap_df_snowpark, self.feat_df_snowpark),
            (self.shap_df_numpy, self.feat_df_numpy),
        ]:
            plot = plot_violin(test_shap_df, test_feat_df)
            self._assert_plot_columns(
                plot,
                ["feature_name", "shap_value"],
            )


if __name__ == "__main__":
    absltest.main()
