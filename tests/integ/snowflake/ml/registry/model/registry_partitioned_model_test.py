from typing import Callable

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn.linear_model import LinearRegression

from snowflake.ml.model import custom_model
from tests.integ.snowflake.ml.registry.model import registry_model_test_base

_NUM_ROWS_PER_PARTITION = 20

# Output sizes for M:N tests (M = _NUM_ROWS_PER_PARTITION)
_OUTPUT_SIZE_EXPAND = 30  # M < N
_OUTPUT_SIZE_EQUAL = 20  # M = N
_OUTPUT_SIZE_COMPRESS = 5  # M > N


class StatelessPartitionedModel(custom_model.CustomModel):
    """A stateless partitioned model: M input rows → N output rows per partition (M < N)."""

    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(input_df[["FEATURE"]], input_df["TARGET"])
        feat_min, feat_max = input_df["FEATURE"].min(), input_df["FEATURE"].max()
        new_feature_values = np.linspace(feat_min, feat_max, _OUTPUT_SIZE_EXPAND)
        new_features = pd.DataFrame({"FEATURE": new_feature_values})
        preds = model.predict(new_features[["FEATURE"]])
        return pd.DataFrame({"NEW_FEATURE": new_features["FEATURE"], "PREDICTION": preds})


class StatefulPartitionedModelExpand(custom_model.CustomModel):
    """A stateful partitioned model: M input rows → N output rows per partition (M < N)."""

    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        partition_id = str(int(input_df["PARTITION_COL"].iloc[0]))
        model = self.context.model_ref(partition_id)
        feat_min, feat_max = input_df["FEATURE"].min(), input_df["FEATURE"].max()
        new_feature_values = np.linspace(feat_min, feat_max, _OUTPUT_SIZE_EXPAND)
        new_features = pd.DataFrame({"FEATURE": new_feature_values})
        preds = model.predict(new_features[["FEATURE"]])
        return pd.DataFrame({"NEW_FEATURE": new_features["FEATURE"], "PREDICTION": preds})


class StatefulPartitionedModelEqual(custom_model.CustomModel):
    """A stateful partitioned model: M input rows → N output rows per partition (M = N)."""

    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        partition_id = str(int(input_df["PARTITION_COL"].iloc[0]))
        model = self.context.model_ref(partition_id)
        feat_min, feat_max = input_df["FEATURE"].min(), input_df["FEATURE"].max()
        new_feature_values = np.linspace(feat_min, feat_max, _OUTPUT_SIZE_EQUAL)
        new_features = pd.DataFrame({"FEATURE": new_feature_values})
        preds = model.predict(new_features[["FEATURE"]])
        return pd.DataFrame({"NEW_FEATURE": new_features["FEATURE"], "PREDICTION": preds})


class StatefulPartitionedModelCompress(custom_model.CustomModel):
    """A stateful partitioned model: M input rows → N output rows per partition (M > N)."""

    @custom_model.partitioned_api
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        partition_id = str(int(input_df["PARTITION_COL"].iloc[0]))
        model = self.context.model_ref(partition_id)
        feat_min, feat_max = input_df["FEATURE"].min(), input_df["FEATURE"].max()
        new_feature_values = np.linspace(feat_min, feat_max, _OUTPUT_SIZE_COMPRESS)
        new_features = pd.DataFrame({"FEATURE": new_feature_values})
        preds = model.predict(new_features[["FEATURE"]])
        return pd.DataFrame({"NEW_FEATURE": new_features["FEATURE"], "PREDICTION": preds})


def _generate_partitioned_data() -> pd.DataFrame:
    """Generate a synthetic dataset with two partitions (PARTITION_COL=1 and PARTITION_COL=2)."""
    np.random.seed(42)
    rows = []
    for partition_id in [1, 2]:
        feature = np.random.rand(_NUM_ROWS_PER_PARTITION) * 10
        target = partition_id * 2.0 * feature + partition_id * 3.0 + np.random.randn(_NUM_ROWS_PER_PARTITION) * 0.1
        for f, t in zip(feature, target):
            rows.append({"PARTITION_COL": partition_id, "FEATURE": f, "TARGET": t})
    return pd.DataFrame(rows)


class TestRegistryPartitionedModelInteg(registry_model_test_base.RegistryModelTestBase):
    """Integration tests for partitioned custom models with M:N input/output ratios."""

    def _fit_models(self, input_df: pd.DataFrame) -> dict[str, LinearRegression]:
        """Pre-fit a LinearRegression model for each partition."""
        fitted_models: dict[str, LinearRegression] = {}
        for partition_id in [1, 2]:
            partition_data = input_df[input_df["PARTITION_COL"] == partition_id]
            lr = LinearRegression()
            lr.fit(partition_data[["FEATURE"]], partition_data["TARGET"])
            fitted_models[str(partition_id)] = lr
        return fitted_models

    def _create_check_fn(
        self,
        input_df: pd.DataFrame,
        output_size: int,
        fitted_models: dict[str, LinearRegression] | None = None,
    ) -> Callable[[pd.DataFrame], None]:
        """Create a check function that verifies partitioned model output."""

        def check_res(res: pd.DataFrame) -> None:
            self.assertIsInstance(res, pd.DataFrame)
            self.assertIn("PREDICTION", res.columns)
            self.assertIn("NEW_FEATURE", res.columns)
            num_partitions = input_df["PARTITION_COL"].nunique()
            self.assertEqual(len(res), num_partitions * output_size)
            for partition_id in [1, 2]:
                partition_data = input_df[input_df["PARTITION_COL"] == partition_id]
                if fitted_models is not None:
                    lr = fitted_models[str(partition_id)]
                else:
                    lr = LinearRegression()
                    lr.fit(partition_data[["FEATURE"]], partition_data["TARGET"])
                feat_min, feat_max = partition_data["FEATURE"].min(), partition_data["FEATURE"].max()
                new_feature_values = np.linspace(feat_min, feat_max, output_size)
                expected = np.sort(lr.predict(new_feature_values.reshape(-1, 1)))
                actual = np.sort(res[res["PARTITION_COL"] == partition_id]["PREDICTION"].astype(float).values)
                np.testing.assert_allclose(actual, expected, rtol=1e-5)

        return check_res

    def test_stateless_partitioned_model(self) -> None:
        """Test stateless partitioned model: M < N."""
        input_df = _generate_partitioned_data()
        self._test_registry_model(
            model=StatelessPartitionedModel(custom_model.ModelContext()),
            prediction_assert_fns={"PREDICT": (input_df, self._create_check_fn(input_df, _OUTPUT_SIZE_EXPAND))},
            sample_input_data=input_df,
            additional_dependencies=["scikit-learn"],
            options={"function_type": "TABLE_FUNCTION"},
            partition_column="PARTITION_COL",
        )

    def test_stateful_partitioned_model_expand(self) -> None:
        """Test stateful partitioned model: M < N."""
        input_df = _generate_partitioned_data()
        fitted_models = self._fit_models(input_df)
        self._test_registry_model(
            model=StatefulPartitionedModelExpand(custom_model.ModelContext(models=fitted_models)),
            prediction_assert_fns={
                "PREDICT": (input_df, self._create_check_fn(input_df, _OUTPUT_SIZE_EXPAND, fitted_models))
            },
            sample_input_data=input_df,
            additional_dependencies=["scikit-learn"],
            options={"function_type": "TABLE_FUNCTION"},
            partition_column="PARTITION_COL",
        )

    def test_stateful_partitioned_model_equal(self) -> None:
        """Test stateful partitioned model: M = N."""
        input_df = _generate_partitioned_data()
        fitted_models = self._fit_models(input_df)
        self._test_registry_model(
            model=StatefulPartitionedModelEqual(custom_model.ModelContext(models=fitted_models)),
            prediction_assert_fns={
                "PREDICT": (input_df, self._create_check_fn(input_df, _OUTPUT_SIZE_EQUAL, fitted_models))
            },
            sample_input_data=input_df,
            additional_dependencies=["scikit-learn"],
            options={"function_type": "TABLE_FUNCTION"},
            partition_column="PARTITION_COL",
        )

    def test_stateful_partitioned_model_compress(self) -> None:
        """Test stateful partitioned model: M > N."""
        input_df = _generate_partitioned_data()
        fitted_models = self._fit_models(input_df)
        self._test_registry_model(
            model=StatefulPartitionedModelCompress(custom_model.ModelContext(models=fitted_models)),
            prediction_assert_fns={
                "PREDICT": (input_df, self._create_check_fn(input_df, _OUTPUT_SIZE_COMPRESS, fitted_models))
            },
            sample_input_data=input_df,
            additional_dependencies=["scikit-learn"],
            options={"function_type": "TABLE_FUNCTION"},
            partition_column="PARTITION_COL",
        )


if __name__ == "__main__":
    absltest.main()
