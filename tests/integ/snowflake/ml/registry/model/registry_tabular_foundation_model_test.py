import os
import tempfile

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets, model_selection

from snowflake.ml.model._model_composer.model_manifest import model_manifest_schema
from tests.integ.snowflake.ml.registry.model import registry_model_test_base

# tabicl is imported lazily in the test body and loads numba code compiled with @njit(cache=True).
# numba's default on-disk cache sits next to the installed package, which is read-only under the Bazel
# test sandbox, so the import otherwise fails with "no locator available". Point numba at a writable dir.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()), "numba_cache"),
)

# tabicl downloads its foundation checkpoint via hf_hub_download, which stages files under
# $HOME/.cache/huggingface by default -- read-only in the Bazel test sandbox, so makedirs fails before
# the mirror is even contacted. Point the HF cache at a writable dir (the HF endpoint is set separately
# by .bazelrc to the Artifactory mirror).
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()), "hf_home"),
)

# tabicl is PyPI-only; serve it in the warehouse through the shared PyPI artifact repository.
_PIP_ARTIFACT_REPOSITORY_MAP = {"pip": "snowflake.snowpark.pypi_shared_repository"}
_TARGET_COLUMN = "TARGET"
_PREDICTION_COLUMN = "PREDICTION"
# A fitted estimator is served bound: a non-partitioned scalar predict, not a table function.
_BOUND_PREDICT_ASSERTS = {
    "is_partitioned": {"predict": False},
    "function_type": {"predict": model_manifest_schema.ModelMethodFunctionTypes.FUNCTION},
}


class TestRegistryTabularFoundationModelInteg(registry_model_test_base.RegistryModelTestBase):
    """End-to-end bound serving of a fitted TabICL estimator in the warehouse.

    A fitted TabICL estimator is served *bound*: its training frame is packaged as the context, the
    estimator is re-fit on load, and ``predict`` is a plain scalar function over named feature columns.
    """

    def test_tabicl_bound_classifier(self) -> None:
        import tabicl

        feature_columns = ["SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH"]
        data = datasets.load_iris()
        features = pd.DataFrame(data.data, columns=feature_columns)
        labels = pd.Series(data.target, name=_TARGET_COLUMN)
        x_train, x_test, y_train, _ = model_selection.train_test_split(
            features, labels, train_size=80, test_size=15, random_state=42, stratify=labels
        )
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        model = tabicl.TabICLClassifier(n_estimators=2, device="cpu", random_state=42)
        model.fit(x_train, y_train.to_numpy())

        training_frame = x_train.copy()
        training_frame[_TARGET_COLUMN] = y_train.to_numpy()
        expected_labels = np.asarray(model.predict(x_test)).astype(int)

        def _check_predict(res: pd.DataFrame) -> None:
            self.assertIn(_PREDICTION_COLUMN, res.columns)
            self.assertEqual(len(res), len(expected_labels))
            served_labels = res[_PREDICTION_COLUMN].astype(float).round().astype(int).to_numpy()
            agreement = float((served_labels == expected_labels).mean())
            self.assertGreaterEqual(agreement, 0.75)

        self._test_registry_model(
            model=model,
            sample_input_data=training_frame,
            prediction_assert_fns={"predict": (x_test, _check_predict)},
            is_partitioned_assert=_BOUND_PREDICT_ASSERTS["is_partitioned"],
            function_type_assert=_BOUND_PREDICT_ASSERTS["function_type"],
            artifact_repository_map=_PIP_ARTIFACT_REPOSITORY_MAP,
            options={"enable_explainability": False},
        )

    @absltest.skip(  # type: ignore[misc]
        "TabICL regressor checkpoint (tabicl-regressor-v2) is not available in the offline CI "
        "Hugging Face mirror yet; test_tabicl_bound_classifier covers the bound-serving path."
    )
    def test_tabicl_bound_regressor(self) -> None:
        import tabicl

        feature_columns = ["FEATURE_0", "FEATURE_1", "FEATURE_2", "FEATURE_3"]
        feature_values, target_values = datasets.make_regression(
            n_samples=95, n_features=len(feature_columns), noise=0.1, random_state=42
        )
        features = pd.DataFrame(feature_values, columns=feature_columns)
        labels = pd.Series(target_values, name=_TARGET_COLUMN)
        x_train, x_test, y_train, _ = model_selection.train_test_split(
            features, labels, train_size=80, test_size=15, random_state=42
        )
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        model = tabicl.TabICLRegressor(n_estimators=2, device="cpu", random_state=42)
        model.fit(x_train, y_train.to_numpy())

        training_frame = x_train.copy()
        training_frame[_TARGET_COLUMN] = y_train.to_numpy()
        expected_predictions = np.asarray(model.predict(x_test), dtype=float)

        def _check_predict(res: pd.DataFrame) -> None:
            self.assertIn(_PREDICTION_COLUMN, res.columns)
            self.assertEqual(len(res), len(expected_predictions))
            served_predictions = res[_PREDICTION_COLUMN].astype(float).to_numpy()
            self.assertTrue(np.all(np.isfinite(served_predictions)))
            # Local and served predictions share the checkpoint and re-fit context, so they should
            # track closely; correlation is robust to small cross-platform torch numeric drift.
            correlation = float(np.corrcoef(served_predictions, expected_predictions)[0, 1])
            self.assertGreaterEqual(correlation, 0.9)

        self._test_registry_model(
            model=model,
            sample_input_data=training_frame,
            prediction_assert_fns={"predict": (x_test, _check_predict)},
            is_partitioned_assert=_BOUND_PREDICT_ASSERTS["is_partitioned"],
            function_type_assert=_BOUND_PREDICT_ASSERTS["function_type"],
            artifact_repository_map=_PIP_ARTIFACT_REPOSITORY_MAP,
            options={"enable_explainability": False},
        )


if __name__ == "__main__":
    absltest.main()
