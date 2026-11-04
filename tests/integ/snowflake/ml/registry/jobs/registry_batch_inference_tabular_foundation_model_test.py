"""Integ: log a TabICL classifier through the generic tabular foundation model handler and score it
with SPCS batch inference (``run_batch``).

The handler serves this model *unbound* (in-context-learning): only the foundation weights and
constructor params are logged, and the labeled context is supplied at inference time inside the input
relation. ``predict`` is therefore a partitioned ``TABLE_FUNCTION`` -- each partition carries both
context (non-null ``TARGET``) and query (null ``TARGET``) rows -- so it runs via ``run_batch`` (a
partitioned table function has no scalar service endpoint) rather than ``create_service``, on a GPU
compute pool (tabular foundation models are torch models served on SPCS GPU compute pools).

The handler auto-generates the generic, width-agnostic signature (a single ``FEATURES`` OBJECT column
plus a scalar ``TARGET``); the test supplies no explicit signature.
"""

import logging
import os
import tempfile

import numpy as np
import pandas as pd
from absl.testing import absltest
from sklearn import datasets

from snowflake import snowpark
from snowflake.ml.model._client.model import batch_inference_job_specs
from snowflake.ml.model._packager.model_env import model_env
from snowflake.snowpark import types as snowpark_types
from tests.integ.snowflake.ml.registry.jobs import registry_batch_inference_test_base

# tabicl is imported lazily below and loads numba code compiled with @njit(cache=True). numba's default
# on-disk cache sits next to the installed package, which is read-only under the Bazel test sandbox, so
# the import otherwise fails with "no locator available". Point numba at a writable dir.
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()), "numba_cache"),
)

# Logging the model downloads the TabICL foundation checkpoint via hf_hub_download, which stages files
# under $HOME/.cache/huggingface by default -- read-only in the Bazel test sandbox, so makedirs fails
# before the mirror is even contacted. Point the HF cache at a writable dir (the HF endpoint is set
# separately by .bazelrc to the Artifactory mirror).
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.environ.get("TEST_TMPDIR", tempfile.gettempdir()), "hf_home"),
)

logger = logging.getLogger(__name__)

# The generic, width-agnostic schema shared by every tabular foundation model: one FEATURES OBJECT
# column ({feature_name: value} per row) plus a scalar TARGET. Non-null TARGET = in-context example;
# null TARGET = a row to score. See the handler for the full contract.
_FEATURES_COLUMN = "FEATURES"
_TARGET_COLUMN = "TARGET"
_PARTITION_COLUMN = "PARTITION_KEY"

# Output columns produced by the partitioned predict table function.
_OUTPUT_FEATURES_COLUMN = "OUTPUT_FEATURES"
_PREDICTION_COLUMN = "PREDICTION"

_IRIS_CLASS_NAMES = {"setosa", "versicolor", "virginica"}
_NUM_CONTEXT_ROWS = 80
_NUM_QUERY_ROWS = 20
_RANDOM_STATE = 0


def _make_tabicl_classifier() -> object:
    """A TabICL classifier (single ``.ckpt`` foundation bundled by the handler at log time)."""
    import tabicl

    return tabicl.TabICLClassifier()


class TestBatchInferenceTabularFoundationModelInteg(registry_batch_inference_test_base.RegistryBatchInferenceTestBase):
    """Batch-inference integ test exercising unbound (in-context-learning) TabICL serving."""

    def _build_iris_classification_input(self, *, partition_value: str) -> snowpark.DataFrame:
        """Build an OBJECT-packed iris classification relation: labeled context + null-label query rows.

        Args:
            partition_value: The value written into the partition column for every row (one ICL block).

        Returns:
            A Snowpark DataFrame with FEATURES (MapType OBJECT), TARGET (nullable STRING), and the
            partition column. Non-null TARGET marks a context row; null TARGET marks a query row.
        """
        iris = datasets.load_iris(as_frame=True)
        features = iris.data  # 4 numeric feature columns
        labels = np.asarray(iris.target_names)[iris.target]  # string class names

        order = np.random.default_rng(_RANDOM_STATE).permutation(len(features))
        context_indices = order[:_NUM_CONTEXT_ROWS]
        query_indices = order[_NUM_CONTEXT_ROWS : _NUM_CONTEXT_ROWS + _NUM_QUERY_ROWS]

        rows: list[list[object]] = []
        for row_index in context_indices:
            feature_map = {str(key): float(value) for key, value in features.iloc[row_index].items()}
            rows.append([feature_map, str(labels[row_index]), partition_value])  # non-null label => context
        for row_index in query_indices:
            feature_map = {str(key): float(value) for key, value in features.iloc[row_index].items()}
            rows.append([feature_map, None, partition_value])  # null label => query row to score

        schema = snowpark_types.StructType(
            [
                snowpark_types.StructField(
                    _FEATURES_COLUMN,
                    snowpark_types.MapType(snowpark_types.StringType(), snowpark_types.VariantType()),
                ),
                snowpark_types.StructField(_TARGET_COLUMN, snowpark_types.StringType()),
                snowpark_types.StructField(_PARTITION_COLUMN, snowpark_types.StringType()),
            ]
        )
        return self.session.create_dataframe(rows, schema=schema)

    def _run_classifier_batch_inference(self, model_id: str, estimator: object) -> None:
        """Log ``estimator`` through the generic handler and score the iris relation via ``run_batch``.

        Args:
            model_id: Short id used to name the model version (one per foundation model).
            estimator: The tabular foundation classifier to log and serve unbound.
        """
        input_df = self._build_iris_classification_input(partition_value="iris")
        job_name, output_stage_location, _ = self._prepare_job_name_and_stage_for_batch_inference()

        def check_output(actual: pd.DataFrame) -> None:
            self.assertIsInstance(actual, pd.DataFrame)
            columns = {column.upper() for column in actual.columns}
            self.assertIn(_OUTPUT_FEATURES_COLUMN, columns)
            self.assertIn(_PREDICTION_COLUMN, columns)
            self.assertIn(_PARTITION_COLUMN, columns)
            # Only the query rows are scored (M:N partitioned table function).
            self.assertEqual(len(actual), _NUM_QUERY_ROWS)
            prediction_column = next(column for column in actual.columns if column.upper() == _PREDICTION_COLUMN)
            predictions = {str(value) for value in actual[prediction_column]}
            self.assertTrue(
                predictions.issubset(_IRIS_CLASS_NAMES),
                f"predictions {predictions} are not iris class labels {_IRIS_CLASS_NAMES}",
            )

        self._test_registry_batch_inference(
            model=estimator,
            X=input_df,
            input_spec=batch_inference_job_specs.InputSpec(
                partition_column=_PARTITION_COLUMN,
                params={"target_column": _TARGET_COLUMN},  # non-null TARGET = context, null = scored
            ),
            output_spec=batch_inference_job_specs.OutputSpec(stage_location=output_stage_location),
            resources_spec=batch_inference_job_specs.ResourcesSpec(gpu_requests="1"),
            function_name="predict",
            job_name=job_name,
            # No sample_input_data / signatures: the handler auto-generates the generic signature.
            options={"cuda_version": model_env.DEFAULT_CUDA_VERSION},
            prediction_assert_fn=check_output,
            skip_row_count_check=True,  # output is only the scored query rows
            model_name=f"model_tfm_{model_id}",
            version_name=f"ver_{self._run_id}",
        )

    def test_batch_inference_tabicl_classifier(self) -> None:
        self._run_classifier_batch_inference("tabicl", _make_tabicl_classifier())


if __name__ == "__main__":
    absltest.main()
