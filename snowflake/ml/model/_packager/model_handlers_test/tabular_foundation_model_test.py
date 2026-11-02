"""Unit tests for the generic tabular foundation model (unbound / ICL) handler.

The handler imports ``tabicl`` only lazily (inside the adapter's ``save_model`` /
``load_model`` paths) and keeps all of its real logic in pure pandas/numpy helpers, so
these tests avoid the heavy ``tabicl`` / ``torch`` / GPU stack entirely:

  - The helper tests call the pure functions directly (no estimator package needed).
  - The adapter tests inject a lightweight fake ``tabicl`` module into ``sys.modules`` so
    ``from tabicl import ...`` and ``LazyType("tabicl.TabICL...").isinstance`` resolve to
    in-process fakes.
"""

import os
import shutil
import sys
import tempfile
import types
from typing import Any, cast
from unittest import mock

import numpy as np
import pandas as pd
from absl.testing import absltest, parameterized

from snowflake.ml.model import custom_model, model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_handler
from snowflake.ml.model._packager.model_env import model_env
from snowflake.ml.model._packager.model_handlers import (
    tabular_foundation_model as tfm_handler,
)
from snowflake.ml.model._packager.model_meta import model_meta, model_meta_schema


def _feature_dtype(spec: Any) -> model_signature.DataType:
    """Return a feature spec's declared ``DataType`` (narrowing to ``FeatureSpec``)."""
    assert isinstance(spec, model_signature.FeatureSpec)
    return spec._dtype


def _feature_nullable(spec: Any) -> bool:
    """Return whether a feature spec is nullable (narrowing to ``FeatureSpec``)."""
    assert isinstance(spec, model_signature.FeatureSpec)
    return spec._nullable


def _blob_options(meta: model_meta.ModelMetadata) -> model_meta_schema.TabularFoundationModelBlobOptions:
    """Return the single model blob's options narrowed to the tabular foundation options type."""
    return cast(model_meta_schema.TabularFoundationModelBlobOptions, meta.models[meta.name].options)


class _FakeTabICLBase:
    """Minimal sklearn-like TabICL estimator whose foundation is an on-disk checkpoint path."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        allow_auto_download: bool = True,
        device: str | None = None,
        checkpoint_version: str = "tabicl-classifier.ckpt",
        n_estimators: int = 8,
        random_state: int = 42,
        **extra: Any,
    ) -> None:
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.device = device
        self.checkpoint_version = checkpoint_version
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._extra = extra

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model_path": self.model_path,
            "allow_auto_download": self.allow_auto_download,
            "device": self.device,
            "checkpoint_version": self.checkpoint_version,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
        params.update(self._extra)
        return params


class TabICLClassifier(_FakeTabICLBase):
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "TabICLClassifier":
        self.classes_ = sorted({str(label) for label in np.asarray(y).tolist()})
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.array([self.classes_[0]] * len(x))


class TabICLRegressor(_FakeTabICLBase):
    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "TabICLRegressor":
        self.mean_ = float(np.mean(np.asarray(y, dtype=float)))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.full(len(x), self.mean_, dtype=float)


def _make_fake_tabicl_module() -> types.ModuleType:
    """Build a fake ``tabicl`` package exposing the classes the TabICL adapter imports."""
    fake_module = types.ModuleType("tabicl")
    fake_module.TabICLClassifier = TabICLClassifier  # type: ignore[attr-defined]
    fake_module.TabICLRegressor = TabICLRegressor  # type: ignore[attr-defined]
    return fake_module


class TabularFoundationModelHelperTest(parameterized.TestCase):
    """Tests for the pure (estimator-package-free) helper functions."""

    def test_object_features_to_frame_reconstructs_and_sorts(self) -> None:
        # Keys intentionally out of order to prove deterministic column sorting.
        series = pd.Series([{"b": 2.0, "a": 1.0}, {"b": 4.0, "a": 3.0}])

        frame = tfm_handler._object_features_to_frame(series)

        self.assertEqual(list(frame.columns), ["a", "b"])
        self.assertEqual(frame["a"].tolist(), [1.0, 3.0])
        self.assertEqual(frame["b"].tolist(), [2.0, 4.0])

    def test_object_features_to_frame_ragged_keys_fill_nan(self) -> None:
        series = pd.Series([{"a": 1.0, "b": 2.0}, {"b": 3.0, "c": 4.0}])

        frame = tfm_handler._object_features_to_frame(series)

        self.assertEqual(list(frame.columns), ["a", "b", "c"])
        self.assertTrue(np.isnan(frame["c"].iloc[0]))
        self.assertTrue(np.isnan(frame["a"].iloc[1]))

    def test_object_features_to_frame_preserves_mixed_dtypes(self) -> None:
        series = pd.Series([{"num": 1.0, "cat": "x"}, {"num": 2.0, "cat": "y"}])

        frame = tfm_handler._object_features_to_frame(series)

        self.assertTrue(pd.api.types.is_numeric_dtype(frame["num"]))
        self.assertFalse(pd.api.types.is_numeric_dtype(frame["cat"]))

    def test_object_features_to_frame_empty(self) -> None:
        frame = tfm_handler._object_features_to_frame(pd.Series([], dtype=object))
        self.assertEqual(frame.shape[1], 0)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("floats", [1.0, 2.5], model_signature.DataType.DOUBLE),
        ("ints", [1, 2, 3], model_signature.DataType.DOUBLE),
        ("bools", [True, False], model_signature.DataType.BOOL),
        ("strings", ["cat", "dog"], model_signature.DataType.STRING),
    )
    def test_infer_datatype(self, values: list[Any], expected: model_signature.DataType) -> None:
        self.assertEqual(tfm_handler._infer_datatype(pd.Series(values)), expected)

    def test_infer_datatype_all_null_defaults_string(self) -> None:
        series = pd.Series([None, None], dtype=object)
        self.assertEqual(tfm_handler._infer_datatype(series), model_signature.DataType.STRING)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("regression", "regression", model_signature.DataType.DOUBLE),
        ("classification", "classification", model_signature.DataType.STRING),
    )
    def test_default_scalar_dtype(self, model_type: str, expected: model_signature.DataType) -> None:
        self.assertEqual(tfm_handler._default_scalar_dtype(model_type), expected)

    def test_split_context_and_query_implicit_null(self) -> None:
        df = pd.DataFrame(
            {
                tfm_handler.FEATURES_COLUMN: [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}, {"a": 4.0}],
                "TARGET": [1.0, np.nan, 2.0, np.nan],
            }
        )

        context_df, query_df = tfm_handler._split_context_and_query(
            df, target_column="TARGET", context_filter="", prediction_filter=""
        )

        self.assertEqual(context_df["TARGET"].tolist(), [1.0, 2.0])
        self.assertEqual(len(query_df), 2)
        self.assertTrue(query_df["TARGET"].isna().all())

    def test_split_context_and_query_explicit_filters(self) -> None:
        df = pd.DataFrame(
            {
                tfm_handler.FEATURES_COLUMN: [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}],
                "TARGET": ["ctx", "__q__", "__q__"],
            }
        )

        context_df, query_df = tfm_handler._split_context_and_query(
            df,
            target_column="TARGET",
            context_filter="TARGET != '__q__'",
            prediction_filter="TARGET == '__q__'",
        )

        self.assertEqual(len(context_df), 1)
        self.assertEqual(context_df["TARGET"].tolist(), ["ctx"])
        self.assertEqual(len(query_df), 2)

    @parameterized.named_parameters(  # type: ignore[misc]
        ("attribute_access", "TARGET.str.startswith('x')"),
        ("scope_reference", "@TARGET"),
        ("unknown_column", "TARGET == UNKNOWN_COLUMN"),
        ("disallowed_call", "`TARGET` or __import__('os')"),
    )
    def test_split_context_and_query_rejects_unsafe_filter(self, unsafe_filter: str) -> None:
        df = pd.DataFrame(
            {
                tfm_handler.FEATURES_COLUMN: [{"a": 1.0}, {"a": 2.0}],
                "TARGET": ["ctx", "__q__"],
            }
        )

        with self.assertRaises(ValueError):
            tfm_handler._split_context_and_query(
                df, target_column="TARGET", context_filter=unsafe_filter, prediction_filter=""
            )

    def test_context_fingerprint_is_deterministic(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        labels = np.array([0, 1])

        first = tfm_handler._context_fingerprint(frame.copy(), labels.copy())
        second = tfm_handler._context_fingerprint(frame.copy(), labels.copy())

        self.assertEqual(first, second)

    def test_context_fingerprint_differs_on_content(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        other = pd.DataFrame({"a": [1.0, 9.0], "b": ["x", "y"]})
        labels = np.array([0, 1])

        self.assertNotEqual(
            tfm_handler._context_fingerprint(frame, labels),
            tfm_handler._context_fingerprint(other, labels),
        )

    def test_build_predict_signature_structure(self) -> None:
        signature = tfm_handler.TabularFoundationModelHandler._build_predict_signature(
            target_column="TARGET",
            target_dtype=model_signature.DataType.STRING,
            prediction_dtype=model_signature.DataType.DOUBLE,
        )

        self.assertEqual(signature.inputs[0].name, tfm_handler.FEATURES_COLUMN)
        self.assertEqual(_feature_dtype(signature.inputs[0]), model_signature.DataType.OBJECT)
        self.assertFalse(_feature_nullable(signature.inputs[0]))

        self.assertEqual(signature.inputs[1].name, "TARGET")
        self.assertEqual(_feature_dtype(signature.inputs[1]), model_signature.DataType.STRING)
        self.assertTrue(_feature_nullable(signature.inputs[1]))

        self.assertEqual(signature.outputs[0].name, tfm_handler.OUTPUT_FEATURES_COLUMN)
        self.assertNotEqual(tfm_handler.OUTPUT_FEATURES_COLUMN, tfm_handler.FEATURES_COLUMN)
        self.assertEqual(_feature_dtype(signature.outputs[0]), model_signature.DataType.OBJECT)
        self.assertEqual(signature.outputs[1].name, tfm_handler.PREDICTION_COLUMN)
        self.assertEqual(_feature_dtype(signature.outputs[1]), model_signature.DataType.DOUBLE)

        param_names = {param.name for param in signature.params}
        self.assertEqual(
            param_names,
            {
                tfm_handler._PARAM_TARGET_COLUMN,
                tfm_handler._PARAM_CONTEXT_FILTER,
                tfm_handler._PARAM_PREDICTION_FILTER,
            },
        )


class TabularFoundationModelGenericHandlerTest(parameterized.TestCase):
    """Handler-generic behavior (signature inference, packaging, validation), exercised via the TabICL fakes."""

    def setUp(self) -> None:
        # Inject a fake ``tabicl`` so ``_find_adapter`` never imports the real torch-backed
        # package while probing ``can_handle`` (matches() calls LazyType.isinstance).
        self._fake_tabicl = _make_fake_tabicl_module()
        modules_patcher = mock.patch.dict(sys.modules, {"tabicl": self._fake_tabicl})
        modules_patcher.start()
        self.addCleanup(modules_patcher.stop)

        # A real on-disk checkpoint so save_foundation copies it instead of downloading.
        checkpoint_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(checkpoint_dir, ignore_errors=True))
        self._checkpoint_path = os.path.join(checkpoint_dir, "source.ckpt")
        with open(self._checkpoint_path, "wb") as checkpoint_file:
            checkpoint_file.write(b"fake-checkpoint-bytes")

    def _new_regressor(self) -> TabICLRegressor:
        return TabICLRegressor(model_path=self._checkpoint_path)

    def _new_classifier(self) -> TabICLClassifier:
        return TabICLClassifier(model_path=self._checkpoint_path)

    def _new_meta(self, name: str = "tfm_model") -> model_meta.ModelMetadata:
        return model_meta.ModelMetadata(name=name, env=model_env.ModelEnv(), model_type="tabular_foundation_model")

    def test_handler_constants(self) -> None:
        handler = tfm_handler.TabularFoundationModelHandler
        self.assertEqual(handler.HANDLER_TYPE, "tabular_foundation_model")
        self.assertEqual(handler.MODEL_BLOB_FILE_OR_DIR, "model")
        self.assertEqual(handler.DEFAULT_TARGET_METHODS, ["predict"])
        self.assertTrue(handler.IS_AUTO_SIGNATURE)

    def test_can_handle_false_for_non_foundation_model(self) -> None:
        self.assertFalse(tfm_handler.TabularFoundationModelHandler.can_handle(object()))

    def _save(
        self,
        model: Any,
        meta: model_meta.ModelMetadata,
        sample_input_data: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> str:
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        tfm_handler.TabularFoundationModelHandler.save_model(
            name=meta.name,
            model=model,
            model_meta=meta,
            model_blobs_dir_path=temp_dir,
            sample_input_data=sample_input_data,
            **kwargs,
        )
        return temp_dir

    def test_save_model_classifier_defaults_to_string(self) -> None:
        meta = self._new_meta()
        self._save(self._new_classifier(), meta)

        signature = meta.signatures["predict"]
        self.assertEqual(_feature_dtype(signature.inputs[1]), model_signature.DataType.STRING)
        self.assertEqual(_feature_dtype(signature.outputs[1]), model_signature.DataType.STRING)

    def test_save_model_regressor_defaults_to_double(self) -> None:
        meta = self._new_meta()
        self._save(self._new_regressor(), meta)

        signature = meta.signatures["predict"]
        self.assertEqual(_feature_dtype(signature.inputs[1]), model_signature.DataType.DOUBLE)
        self.assertEqual(_feature_dtype(signature.outputs[1]), model_signature.DataType.DOUBLE)

    def test_save_model_marks_predict_partitioned(self) -> None:
        meta = self._new_meta()
        self._save(self._new_regressor(), meta)

        self.assertEqual(
            meta.function_properties["predict"],
            {model_meta_schema.FunctionProperties.PARTITIONED.value: True},
        )

    def test_save_model_records_blob_options(self) -> None:
        meta = self._new_meta()
        self._save(self._new_classifier(), meta)

        options = _blob_options(meta)
        self.assertEqual(meta.models[meta.name].model_type, "tabular_foundation_model")
        self.assertEqual(options["adapter"], "tabicl")
        self.assertEqual(options["estimator_type"], "TabICLClassifier")
        self.assertEqual(options["serving_mode"], "unbound")
        self.assertEqual(options["model_type"], "classification")
        self.assertEqual(options["target_column"], tfm_handler.DEFAULT_TARGET_COLUMN)
        self.assertEqual(meta.task, model_types.Task.UNKNOWN)

    def test_save_model_always_emits_gpu_runtime(self) -> None:
        meta = self._new_meta()
        self._save(self._new_regressor(), meta)
        # cuda_version drives emission of a `gpu` runtime in model.yaml; always needed.
        self.assertIsNotNone(meta.env.cuda_version)

    def test_save_model_sample_refines_regressor_double(self) -> None:
        meta = self._new_meta()
        sample = pd.DataFrame(
            {
                tfm_handler.FEATURES_COLUMN: [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}, {"a": 5.0, "b": 6.0}],
                "TARGET": [10.0, 20.0, np.nan],
            }
        )

        self._save(self._new_regressor(), meta, sample_input_data=sample)

        signature = meta.signatures["predict"]
        self.assertEqual(_feature_dtype(signature.inputs[1]), model_signature.DataType.DOUBLE)
        self.assertEqual(_feature_dtype(signature.outputs[1]), model_signature.DataType.DOUBLE)

    def test_save_model_sample_refines_classifier_string(self) -> None:
        meta = self._new_meta()
        sample = pd.DataFrame(
            {
                tfm_handler.FEATURES_COLUMN: [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}],
                "TARGET": ["cat", "dog", None],
            }
        )

        self._save(self._new_classifier(), meta, sample_input_data=sample)

        signature = meta.signatures["predict"]
        self.assertEqual(_feature_dtype(signature.inputs[1]), model_signature.DataType.STRING)
        self.assertEqual(_feature_dtype(signature.outputs[1]), model_signature.DataType.STRING)

    def test_save_model_prediction_type_override(self) -> None:
        meta = self._new_meta()
        # Classifier would default PREDICTION to STRING; the explicit override wins.
        self._save(self._new_classifier(), meta, prediction_type=float)

        signature = meta.signatures["predict"]
        self.assertEqual(_feature_dtype(signature.outputs[1]), model_signature.DataType.DOUBLE)

    def test_save_model_custom_target_column(self) -> None:
        meta = self._new_meta()
        self._save(self._new_regressor(), meta, target_column="LABEL")

        signature = meta.signatures["predict"]
        self.assertEqual(signature.inputs[1].name, "LABEL")
        self.assertEqual(_blob_options(meta)["target_column"], "LABEL")

    def test_save_model_honors_user_signature(self) -> None:
        meta = self._new_meta()
        user_signature = tfm_handler.TabularFoundationModelHandler._build_predict_signature(
            target_column="TARGET",
            target_dtype=model_signature.DataType.INT64,
            prediction_dtype=model_signature.DataType.INT64,
        )
        meta.signatures = {"predict": user_signature}

        self._save(self._new_regressor(), meta)

        # Not overwritten by the default DOUBLE inference.
        self.assertEqual(_feature_dtype(meta.signatures["predict"].outputs[1]), model_signature.DataType.INT64)

    def test_save_model_user_signature_requires_predict(self) -> None:
        meta = self._new_meta()
        meta.signatures = {"not_predict": mock.MagicMock()}

        with self.assertRaisesRegex(ValueError, "predict"):
            self._save(self._new_regressor(), meta)

    def test_save_model_rejects_explainability(self) -> None:
        meta = self._new_meta()
        with self.assertRaisesRegex(ValueError, "Explainability"):
            self._save(self._new_regressor(), meta, enable_explainability=True)

    def test_save_model_sample_missing_features_column(self) -> None:
        meta = self._new_meta()
        sample = pd.DataFrame({"TARGET": [1.0, np.nan]})

        with self.assertRaisesRegex(ValueError, tfm_handler.FEATURES_COLUMN):
            self._save(self._new_regressor(), meta, sample_input_data=sample)


class TabICLAdapterHandlerTest(parameterized.TestCase):
    """Handler behavior for the TabICL adapter (touches the faked tabicl estimators)."""

    def setUp(self) -> None:
        # Inject a fake ``tabicl`` so ``_find_adapter`` never imports the real torch-backed
        # package while probing ``can_handle`` (matches() calls LazyType.isinstance).
        self._fake_tabicl = _make_fake_tabicl_module()
        modules_patcher = mock.patch.dict(sys.modules, {"tabicl": self._fake_tabicl})
        modules_patcher.start()
        self.addCleanup(modules_patcher.stop)

        # A real on-disk checkpoint so save_foundation copies it instead of downloading.
        checkpoint_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(checkpoint_dir, ignore_errors=True))
        self._checkpoint_path = os.path.join(checkpoint_dir, "source.ckpt")
        with open(self._checkpoint_path, "wb") as checkpoint_file:
            checkpoint_file.write(b"fake-checkpoint-bytes")

    def _new_classifier(self) -> TabICLClassifier:
        return TabICLClassifier(model_path=self._checkpoint_path)

    def _new_regressor(self) -> TabICLRegressor:
        return TabICLRegressor(model_path=self._checkpoint_path)

    def _new_meta(self, name: str = "tabicl_model") -> model_meta.ModelMetadata:
        return model_meta.ModelMetadata(name=name, env=model_env.ModelEnv(), model_type="tabular_foundation_model")

    def _save(self, model: Any, meta: model_meta.ModelMetadata) -> str:
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        tfm_handler.TabularFoundationModelHandler.save_model(
            name=meta.name,
            model=model,
            model_meta=meta,
            model_blobs_dir_path=temp_dir,
        )
        return temp_dir

    def test_can_handle_true_for_tabicl(self) -> None:
        self.assertTrue(tfm_handler.TabularFoundationModelHandler.can_handle(self._new_classifier()))
        self.assertTrue(tfm_handler.TabularFoundationModelHandler.can_handle(self._new_regressor()))

    def test_can_handle_false_for_custom_model_wrapping_tabicl(self) -> None:
        # Dispatch keys on the top-level object's type, so a CustomModel that wraps a TabICL estimator is
        # owned by the custom handler, never this one. The foundation-model save/load path (including the
        # bound refit-from-context deserialization) therefore never runs for it, so an existing
        # TabICL-based custom model in production is unaffected.
        tabicl_estimator = self._new_classifier()

        class TabICLCustomModel(custom_model.CustomModel):
            def __init__(self, context: custom_model.ModelContext | None = None) -> None:
                super().__init__(context)
                self.estimator = tabicl_estimator

            @custom_model.inference_api
            def predict(self, X: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({tfm_handler.PREDICTION_COLUMN: [0.0] * len(X)})

        wrapped = TabICLCustomModel()

        self.assertFalse(tfm_handler.TabularFoundationModelHandler.can_handle(wrapped))
        self.assertIsNone(tfm_handler._find_adapter(wrapped))

        resolved_handler = model_handler.find_handler(wrapped)
        assert resolved_handler is not None
        self.assertEqual(resolved_handler.HANDLER_TYPE, "custom")

    def test_supported_target_platforms_unrestricted(self) -> None:
        # TabICL's checkpoint is small enough to run in a Warehouse: no platform restriction.
        handler = tfm_handler.TabularFoundationModelHandler
        self.assertIsNone(handler.get_supported_target_platforms(self._new_classifier()))
        self.assertIsNone(handler.get_supported_target_platforms(self._new_regressor()))

    def test_save_model_records_tabicl_blob_options(self) -> None:
        meta = self._new_meta()
        self._save(self._new_classifier(), meta)

        options = _blob_options(meta)
        self.assertEqual(options["adapter"], "tabicl")
        self.assertEqual(options["estimator_type"], "TabICLClassifier")
        self.assertEqual(options["model_type"], "classification")
        self.assertEqual(options["serving_mode"], "unbound")

    def test_save_model_regressor_defaults_to_double(self) -> None:
        meta = self._new_meta()
        self._save(self._new_regressor(), meta)

        signature = meta.signatures["predict"]
        self.assertEqual(_feature_dtype(signature.inputs[1]), model_signature.DataType.DOUBLE)
        self.assertEqual(_feature_dtype(signature.outputs[1]), model_signature.DataType.DOUBLE)

    def test_save_model_bundles_checkpoint(self) -> None:
        meta = self._new_meta()
        temp_dir = self._save(self._new_classifier(), meta)

        bundled = os.path.join(temp_dir, meta.name, "model", "foundation", "checkpoint.ckpt")
        self.assertTrue(os.path.isfile(bundled))

    def test_save_model_declares_tabicl_pip_dependency(self) -> None:
        meta = self._new_meta()
        self._save(self._new_regressor(), meta)
        self.assertTrue(any("tabicl" in requirement for requirement in meta.env.pip_requirements))

    def test_load_model_rebinds_bundled_checkpoint_offline(self) -> None:
        meta = self._new_meta()
        temp_dir = self._save(self._new_classifier(), meta)

        loaded = tfm_handler.TabularFoundationModelHandler.load_model(
            name=meta.name, model_meta=meta, model_blobs_dir_path=temp_dir
        )

        self.assertEqual(type(loaded).__name__, "TabICLClassifier")
        params = loaded.get_params()
        self.assertTrue(params["model_path"].endswith(os.path.join("foundation", "checkpoint.ckpt")))
        self.assertFalse(params["allow_auto_download"])
        # Non-foundation params round-trip through init_params.json.
        self.assertEqual(params["n_estimators"], 8)
        self.assertEqual(params["random_state"], 42)


class TabularFoundationModelBoundTest(parameterized.TestCase):
    """Handler behavior for the bound (pre-fitted, static scalar) serving path.

    Bound mode is handler-generic; it is exercised through the TabICL fakes, which must be fit before
    logging (so ``predict`` works during signature inference and the stored context can be re-fit).
    """

    def setUp(self) -> None:
        self._fake_tabicl = _make_fake_tabicl_module()
        modules_patcher = mock.patch.dict(sys.modules, {"tabicl": self._fake_tabicl})
        modules_patcher.start()
        self.addCleanup(modules_patcher.stop)

        # A real on-disk checkpoint so save_foundation copies it instead of downloading.
        checkpoint_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(checkpoint_dir, ignore_errors=True))
        self._checkpoint_path = os.path.join(checkpoint_dir, "source.ckpt")
        with open(self._checkpoint_path, "wb") as checkpoint_file:
            checkpoint_file.write(b"fake-checkpoint-bytes")

    _TARGET = tfm_handler.DEFAULT_TARGET_COLUMN

    def _regression_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0], self._TARGET: [10.0, 20.0, 30.0, 40.0]}
        )

    def _classification_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0], self._TARGET: ["x", "y", "x", "y"]})

    def _unfitted_regressor(self) -> TabICLRegressor:
        return TabICLRegressor(model_path=self._checkpoint_path)

    def _fitted_regressor(self, train: pd.DataFrame) -> TabICLRegressor:
        model = TabICLRegressor(model_path=self._checkpoint_path)
        model.fit(train.drop(columns=[self._TARGET]), train[self._TARGET].to_numpy())
        return model

    def _fitted_classifier(self, train: pd.DataFrame) -> TabICLClassifier:
        model = TabICLClassifier(model_path=self._checkpoint_path)
        model.fit(train.drop(columns=[self._TARGET]), train[self._TARGET].to_numpy())
        return model

    def _new_meta(self, name: str = "tfm_bound") -> model_meta.ModelMetadata:
        return model_meta.ModelMetadata(name=name, env=model_env.ModelEnv(), model_type="tabular_foundation_model")

    def _save(
        self, model: Any, meta: model_meta.ModelMetadata, sample_input_data: pd.DataFrame | None, **kwargs: Any
    ) -> str:
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        tfm_handler.TabularFoundationModelHandler.save_model(
            name=meta.name,
            model=model,
            model_meta=meta,
            model_blobs_dir_path=temp_dir,
            sample_input_data=sample_input_data,
            **kwargs,
        )
        return temp_dir

    def test_fitted_estimator_auto_detects_bound(self) -> None:
        # No `bound` flag: a fitted estimator is detected and served bound.
        train = self._regression_frame()
        meta = self._new_meta()
        self._save(self._fitted_regressor(train), meta, train)
        self.assertEqual(_blob_options(meta)["serving_mode"], "bound")

    def test_adapter_is_fitted_detects_fit_state(self) -> None:
        unfitted = self._unfitted_regressor()
        fitted = self._fitted_regressor(self._regression_frame())
        unfitted_adapter = tfm_handler._find_adapter(unfitted)
        fitted_adapter = tfm_handler._find_adapter(fitted)
        assert unfitted_adapter is not None and fitted_adapter is not None
        self.assertFalse(unfitted_adapter.is_fitted(unfitted))
        self.assertTrue(fitted_adapter.is_fitted(fitted))

    def test_unfitted_estimator_serves_unbound(self) -> None:
        # Mode follows the fitted state: a bare (unfitted) estimator stays unbound.
        meta = self._new_meta()
        self._save(self._unfitted_regressor(), meta, None)
        options = _blob_options(meta)
        self.assertEqual(options["serving_mode"], "unbound")
        self.assertNotIn("context_file", options)
        self.assertEqual(
            meta.function_properties["predict"],
            {model_meta_schema.FunctionProperties.PARTITIONED.value: True},
        )

    def test_bound_save_builds_static_scalar_signature(self) -> None:
        train = self._regression_frame()
        meta = self._new_meta()
        self._save(self._fitted_regressor(train), meta, train)

        signature = meta.signatures["predict"]
        # Concrete named feature columns, not the generic OBJECT input.
        self.assertEqual([spec.name for spec in signature.inputs], ["a", "b"])
        self.assertNotIn(tfm_handler.FEATURES_COLUMN, [spec.name for spec in signature.inputs])
        for spec in signature.inputs:
            self.assertEqual(_feature_dtype(spec), model_signature.DataType.DOUBLE)
        self.assertEqual(len(signature.outputs), 1)
        self.assertEqual(_feature_dtype(signature.outputs[0]), model_signature.DataType.DOUBLE)
        # A plain scalar signature carries none of the unbound inference params.
        self.assertFalse(signature.params)

    def test_bound_save_is_not_partitioned(self) -> None:
        train = self._regression_frame()
        meta = self._new_meta()
        self._save(self._fitted_regressor(train), meta, train)
        self.assertNotIn("predict", meta.function_properties)

    def test_bound_save_records_blob_options_and_stores_context(self) -> None:
        train = self._regression_frame()
        meta = self._new_meta()
        temp_dir = self._save(self._fitted_regressor(train), meta, train)

        options = _blob_options(meta)
        self.assertEqual(options["serving_mode"], "bound")
        self.assertEqual(options["context_file"], "context.json")
        context_path = os.path.join(temp_dir, meta.name, "model", "context.json")
        self.assertTrue(os.path.isfile(context_path))
        stored = pd.read_json(context_path, orient="table").reset_index(drop=True)
        self.assertEqual(set(stored.columns), {"a", "b", self._TARGET})
        self.assertEqual(len(stored), 4)

    def test_bound_classifier_output_defaults_string(self) -> None:
        train = self._classification_frame()
        meta = self._new_meta()
        self._save(self._fitted_classifier(train), meta, train)
        self.assertEqual(_feature_dtype(meta.signatures["predict"].outputs[0]), model_signature.DataType.STRING)

    def test_bound_prediction_type_override(self) -> None:
        train = self._regression_frame()
        meta = self._new_meta()
        self._save(self._fitted_regressor(train), meta, train, prediction_type=str)
        self.assertEqual(_feature_dtype(meta.signatures["predict"].outputs[0]), model_signature.DataType.STRING)

    def test_bound_custom_target_column(self) -> None:
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "LABEL": [7.0, 8.0, 9.0]})
        model = TabICLRegressor(model_path=self._checkpoint_path)
        model.fit(train.drop(columns=["LABEL"]), train["LABEL"].to_numpy())
        meta = self._new_meta()
        self._save(model, meta, train, target_column="LABEL")

        signature = meta.signatures["predict"]
        self.assertEqual([spec.name for spec in signature.inputs], ["a", "b"])
        self.assertEqual(_blob_options(meta)["target_column"], "LABEL")

    def test_bound_requires_sample_input_data(self) -> None:
        meta = self._new_meta()
        model = self._fitted_regressor(self._regression_frame())
        with self.assertRaisesRegex(ValueError, "require sample_input_data"):
            self._save(model, meta, None)

    def test_bound_requires_target_column(self) -> None:
        meta = self._new_meta()
        model = self._fitted_regressor(self._regression_frame())
        no_target = self._regression_frame().drop(columns=[self._TARGET])
        with self.assertRaisesRegex(ValueError, "target column"):
            self._save(model, meta, no_target)

    def test_bound_load_refits_and_predicts(self) -> None:
        train = self._regression_frame()
        meta = self._new_meta()
        temp_dir = self._save(self._fitted_regressor(train), meta, train)

        loaded = tfm_handler.TabularFoundationModelHandler.load_model(
            name=meta.name, model_meta=meta, model_blobs_dir_path=temp_dir
        )
        self.assertEqual(type(loaded).__name__, "TabICLRegressor")
        predictions = loaded.predict(pd.DataFrame({"a": [1.0, 2.0], "b": [5.0, 6.0]}))
        self.assertEqual(len(predictions), 2)
        # The fake regressor predicts the mean of the training targets (10..40 -> 25.0).
        self.assertEqual(float(predictions[0]), 25.0)

    def test_bound_predict_helper_returns_scalar_column(self) -> None:
        model = self._fitted_regressor(self._regression_frame())
        result = tfm_handler._bound_predict(model, pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}))
        self.assertEqual(list(result.columns), [tfm_handler.PREDICTION_COLUMN])
        self.assertEqual(len(result), 3)

    def test_bound_convert_as_custom_model_scores_scalar(self) -> None:
        train = self._regression_frame()
        meta = self._new_meta()
        temp_dir = self._save(self._fitted_regressor(train), meta, train)
        loaded = tfm_handler.TabularFoundationModelHandler.load_model(
            name=meta.name, model_meta=meta, model_blobs_dir_path=temp_dir
        )

        custom = tfm_handler.TabularFoundationModelHandler.convert_as_custom_model(loaded, meta)
        predict = getattr(custom, "predict", None)
        assert callable(predict)
        output = predict(pd.DataFrame({"a": [1.0, 2.0], "b": [5.0, 6.0]}))
        self.assertIn(tfm_handler.PREDICTION_COLUMN, output.columns)
        self.assertEqual(len(output), 2)


if __name__ == "__main__":
    absltest.main()
