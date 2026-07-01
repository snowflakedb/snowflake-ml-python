import gc
import json
import logging
import os
import tempfile
import warnings
from importlib import metadata as importlib_metadata
from typing import Any, Mapping, Optional, cast
from unittest import mock

import numpy as np
import numpy.typing as npt
import pandas as pd
from absl.testing import absltest, parameterized
from packaging import requirements
from pandas.testing import assert_frame_equal

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_handlers.sentence_transformers import (
    _ALLOWED_TARGET_METHODS,
    _DEFAULT_WRAPPER_TARGET_METHODS,
    SentenceTransformerHandler,
    _auto_infer_signature,
    _capture_model_truncate_dim,
    _encode_sentences_with_nulls,
    _get_available_default_methods,
    _get_embedding_dim_from_config,
    _is_null_sentence,
    _supports_encode_truncate_dim_param,
    _supports_init_truncate_dim,
    _validate_sentence_transformers_signatures,
)
from snowflake.ml.model._packager.model_meta import model_meta_schema
from snowflake.ml.model._signatures import utils as model_signature_utils

MODEL_NAMES = ["intfloat/e5-base-v2"]
SENTENCE_TRANSFORMERS_CACHE_DIR = "SENTENCE_TRANSFORMERS_HOME"


class SentenceTransformerHandlerTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        cls.cache_dir = tempfile.TemporaryDirectory()
        cls._original_cache = os.getenv(SENTENCE_TRANSFORMERS_CACHE_DIR, None)
        cls._original_hf_home = os.getenv("HF_HOME", None)
        cls._original_transformers_cache = os.getenv("TRANSFORMERS_CACHE", None)
        os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR] = cls.cache_dir.name
        os.environ["HF_HOME"] = cls.cache_dir.name
        os.environ["TRANSFORMERS_CACHE"] = cls.cache_dir.name

        import sentence_transformers

        cls._st_model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])
        cls._st_model_truncated = None
        if _supports_init_truncate_dim():
            cls._st_model_truncated = sentence_transformers.SentenceTransformer(MODEL_NAMES[0], truncate_dim=128)

        cls._wrapper_snapshot_dir = os.path.join(cls.cache_dir.name, "wrapper_snapshot")
        cls._st_model.save(cls._wrapper_snapshot_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._original_cache:
            os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR] = cls._original_cache
        else:
            os.environ.pop(SENTENCE_TRANSFORMERS_CACHE_DIR, None)
        if cls._original_hf_home:
            os.environ["HF_HOME"] = cls._original_hf_home
        else:
            os.environ.pop("HF_HOME", None)
        if cls._original_transformers_cache:
            os.environ["TRANSFORMERS_CACHE"] = cls._original_transformers_cache
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)
        del cls._st_model
        if cls._st_model_truncated is not None:
            del cls._st_model_truncated
        gc.collect()
        cls.cache_dir.cleanup()

    def test_convert_as_custom_model_unsupported_method_raises(self) -> None:
        """Test that convert_as_custom_model raises ValueError for unsupported methods.

        This test verifies the fix for the missing-raise bug where unsupported methods
        would silently construct a ValueError but not raise it.
        """
        model = self._st_model

        # Create a mock model_meta with an unsupported method
        with tempfile.TemporaryDirectory() as tmpdir:
            # First save a valid model to get proper metadata
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                sample_input_data=pd.DataFrame({"text": ["test sentence"]}),
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta

            # Manually add an unsupported method to signatures
            pk.meta.signatures["unsupported_method"] = pk.meta.signatures["encode"]

            # Verify that convert_as_custom_model now raises ValueError
            with self.assertRaises(ValueError) as ctx:
                SentenceTransformerHandler.convert_as_custom_model(
                    raw_model=model,
                    model_meta=pk.meta,
                )
            self.assertIn("unsupported_method", str(ctx.exception))
            self.assertIn("not supported", str(ctx.exception))

    def test_allowed_target_methods(self) -> None:
        """Test that _ALLOWED_TARGET_METHODS contains expected methods."""
        # Both singular (new) and plural (old) naming conventions are supported
        self.assertEqual(
            _ALLOWED_TARGET_METHODS,
            [
                "encode",
                "encode_query",
                "encode_document",
                "encode_queries",
                "encode_documents",
            ],
        )

    def test_get_available_default_methods(self) -> None:
        """Test that _get_available_default_methods returns only methods that exist on the model."""
        model = self._st_model
        available_methods = _get_available_default_methods(model)

        # 'encode' should always be available
        self.assertIn("encode", available_methods)

        # All returned methods should be callable on the model
        for method_name in available_methods:
            method = getattr(model, method_name, None)
            self.assertIsNotNone(method, f"Method {method_name} should exist on model")
            self.assertTrue(callable(method), f"Method {method_name} should be callable")

        # All returned methods should be in _ALLOWED_TARGET_METHODS
        for method_name in available_methods:
            self.assertIn(method_name, _ALLOWED_TARGET_METHODS)

    def test_auto_infer_signature(self) -> None:
        """Test that _auto_infer_signature creates correct signatures."""
        embedding_dim = 384

        # Test encode method
        sig = _auto_infer_signature(
            "encode",
            embedding_dim,
            batch_size=None,
            include_truncate_dim_param=True,
        )
        assert sig is not None  # Type narrowing for mypy
        self.assertEqual(len(sig.inputs), 1)
        self.assertEqual(sig.inputs[0].name, "sentence")
        assert isinstance(sig.inputs[0], model_signature.FeatureSpec)
        self.assertEqual(sig.inputs[0]._dtype, model_signature.DataType.STRING)
        self.assertEqual(len(sig.outputs), 1)
        self.assertEqual(sig.outputs[0].name, "output")
        assert isinstance(sig.outputs[0], model_signature.FeatureSpec)
        self.assertEqual(sig.outputs[0]._dtype, model_signature.DataType.DOUBLE)
        self.assertEqual(sig.outputs[0]._shape, (embedding_dim,))

        # Verify runtime params are present
        self.assertEqual(len(sig.params), 2)
        self.assertEqual(sig.params[0].name, "batch_size")
        assert isinstance(sig.params[0], model_signature.ParamSpec)
        self.assertEqual(sig.params[0]._dtype, model_signature.DataType.INT64)
        self.assertIsNone(sig.params[0].default_value)
        self.assertEqual(sig.params[1].name, "truncate_dim")
        assert isinstance(sig.params[1], model_signature.ParamSpec)
        self.assertEqual(sig.params[1]._dtype, model_signature.DataType.INT64)
        self.assertEqual(sig.params[1].default_value, None)

        sig_batch_only = _auto_infer_signature("encode", embedding_dim, batch_size=None)
        assert sig_batch_only is not None
        self.assertEqual(len(sig_batch_only.params), 1)

        sig_custom_batch_size = _auto_infer_signature(
            "encode",
            embedding_dim,
            batch_size=64,
            include_truncate_dim_param=True,
        )
        assert sig_custom_batch_size is not None
        self.assertEqual(sig_custom_batch_size.params[0].default_value, 64)
        self.assertEqual(sig_custom_batch_size.outputs[0]._shape, (embedding_dim,))
        self.assertIsNone(sig_custom_batch_size.params[1].default_value)

        # Test all allowed methods
        for method in _ALLOWED_TARGET_METHODS:
            method_sig = _auto_infer_signature(
                method,
                embedding_dim,
                batch_size=None,
                include_truncate_dim_param=True,
            )
            self.assertIsNotNone(method_sig, f"Should infer signature for {method}")

        # Test unsupported method returns None
        unsupported_sig = _auto_infer_signature("unsupported_method", embedding_dim, batch_size=32)
        self.assertIsNone(unsupported_sig)

    def test_validate_sentence_transformers_signatures_valid(self) -> None:
        """Test valid signatures for all supported methods."""
        # Valid signature with encode only (using realistic shape for embeddings)
        valid_encode_only = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=(384,))
                ],
            )
        }
        _validate_sentence_transformers_signatures(valid_encode_only)

        # Valid signature with all three methods (using realistic shape for embeddings)
        valid_all_methods = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="text", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(name="embedding", dtype=model_signature.DataType.FLOAT, shape=(384,))
                ],
            ),
            "encode_queries": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="query", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(
                        name="query_embedding", dtype=model_signature.DataType.FLOAT, shape=(384,)
                    )
                ],
            ),
            "encode_documents": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="doc", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(
                        name="doc_embedding", dtype=model_signature.DataType.FLOAT, shape=(384,)
                    )
                ],
            ),
        }
        _validate_sentence_transformers_signatures(valid_all_methods)

    def test_is_null_sentence(self) -> None:
        self.assertTrue(_is_null_sentence(None))
        self.assertTrue(_is_null_sentence(pd.NA))
        self.assertTrue(_is_null_sentence(float("nan")))
        self.assertFalse(_is_null_sentence("hello"))
        self.assertFalse(_is_null_sentence(""))
        self.assertFalse(_is_null_sentence(["hello"]))

    def test_encode_sentences_with_nulls(self) -> None:
        def fake_encode(sentences: list[str], **kwargs: Any) -> npt.NDArray[Any]:
            del kwargs
            return np.array([[float(index), float(index + 1)] for index, _ in enumerate(sentences)])

        outputs = _encode_sentences_with_nulls(
            ["a", None, pd.NA, "b"],
            fake_encode,
            {},
        )
        self.assertEqual(outputs[0], [0.0, 1.0])
        self.assertIsNone(outputs[1])
        self.assertIsNone(outputs[2])
        self.assertEqual(outputs[3], [1.0, 2.0])

        all_valid_outputs = _encode_sentences_with_nulls(["a", "b"], fake_encode, {})
        self.assertEqual(all_valid_outputs, [[0.0, 1.0], [1.0, 2.0]])

        all_null_outputs = _encode_sentences_with_nulls([None, pd.NA], fake_encode, {})
        self.assertEqual(all_null_outputs, [None, None])

    def test_null_sentence_custom_model_returns_none_embedding(self) -> None:
        model = self._st_model
        sentences = pd.DataFrame(
            {
                "sentence": [
                    "Why don't scientists trust atoms? Because they make up everything.",
                    None,
                    "Parallel lines have so much in common. It's a shame they'll never meet.",
                ]
            }
        )
        expected_embeddings = model.encode(
            [sentences.iloc[0, 0], sentences.iloc[2, 0]],
        ).tolist()
        expected = pd.DataFrame(
            {
                "output": [
                    expected_embeddings[0],
                    None,
                    expected_embeddings[1],
                ]
            }
        )

        sig = {"encode": model_signature.infer_signature(sentences.iloc[[0, 2]], expected.iloc[[0, 2]])}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                signatures=sig,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
                pk.load(as_custom_model=True)
                assert pk.model
                predict_method = getattr(pk.model, "encode", None)
                assert callable(predict_method)
                embeddings_load = predict_method(sentences)
                embeddings_load.columns = expected.columns
                self.assertIsNone(embeddings_load.iloc[1, 0])
                assert_frame_equal(embeddings_load.iloc[[0, 2]], expected.iloc[[0, 2]])
                assert_frame_equal(embeddings_load.iloc[[1]], expected.iloc[[1]])

    def test_validate_sentence_transformers_signatures_empty(self) -> None:
        """Test that empty signatures raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            _validate_sentence_transformers_signatures({})
        self.assertIn("At least one signature", str(ctx.exception))

    def test_validate_sentence_transformers_signatures_unsupported_method(self) -> None:
        """Test that unsupported method names raise ValueError."""
        invalid_signature = {
            "another_method": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=(384,))
                ],
            )
        }
        with self.assertRaises(ValueError) as ctx:
            _validate_sentence_transformers_signatures(invalid_signature)
        self.assertIn("Unsupported target methods", str(ctx.exception))
        self.assertIn("another_method", str(ctx.exception))

    def test_validate_sentence_transformers_signatures_multiple_inputs(self) -> None:
        """Test that multiple inputs raise ValueError."""
        invalid_signature = {
            "encode": model_signature.ModelSignature(
                inputs=[
                    model_signature.FeatureSpec(name="input1", dtype=model_signature.DataType.STRING, shape=None),
                    model_signature.FeatureSpec(name="input2", dtype=model_signature.DataType.STRING, shape=None),
                ],
                outputs=[
                    model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=(384,))
                ],
            )
        }
        with self.assertRaises(ValueError) as ctx:
            _validate_sentence_transformers_signatures(invalid_signature)
        self.assertIn("exactly 1 input column", str(ctx.exception))

    def test_validate_sentence_transformers_signatures_multiple_outputs(self) -> None:
        """Test that multiple outputs raise ValueError."""
        invalid_signature = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(name="output1", dtype=model_signature.DataType.FLOAT, shape=(384,)),
                    model_signature.FeatureSpec(name="output2", dtype=model_signature.DataType.FLOAT, shape=(384,)),
                ],
            )
        }
        with self.assertRaises(ValueError) as ctx:
            _validate_sentence_transformers_signatures(invalid_signature)
        self.assertIn("exactly 1 output column", str(ctx.exception))

    def test_validate_sentence_transformers_signatures_with_shape(self) -> None:
        """Test that input with shape raises ValueError."""
        invalid_signature = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING, shape=(1,))],
                outputs=[
                    model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=(384,))
                ],
            )
        }
        with self.assertRaises(ValueError) as ctx:
            _validate_sentence_transformers_signatures(invalid_signature)
        self.assertIn("does not support input shape", str(ctx.exception))

    def test_validate_sentence_transformers_signatures_non_string_input(self) -> None:
        """Test that non-STRING input raises ValueError."""
        invalid_signature = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.INT32, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=(384,))
                ],
            )
        }
        with self.assertRaises(ValueError) as ctx:
            _validate_sentence_transformers_signatures(invalid_signature)
        self.assertIn("only accepts STRING input", str(ctx.exception))

    def test_convert_as_custom_model_multi_method(self) -> None:
        """Test that convert_as_custom_model handles multiple target_methods for SentenceTransformers.

        This test verifies that multiple encode methods work correctly when they exist on the model.
        Note: encode_queries and encode_documents may not exist in all sentence-transformers versions,
        so we test with encode only (which always exists).
        """
        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model - only use 'encode' which is guaranteed to exist
            sentences = pd.DataFrame({"text": ["test sentence 1", "test sentence 2"]})

            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                sample_input_data=sentences,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(target_methods=["encode"]),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(as_custom_model=True)
            assert pk.model is not None
            assert pk.meta is not None

            # Verify encode method is in signatures
            self.assertIn("encode", pk.meta.signatures)

            # Verify encode method is available and callable
            method = getattr(pk.model, "encode", None)
            self.assertIsNotNone(method, "Method 'encode' should exist")
            self.assertTrue(callable(method), "Method 'encode' should be callable")

            # Call the method and verify it returns a DataFrame
            assert method is not None  # Type narrowing for mypy
            result = method(sentences)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(sentences))

    def test_auto_signature_inference(self) -> None:
        """Test that signatures are automatically inferred when no sample_input_data or signatures are provided.

        This tests the IS_AUTO_SIGNATURE = True functionality for SentenceTransformer models.
        """
        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model WITHOUT sample_input_data or signatures - should auto-infer
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(target_methods=["encode"]),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            # Verify that signature was auto-inferred
            self.assertIn("encode", pk.meta.signatures)
            sig = pk.meta.signatures["encode"]

            # Verify signature structure
            self.assertEqual(len(sig.inputs), 1)
            self.assertEqual(len(sig.outputs), 1)

            # Verify input is STRING type
            self.assertEqual(sig.inputs[0].name, "sentence")
            assert isinstance(sig.inputs[0], model_signature.FeatureSpec)
            self.assertEqual(sig.inputs[0]._dtype, model_signature.DataType.STRING)

            # Verify output is DOUBLE type with correct embedding dimension
            self.assertEqual(sig.outputs[0].name, "output")
            assert isinstance(sig.outputs[0], model_signature.FeatureSpec)
            self.assertEqual(sig.outputs[0]._dtype, model_signature.DataType.DOUBLE)

            # Verify embedding dimension matches model
            expected_dim = model.get_sentence_embedding_dimension()
            self.assertEqual(sig.outputs[0]._shape, (expected_dim,))

            # Verify batch_size param is present with default value (None = system decides)
            batch_size_param = next(p for p in sig.params if p.name == "batch_size")
            assert isinstance(batch_size_param, model_signature.ParamSpec)
            self.assertEqual(batch_size_param._dtype, model_signature.DataType.INT64)
            self.assertIsNone(batch_size_param.default_value)

            if _supports_encode_truncate_dim_param():
                self.assertEqual(len(sig.params), 2)
                truncate_dim_param = next(p for p in sig.params if p.name == "truncate_dim")
                assert isinstance(truncate_dim_param, model_signature.ParamSpec)
                self.assertEqual(truncate_dim_param._dtype, model_signature.DataType.INT64)
                self.assertIsNone(truncate_dim_param.default_value)
            else:
                self.assertEqual(len(sig.params), 1)

            # Test that the model works correctly with auto-inferred signature
            pk.load(as_custom_model=True)
            assert pk.model is not None
            test_sentences = pd.DataFrame({"sentence": ["Hello world", "Test sentence"]})
            predict_method = getattr(pk.model, "encode", None)
            assert callable(predict_method)
            result = predict_method(test_sentences)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)

    def test_auto_signature_multiple_methods(self) -> None:
        """Test auto signature inference with multiple target methods."""
        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # Only test with 'encode' since encode_queries/encode_documents may not exist in all versions
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(target_methods=["encode"]),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            # Verify encode method signature
            self.assertIn("encode", pk.meta.signatures)
            encode_sig = pk.meta.signatures["encode"]
            expected_dim = model.get_sentence_embedding_dimension()
            self.assertEqual(encode_sig.outputs[0]._shape, (expected_dim,))

    def test_auto_signature_encode_query(self) -> None:
        """Test auto signature inference for encode_query method.

        The encode_query method is used in asymmetric semantic search models
        to encode query strings. It applies query-specific preprocessing (e.g., adding
        "query: " prefix for some models) before encoding.

        This test verifies:
        1. Auto signature is correctly inferred for encode_query
        2. The model can be saved without sample_input_data
        3. The loaded model can perform inference with the auto-inferred signature

        Note: sentence-transformers >= 3.0 uses singular names (encode_query)
        while older versions may use plural names (encode_queries).
        """
        model = self._st_model

        # Check if encode_query method exists on this model (singular, new naming)
        if not hasattr(model, "encode_query") or not callable(getattr(model, "encode_query", None)):
            self.skipTest("Model does not support encode_query method")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model with encode_query target method - no sample_input_data
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(target_methods=["encode_query"]),
            )

            # Load and verify signature was auto-inferred
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            # Verify encode_query signature structure
            self.assertIn("encode_query", pk.meta.signatures)
            sig = pk.meta.signatures["encode_query"]

            # Input should be STRING type with name "sentence"
            self.assertEqual(len(sig.inputs), 1)
            self.assertEqual(sig.inputs[0].name, "sentence")
            assert isinstance(sig.inputs[0], model_signature.FeatureSpec)
            self.assertEqual(sig.inputs[0]._dtype, model_signature.DataType.STRING)

            # Output should be DOUBLE type with correct embedding dimension
            self.assertEqual(len(sig.outputs), 1)
            self.assertEqual(sig.outputs[0].name, "output")
            assert isinstance(sig.outputs[0], model_signature.FeatureSpec)
            self.assertEqual(sig.outputs[0]._dtype, model_signature.DataType.DOUBLE)

            expected_dim = model.get_sentence_embedding_dimension()
            self.assertEqual(sig.outputs[0]._shape, (expected_dim,))

            # Test inference with auto-inferred signature
            pk.load(as_custom_model=True)
            assert pk.model is not None

            test_queries = pd.DataFrame({"sentence": ["What is machine learning?", "How does AI work?"]})
            predict_method = getattr(pk.model, "encode_query", None)
            assert callable(predict_method)
            result = predict_method(test_queries)

            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)
            self.assertIn("output", result.columns)
            # Verify embedding dimension
            self.assertEqual(len(result["output"].iloc[0]), expected_dim)

    def test_auto_signature_encode_document(self) -> None:
        """Test auto signature inference for encode_document method.

        The encode_document method is used in asymmetric semantic search models
        to encode document/passage strings. It applies document-specific preprocessing
        (e.g., adding "passage: " prefix for some models) before encoding.

        This test verifies:
        1. Auto signature is correctly inferred for encode_document
        2. The model can be saved without sample_input_data
        3. The loaded model can perform inference with the auto-inferred signature

        Note: sentence-transformers >= 3.0 uses singular names (encode_document)
        while older versions may use plural names (encode_documents).
        """
        model = self._st_model

        # Check if encode_document method exists on this model (singular, new naming)
        if not hasattr(model, "encode_document") or not callable(getattr(model, "encode_document", None)):
            self.skipTest("Model does not support encode_document method")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model with encode_document target method - no sample_input_data
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(target_methods=["encode_document"]),
            )

            # Load and verify signature was auto-inferred
            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            # Verify encode_document signature structure
            self.assertIn("encode_document", pk.meta.signatures)
            sig = pk.meta.signatures["encode_document"]

            self.assertEqual(len(sig.inputs), 1)
            self.assertEqual(sig.inputs[0].name, "sentence")
            assert isinstance(sig.inputs[0], model_signature.FeatureSpec)
            self.assertEqual(sig.inputs[0]._dtype, model_signature.DataType.STRING)

            # Output should be DOUBLE type with correct embedding dimension
            self.assertEqual(len(sig.outputs), 1)
            self.assertEqual(sig.outputs[0].name, "output")
            assert isinstance(sig.outputs[0], model_signature.FeatureSpec)
            self.assertEqual(sig.outputs[0]._dtype, model_signature.DataType.DOUBLE)

            expected_dim = model.get_sentence_embedding_dimension()
            self.assertEqual(sig.outputs[0]._shape, (expected_dim,))

            # Test inference with auto-inferred signature
            pk.load(as_custom_model=True)
            assert pk.model is not None

            test_docs = pd.DataFrame(
                {
                    "sentence": [
                        "Machine learning is a subset of artificial intelligence.",
                        "Neural networks are inspired by biological neurons.",
                    ]
                }
            )
            predict_method = getattr(pk.model, "encode_document", None)
            assert callable(predict_method)
            result = predict_method(test_docs)

            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)
            self.assertIn("output", result.columns)
            # Verify embedding dimension
            self.assertEqual(len(result["output"].iloc[0]), expected_dim)

    def test_auto_signature_all_methods(self) -> None:
        """Test auto signature inference with all three methods together.

        This test verifies that when saving a model with all three target methods
        (encode, encode_query, encode_document), each method gets its own
        correctly inferred signature with the same structure but independent entries.

        Note: sentence-transformers >= 3.0 uses singular names (encode_query, encode_document).
        """
        model = self._st_model

        # Check if all methods exist (using singular names for new sentence-transformers)
        has_encode_query = hasattr(model, "encode_query") and callable(getattr(model, "encode_query", None))
        has_encode_document = hasattr(model, "encode_document") and callable(getattr(model, "encode_document", None))

        if not has_encode_query or not has_encode_document:
            self.skipTest("Model does not support all encode methods")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model with all three target methods (singular names)
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(
                    target_methods=["encode", "encode_query", "encode_document"]
                ),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            expected_dim = model.get_sentence_embedding_dimension()

            # Verify all three methods have signatures
            for method_name in ["encode", "encode_query", "encode_document"]:
                self.assertIn(method_name, pk.meta.signatures)
                sig = pk.meta.signatures[method_name]

                # All methods should have same signature structure
                self.assertEqual(len(sig.inputs), 1)
                self.assertEqual(sig.inputs[0].name, "sentence")
                assert isinstance(sig.inputs[0], model_signature.FeatureSpec)
                self.assertEqual(sig.inputs[0]._dtype, model_signature.DataType.STRING)

                self.assertEqual(len(sig.outputs), 1)
                self.assertEqual(sig.outputs[0].name, "output")
                assert isinstance(sig.outputs[0], model_signature.FeatureSpec)
                self.assertEqual(sig.outputs[0]._dtype, model_signature.DataType.DOUBLE)
                self.assertEqual(sig.outputs[0]._shape, (expected_dim,))

            # Test inference with all methods
            pk.load(as_custom_model=True)
            assert pk.model is not None

            test_data = pd.DataFrame({"sentence": ["Test sentence"]})

            # All methods should work
            for method_name in ["encode", "encode_query", "encode_document"]:
                predict_method = getattr(pk.model, method_name, None)
                self.assertIsNotNone(predict_method, f"Method '{method_name}' should exist")
                assert callable(predict_method)
                result = predict_method(test_data)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertEqual(len(result), 1)

    def test_save_model_target_methods_validation_unsupported(self) -> None:
        """Test that save_model rejects unsupported target_methods with clear error message.

        Per the design doc (Section 4.2), save_model should:
        - Accept any non-empty subset of ALLOWED_TARGET_METHODS
        - Reject methods not in ALLOWED_TARGET_METHODS with clear error
        """
        model = self._st_model
        sentences = pd.DataFrame({"text": ["test sentence"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                    name="model",
                    model=model,
                    sample_input_data=sentences,
                    metadata={"author": "test", "version": "1"},
                    options=model_types.SentenceTransformersSaveOptions(target_methods=["unsupported_method"]),
                )
            # Verify error message mentions the unsupported method
            self.assertIn("unsupported_method", str(ctx.exception).lower())

    def test_save_model_target_methods_validation_empty(self) -> None:
        """Test that save_model rejects empty target_methods.

        Per the design doc (Section 4.2), save_model should raise ValueError
        if target_methods is empty.
        """
        model = self._st_model
        sentences = pd.DataFrame({"text": ["test sentence"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                    name="model",
                    model=model,
                    sample_input_data=sentences,
                    metadata={"author": "test", "version": "1"},
                    options=model_types.SentenceTransformersSaveOptions(target_methods=[]),
                )
            self.assertIn("at least one target method", str(ctx.exception).lower())

    def test_save_model_target_methods_validation_subset(self) -> None:
        """Test that save_model accepts any valid subset of ALLOWED_TARGET_METHODS.

        Per the design doc (Section 4.2), target_methods must be a subset of
        ALLOWED_TARGET_METHODS. This test verifies that various valid subsets work.
        """
        model = self._st_model
        sentences = pd.DataFrame({"text": ["test sentence"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with only encode (always available)
            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                sample_input_data=sentences,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(target_methods=["encode"]),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
            pk.load()
            assert pk.meta is not None
            self.assertIn("encode", pk.meta.signatures)
            self.assertEqual(len(pk.meta.signatures), 1)

    def test_save_model_target_methods_validation_multiple_methods(self) -> None:
        """Test that save_model accepts multiple valid target_methods.

        Per the design doc (Section 3.1), multiple methods can be specified
        as long as they are all in ALLOWED_TARGET_METHODS.
        """
        model = self._st_model

        # Check which methods are available on this model
        has_encode_query = hasattr(model, "encode_query") and callable(getattr(model, "encode_query", None))
        has_encode_document = hasattr(model, "encode_document") and callable(getattr(model, "encode_document", None))

        if not has_encode_query or not has_encode_document:
            self.skipTest("Model does not support encode_query/encode_document methods")

        sentences = pd.DataFrame({"text": ["test sentence"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with multiple methods
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                sample_input_data=sentences,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(
                    target_methods=["encode", "encode_query", "encode_document"]
                ),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            # All specified methods should have signatures
            self.assertIn("encode", pk.meta.signatures)
            self.assertIn("encode_query", pk.meta.signatures)
            self.assertIn("encode_document", pk.meta.signatures)
            self.assertEqual(len(pk.meta.signatures), 3)

    def test_save_model_target_methods_non_callable(self) -> None:
        """Test that save_model rejects target_methods that don't exist on the model.

        Even if a method is in ALLOWED_TARGET_METHODS, it must be callable on the
        actual model instance.
        """
        model = self._st_model
        sentences = pd.DataFrame({"text": ["test sentence"]})

        # Try to use encode_queries (plural) which may not exist on newer models
        if hasattr(model, "encode_queries") and callable(getattr(model, "encode_queries", None)):
            self.skipTest("Model has encode_queries method, cannot test non-callable scenario")

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                    name="model",
                    model=model,
                    sample_input_data=sentences,
                    metadata={"author": "test", "version": "1"},
                    options=model_types.SentenceTransformersSaveOptions(target_methods=["encode_queries"]),
                )
            self.assertIn("encode_queries", str(ctx.exception))

    def test_backward_compatibility_plural_method_names(self) -> None:
        """Test backward compatibility with plural method names (encode_queries, encode_documents).

        Per the design doc, both singular (new sentence-transformers >= 3.0) and
        plural (older versions) naming conventions are supported for backward compatibility.
        The _ALLOWED_TARGET_METHODS includes both naming conventions.
        """
        model = self._st_model

        # Check if plural methods exist on this model (older sentence-transformers)
        has_encode_queries = hasattr(model, "encode_queries") and callable(getattr(model, "encode_queries", None))
        has_encode_documents = hasattr(model, "encode_documents") and callable(getattr(model, "encode_documents", None))

        if not has_encode_queries and not has_encode_documents:
            self.skipTest("Model does not support plural method names (encode_queries/encode_documents)")

        sentences = pd.DataFrame({"text": ["test sentence"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            target_methods = ["encode"]
            if has_encode_queries:
                target_methods.append("encode_queries")
            if has_encode_documents:
                target_methods.append("encode_documents")

            # Save model with plural method names
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                sample_input_data=sentences,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(target_methods=target_methods),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            # Verify signatures were created for plural methods
            for method in target_methods:
                self.assertIn(method, pk.meta.signatures, f"Method '{method}' should have a signature")

            # Verify model can be loaded as custom model and methods work
            pk.load(as_custom_model=True)
            assert pk.model is not None

            for method in target_methods:
                predict_method = getattr(pk.model, method, None)
                self.assertIsNotNone(predict_method, f"Method '{method}' should exist on custom model")
                assert predict_method is not None  # Type narrowing for mypy
                self.assertTrue(callable(predict_method), f"Method '{method}' should be callable")
                result = predict_method(sentences)
                self.assertIsInstance(result, pd.DataFrame)

    def test_backward_compatibility_encode_only(self) -> None:
        """Test backward compatibility: existing code using only encode continues to work.

        This test ensures that users who only rely on encode (the original behavior)
        continue to have their code work without any changes.
        """
        model = self._st_model
        sentences = pd.DataFrame({"text": ["test sentence 1", "test sentence 2"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with explicit encode only (mimics existing user code)
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                sample_input_data=sentences,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(target_methods=["encode"]),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            # Only encode should be in signatures
            self.assertEqual(list(pk.meta.signatures.keys()), ["encode"])

            # Load as custom model and verify encode works
            pk.load(as_custom_model=True)
            assert pk.model is not None

            encode_method = getattr(pk.model, "encode", None)
            assert callable(encode_method)
            result = encode_method(sentences)

            # Verify output matches direct model.encode()
            expected = model.encode(sentences["text"].tolist())
            self.assertEqual(len(result), len(sentences))
            self.assertEqual(len(result.iloc[0, 0]), len(expected[0]))

    def test_save_model_captures_tokenizers_dependency(self) -> None:
        """Verify save captures blob options and dependency pins for SentenceTransformer models.

        Checks that the hub model id is stored in blob options and that tokenizers is pinned
        alongside transformers to prevent version mismatches in deployment environments
        (e.g. transformers==4.41.2 requires tokenizers>=0.19,<0.20).
        """
        import tokenizers
        import transformers

        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                sample_input_data=pd.DataFrame({"text": ["test sentence"]}),
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            blob_options = cast(
                model_meta_schema.SentenceTransformersModelBlobOptions,
                pk.meta.models["model"].options,
            )
            self.assertEqual(blob_options["model"], MODEL_NAMES[0])

            deps = {
                requirements.Requirement(dep).name: requirements.Requirement(dep)
                for dep in pk.meta.env.conda_dependencies
            }
            self.assertIn("tokenizers", deps)
            self.assertIn("transformers", deps)
            self.assertIn("sentence-transformers", deps)

            self.assertTrue(deps["tokenizers"].specifier.contains(tokenizers.__version__))
            self.assertTrue(deps["transformers"].specifier.contains(transformers.__version__))

    @parameterized.parameters(  # type: ignore[misc]
        {"transformers_version": "4.41.2", "tokenizers_version": "0.19.1", "sentence_transformers_version": "2.7.0"},
        {"transformers_version": "5.3.0", "tokenizers_version": "0.22.2", "sentence_transformers_version": "5.2.0"},
    )
    def test_save_model_pins_compatible_version_pairs(
        self,
        transformers_version: str,
        tokenizers_version: str,
        sentence_transformers_version: str,
    ) -> None:
        """Verify that tokenizers and transformers are pinned as a compatible pair for different version combos."""
        model = self._st_model

        original_distribution = importlib_metadata.distribution

        def fake_distribution(name: str) -> importlib_metadata.Distribution:
            versions = {
                "tokenizers": tokenizers_version,
                "transformers": transformers_version,
                "sentence-transformers": sentence_transformers_version,
            }
            if name in versions:
                dist = mock.MagicMock(spec=importlib_metadata.Distribution)
                dist.version = versions[name]
                return dist
            return original_distribution(name)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch(
                "snowflake.ml._internal.env_utils.importlib_metadata.distribution", side_effect=fake_distribution
            ):
                model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                    name="model",
                    model=model,
                    sample_input_data=pd.DataFrame({"text": ["test sentence"]}),
                    metadata={"author": "test", "version": "1"},
                    options=model_types.SentenceTransformersSaveOptions(),
                )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            deps = {
                requirements.Requirement(dep).name: requirements.Requirement(dep)
                for dep in pk.meta.env.conda_dependencies
            }
            self.assertTrue(deps["tokenizers"].specifier.contains(tokenizers_version))
            self.assertTrue(deps["transformers"].specifier.contains(transformers_version))
            self.assertTrue(deps["sentence-transformers"].specifier.contains(sentence_transformers_version))

    def test_sentence_transformers(self) -> None:
        import sentence_transformers

        # Sample Data
        sentences = pd.DataFrame(
            {
                "SENTENCES": [
                    "Why don't scientists trust atoms? Because they make up everything.",
                    "I told my wife she should embrace her mistakes. She gave me a hug.",
                    "Im reading a book on anti-gravity. Its impossible to put down!",
                    "Did you hear about the mathematician who's afraid of negative numbers?",
                    "Parallel lines have so much in common. It's a shame they'll never meet.",
                ]
            }
        )
        model = self._st_model
        embeddings = pd.DataFrame({"EMBEDDINGS": model.encode(sentences["SENTENCES"].tolist()).tolist()})

        sig = {"encode": model_signature.infer_signature(sentences, embeddings)}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test whether an unsupported target method in signatures raises ValueError
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures={**sig, "another_encode": sig["encode"]},
                    metadata={"author": "halu", "version": "1"},
                    options=model_types.SentenceTransformersSaveOptions(),
                )

            with self.assertRaises(NotImplementedError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures=sig,
                    metadata={"author": "halu", "version": "1"},
                    options={"enable_explainability": True},
                )

            model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                name="model1",
                model=model,
                signatures=sig,
                metadata={"author": "halu", "version": "2"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            model_packager.ModelPackager(os.path.join(tmpdir, "model2")).save(
                name="model2",
                model=model,
                sample_input_data=sentences,
                metadata={"author": "halu", "version": "2"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, sentence_transformers.SentenceTransformer)
                embeddings_load = pd.DataFrame(
                    {sig["encode"].outputs[0].name: pk.model.encode(sentences["SENTENCES"].tolist()).tolist()}
                )
                embeddings_load = model_signature_utils.rename_pandas_df(embeddings_load, sig["encode"].outputs)
                assert_frame_equal(embeddings_load, embeddings)
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model1"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                self.assertEqual(sig["encode"], pk.meta.signatures["encode"])
                predict_method = getattr(pk.model, "encode", None)
                assert callable(predict_method)
                embeddings_load = predict_method(sentences)
                assert_frame_equal(embeddings_load, embeddings)

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model2"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, sentence_transformers.SentenceTransformer)
                embeddings_load = pd.DataFrame(
                    {sig["encode"].outputs[0].name: pk.model.encode(sentences["SENTENCES"].tolist()).tolist()}
                )
                assert_frame_equal(embeddings_load, embeddings)
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model2"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                predict_method = getattr(pk.model, "encode", None)
                assert callable(predict_method)
                embeddings_load = predict_method(sentences)
                embeddings.columns = embeddings_load.columns
                assert_frame_equal(embeddings_load, embeddings)

        return None

    def test_batch_size_param_auto_inferred_custom_value(self) -> None:
        """Test that auto-inferred signature includes batch_size param with the logged value as default."""
        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(batch_size=64),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            sig = pk.meta.signatures["encode"]
            expected_param_count = 2 if _supports_encode_truncate_dim_param() else 1
            self.assertEqual(len(sig.params), expected_param_count)
            batch_size_param = next(p for p in sig.params if p.name == "batch_size")
            assert isinstance(batch_size_param, model_signature.ParamSpec)
            self.assertEqual(batch_size_param.default_value, 64)
            if _supports_encode_truncate_dim_param():
                truncate_dim_param = next(p for p in sig.params if p.name == "truncate_dim")
                assert isinstance(truncate_dim_param, model_signature.ParamSpec)
                self.assertIsNone(truncate_dim_param.default_value)

    def test_batch_size_param_explicit_signature_no_injection(self) -> None:
        """Test that explicit signatures do NOT get batch_size param injected."""
        model = self._st_model
        sentences = pd.DataFrame({"SENTENCES": ["test sentence"]})
        embeddings = pd.DataFrame({"EMBEDDINGS": model.encode(["test sentence"]).tolist()})
        sig = {"encode": model_signature.infer_signature(sentences, embeddings)}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                signatures=sig,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            # Explicit signature should have no params added
            self.assertEqual(len(pk.meta.signatures["encode"].params), 0)

    def test_batch_size_param_explicit_signature_warning(self) -> None:
        """Test that a warning is emitted when explicit signatures AND batch_size option are both provided."""
        model = self._st_model
        sentences = pd.DataFrame({"SENTENCES": ["test sentence"]})
        embeddings = pd.DataFrame({"EMBEDDINGS": model.encode(["test sentence"]).tolist()})
        sig = {"encode": model_signature.infer_signature(sentences, embeddings)}

        # Warning fires even when batch_size matches the default (32) — intent matters, not value
        for explicit_batch_size in [32, 128]:
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertLogs(
                    "snowflake.ml.model._packager.model_handlers.sentence_transformers", level=logging.WARNING
                ) as log_ctx:
                    model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                        name="model",
                        model=model,
                        signatures=sig,
                        metadata={"author": "test", "version": "1"},
                        options=model_types.SentenceTransformersSaveOptions(batch_size=explicit_batch_size),
                    )
                self.assertTrue(any("batch_size" in msg and "will not be added" in msg for msg in log_ctx.output))

    def test_batch_size_param_explicit_signature_paramspec_precedence(self) -> None:
        """Test that a ParamSpec default in explicit signatures takes precedence over blob options."""
        model = self._st_model
        sentences = pd.DataFrame({"SENTENCES": ["test sentence"]})
        embeddings = pd.DataFrame({"EMBEDDINGS": model.encode(["test sentence"]).tolist()})

        paramspec_batch_size = 64
        blob_batch_size = 16

        sig = {
            "encode": model_signature.ModelSignature(
                inputs=model_signature.infer_signature(sentences, embeddings).inputs,
                outputs=model_signature.infer_signature(sentences, embeddings).outputs,
                params=[
                    model_signature.ParamSpec(
                        name="batch_size",
                        dtype=model_signature.DataType.INT64,
                        default_value=paramspec_batch_size,
                    ),
                ],
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                signatures=sig,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(batch_size=blob_batch_size),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(as_custom_model=True)
            assert pk.model is not None
            assert pk.meta is not None

            batch_size_param = next(p for p in pk.meta.signatures["encode"].params if p.name == "batch_size")
            assert isinstance(batch_size_param, model_signature.ParamSpec)
            self.assertEqual(batch_size_param.default_value, paramspec_batch_size)

    def test_batch_size_param_runtime_override(self) -> None:
        """Test that the inference method respects a non-default batch_size kwarg."""
        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(batch_size=32),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(as_custom_model=True)
            assert pk.model is not None

            test_data = pd.DataFrame({"sentence": ["Hello world", "Test sentence"]})

            # Call with runtime override
            encode_method = getattr(pk.model, "encode", None)
            assert callable(encode_method)
            result = encode_method(test_data, batch_size=1)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)

    def test_batch_size_param_sample_input_data(self) -> None:
        """Test that batch_size param is added when using sample_input_data for signature inference."""
        model = self._st_model
        sentences = pd.DataFrame({"text": ["test sentence 1", "test sentence 2"]})

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                sample_input_data=sentences,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(batch_size=16),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            sig = pk.meta.signatures["encode"]
            expected_param_count = 2 if _supports_encode_truncate_dim_param() else 1
            self.assertEqual(len(sig.params), expected_param_count)
            batch_size_param = next(p for p in sig.params if p.name == "batch_size")
            self.assertEqual(batch_size_param.name, "batch_size")
            assert isinstance(batch_size_param, model_signature.ParamSpec)
            self.assertEqual(batch_size_param.default_value, 16)
            if _supports_encode_truncate_dim_param():
                truncate_dim_param = next(p for p in sig.params if p.name == "truncate_dim")
                assert isinstance(truncate_dim_param, model_signature.ParamSpec)
                self.assertIsNone(truncate_dim_param.default_value)

    def test_model_truncate_dim_captured_in_blob(self) -> None:
        """Test that model.truncate_dim is captured in blob metadata without a save option."""
        if self._st_model_truncated is None:
            self.skipTest("sentence-transformers version does not support truncate_dim at init")

        model = self._st_model_truncated
        self.assertEqual(_capture_model_truncate_dim(model), 128)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None
            blob_options = pk.meta.models["model"].options
            self.assertEqual(blob_options.get("truncate_dim"), 128)
            self.assertEqual(pk.meta.signatures["encode"].outputs[0]._shape, (128,))

    def test_model_truncate_dim_not_in_blob_when_unset(self) -> None:
        """Test that blob metadata omits truncate_dim when the model has no init override."""
        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None
            blob_options = pk.meta.models["model"].options
            self.assertNotIn("truncate_dim", blob_options)

    def test_load_model_applies_blob_truncate_dim(self) -> None:
        """Test that load_model applies captured truncate_dim via SentenceTransformer __init__."""
        import sentence_transformers

        if self._st_model_truncated is None:
            self.skipTest("sentence-transformers version does not support truncate_dim at init")

        model = self._st_model_truncated

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.model is not None
            loaded_model = pk.model
            assert isinstance(loaded_model, sentence_transformers.SentenceTransformer)
            self.assertEqual(getattr(loaded_model, "truncate_dim", None), 128)
            self.assertEqual(loaded_model.get_sentence_embedding_dimension(), 128)

    def test_truncate_dim_param_when_supported(self) -> None:
        """Test that truncate_dim param is added only when sentence-transformers supports encode override."""
        if not _supports_encode_truncate_dim_param():
            self.skipTest("sentence-transformers version does not support encode(truncate_dim=...)")

        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            sig = pk.meta.signatures["encode"]
            truncate_dim_param = next(p for p in sig.params if p.name == "truncate_dim")
            assert isinstance(truncate_dim_param, model_signature.ParamSpec)
            self.assertIsNone(truncate_dim_param.default_value)

    def test_truncate_dim_param_default_none_when_model_init_truncated(self) -> None:
        """Test runtime truncate_dim ParamSpec stays None; init truncation is blob-only."""
        if not _supports_encode_truncate_dim_param():
            self.skipTest("sentence-transformers version does not support encode(truncate_dim=...)")
        if self._st_model_truncated is None:
            self.skipTest("sentence-transformers version does not support truncate_dim at init")

        model = self._st_model_truncated

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            truncate_dim_param = next(p for p in pk.meta.signatures["encode"].params if p.name == "truncate_dim")
            assert isinstance(truncate_dim_param, model_signature.ParamSpec)
            self.assertIsNone(truncate_dim_param.default_value)
            self.assertEqual(pk.meta.models["model"].options.get("truncate_dim"), 128)

    def test_truncate_dim_param_explicit_signature_no_injection(self) -> None:
        """Test that explicit signatures do NOT get truncate_dim param injected."""
        if not _supports_encode_truncate_dim_param():
            self.skipTest("sentence-transformers version does not support encode(truncate_dim=...)")

        model = self._st_model
        sentences = pd.DataFrame({"SENTENCES": ["test sentence"]})
        embeddings = pd.DataFrame({"EMBEDDINGS": model.encode(["test sentence"]).tolist()})
        sig = {"encode": model_signature.infer_signature(sentences, embeddings)}

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                signatures=sig,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            self.assertEqual(len(pk.meta.signatures["encode"].params), 0)

    def test_truncate_dim_param_explicit_signature_paramspec_precedence(self) -> None:
        """Test that a ParamSpec default in explicit signatures is used for inference defaults."""
        if not _supports_encode_truncate_dim_param():
            self.skipTest("sentence-transformers version does not support encode(truncate_dim=...)")

        model = self._st_model
        sentences = pd.DataFrame({"SENTENCES": ["test sentence"]})
        embeddings = pd.DataFrame({"EMBEDDINGS": model.encode(["test sentence"]).tolist()})

        paramspec_truncate_dim = 64

        sig = {
            "encode": model_signature.ModelSignature(
                inputs=model_signature.infer_signature(sentences, embeddings).inputs,
                outputs=model_signature.infer_signature(sentences, embeddings).outputs,
                params=[
                    model_signature.ParamSpec(
                        name="truncate_dim",
                        dtype=model_signature.DataType.INT64,
                        default_value=paramspec_truncate_dim,
                    ),
                ],
            )
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                signatures=sig,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(as_custom_model=True)
            assert pk.model is not None

            encode_method = getattr(pk.model, "encode", None)
            assert callable(encode_method)

            test_data = pd.DataFrame({"sentence": ["Hello world"]})
            result = encode_method(test_data)
            self.assertEqual(len(result.iloc[0, 0]), paramspec_truncate_dim)

    def test_truncate_dim_param_runtime_override(self) -> None:
        """Test that the inference method respects a non-default truncate_dim kwarg from params."""
        if not _supports_encode_truncate_dim_param():
            self.skipTest("sentence-transformers version does not support encode(truncate_dim=...)")

        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(as_custom_model=True)
            assert pk.model is not None

            test_data = pd.DataFrame({"sentence": ["Hello world", "Test sentence"]})

            encode_method = getattr(pk.model, "encode", None)
            assert callable(encode_method)
            result = encode_method(test_data, truncate_dim=64)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)
            self.assertEqual(len(result.iloc[0, 0]), 64)

    def test_get_embedding_dim_from_config_standard(self) -> None:
        """Test embedding dimension extraction from standard sentence-transformers config."""
        with tempfile.TemporaryDirectory() as snapshot_dir:
            modules = [
                {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            ]
            with open(os.path.join(snapshot_dir, "modules.json"), "w") as f:
                json.dump(modules, f)

            pooling_dir = os.path.join(snapshot_dir, "1_Pooling")
            os.makedirs(pooling_dir)
            pooling_config = {
                "word_embedding_dimension": 384,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
            }
            with open(os.path.join(pooling_dir, "config.json"), "w") as f:
                json.dump(pooling_config, f)

            dim = _get_embedding_dim_from_config(snapshot_dir)
            self.assertEqual(dim, 384)

    def test_get_embedding_dim_from_config_dense_module(self) -> None:
        """Test that a downstream Dense module overrides the pooling dimension."""
        with tempfile.TemporaryDirectory() as snapshot_dir:
            modules = [
                {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"idx": 2, "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
            ]
            with open(os.path.join(snapshot_dir, "modules.json"), "w") as f:
                json.dump(modules, f)

            pooling_dir = os.path.join(snapshot_dir, "1_Pooling")
            os.makedirs(pooling_dir)
            with open(os.path.join(pooling_dir, "config.json"), "w") as f:
                json.dump(
                    {
                        "word_embedding_dimension": 768,
                        "pooling_mode_mean_tokens": True,
                    },
                    f,
                )

            dense_dir = os.path.join(snapshot_dir, "2_Dense")
            os.makedirs(dense_dir)
            with open(os.path.join(dense_dir, "config.json"), "w") as f:
                json.dump({"in_features": 768, "out_features": 512}, f)

            dim = _get_embedding_dim_from_config(snapshot_dir)
            self.assertEqual(dim, 512)

    def test_get_embedding_dim_from_config_multiple_pooling_modes(self) -> None:
        """Test that multiple enabled pooling modes multiply the dimension."""
        with tempfile.TemporaryDirectory() as snapshot_dir:
            modules = [
                {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            ]
            with open(os.path.join(snapshot_dir, "modules.json"), "w") as f:
                json.dump(modules, f)

            pooling_dir = os.path.join(snapshot_dir, "1_Pooling")
            os.makedirs(pooling_dir)
            pooling_config = {
                "word_embedding_dimension": 768,
                "pooling_mode_cls_token": True,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
            }
            with open(os.path.join(pooling_dir, "config.json"), "w") as f:
                json.dump(pooling_config, f)

            dim = _get_embedding_dim_from_config(snapshot_dir)
            self.assertEqual(dim, 768 * 2)

    def test_get_embedding_dim_from_config_missing_modules_json(self) -> None:
        with tempfile.TemporaryDirectory() as snapshot_dir:
            dim = _get_embedding_dim_from_config(snapshot_dir)
            self.assertIsNone(dim)

    def test_get_embedding_dim_from_config_no_pooling_module(self) -> None:
        with tempfile.TemporaryDirectory() as snapshot_dir:
            modules = [{"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"}]
            with open(os.path.join(snapshot_dir, "modules.json"), "w") as f:
                json.dump(modules, f)

            dim = _get_embedding_dim_from_config(snapshot_dir)
            self.assertIsNone(dim)

    def test_get_embedding_dim_from_config_missing_pooling_config(self) -> None:
        with tempfile.TemporaryDirectory() as snapshot_dir:
            modules = [
                {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            ]
            with open(os.path.join(snapshot_dir, "modules.json"), "w") as f:
                json.dump(modules, f)

            dim = _get_embedding_dim_from_config(snapshot_dir)
            self.assertIsNone(dim)

    def _mock_lazy_st_repo_download(
        self,
        *,
        local_dir: str,
        filename: str,
        modules: list[dict[str, object]],
        pooling_config: Optional[Mapping[str, Any]] = None,
        dense_config: Optional[Mapping[str, Any]] = None,
    ) -> str:
        if filename == "modules.json":
            with open(os.path.join(local_dir, "modules.json"), "w") as f:
                json.dump(modules, f)
            return os.path.join(local_dir, "modules.json")
        if filename == "1_Pooling/config.json" and pooling_config is not None:
            pooling_dir = os.path.join(local_dir, "1_Pooling")
            os.makedirs(pooling_dir, exist_ok=True)
            config_path = os.path.join(pooling_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(pooling_config, f)
            return config_path
        if filename == "2_Dense/config.json" and dense_config is not None:
            dense_dir = os.path.join(local_dir, "2_Dense")
            os.makedirs(dense_dir, exist_ok=True)
            config_path = os.path.join(dense_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(dense_config, f)
            return config_path
        raise AssertionError(f"Unexpected download: {filename}")

    @mock.patch("huggingface_hub.hf_hub_download")
    @mock.patch("huggingface_hub.HfApi")
    def test_save_wrapper_lazy_upload_skips_copytree_and_sets_lazy_hf_upload(
        self,
        mock_hf_api: mock.Mock,
        mock_hf_hub_download: mock.Mock,
    ) -> None:
        """Lazy ST wrapper logging should defer weight copies and attach upload metadata."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        modules = [
            {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        ]
        pooling_config = {"word_embedding_dimension": 384, "pooling_mode_mean_tokens": True}

        def fake_download(
            *,
            repo_id: str,
            filename: str,
            revision: object,
            token: object,
            local_dir: str,
            **kwargs: object,
        ) -> str:
            del repo_id, revision, token, kwargs
            return self._mock_lazy_st_repo_download(
                local_dir=local_dir,
                filename=filename,
                modules=modules,
                pooling_config=pooling_config,
            )

        mock_hf_hub_download.side_effect = fake_download
        mock_hf_api.return_value.model_info.return_value.siblings = [
            mock.Mock(rfilename="modules.json", size=120),
            mock.Mock(rfilename="1_Pooling/config.json", size=200),
            mock.Mock(rfilename="model.safetensors", size=1000),
        ]

        wrapper = snowml_huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
            compute_pool_for_log=None,
            lazy_upload=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            packager = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            with mock.patch("shutil.copytree") as mock_copytree:
                packager.save(
                    name="model",
                    model=wrapper,
                    metadata={"author": "test", "version": "1"},
                    options=model_types.SentenceTransformersSaveOptions(),
                )
                mock_copytree.assert_not_called()

            assert packager.meta is not None
            lazy_hf_upload = getattr(packager.meta, "_lazy_hf_upload", None)
            self.assertIsNotNone(lazy_hf_upload)
            assert lazy_hf_upload is not None
            self.assertEqual(
                lazy_hf_upload.files,
                ["modules.json", "1_Pooling/config.json", "model.safetensors"],
            )
            self.assertEqual(
                lazy_hf_upload.file_sizes,
                {"modules.json": 120, "1_Pooling/config.json": 200, "model.safetensors": 1000},
            )
            blob_options = cast(
                model_meta_schema.SentenceTransformersModelBlobOptions,
                packager.meta.models["model"].options,
            )
            self.assertTrue(blob_options.get("is_repo_downloaded", False))

    @mock.patch("huggingface_hub.hf_hub_download")
    @mock.patch("huggingface_hub.HfApi")
    def test_save_wrapper_lazy_upload_auto_infers_signature(
        self,
        mock_hf_api: mock.Mock,
        mock_hf_hub_download: mock.Mock,
    ) -> None:
        """Lazy ST wrapper save should auto-infer signatures after config download."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        modules = [
            {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        ]
        pooling_config = {"word_embedding_dimension": 384, "pooling_mode_mean_tokens": True}

        def fake_download(
            *,
            repo_id: str,
            filename: str,
            revision: object,
            token: object,
            local_dir: str,
            **kwargs: object,
        ) -> str:
            del repo_id, revision, token, kwargs
            return self._mock_lazy_st_repo_download(
                local_dir=local_dir,
                filename=filename,
                modules=modules,
                pooling_config=pooling_config,
            )

        mock_hf_hub_download.side_effect = fake_download
        mock_hf_api.return_value.model_info.return_value.siblings = [
            mock.Mock(rfilename="modules.json", size=120),
            mock.Mock(rfilename="1_Pooling/config.json", size=200),
            mock.Mock(rfilename="model.safetensors", size=1000),
        ]

        wrapper = snowml_huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
            compute_pool_for_log=None,
            lazy_upload=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=wrapper,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(meta_only=True)
            assert pk.meta is not None

            for method_name in _DEFAULT_WRAPPER_TARGET_METHODS:
                self.assertIn(method_name, pk.meta.signatures)
                sig = pk.meta.signatures[method_name]
                assert isinstance(sig.outputs[0], model_signature.FeatureSpec)
                self.assertEqual(sig.outputs[0]._shape, (384,))

    @mock.patch("huggingface_hub.hf_hub_download")
    @mock.patch("huggingface_hub.HfApi")
    def test_save_wrapper_lazy_upload_dense_module_dim(
        self,
        mock_hf_api: mock.Mock,
        mock_hf_hub_download: mock.Mock,
    ) -> None:
        """Lazy ST wrapper save should prefer Dense out_features when present."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        modules = [
            {"idx": 0, "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
        ]
        pooling_config = {"word_embedding_dimension": 768, "pooling_mode_mean_tokens": True}
        dense_config = {"out_features": 256}

        def fake_download(
            *,
            repo_id: str,
            filename: str,
            revision: object,
            token: object,
            local_dir: str,
            **kwargs: object,
        ) -> str:
            del repo_id, revision, token, kwargs
            return self._mock_lazy_st_repo_download(
                local_dir=local_dir,
                filename=filename,
                modules=modules,
                pooling_config=pooling_config,
                dense_config=dense_config,
            )

        mock_hf_hub_download.side_effect = fake_download
        mock_hf_api.return_value.model_info.return_value.siblings = [
            mock.Mock(rfilename="modules.json", size=120),
            mock.Mock(rfilename="1_Pooling/config.json", size=200),
            mock.Mock(rfilename="2_Dense/config.json", size=150),
            mock.Mock(rfilename="model.safetensors", size=1000),
        ]

        wrapper = snowml_huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
            compute_pool_for_log=None,
            lazy_upload=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=wrapper,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(target_methods=["encode"]),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(meta_only=True)
            assert pk.meta is not None

            sig = pk.meta.signatures["encode"]
            assert isinstance(sig.outputs[0], model_signature.FeatureSpec)
            self.assertEqual(sig.outputs[0]._shape, (256,))

    def test_can_handle_wrapper(self) -> None:
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        wrapper = snowml_huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.assertTrue(SentenceTransformerHandler.can_handle(wrapper))

    def test_can_handle_real_model(self) -> None:
        model = self._st_model
        self.assertTrue(SentenceTransformerHandler.can_handle(model))

    def test_cast_model_wrapper(self) -> None:
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        wrapper = snowml_huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        result = SentenceTransformerHandler.cast_model(wrapper)
        self.assertIs(result, wrapper)

    def test_save_and_load_wrapper_with_snapshot(self) -> None:
        """Test save/load round-trip for a SentenceTransformer wrapper with a real snapshot."""
        import sentence_transformers

        from snowflake.ml.model.models import huggingface as snowml_huggingface

        wrapper = snowml_huggingface.SentenceTransformer(
            model=MODEL_NAMES[0],
            compute_pool_for_log=None,
            lazy_upload=False,
        )
        wrapper.repo_snapshot_dir = self._wrapper_snapshot_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=wrapper,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.model is not None
            assert pk.meta is not None

            assert isinstance(pk.model, sentence_transformers.SentenceTransformer)

            self.assertIn("encode", pk.meta.signatures)
            sig = pk.meta.signatures["encode"]
            self.assertEqual(len(sig.inputs), 1)
            self.assertEqual(len(sig.outputs), 1)

            test_sentences = ["Hello world", "Test sentence"]
            embeddings = pk.model.encode(test_sentences)
            self.assertEqual(len(embeddings), 2)

    def test_save_wrapper_sets_is_repo_downloaded(self) -> None:
        """Test that saving a local-mode wrapper with a snapshot sets is_repo_downloaded in metadata."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        wrapper = snowml_huggingface.SentenceTransformer(
            model=MODEL_NAMES[0],
            compute_pool_for_log=None,
            lazy_upload=False,
        )
        wrapper.repo_snapshot_dir = self._wrapper_snapshot_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=wrapper,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            blob_options = cast(
                model_meta_schema.SentenceTransformersModelBlobOptions,
                pk.meta.models["model"].options,
            )
            self.assertTrue(blob_options.get("is_repo_downloaded", False))
            self.assertEqual(blob_options["model"], MODEL_NAMES[0])

    def test_save_wrapper_without_snapshot_sets_is_repo_downloaded_false(self) -> None:
        """Test that saving a wrapper without a snapshot sets is_repo_downloaded = False."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        expected_dim = self._st_model.get_sentence_embedding_dimension()
        encode_sig = _auto_infer_signature("encode", expected_dim, batch_size=None)
        assert encode_sig is not None

        wrapper = snowml_huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
            compute_pool_for_log="some_pool",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=wrapper,
                signatures={"encode": encode_sig},
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(meta_only=True)
            assert pk.meta is not None

            blob_options = cast(
                model_meta_schema.SentenceTransformersModelBlobOptions,
                pk.meta.models["model"].options,
            )
            self.assertFalse(blob_options.get("is_repo_downloaded", False))
            self.assertEqual(blob_options["model"], "sentence-transformers/all-MiniLM-L6-v2")

    def test_save_wrapper_auto_infers_signature_from_config(self) -> None:
        """Test that saving a wrapper auto-infers signature from snapshot config files."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        expected_dim = self._st_model.get_sentence_embedding_dimension()

        wrapper = snowml_huggingface.SentenceTransformer(
            model=MODEL_NAMES[0],
            compute_pool_for_log=None,
            lazy_upload=False,
        )
        wrapper.repo_snapshot_dir = self._wrapper_snapshot_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=wrapper,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            for method_name in _DEFAULT_WRAPPER_TARGET_METHODS:
                self.assertIn(method_name, pk.meta.signatures)
                sig = pk.meta.signatures[method_name]
                assert isinstance(sig.outputs[0], model_signature.FeatureSpec)
                self.assertEqual(sig.outputs[0]._shape, (expected_dim,))
            for method_name in ("encode_queries", "encode_documents"):
                self.assertNotIn(method_name, pk.meta.signatures)

    def test_save_wrapper_target_methods_subset(self) -> None:
        """Test that wrapper save honors target_methods when provided."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        expected_dim = self._st_model.get_sentence_embedding_dimension()

        wrapper = snowml_huggingface.SentenceTransformer(
            model=MODEL_NAMES[0],
            compute_pool_for_log=None,
            lazy_upload=False,
        )
        wrapper.repo_snapshot_dir = self._wrapper_snapshot_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=wrapper,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(target_methods=["encode"]),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(meta_only=True)
            assert pk.meta is not None

            self.assertEqual(list(pk.meta.signatures.keys()), ["encode"])
            sig = pk.meta.signatures["encode"]
            assert isinstance(sig.outputs[0], model_signature.FeatureSpec)
            self.assertEqual(sig.outputs[0]._shape, (expected_dim,))

    def test_save_wrapper_target_methods_invalid(self) -> None:
        """Test that wrapper save rejects unsupported target_methods."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        wrapper = snowml_huggingface.SentenceTransformer(
            model=MODEL_NAMES[0],
            compute_pool_for_log=None,
            lazy_upload=False,
        )
        wrapper.repo_snapshot_dir = self._wrapper_snapshot_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                    name="model",
                    model=wrapper,
                    metadata={"author": "test", "version": "1"},
                    options=model_types.SentenceTransformersSaveOptions(target_methods=["unsupported_method"]),
                )
            self.assertEqual(
                ctx.exception.args[0],
                "Unsupported model methods: ['unsupported_method']. "
                "SentenceTransformer model methods must be one of: "
                "['encode', 'encode_query', 'encode_document', 'encode_queries', 'encode_documents'].",
            )

    def test_save_wrapper_dependency_unpinned(self) -> None:
        """Test that wrapper save does not pin sentence-transformers to a local version."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        wrapper = snowml_huggingface.SentenceTransformer(
            model=MODEL_NAMES[0],
            compute_pool_for_log=None,
            lazy_upload=False,
        )
        wrapper.repo_snapshot_dir = self._wrapper_snapshot_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=wrapper,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(meta_only=True)
            assert pk.meta is not None

            deps = {
                requirements.Requirement(dep).name: requirements.Requirement(dep)
                for dep in pk.meta.env.conda_dependencies
            }
            self.assertIn("sentence-transformers", deps)
            self.assertFalse(
                any(spec.operator == "==" for spec in deps["sentence-transformers"].specifier),
                "Wrapper should not pin sentence-transformers to an exact version",
            )

    def test_save_wrapper_multi_method_inference(self) -> None:
        """Test wrapper save registers all default methods and each is callable."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        wrapper = snowml_huggingface.SentenceTransformer(
            model=MODEL_NAMES[0],
            compute_pool_for_log=None,
            lazy_upload=False,
        )
        wrapper.repo_snapshot_dir = self._wrapper_snapshot_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=wrapper,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(as_custom_model=True)
            assert pk.model is not None
            assert pk.meta is not None

            test_data = pd.DataFrame({"sentence": ["Test sentence"]})
            for method_name in _DEFAULT_WRAPPER_TARGET_METHODS:
                predict_method = getattr(pk.model, method_name, None)
                self.assertIsNotNone(predict_method, f"Method '{method_name}' should exist")
                assert callable(predict_method)
                result = predict_method(test_data)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertEqual(len(result), 1)

    def test_save_wrapper_local_mode_fails_without_snapshot(self) -> None:
        """Test that local-mode wrapper save fails when no snapshot is available."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        wrapper = snowml_huggingface.SentenceTransformer(
            model="sentence-transformers/all-MiniLM-L6-v2",
            compute_pool_for_log="some_pool",
        )
        wrapper.compute_pool_for_log = None
        wrapper.repo_snapshot_dir = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                ValueError,
                "Unable to determine the model's embedding dimension from snapshot config files. "
                "Please provide sample_input_data or signatures explicitly.",
            ):
                model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                    name="model",
                    model=wrapper,
                    metadata={"author": "test", "version": "1"},
                    options=model_types.SentenceTransformersSaveOptions(),
                )

    def test_save_wrapper_remote_mode_skips_signature_inference(self) -> None:
        """Test that remote-mode wrapper save does not auto-infer signatures from snapshot.

        Remote mode skips signature inference, so a save without explicit signatures fails
        because model metadata requires at least one signature.
        """
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        wrapper = snowml_huggingface.SentenceTransformer(
            model=MODEL_NAMES[0],
            compute_pool_for_log="some_pool",
        )
        wrapper.repo_snapshot_dir = self._wrapper_snapshot_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "The meta data is not ready to save."):
                model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                    name="model",
                    model=wrapper,
                    metadata={"author": "test", "version": "1"},
                    options=model_types.SentenceTransformersSaveOptions(),
                )

    def test_save_wrapper_remote_mode_skips_snapshot_copy(self) -> None:
        """Test that remote-mode wrapper save does not copy snapshot artifacts."""
        from snowflake.ml.model.models import huggingface as snowml_huggingface

        expected_dim = self._st_model.get_sentence_embedding_dimension()
        encode_sig = _auto_infer_signature("encode", expected_dim, batch_size=None)
        assert encode_sig is not None

        wrapper = snowml_huggingface.SentenceTransformer(
            model=MODEL_NAMES[0],
            compute_pool_for_log="some_pool",
        )
        wrapper.repo_snapshot_dir = self._wrapper_snapshot_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=wrapper,
                signatures={"encode": encode_sig},
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(meta_only=True)
            assert pk.meta is not None

            blob_options = pk.meta.models["model"].options
            self.assertFalse(blob_options.get("is_repo_downloaded", False))

            model_blob_dir = os.path.join(tmpdir, "model", "models", "model", "model")
            self.assertFalse(os.path.isfile(os.path.join(model_blob_dir, "modules.json")))

    def test_batch_size_none_warehouse_omits_batch_size(self) -> None:
        """In warehouse (stored procedure), batch_size=None should NOT be passed to model.encode."""
        import sentence_transformers

        model = self._st_model
        embedding_dim = model.get_sentence_embedding_dimension()
        assert embedding_dim is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            test_data = pd.DataFrame({"sentence": ["Hello world", "Test sentence"]})

            def fake_encode(sentences: list[str], **kwargs: Any) -> npt.NDArray[np.float32]:
                return np.zeros((len(sentences), embedding_dim), dtype=np.float32)

            # Patch encode at the class level so the reloaded model instance is covered, and load
            # inside the patch so the inference wrapper captures the patched method.
            with mock.patch(
                "snowflake.snowpark._internal.utils.is_in_stored_procedure", return_value=True
            ), mock.patch.object(
                sentence_transformers.SentenceTransformer, "encode", side_effect=fake_encode
            ) as mock_encode:
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
                pk.load(as_custom_model=True)
                assert pk.model is not None
                encode_method = getattr(pk.model, "encode", None)
                assert callable(encode_method)

                encode_method(test_data)
                mock_encode.assert_called_once()
                self.assertNotIn("batch_size", mock_encode.call_args.kwargs)

    def test_batch_size_none_spcs_omits_batch_size(self) -> None:
        """Outside the warehouse (SPCS/local), batch_size=None is also NOT passed to model.encode."""
        import sentence_transformers

        model = self._st_model
        embedding_dim = model.get_sentence_embedding_dimension()
        assert embedding_dim is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            test_data = pd.DataFrame({"sentence": ["a", "b", "c"]})

            def fake_encode(sentences: list[str], **kwargs: Any) -> npt.NDArray[np.float32]:
                return np.zeros((len(sentences), embedding_dim), dtype=np.float32)

            with mock.patch(
                "snowflake.snowpark._internal.utils.is_in_stored_procedure", return_value=False
            ), mock.patch.object(
                sentence_transformers.SentenceTransformer, "encode", side_effect=fake_encode
            ) as mock_encode:
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
                pk.load(as_custom_model=True)
                assert pk.model is not None
                encode_method = getattr(pk.model, "encode", None)
                assert callable(encode_method)

                encode_method(test_data)
                mock_encode.assert_called_once()
                self.assertNotIn("batch_size", mock_encode.call_args.kwargs)

    def test_batch_size_explicit_override_always_used(self) -> None:
        """Explicit batch_size kwarg should always be used regardless of environment."""
        import sentence_transformers

        model = self._st_model
        embedding_dim = model.get_sentence_embedding_dimension()
        assert embedding_dim is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(batch_size=64),
            )

            test_data = pd.DataFrame({"sentence": ["Hello"]})

            def fake_encode(sentences: list[str], **kwargs: Any) -> npt.NDArray[np.float32]:
                return np.zeros((len(sentences), embedding_dim), dtype=np.float32)

            # Even in the warehouse (stored procedure) the explicit batch_size must win.
            with mock.patch(
                "snowflake.snowpark._internal.utils.is_in_stored_procedure", return_value=True
            ), mock.patch.object(
                sentence_transformers.SentenceTransformer, "encode", side_effect=fake_encode
            ) as mock_encode:
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
                pk.load(as_custom_model=True)
                assert pk.model is not None
                encode_method = getattr(pk.model, "encode", None)
                assert callable(encode_method)

                encode_method(test_data)
                mock_encode.assert_called_once()
                self.assertEqual(mock_encode.call_args.kwargs["batch_size"], 64)

    def test_batch_size_none_empty_input_returns_empty(self) -> None:
        """Empty input with batch_size=None returns an empty frame without error."""
        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load(as_custom_model=True)
            assert pk.model is not None
            encode_method = getattr(pk.model, "encode", None)
            assert callable(encode_method)

            result = encode_method(pd.DataFrame({"sentence": pd.Series([], dtype=object)}))
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 0)

    def test_batch_size_none_warehouse_real_encode_succeeds(self) -> None:
        """Warehouse path omits batch_size; verify real sentence-transformers encode still works."""
        model = self._st_model
        embedding_dim = model.get_sentence_embedding_dimension()
        assert embedding_dim is not None

        with tempfile.TemporaryDirectory() as tmpdir:
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            with mock.patch("snowflake.snowpark._internal.utils.is_in_stored_procedure", return_value=True):
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
                pk.load(as_custom_model=True)
                assert pk.model is not None
                encode_method = getattr(pk.model, "encode", None)
                assert callable(encode_method)

                result = encode_method(pd.DataFrame({"sentence": ["Hello world", "Test sentence"]}))
                self.assertIsInstance(result, pd.DataFrame)
                self.assertEqual(len(result), 2)
                self.assertEqual(len(result.iloc[0, 0]), embedding_dim)

    def test_batch_size_metadata_persistence(self) -> None:
        """Verify blob_options key presence depends on whether batch_size was explicitly set."""
        model = self._st_model

        with tempfile.TemporaryDirectory() as tmpdir:
            # No explicit batch_size → key should be absent from blob_options
            model_packager.ModelPackager(os.path.join(tmpdir, "model_default")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )
            pk_default = model_packager.ModelPackager(os.path.join(tmpdir, "model_default"))
            pk_default.load()
            assert pk_default.meta is not None
            blob_options = pk_default.meta.models["model"].options
            self.assertNotIn("batch_size", blob_options)

            # Explicit batch_size=64 → key should be present
            model_packager.ModelPackager(os.path.join(tmpdir, "model_explicit")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(batch_size=64),
            )
            pk_explicit = model_packager.ModelPackager(os.path.join(tmpdir, "model_explicit"))
            pk_explicit.load()
            assert pk_explicit.meta is not None
            blob_options_explicit = pk_explicit.meta.models["model"].options
            self.assertEqual(blob_options_explicit.get("batch_size"), 64)


if __name__ == "__main__":
    absltest.main()
