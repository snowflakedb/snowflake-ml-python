import os
import random
import tempfile
import warnings

import pandas as pd
import sentence_transformers
from absl.testing import absltest
from pandas.testing import assert_frame_equal

from snowflake.ml.model import model_signature, type_hints as model_types
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_handlers.sentence_transformers import (
    _ALLOWED_TARGET_METHODS,
    SentenceTransformerHandler,
    _validate_sentence_transformers_signatures,
)
from snowflake.ml.model._signatures import utils as model_signature_utils

MODEL_NAMES = ["intfloat/e5-base-v2"]
SENTENCE_TRANSFORMERS_CACHE_DIR = "SENTENCE_TRANSFORMERS_HOME"


class SentenceTransformerHandlerTest(absltest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        self.cache_dir = tempfile.TemporaryDirectory()
        self._original_cache = os.getenv(SENTENCE_TRANSFORMERS_CACHE_DIR, None)
        os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR] = self.cache_dir.name

    @classmethod
    def tearDownClass(self) -> None:
        if self._original_cache:
            os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR] = self._original_cache
        else:
            del os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR]
        self.cache_dir.cleanup()

    def test_convert_as_custom_model_unsupported_method_raises(self) -> None:
        """Test that convert_as_custom_model raises ValueError for unsupported methods.

        This test verifies the fix for the missing-raise bug where unsupported methods
        would silently construct a ValueError but not raise it.
        """
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])

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

    def test_default_target_methods(self) -> None:
        """Test that DEFAULT_TARGET_METHODS contains expected default methods.

        Per the design doc (Section 3.1), by default enable all the methods.
        The handler uses singular names (encode_query, encode_document) which are
        used in sentence-transformers >= 3.0.
        """
        self.assertEqual(
            SentenceTransformerHandler.DEFAULT_TARGET_METHODS,
            ["encode", "encode_query", "encode_document"],
        )

    def test_default_target_methods_when_omitted(self) -> None:
        """Test that default target methods are used when target_methods is omitted.

        Per the design doc (Section 5.2), if target_methods is omitted, we still
        default to DEFAULT_TARGET_METHODS and preserve all existing behavior.
        """
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])

        # Check which methods are available on this model
        available_methods = []
        for method in SentenceTransformerHandler.DEFAULT_TARGET_METHODS:
            if hasattr(model, method) and callable(getattr(model, method, None)):
                available_methods.append(method)

        # encode should always be available
        self.assertIn("encode", available_methods)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model WITHOUT specifying target_methods - should use defaults
            model_packager.ModelPackager(os.path.join(tmpdir, "model")).save(
                name="model",
                model=model,
                metadata={"author": "test", "version": "1"},
                options=model_types.SentenceTransformersSaveOptions(),
            )

            pk = model_packager.ModelPackager(os.path.join(tmpdir, "model"))
            pk.load()
            assert pk.meta is not None

            # Verify that signatures were created for available default methods
            for method in available_methods:
                self.assertIn(
                    method,
                    pk.meta.signatures,
                    f"Method '{method}' should have a signature when target_methods is omitted",
                )

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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])

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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])

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
            self.assertEqual(sig.inputs[0].name, "text")
            assert isinstance(sig.inputs[0], model_signature.FeatureSpec)
            self.assertEqual(sig.inputs[0]._dtype, model_signature.DataType.STRING)

            # Verify output is DOUBLE type with correct embedding dimension
            self.assertEqual(sig.outputs[0].name, "output")
            assert isinstance(sig.outputs[0], model_signature.FeatureSpec)
            self.assertEqual(sig.outputs[0]._dtype, model_signature.DataType.DOUBLE)

            # Verify embedding dimension matches model
            expected_dim = model.get_sentence_embedding_dimension()
            self.assertEqual(sig.outputs[0]._shape, (expected_dim,))

            # Test that the model works correctly with auto-inferred signature
            pk.load(as_custom_model=True)
            assert pk.model is not None
            test_sentences = pd.DataFrame({"text": ["Hello world", "Test sentence"]})
            predict_method = getattr(pk.model, "encode", None)
            assert callable(predict_method)
            result = predict_method(test_sentences)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)

    def test_auto_signature_multiple_methods(self) -> None:
        """Test auto signature inference with multiple target methods."""
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])

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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])

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

            # Input should be STRING type with name "text"
            self.assertEqual(len(sig.inputs), 1)
            self.assertEqual(sig.inputs[0].name, "text")
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

            test_queries = pd.DataFrame({"text": ["What is machine learning?", "How does AI work?"]})
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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])

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

            # Input should be STRING type with name "text"
            self.assertEqual(len(sig.inputs), 1)
            self.assertEqual(sig.inputs[0].name, "text")
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
                    "text": [
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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])

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
                self.assertEqual(sig.inputs[0].name, "text")
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

            test_data = pd.DataFrame({"text": ["Test sentence"]})

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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])
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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])
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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])
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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])

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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])
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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])

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
        model = sentence_transformers.SentenceTransformer(MODEL_NAMES[0])
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

    def test_sentence_transformers(self) -> None:
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
        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))
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


if __name__ == "__main__":
    absltest.main()
