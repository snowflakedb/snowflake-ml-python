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
        self.assertEqual(_ALLOWED_TARGET_METHODS, ["encode", "encode_queries", "encode_documents"])

    def test_validate_sentence_transformers_signatures_valid(self) -> None:
        """Test valid signatures for all supported methods."""
        # Valid signature with encode only
        valid_encode_only = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=None)],
            )
        }
        _validate_sentence_transformers_signatures(valid_encode_only)

        # Valid signature with all three methods
        valid_all_methods = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="text", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(name="embedding", dtype=model_signature.DataType.FLOAT, shape=None)
                ],
            ),
            "encode_queries": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="query", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(
                        name="query_embedding", dtype=model_signature.DataType.FLOAT, shape=None
                    )
                ],
            ),
            "encode_documents": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="doc", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(name="doc_embedding", dtype=model_signature.DataType.FLOAT, shape=None)
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
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=None)],
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
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=None)],
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
                    model_signature.FeatureSpec(name="output1", dtype=model_signature.DataType.FLOAT, shape=None),
                    model_signature.FeatureSpec(name="output2", dtype=model_signature.DataType.FLOAT, shape=None),
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
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=None)],
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
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=None)],
            )
        }
        with self.assertRaises(ValueError) as ctx:
            _validate_sentence_transformers_signatures(invalid_signature)
        self.assertIn("only accepts STRING input", str(ctx.exception))

    def test_sentence_transformers(self) -> None:
        # Sample Data
        sentences = pd.DataFrame(
            {
                "SENTENCES": [
                    "Why don’t scientists trust atoms? Because they make up everything.",
                    "I told my wife she should embrace her mistakes. She gave me a hug.",
                    "Im reading a book on anti-gravity. Its impossible to put down!",
                    "Did you hear about the mathematician who’s afraid of negative numbers?",
                    "Parallel lines have so much in common. It’s a shame they’ll never meet.",
                ]
            }
        )
        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))
        embeddings = pd.DataFrame({"EMBEDDINGS": model.encode(sentences["SENTENCES"].tolist()).tolist()})

        sig = {"encode": model_signature.infer_signature(sentences, embeddings)}

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test whether an unsupported target method by the model class works
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
