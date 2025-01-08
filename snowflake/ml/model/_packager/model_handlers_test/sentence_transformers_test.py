import os
import random
import tempfile
import warnings

import pandas as pd
import sentence_transformers
from absl.testing import absltest
from pandas.testing import assert_frame_equal

from snowflake.ml.model import model_signature
from snowflake.ml.model._packager import model_packager
from snowflake.ml.model._packager.model_handlers.sentence_transformers import (
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

    def test_validate_sentence_transformers_signatures(self) -> None:
        """
        Test the _validate_sentence_transformers_signatures function to ensure it correctly validates
        the signatures of sentence transformers models.

        This test checks the following cases:
        - A valid signature with the correct method name, input, and output types.
        - An invalid signature with an incorrect method name.
        - An invalid signature with multiple input features.
        - An invalid signature with multiple output features.
        - An invalid signature with an input feature having a specific shape.
        - An invalid signature with an input feature of a different data type.

        Raises:
            ValueError: If the signature is invalid.
        """
        valid_signature = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=None)],
            )
        }

        invalid_signature_1 = {
            "another_method": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=None)],
            )
        }

        invalid_signature_2 = {
            "encode": model_signature.ModelSignature(
                inputs=[
                    model_signature.FeatureSpec(name="input1", dtype=model_signature.DataType.STRING, shape=None),
                    model_signature.FeatureSpec(name="input2", dtype=model_signature.DataType.STRING, shape=None),
                ],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=None)],
            )
        }

        invalid_signature_3 = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING, shape=None)],
                outputs=[
                    model_signature.FeatureSpec(name="output1", dtype=model_signature.DataType.FLOAT, shape=None),
                    model_signature.FeatureSpec(name="output2", dtype=model_signature.DataType.FLOAT, shape=None),
                ],
            )
        }

        invalid_signature_4 = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.STRING, shape=(1,))],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=None)],
            )
        }

        invalid_signature_5 = {
            "encode": model_signature.ModelSignature(
                inputs=[model_signature.FeatureSpec(name="input", dtype=model_signature.DataType.INT32, shape=None)],
                outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT, shape=None)],
            )
        }

        _validate_sentence_transformers_signatures(valid_signature)

        with self.assertRaises(ValueError):
            _validate_sentence_transformers_signatures(invalid_signature_1)

        with self.assertRaises(ValueError):
            _validate_sentence_transformers_signatures(invalid_signature_2)

        with self.assertRaises(ValueError):
            _validate_sentence_transformers_signatures(invalid_signature_3)

        with self.assertRaises(ValueError):
            _validate_sentence_transformers_signatures(invalid_signature_4)

        with self.assertRaises(ValueError):
            _validate_sentence_transformers_signatures(invalid_signature_5)

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
            )

            model_packager.ModelPackager(os.path.join(tmpdir, "model2")).save(
                name="model2",
                model=model,
                sample_input_data=sentences,
                metadata={"author": "halu", "version": "2"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("error")

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
