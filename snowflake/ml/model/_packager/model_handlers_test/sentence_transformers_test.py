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
    _sentence_transformer_encode,
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
        embeddings = _sentence_transformer_encode(model, sentences)

        sig = {"encode": model_signature.infer_signature(sentences, embeddings)}
        embeddings = model_signature_utils.rename_pandas_df(embeddings, sig["encode"].outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test whether an unsupported target method by the model class works
            with self.assertRaises(ValueError):
                model_packager.ModelPackager(os.path.join(tmpdir, "model1")).save(
                    name="model1",
                    model=model,
                    signatures={**sig, "another_encode": sig["encode"]},
                    metadata={"author": "halu", "version": "1"},
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

                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model2"))
                pk.load()
                assert pk.model
                assert pk.meta
                assert isinstance(pk.model, sentence_transformers.SentenceTransformer)
                embeddings_load = _sentence_transformer_encode(pk.model, sentences)
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
                embeddings_load = _sentence_transformer_encode(pk.model, sentences)
                embeddings_load = model_signature_utils.rename_pandas_df(embeddings_load, sig["encode"].outputs)
                assert_frame_equal(embeddings_load, embeddings)
                pk = model_packager.ModelPackager(os.path.join(tmpdir, "model2"))
                pk.load(as_custom_model=True)
                assert pk.model
                assert pk.meta
                self.assertEqual(sig["encode"], pk.meta.signatures["encode"])
                predict_method = getattr(pk.model, "encode", None)
                assert callable(predict_method)
                embeddings_load = predict_method(sentences)
                assert_frame_equal(embeddings_load, embeddings)

        return None


if __name__ == "__main__":
    absltest.main()
