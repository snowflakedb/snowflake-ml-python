import os
import random
import tempfile

import numpy as np
import pandas as pd
from absl.testing import absltest

from snowflake.ml.model._signatures import snowpark_handler
from tests.integ.snowflake.ml.registry.model import registry_model_test_base
from tests.integ.snowflake.ml.test_utils import dataframe_utils

MODEL_NAMES = ["intfloat/e5-base-v2"]  # cant load models in parallel
SENTENCE_TRANSFORMERS_CACHE_DIR = "SENTENCE_TRANSFORMERS_HOME"


class TestRegistrySentenceTransformerModelInteg(registry_model_test_base.RegistryModelTestBase):
    @classmethod
    def setUpClass(self) -> None:
        self.cache_dir = tempfile.TemporaryDirectory()
        self._original_cache_dir = os.getenv(SENTENCE_TRANSFORMERS_CACHE_DIR, None)
        os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR] = self.cache_dir.name

    @classmethod
    def tearDownClass(self) -> None:
        if self._original_cache_dir:
            os.environ[SENTENCE_TRANSFORMERS_CACHE_DIR] = self._original_cache_dir
        self.cache_dir.cleanup()

    def test_sentence_transformers(
        self,
    ) -> None:
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
        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))
        embeddings = pd.DataFrame({"output_feature_0": model.encode(sentences["SENTENCES"].tolist()).tolist()})
        self._test_registry_model(
            model=model,
            sample_input_data=sentences,
            prediction_assert_fns={
                "encode": (
                    sentences,
                    lambda res: res.equals(embeddings),
                ),
            },
            additional_dependencies=["datasets>=2.15"],
        )

    def test_sentence_transformers_auto_signature(self) -> None:
        """Test auto-signature inference without providing sample_input_data.

        This test verifies that when no sample_input_data is provided, the model
        handler automatically infers the signature from the model's embedding
        dimension. It tests all available methods (encode, encode_query, encode_document)
        in a single model log operation to reduce test time.

        The auto-inferred signature expects:
        - Input: a column named "text" with STRING type
        - Output: a column named "output" with DOUBLE type and shape (embedding_dim,)
        """
        import sentence_transformers

        from snowflake.ml.model import type_hints as model_types

        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))

        # Test data must match the auto-inferred signature input column name ("text")
        test_sentences = pd.DataFrame(
            {
                "text": [
                    "Why don't scientists trust atoms? Because they make up everything.",
                    "I told my wife she should embrace her mistakes. She gave me a hug.",
                    "Im reading a book on anti-gravity. Its impossible to put down!",
                ]
            }
        )

        # Build prediction_assert_fns and target_methods based on available methods
        prediction_assert_fns = {}
        target_methods = []

        # encode is always available
        expected_encode = model.encode(test_sentences["text"].tolist())

        def check_encode(res: pd.DataFrame) -> bool:
            if "output" not in res.columns:
                return False
            actual = np.array(res["output"].tolist())
            return np.allclose(actual, expected_encode, atol=1e-6)

        prediction_assert_fns["encode"] = (test_sentences, check_encode)
        target_methods.append("encode")

        # Check for encode_query (sentence-transformers >= 3.0)
        if hasattr(model, "encode_query") and callable(getattr(model, "encode_query", None)):
            expected_query = model.encode_query(test_sentences["text"].tolist())

            def check_query(res: pd.DataFrame) -> bool:
                if "output" not in res.columns:
                    return False
                actual = np.array(res["output"].tolist())
                return np.allclose(actual, expected_query, atol=1e-6)

            prediction_assert_fns["encode_query"] = (test_sentences, check_query)
            target_methods.append("encode_query")

        # Check for encode_document (sentence-transformers >= 3.0)
        if hasattr(model, "encode_document") and callable(getattr(model, "encode_document", None)):
            expected_doc = model.encode_document(test_sentences["text"].tolist())

            def check_doc(res: pd.DataFrame) -> bool:
                if "output" not in res.columns:
                    return False
                actual = np.array(res["output"].tolist())
                return np.allclose(actual, expected_doc, atol=1e-6)

            prediction_assert_fns["encode_document"] = (test_sentences, check_doc)
            target_methods.append("encode_document")

        # Log model once with all available methods and test all of them
        self._test_registry_model(
            model=model,
            sample_input_data=None,  # No sample data - triggers auto-signature inference
            prediction_assert_fns=prediction_assert_fns,
            options=model_types.SentenceTransformersSaveOptions(target_methods=target_methods),
            additional_dependencies=["datasets>=2.15"],
        )

    def test_sentence_transformers_sp(self) -> None:
        import sentence_transformers

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
        sentences_sp = snowpark_handler.SnowparkDataFrameHandler.convert_from_df(self.session, sentences)
        model = sentence_transformers.SentenceTransformer(random.choice(MODEL_NAMES))
        embeddings = pd.DataFrame({"output_feature_0": model.encode(sentences["SENTENCES"].tolist()).tolist()})
        y_df_expected = pd.concat([sentences_sp.to_pandas(), embeddings], axis=1)

        self._test_registry_model(
            model=model,
            sample_input_data=sentences_sp,
            prediction_assert_fns={
                "encode": (
                    sentences_sp,
                    lambda res: dataframe_utils.check_sp_df_res(res, y_df_expected, atol=1e-6),
                ),
            },
            additional_dependencies=["datasets>=2.15"],
        )


if __name__ == "__main__":
    absltest.main()
